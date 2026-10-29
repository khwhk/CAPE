import argparse

import json

import sys

from pathlib import Path

import torch

from torch.nn import functional as F

from torch.utils.data import DataLoader, Subset

from exp.recipes import rescue as previous

from exp.recipes.core import inputs, persistence_metrics, subset, write_json

from layers.embedding import TransformerDiffusionConfig

from layers.history import RankHistoryEDM

base = previous.base

EXTRA = dict(
    temporal_response=False,
    temporal_mode="both",
    temporal_bins=24,
    temporal_scale=1.0,
    temporal_cutoff=0.0,
    project_mse=False,
    refine_anchor_weight=0.0,
    refine_smooth_mae=0.0,
)


def arguments(job, output, device="cuda:0"):
    a = previous.arguments(job, output, device)
    for k, v in EXTRA.items():
        vars(a).setdefault(k, v)
    return a


def make_model(args, protocol, config=None):
    old = previous.RescueHistoryEDM
    kw = {k: getattr(args, k) for k in EXTRA if k.startswith("temporal_")}
    previous.RescueHistoryEDM = lambda *a, **k: RankHistoryEDM(
        *a, temporal_period=args.history_period, **kw, **k
    )
    try:
        return previous.make_model(args, protocol, config)
    finally:
        previous.RescueHistoryEDM = old


def load_model(job, args, protocol):
    checkpoint = job.get("checkpoint") or job.get("init_checkpoint")
    saved = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        if checkpoint
        else None
    )
    config = (
        TransformerDiffusionConfig(**saved["model_config"])
        if job.get("checkpoint")
        else None
    )
    model = make_model(args, protocol, config).to(args.device)
    missing = []
    if saved:
        base.adaptive.native.verify_prefix(saved, args, protocol)
        for k in ("history_condition", "history_period"):
            if saved["args"][k] != getattr(args, k):
                raise ValueError("Warm-start history layout changed: " + k)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        allowed = (
            "state_response.temporal.",
            "state_response.history_basis.",
            "state_response.history_shift.",
        )
        if (
            unexpected
            or any(not k.startswith(allowed) for k in missing)
            or (job.get("checkpoint") and missing)
        ):
            raise ValueError(str((missing, unexpected)))
    return model, saved, dict(missing=missing, removed=[])


def selected_parameters(model, scope):
    if scope == "temporal":
        return {
            n
            for n, _ in model.named_parameters()
            if n.startswith("state_response.temporal.")
        }
    if scope == "conditioning":
        return {
            n
            for n, _ in model.named_parameters()
            if n.startswith(("conditioner.", "exogenous.", "state_response."))
        }
    return previous.selected_parameters(model, scope)


def fixed_train(model, data, args):
    chosen = selected_parameters(model, args.train_scope)
    old = {n: p.requires_grad for n, p in model.named_parameters()}
    if not chosen:
        raise ValueError("Empty training scope")
    try:
        for n, p in model.named_parameters():
            p.requires_grad_(n in chosen)
        return previous.original_train(model, data, args)
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(old[n])


def projected_gradients(primary, secondary):
    """Remove the secondary component that conflicts with the MSE gradient."""
    dot = sum((a * b).sum() for a, b in zip(primary, secondary))
    norm = sum(a.square().sum() for a in primary)
    coefficient = dot.clamp_max(0) / norm.clamp_min(1e-20)
    return [a + b - coefficient * a for a, b in zip(primary, secondary)], bool(
        dot.detach() < 0
    )


def fixed_refine(model, data, args):
    model.eval()
    named = dict(model.named_parameters())
    chosen = selected_parameters(model, args.refine_scope)
    if not chosen:
        raise ValueError("Empty refinement scope")
    old = {n: p.requires_grad for n, p in named.items()}
    before = {n: p.detach().clone() for n, p in named.items()}
    for n, p in named.items():
        p.requires_grad_(n in chosen)
    names = sorted(chosen)
    params = [named[n] for n in names]
    ema = [p.detach().clone() for p in params] if args.refine_ema_decay else None
    optimizer = torch.optim.AdamW(
        params, lr=args.refine_lr, weight_decay=args.refine_weight_decay
    )
    scales = persistence_metrics(subset(data, min(2048, len(data))))
    weights, audit = base.adaptive.sampling_weights(data, args)
    history = []
    conflicts = 0
    steps = 0
    try:
        for epoch in range(1, args.refine_epochs + 1):
            ids = base.adaptive.epoch_indices(
                weights, args.refine_origins, args.seed, epoch
            )
            loader = DataLoader(Subset(data, ids), batch_size=args.refine_batch_size)
            torch.manual_seed(args.seed + 5100 + epoch)
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for i, batch in enumerate(loader):
                past, future, target, zero = inputs(batch, args)
                samples = model.differentiable_sample(
                    past,
                    future,
                    zero,
                    args.refine_samples,
                    seed=args.seed + 71000 + epoch * 100000 + i,
                )
                mse, mae, crps = previous.previous.point_losses(
                    samples, target, args.samples
                )
                primary = mse / max(scales["mse"], 1e-6)
                if args.refine_smooth_mae:
                    mae = F.smooth_l1_loss(
                        samples.mean(1), target, beta=args.refine_smooth_mae
                    )
                secondary = args.refine_mae_weight * mae / max(
                    scales["mae"], 1e-6
                ) + args.refine_crps_weight * crps / max(scales["crps"], 1e-6)
                if args.refine_coarse_weight:
                    period = min(args.history_period, target.shape[1])
                    error = F.avg_pool1d(
                        (samples.mean(1) - target).transpose(1, 2),
                        period,
                        ceil_mode=True,
                    )
                    secondary = (
                        secondary
                        + args.refine_coarse_weight
                        * error.square().mean()
                        / max(scales["mse"], 1e-6)
                    )
                if args.refine_denoise_weight:
                    secondary = (
                        secondary
                        + args.refine_denoise_weight
                        * model.distribution_loss(past, future, target, zero)
                    )
                if args.refine_anchor_weight:
                    secondary = secondary + args.refine_anchor_weight * sum(
                        (named[n] - before[n]).square().sum() for n in names
                    )
                loss = primary + secondary
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite rank refinement")
                start = (
                    i
                    // args.refine_accumulate
                    * args.refine_accumulate
                    * args.refine_batch_size
                )
                count = min(
                    args.refine_accumulate * args.refine_batch_size, len(ids) - start
                )
                fraction = len(past) / count
                if args.project_mse:
                    gm = torch.autograd.grad(
                        primary, params, retain_graph=True, allow_unused=True
                    )
                    ga = torch.autograd.grad(secondary, params, allow_unused=True)
                    gm = [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(gm, params)
                    ]
                    ga = [
                        g if g is not None else torch.zeros_like(p)
                        for g, p in zip(ga, params)
                    ]
                    gradients, conflict = projected_gradients(gm, ga)
                    conflicts += conflict
                    steps += 1
                    for p, g in zip(params, gradients):
                        if p.grad is None:
                            p.grad = g * fraction
                        else:
                            p.grad.add_(g, alpha=fraction)
                else:
                    (loss * fraction).backward()
                if (i + 1) % args.refine_accumulate == 0 or i + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        for e, p in zip(ema, params):
                            e.lerp_(p.detach(), 1 - args.refine_ema_decay)
                total += loss.detach().item() * len(past)
            history.append(dict(epoch=epoch, loss=total / len(ids), origins=len(ids)))
            print("rank_refine", history[-1], flush=True)
        if ema is not None:
            with torch.no_grad():
                for p, e in zip(params, ema):
                    p.copy_(e)
    finally:
        for n, p in named.items():
            p.requires_grad_(old[n])
    frozen = all(
        torch.equal(before[n], p.detach()) for n, p in named.items() if n not in chosen
    )
    if not frozen:
        raise RuntimeError("Frozen parameters changed")
    audit.update(
        scope=args.refine_scope,
        project_mse=args.project_mse,
        conflicting_batches=conflicts,
        projected_batches=steps,
        frozen_parameters_unchanged=frozen,
        parameter_change_l2=sum(
            (p.detach() - before[n]).square().sum().item() for n, p in named.items()
        )
        ** 0.5,
        mse_coefficient=1.0,
        note="Training gradient heuristic; no guarantee of test MSE. Strict external admission remains necessary.",
    )
    return history, audit


def run(job, output, device="cuda:0", smoke=False):
    hooks = dict(
        arguments=arguments,
        make_model=make_model,
        load_model=load_model,
        fixed_train=fixed_train,
        fixed_refine=fixed_refine,
        score=previous.previous.local.gap.score,
    )
    old = {k: getattr(base, k) for k in hooks}
    try:
        for k, v in hooks.items():
            setattr(base, k, v)
        result = base.run(job, output, device, smoke)
        result.update(
            worker="history_rank",
            test_lead_metrics=dict(previous.previous.local.gap.lead_metrics),
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        for k, v in old.items():
            setattr(base, k, v)
