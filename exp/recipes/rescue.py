import argparse

import copy

import json

from pathlib import Path

import sys

import torch

from torch.nn import functional as F

from torch.utils.data import DataLoader, Subset

from exp.recipes import guarded as previous

from exp.recipes.core import inputs, persistence_metrics, subset, write_json

from layers.embedding import TransformerDiffusionConfig

from layers.history import RescueHistoryEDM

base = previous.base

EXTRA = dict(
    response_basis=False,
    basis_rank=16,
    basis_regimes=1,
    basis_scale=0.5,
    basis_cutoff=0.0,
    train_scope="all",
    refine_coarse_weight=0.0,
    refine_ema_decay=0.0,
    batch_guard_margin=0.01,
)

original_train = base.fixed_train


def arguments(job, output, device="cuda:0"):
    a = previous.arguments(job, output, device)
    for k, v in EXTRA.items():
        vars(a).setdefault(k, v)
    return a


def make_model(args, protocol, config=None):
    old = previous.GuardedHistoryEDM
    options = {
        k: getattr(args, k)
        for k in EXTRA
        if k.startswith("basis_") or k == "response_basis"
    }
    previous.GuardedHistoryEDM = lambda *a, **kw: RescueHistoryEDM(
        *a, basis_period=args.history_period, **options, **kw
    )
    try:
        return previous.make_model(args, protocol, config)
    finally:
        previous.GuardedHistoryEDM = old


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
        for key in ("history_condition", "history_period"):
            if saved["args"][key] != getattr(args, key):
                raise ValueError("Warm-start history layout changed: " + key)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        allowed = ("state_response.history_basis.", "state_response.history_shift.")
        if (
            unexpected
            or any(not k.startswith(allowed) for k in missing)
            or (job.get("checkpoint") and missing)
        ):
            raise ValueError(str((missing, unexpected)))
    return model, saved, dict(missing=missing, removed=[])


def selected_parameters(model, scope):
    prefixes = {
        "basis": ("state_response.history_basis.",),
        "response": ("state_response.",),
        "output_head": ("denoiser.output.", "denoiser.global_denoiser.output."),
    }
    chosen = {
        n
        for n, _ in model.named_parameters()
        if scope == "all" or n.startswith(prefixes[scope])
    }
    if not chosen:
        raise ValueError("Empty scope: " + scope)
    return chosen


def fixed_train(model, data, args):
    chosen = selected_parameters(model, args.train_scope)
    old = {n: p.requires_grad for n, p in model.named_parameters()}
    try:
        for n, p in model.named_parameters():
            p.requires_grad_(n in chosen)
        return original_train(model, data, args)
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(old[n])


def objective(samples, ref_samples, target, scales, args):
    mse, mae, crps = previous.point_losses(samples, target, args.samples)
    loss = mse / max(scales["mse"], 1e-6) + args.refine_mae_weight * mae / max(
        scales["mae"], 1e-6
    )
    loss = loss + args.refine_crps_weight * crps / max(scales["crps"], 1e-6)
    penalty = samples.new_zeros(())
    if args.guard_weight:
        reference = previous.point_losses(ref_samples, target, args.samples)[0]
        penalty = (mse - reference - args.batch_guard_margin * scales["mse"]).clamp_min(
            0
        ) / max(scales["mse"], 1e-6)
        loss = loss + args.guard_weight * penalty
    if args.refine_coarse_weight:
        period = min(args.history_period, target.shape[1])
        predicted = F.avg_pool1d(
            samples.mean(1).transpose(1, 2), period, ceil_mode=True
        )
        actual = F.avg_pool1d(target.transpose(1, 2), period, ceil_mode=True)
        loss = loss + args.refine_coarse_weight * (
            predicted - actual
        ).square().mean() / max(scales["mse"], 1e-6)
    return loss, penalty


def fixed_refine(model, data, args):
    reference = None
    if args.guard_weight:
        if args.guard_checkpoint:
            saved = torch.load(
                args.guard_checkpoint, map_location="cpu", weights_only=True
            )
            a = arguments(dict(arguments=saved["args"]), "/unused", args.device)
            with torch.random.fork_rng(devices=[]):
                reference = make_model(
                    a,
                    saved["protocol"],
                    TransformerDiffusionConfig(**saved["model_config"]),
                ).to(args.device)
            reference.load_state_dict(saved["state_dict"], strict=True)
            reference.config = model.config
        else:
            reference = copy.deepcopy(model)
        reference.eval()
    model.eval()
    params = dict(model.named_parameters())
    chosen = selected_parameters(model, args.refine_scope)
    old_grad = {n: p.requires_grad for n, p in params.items()}
    before = {n: p.detach().cpu().clone() for n, p in params.items()}
    for n, p in params.items():
        p.requires_grad_(n in chosen)
    ema = (
        {n: params[n].detach().clone() for n in chosen}
        if args.refine_ema_decay
        else None
    )
    optimizer = torch.optim.AdamW(
        [params[n] for n in sorted(chosen)],
        lr=args.refine_lr,
        weight_decay=args.refine_weight_decay,
    )
    scales = persistence_metrics(subset(data, min(2048, len(data))))
    weights, audit = base.adaptive.sampling_weights(data, args)
    history = []
    try:
        for epoch in range(1, args.refine_epochs + 1):
            ids = base.adaptive.epoch_indices(
                weights, args.refine_origins, args.seed, epoch
            )
            loader = DataLoader(Subset(data, ids), batch_size=args.refine_batch_size)
            torch.manual_seed(args.seed + 5100 + epoch)
            optimizer.zero_grad(set_to_none=True)
            total, guard = 0.0, 0.0
            for i, batch in enumerate(loader):
                past, future, target, zero = inputs(batch, args)
                seed = args.seed + 71000 + epoch * 100000 + i
                ref_samples = None
                if reference is not None:
                    with torch.no_grad():
                        ref_samples = reference.differentiable_sample(
                            past, future, zero, args.refine_samples, seed=seed
                        )
                samples = model.differentiable_sample(
                    past, future, zero, args.refine_samples, seed=seed
                )
                loss, penalty = objective(samples, ref_samples, target, scales, args)
                loss = loss + args.refine_denoise_weight * model.distribution_loss(
                    past, future, target, zero
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite rescue refinement")
                start = (
                    i
                    // args.refine_accumulate
                    * args.refine_accumulate
                    * args.refine_batch_size
                )
                count = min(
                    args.refine_accumulate * args.refine_batch_size, len(ids) - start
                )
                (loss * len(past) / count).backward()
                if (i + 1) % args.refine_accumulate == 0 or i + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_([params[n] for n in chosen], 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        for n in chosen:
                            ema[n].lerp_(params[n].detach(), 1 - args.refine_ema_decay)
                total += loss.detach().item() * len(past)
                guard += penalty.detach().item() * len(past)
            history.append(
                dict(
                    epoch=epoch,
                    loss=total / len(ids),
                    soft_batch_guard=guard / len(ids),
                    origins=len(ids),
                )
            )
            print("rescue_refine", history[-1], flush=True)
        if ema is not None:
            with torch.no_grad():
                for n in chosen:
                    params[n].copy_(ema[n])
    finally:
        for n, p in params.items():
            p.requires_grad_(old_grad[n])
    frozen = all(
        torch.equal(before[n], p.detach().cpu())
        for n, p in params.items()
        if n not in chosen
    )
    if not frozen:
        raise RuntimeError("Frozen parameters changed")
    change = (
        sum(
            (p.detach().cpu() - before[n]).square().sum().item()
            for n, p in params.items()
        )
        ** 0.5
    )
    audit.update(
        scope=args.refine_scope,
        frozen_parameters_unchanged=frozen,
        parameter_change_l2=change,
        mse_coefficient=1.0,
        guard_weight=args.guard_weight,
        guard_margin=args.batch_guard_margin,
        guard_reduction="batch-average, never per-origin hinge",
        reference_inference_model=False,
        final_weights="fixed-budget EMA" if ema else "fixed-budget last weights",
    )
    return history, audit


def run(job, output, device="cuda:0", smoke=False):
    hooks = dict(
        arguments=arguments,
        make_model=make_model,
        load_model=load_model,
        fixed_train=fixed_train,
        fixed_refine=fixed_refine,
        score=previous.local.gap.score,
    )
    old = {k: getattr(base, k) for k in hooks}
    try:
        for k, v in hooks.items():
            setattr(base, k, v)
        result = base.run(job, output, device, smoke)
        result.update(
            worker="history_rescue",
            test_lead_metrics=dict(previous.local.gap.lead_metrics),
            source_legacy_validation_selected=job.get(
                "source_legacy_validation_selected", False
            ),
            selection_note="Training exploration is unconstrained or softly batch-constrained; publication requires strict 3-seed MSE non-regression.",
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        for k, v in old.items():
            setattr(base, k, v)
