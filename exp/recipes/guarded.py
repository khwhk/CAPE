import argparse

import copy

import json

from pathlib import Path

import sys

import torch

from torch.utils.data import DataLoader, Subset

from exp.recipes import local as local

from exp.recipes.core import inputs, persistence_metrics, subset, write_json

from layers.history import GuardedHistoryEDM

from layers.embedding import TransformerDiffusionConfig

from layers.objectives import fair_sample_losses

base = local.gap.base

EXTRA = dict(
    cycle_mix=0.0,
    cycle_decay=72.0,
    cycle_week_mix=0.0,
    state_shift=False,
    state_shift_scale=0.1,
    state_shift_cutoff=0.0,
    state_shift_groups=None,
    guard_weight=10.0,
    guard_checkpoint=None,
)


def arguments(job, output, device="cuda:0"):
    args = local.arguments(job, output, device)
    for k, v in EXTRA.items():
        vars(args).setdefault(k, v)
    return args


def make_model(args, protocol, config=None):
    original = local.HistoryLocalEDM
    options = {k: getattr(args, k) for k in EXTRA if not k.startswith("guard_")}
    options["wind_index"] = (
        protocol["columns"].index("Wspd_w") if "Wspd_w" in protocol["columns"] else 1
    )
    local.HistoryLocalEDM = lambda *a, **kw: GuardedHistoryEDM(*a, **options, **kw)
    try:
        return local.make_model(args, protocol, config)
    finally:
        local.HistoryLocalEDM = original


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
        for field in ("history_condition", "history_period"):
            if saved["args"][field] != getattr(args, field):
                raise ValueError("History layout changed: " + field)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        if (
            unexpected
            or any(not k.startswith("state_response.history_shift.") for k in missing)
            or (job.get("checkpoint") and missing)
        ):
            raise ValueError(str((missing, unexpected)))
    return model, saved, dict(missing=missing, removed=[])


def point_losses(samples, target, evaluation_samples):
    mse, crps = fair_sample_losses(samples, target)
    # Unbiased estimate of MSE of the SAME finite-IID mean used for evaluation.
    mse = mse + samples.var(1, unbiased=True).mean() / evaluation_samples
    return mse, (samples.mean(1) - target).abs().mean(), crps


def constrained_objective(samples, reference_samples, target, scales, args):
    mse, mae, crps = point_losses(samples, target, args.samples)
    reference_mse = point_losses(reference_samples, target, args.samples)[0]
    penalty = (mse - reference_mse).clamp_min(0) / max(scales["mse"], 1e-6)
    loss = (
        mse / max(scales["mse"], 1e-6)
        + args.refine_mae_weight * mae / max(scales["mae"], 1e-6)
        + args.guard_weight * penalty
        + args.refine_crps_weight * crps / max(scales["crps"], 1e-6)
    )
    return loss, penalty


def update_state_audit(groups, past, prediction, target):
    y = past[..., :1]
    n = max(2, past.shape[1] // 8)
    scale = y.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
    low = y[:, -n:].mean((1, 2)) < 0
    calm = (y[:, -n:].diff(dim=1).square().mean(1).sqrt() / scale).flatten() < 0.25
    residual = prediction - target
    for i in range(len(past)):
        name = ("low" if low[i] else "high") + ("_calm" if calm[i] else "_changing")
        g = groups.setdefault(
            name, dict(origins=0, points=0, error=0.0, absolute=0.0, squared=0.0)
        )
        e = residual[i]
        g["origins"] += 1
        g["points"] += e.numel()
        g["error"] += e.sum().item()
        g["absolute"] += e.abs().sum().item()
        g["squared"] += e.square().sum().item()


def fixed_refine(model, data, args):
    saved = (
        torch.load(args.guard_checkpoint, map_location="cpu", weights_only=True)
        if args.guard_checkpoint
        else None
    )
    if saved:
        refargs = arguments(dict(arguments=saved["args"]), "/unused", args.device)
        with torch.random.fork_rng(devices=[]):
            reference = make_model(
                refargs,
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
    scope = args.refine_scope
    prefixes = {
        "output_head": ("denoiser.output.", "denoiser.global_denoiser.output."),
        "response_shift": ("state_response.response.2.",),
        "state_shift": ("state_response.history_shift.",),
    }
    selected = (
        set(params)
        if scope == "all"
        else {n for n in params if n.startswith(prefixes[scope])}
    )
    if not selected:
        raise ValueError("Empty refinement scope")
    original_grad = {n: p.requires_grad for n, p in params.items()}
    before = {n: p.detach().cpu().clone() for n, p in params.items()}
    handles = []
    for n, p in params.items():
        p.requires_grad_(n in selected)
        if scope == "response_shift" and n in selected:
            handles.append(
                p.register_hook(
                    lambda g: torch.cat((g[:1], torch.zeros_like(g[1:])), 0)
                )
            )
    scales = persistence_metrics(subset(data, min(2048, len(data))))
    weights, audit = base.adaptive.sampling_weights(data, args)
    optimizer = torch.optim.AdamW(
        [params[n] for n in sorted(selected)], lr=args.refine_lr, weight_decay=0.0
    )
    history, groups = [], {}
    try:
        for epoch in range(1, args.refine_epochs + 1):
            ids = base.adaptive.epoch_indices(
                weights, args.refine_origins, args.seed, epoch
            )
            loader = DataLoader(Subset(data, ids), batch_size=args.refine_batch_size)
            torch.manual_seed(args.seed + 5100 + epoch)
            optimizer.zero_grad(set_to_none=True)
            total, penalties = 0.0, 0.0
            for i, batch in enumerate(loader):
                past, future, target, zero = inputs(batch, args)
                noise_seed = args.seed + 71000 + epoch * 100000 + i
                with torch.no_grad():
                    reference_samples = reference.differentiable_sample(
                        past, future, zero, args.refine_samples, seed=noise_seed
                    )
                    if epoch == 1:
                        update_state_audit(
                            groups, past, reference_samples.mean(1), target
                        )
                samples = model.differentiable_sample(
                    past, future, zero, args.refine_samples, seed=noise_seed
                )
                loss, penalty = constrained_objective(
                    samples, reference_samples, target, scales, args
                )
                loss = loss + args.refine_denoise_weight * model.distribution_loss(
                    past, future, target, zero
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite guarded loss")
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
                    torch.nn.utils.clip_grad_norm_([params[n] for n in selected], 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total += loss.detach().item() * len(past)
                penalties += penalty.detach().item() * len(past)
            history.append(
                dict(
                    epoch=epoch,
                    loss=total / len(ids),
                    mse_guard=penalties / len(ids),
                    origins=len(ids),
                )
            )
            print("guarded_refine", history[-1], flush=True)
    finally:
        for h in handles:
            h.remove()
        for n, p in params.items():
            p.requires_grad_(original_grad[n])
    frozen_ok = all(
        torch.equal(before[n], p.detach().cpu())
        for n, p in params.items()
        if n not in selected
    )
    gain_ok = scope != "response_shift" or all(
        torch.equal(before[n][1:], params[n].detach().cpu()[1:]) for n in selected
    )
    if not frozen_ok or not gain_ok:
        raise RuntimeError("Frozen parameter changed")
    audit.update(
        parameter_scope=dict(
            scope=scope,
            names=sorted(selected),
            frozen_unchanged=frozen_ok,
            gain_unchanged=gain_ok,
        ),
        guard_checkpoint=args.guard_checkpoint,
        mse_coefficient=1.0,
        guard_weight=args.guard_weight,
        paired_sampling=True,
        reference_state_errors={
            k: dict(
                origins=g["origins"],
                bias=g["error"] / g["points"],
                mae=g["absolute"] / g["points"],
                mse=g["squared"] / g["points"],
            )
            for k, g in groups.items()
        },
        state_groups="Train only; recent Y mean below/above zero in train-prefix standardized units; recent roughness/history std below/above .25.",
    )
    return history, audit


def run(job, output, device="cuda:0", smoke=False):
    hooks = dict(
        arguments=arguments,
        make_model=make_model,
        load_model=load_model,
        fixed_refine=fixed_refine,
        score=local.gap.score,
    )
    previous = {k: getattr(base, k) for k in hooks}
    try:
        for k, v in hooks.items():
            setattr(base, k, v)
        result = base.run(job, output, device, smoke)
        result.update(
            worker="history_guarded",
            test_lead_metrics=dict(local.gap.lead_metrics),
            source_legacy_validation_selected=job.get(
                "source_legacy_validation_selected", False
            ),
            guard_note="Training constraint only; strict paired-seed MSE admission is enforced by controller.",
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        for k, v in previous.items():
            setattr(base, k, v)
