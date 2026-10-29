import argparse

import json

import sys

from pathlib import Path

import torch

from torch.nn import functional as F

from exp.recipes import switches as previous

from exp.recipes.core import write_json

from layers.embedding import TransformerDiffusionConfig

from layers.history import BroadHistoryEDM

base, rank = previous.base, previous.rank

MODEL_DEFAULTS = dict(
    multiscale_response=False,
    multiscale_bins=[6, 24, 72],
    multiscale_width=48,
    multiscale_period=24,
    multiscale_dropout=0.0,
    multiscale_shift=0.5,
    multiscale_gain=0.1,
    multiscale_cutoff=0.0,
)

LOSS_DEFAULTS = dict(
    block_mse_weight=0.0,
    worst_block_weight=0.0,
    loss_block_size=24,
    spectral_weight=0.0,
    spectral_power=-1.0,
)


def arguments(job, output, device="cuda:0"):
    args = previous.arguments(job, output, device)
    for k, v in {**MODEL_DEFAULTS, **LOSS_DEFAULTS}.items():
        vars(args).setdefault(k, v)
    return args


def make_model(args, protocol, config=None):
    old = previous.SwitchHistoryEDM
    options = {k: getattr(args, k) for k in MODEL_DEFAULTS}
    previous.SwitchHistoryEDM = lambda *a, **kw: BroadHistoryEDM(*a, **options, **kw)
    try:
        return previous.make_model(args, protocol, config)
    finally:
        previous.SwitchHistoryEDM = old


def load_model(job, args, protocol):
    path = job.get("checkpoint") or job.get("init_checkpoint")
    saved = torch.load(path, map_location="cpu", weights_only=True) if path else None
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
                raise ValueError("Warm-start layout changed: " + k)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        allowed = (
            "state_response.multiscale.",
            "state_response.patch_response.",
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
    if scope == "multiscale":
        return {
            n
            for n, _ in model.named_parameters()
            if n.startswith("state_response.multiscale.")
        }
    if scope == "history_conditioning":
        return {
            n
            for n, _ in model.named_parameters()
            if n.startswith(("history_conditioner.", "exogenous.", "state_response."))
        }
    return previous.selected_parameters(model, scope)


def fixed_train(model, data, args):
    old = rank.selected_parameters
    rank.selected_parameters = selected_parameters
    try:
        return previous.previous.fixed_train(model, data, args)
    finally:
        rank.selected_parameters = old


def point_losses(samples, target, count, args, original):
    mse, mae, crps = previous.point_losses(samples, target, count, args, original)
    mean_error = samples.mean(1) - target
    if args.block_mse_weight or args.worst_block_weight:
        size = min(args.loss_block_size, target.shape[1])
        pooled = F.avg_pool1d(
            samples.squeeze(-1).flatten(0, 1)[:, None], size, ceil_mode=True
        ).squeeze(1)
        pooled = pooled.reshape(samples.shape[0], samples.shape[1], -1)
        truth = F.avg_pool1d(target.transpose(1, 2), size, ceil_mode=True).squeeze(1)
        error = (pooled.mean(1) - truth).square()
        variance = pooled.var(1, unbiased=True)
        corrected = error - variance / samples.shape[1] + variance / count
        mse = mse + args.block_mse_weight * corrected.mean()
        # Training-only worst quarter of lead blocks, not a changed test metric.
        mse = (
            mse
            + args.worst_block_weight
            * error.topk(max(1, error.shape[-1] // 4), dim=-1).values.mean()
        )
    if args.spectral_weight:
        frequency = torch.arange(
            mean_error.shape[1] // 2 + 1, device=target.device, dtype=target.dtype
        )
        weight = (1 + frequency).pow(args.spectral_power)
        weight = weight / weight.mean()
        spectrum = torch.fft.rfft(mean_error.squeeze(-1), norm="ortho")
        mse = mse + args.spectral_weight * (spectrum.abs().square() * weight).mean()
    return mse, mae, crps


def fixed_refine(model, data, args):
    module = previous.previous.point_module
    original, old_scope = module.point_losses, rank.selected_parameters
    module.point_losses = lambda x, y, n: point_losses(x, y, n, args, original)
    rank.selected_parameters = selected_parameters
    try:
        history, audit = rank.fixed_refine(model, data, args)
        audit.update({k: getattr(args, k) for k in LOSS_DEFAULTS})
        audit["loss_data"] = "Training prefix only; uniform test metrics unchanged"
        return history, audit
    finally:
        module.point_losses, rank.selected_parameters = original, old_scope


def run(job, output, device="cuda:0", smoke=False):
    hooks = dict(
        arguments=arguments,
        make_model=make_model,
        load_model=load_model,
        fixed_train=fixed_train,
        fixed_refine=fixed_refine,
        score=rank.previous.previous.local.gap.score,
    )
    old = {k: getattr(base, k) for k in hooks}
    try:
        for k, v in hooks.items():
            setattr(base, k, v)
        result = base.run(job, output, device, smoke)
        result.update(
            worker="history_broad",
            test_lead_metrics=dict(rank.previous.previous.local.gap.lead_metrics),
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        for k, v in old.items():
            setattr(base, k, v)
