import argparse

import json

import sys

from pathlib import Path

import torch

from exp.recipes import horizon as previous

from exp.recipes.core import write_json

from layers.embedding import TransformerDiffusionConfig

from layers.history import SwitchHistoryEDM

rank = previous.previous

base = previous.base

EXTRA = dict(
    patch_response=False,
    response_patch=12,
    response_width=32,
    response_period=24,
    patch_scale=0.5,
    patch_gain=0.0,
    patch_curve=0.0,
    patch_cutoff=0.0,
    typical_mae_tau=0.0,
    tail_mse_weight=0.0,
    tail_mse_threshold=1.0,
    difference_weight=0.0,
    blend_ratio=0.0,
)

MODEL_OPTIONS = tuple(list(EXTRA)[:8])


def arguments(job, output, device="cuda:0"):
    args = previous.arguments(job, output, device)
    for key, value in EXTRA.items():
        vars(args).setdefault(key, value)
    return args


def make_model(args, protocol, config=None):
    old = rank.RankHistoryEDM
    options = {k: getattr(args, k) for k in MODEL_OPTIONS}
    rank.RankHistoryEDM = lambda *a, **kw: SwitchHistoryEDM(*a, **options, **kw)
    try:
        return rank.make_model(args, protocol, config)
    finally:
        rank.RankHistoryEDM = old


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
    audit = dict(missing=[], removed=[])
    if saved:
        base.adaptive.native.verify_prefix(saved, args, protocol)
        for key in ("history_condition", "history_period"):
            if saved["args"][key] != getattr(args, key):
                raise ValueError("Warm-start layout changed: " + key)
        state = dict(saved["state_dict"])
        if job.get("blend_checkpoint") and not job.get("checkpoint"):
            other = torch.load(
                job["blend_checkpoint"], map_location="cpu", weights_only=True
            )
            base.adaptive.native.verify_prefix(other, args, protocol)
            if state.keys() != other["state_dict"].keys() or any(
                state[k].shape != other["state_dict"][k].shape for k in state
            ):
                raise ValueError("Incompatible checkpoint interpolation")
            params = dict(model.named_parameters())
            state = {
                k: (
                    torch.lerp(v, other["state_dict"][k], args.blend_ratio)
                    if k in params
                    else v
                )
                for k, v in state.items()
            }
            audit.update(
                blend_checkpoint=job["blend_checkpoint"],
                blend_ratio=args.blend_ratio,
                inference_models=1,
            )
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = (
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
        audit["missing"] = missing
    return model, saved, audit


original_selected = rank.selected_parameters


def selected_parameters(model, scope):
    if scope == "patch":
        return {
            n
            for n, _ in model.named_parameters()
            if n.startswith("state_response.patch_response.")
        }
    return original_selected(model, scope)


def fixed_train(model, data, args):
    old = rank.selected_parameters
    rank.selected_parameters = selected_parameters
    try:
        return previous.fixed_train(model, data, args)
    finally:
        rank.selected_parameters = old


def point_losses(samples, target, count, args, original):
    mse, mae, crps = previous.weighted_point_losses(
        samples, target, count, args, original
    )
    error = samples.mean(1) - target
    if args.typical_mae_tau:
        weight = (-error.detach().abs() / args.typical_mae_tau).exp()
        mae = (weight * error.abs()).sum() / weight.sum().clamp_min(1e-8)
    if args.tail_mse_weight:
        mse = (
            mse
            + args.tail_mse_weight
            * (error.abs() - args.tail_mse_threshold).clamp_min(0).square().mean()
        )
    if args.difference_weight:
        mse = mse + args.difference_weight * error.diff(dim=1).square().mean()
    return mse, mae, crps


def fixed_refine(model, data, args):
    original = previous.point_module.point_losses
    old_scope = rank.selected_parameters
    previous.point_module.point_losses = lambda x, y, n: point_losses(
        x, y, n, args, original
    )
    rank.selected_parameters = selected_parameters
    try:
        history, audit = rank.fixed_refine(model, data, args)
        audit.update(
            typical_mae_tau=args.typical_mae_tau,
            tail_mse_weight=args.tail_mse_weight,
            difference_weight=args.difference_weight,
            loss_data="training prefix only; reported metrics unchanged",
        )
        return history, audit
    finally:
        previous.point_module.point_losses = original
        rank.selected_parameters = old_scope


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
            worker="history_switches",
            test_lead_metrics=dict(rank.previous.previous.local.gap.lead_metrics),
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        for k, v in old.items():
            setattr(base, k, v)
