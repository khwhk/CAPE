import argparse

import json

import sys

from pathlib import Path

import torch

from torch.nn import functional as F

from torch.utils.data import DataLoader

from exp.recipes import rank as previous

from exp.recipes.core import inputs, subset, write_json

base = previous.base

point_module = previous.previous.previous

EXTRA = dict(
    residual_gain=0.0,
    residual_ridge=1.0,
    residual_origins=1024,
    residual_samples=32,
    lead_weight=0.0,
    target_tail_weight=0.0,
)


def arguments(job, output, device="cuda:0"):
    a = previous.arguments(job, output, device)
    for k, v in EXTRA.items():
        vars(a).setdefault(k, v)
    return a


def ridge_delta(x, y, ridge):
    """Fit in standardized training-feature coordinates, return raw-coordinate map."""
    x, y = x.double(), y.double()
    xm, ym = x.mean(0), y.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(0.05)
    z = (x - xm) / scale
    gram = z.T @ z / len(z) + ridge * torch.eye(
        z.shape[1], device=z.device, dtype=z.dtype
    )
    coefficient = torch.linalg.solve(gram, z.T @ (y - ym) / len(z)) / scale[:, None]
    return coefficient, ym - xm @ coefficient


@torch.no_grad()
def initialize_response(model, data, args):
    """Initialize an INTERNAL denoiser branch using training-origin mean residuals."""
    model.eval()
    head = model.state_response.temporal.smooth
    features, errors = [], []
    loader = DataLoader(
        subset(data, min(args.residual_origins, len(data))), batch_size=8
    )
    for i, batch in enumerate(loader):
        past, future, target, zero = inputs(batch, args)
        generated = model.sample(
            past,
            future,
            zero,
            n_samples=args.residual_samples,
            seed=args.seed + 83000 + i,
            sample_batch_size=args.sample_batch_size,
        )
        mean, scale = model.coordinates(past)
        history = torch.cat(((past[..., :1] - mean) / scale, past[..., 1:]), -1)
        features.append(
            F.adaptive_avg_pool1d(
                history.transpose(1, 2), model.state_response.temporal.bins
            )
            .flatten(1)
            .cpu()
        )
        residual = (target - generated.mean(1)) / scale
        errors.append(
            F.adaptive_avg_pool1d(residual.transpose(1, 2), head.out_features)
            .squeeze(1)
            .cpu()
        )
    x, y = torch.cat(features), torch.cat(errors)
    coefficient, bias = ridge_delta(x, y, args.residual_ridge)
    head.weight.add_(coefficient.T.to(head.weight), alpha=args.residual_gain)
    head.bias.add_(bias.to(head.bias), alpha=args.residual_gain)
    return dict(
        stage="train_residual_initialization",
        origins=len(x),
        samples=args.residual_samples,
        ridge=args.residual_ridge,
        gain=args.residual_gain,
        coefficient_norm=float(coefficient.norm()),
        source="original training prefix only",
        inference="internal temporal response at every denoising step; no posthoc sample correction",
    )


def fixed_train(model, data, args):
    history = []
    if args.residual_gain:
        audit = initialize_response(model, data, args)
        history.append(audit)
        print("response_initialization", audit, flush=True)
    return history + previous.fixed_train(model, data, args)


def weighted_point_losses(samples, target, evaluation_samples, args, original):
    mse, mae, crps = original(samples, target, evaluation_samples)
    if not args.lead_weight and not args.target_tail_weight:
        return mse, mae, crps
    lead = torch.linspace(
        -1, 1, target.shape[1], device=target.device, dtype=target.dtype
    )[None, :, None]
    weight = (1 + args.lead_weight * lead) * (
        1 + args.target_tail_weight * target.abs().clamp_max(3)
    )
    weight = weight / weight.mean()
    variance = samples.var(1, unbiased=True)
    # Weights depend on training labels/lead, NEVER on the random sample residual.
    mse = (
        (samples.mean(1) - target).square()
        - variance / samples.shape[1]
        + variance / evaluation_samples
    )
    return (mse * weight).mean(), mae, crps


def fixed_refine(model, data, args):
    original = point_module.point_losses
    point_module.point_losses = lambda samples, target, count: weighted_point_losses(
        samples, target, count, args, original
    )
    try:
        history, audit = previous.fixed_refine(model, data, args)
        audit.update(
            lead_weight=args.lead_weight,
            target_tail_weight=args.target_tail_weight,
            evaluation_weights="uniform; training weights do not alter reported metrics",
        )
        return history, audit
    finally:
        point_module.point_losses = original


def run(job, output, device="cuda:0", smoke=False):
    hooks = dict(
        arguments=arguments,
        make_model=previous.make_model,
        load_model=previous.load_model,
        fixed_train=fixed_train,
        fixed_refine=fixed_refine,
        score=previous.previous.previous.local.gap.score,
    )
    old = {k: getattr(base, k) for k in hooks}
    try:
        for k, v in hooks.items():
            setattr(base, k, v)
        result = base.run(job, output, device, smoke)
        result.update(
            worker="history_horizon",
            test_lead_metrics=dict(previous.previous.previous.local.gap.lead_metrics),
        )
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        for k, v in old.items():
            setattr(base, k, v)
