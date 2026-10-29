import argparse

import json

from pathlib import Path

import sys

import torch

from torch.utils.data import DataLoader

from exp.recipes import explore as base

from exp.recipes.core import inputs, write_json

from layers.embedding import TransformerDiffusionConfig

from layers.history import HistoryGapEDM

from utils.metrics import evaluate_samples

original_arguments, original_model = base.arguments, base.make_model

lead_metrics = {}


def arguments(job, output, device="cuda:0"):
    args = original_arguments(job, output, device)
    vars(args).setdefault("target_anchor", "original")
    vars(args).setdefault("accumulated_history", False)
    return args


def make_model(args, protocol, config=None):
    original = base.HistoryExploreEDM
    base.HistoryExploreEDM = lambda *a, **kw: HistoryGapEDM(
        *a,
        target_anchor=args.target_anchor,
        accumulated_history=args.accumulated_history,
        **kw
    )
    try:
        return original_model(args, protocol, config)
    finally:
        base.HistoryExploreEDM = original


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
    changes = dict(missing=[], removed=[])
    if saved:
        base.adaptive.native.verify_prefix(saved, args, protocol)
        if job.get("checkpoint"):
            model.load_state_dict(saved["state_dict"], strict=True)
        else:
            # These targeted warm starts never change history-condition type/period.
            for field in ("history_condition", "history_period"):
                if saved["args"][field] != getattr(args, field):
                    raise ValueError("Warm-start history layout changed: " + field)
            missing, unexpected = model.load_state_dict(
                saved["state_dict"], strict=False
            )
            if unexpected or any(
                not k.startswith("exogenous.accumulation.") for k in missing
            ):
                raise ValueError(str((missing, unexpected)))
            changes["missing"] = missing
    return model, saved, changes


@torch.no_grad()
def score(model, data, args, seed, samples):
    model.eval()
    forecasts, truths = [], []
    for i, batch in enumerate(DataLoader(data, batch_size=args.eval_batch_size)):
        past, future, target, zero = inputs(batch, args)
        forecasts.append(
            model.sample(
                past,
                future,
                zero,
                n_samples=samples,
                seed=seed + i,
                sample_batch_size=args.sample_batch_size,
            ).cpu()
        )
        truths.append(target.cpu())
    forecasts, truths = torch.cat(forecasts), torch.cat(truths)
    lead_metrics.clear()
    for name, index in zip(
        ("near", "middle", "far"), torch.tensor_split(torch.arange(truths.shape[1]), 3)
    ):
        lead_metrics[name] = dict(
            first_lead=int(index[0]) + 1,
            last_lead=int(index[-1]) + 1,
            metrics=evaluate_samples(forecasts[:, :, index], truths[:, index]),
        )
    return evaluate_samples(forecasts, truths)


def run(job, output, device="cuda:0", smoke=False):
    replacements = dict(
        arguments=arguments, make_model=make_model, load_model=load_model, score=score
    )
    saved_hooks = {k: getattr(base, k) for k in replacements}
    try:
        for k, v in replacements.items():
            setattr(base, k, v)
        result = base.run(job, output, device, smoke)
        result.update(worker="history_gap", test_lead_metrics=dict(lead_metrics))
        write_json(Path(output) / "summary.json", result)
        return result
    finally:
        for k, v in saved_hooks.items():
            setattr(base, k, v)
