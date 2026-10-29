import argparse

import copy

import json

import random

import sys

from dataclasses import asdict, replace

from pathlib import Path

import numpy as np

import torch

from torch.utils.data import DataLoader, Subset

from exp.recipes import adaptive as adaptive

from exp.recipes.core import (
    inputs,
    load_windows,
    persistence_metrics,
    score,
    subset,
    train_epoch,
    write_json,
)

from layers.embedding import TransformerDiffusionConfig

from layers.history import HistoryExploreEDM

from layers.objectives import fair_sample_losses


def make_model(args, protocol, config=None):
    original = adaptive.HistoryAdaptiveEDM
    adaptive.HistoryAdaptiveEDM = lambda *a, **kw: HistoryExploreEDM(
        *a,
        multiscale_history=args.multiscale_history,
        channel_response=args.channel_response,
        **kw
    )
    try:
        return adaptive.make_model(args, protocol, config)
    finally:
        adaptive.HistoryAdaptiveEDM = original


def fixed_train(model, data, args):
    history = []
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, args.epochs), eta_min=args.lr * 0.05
    )
    ema = copy.deepcopy(model).eval() if args.ema_decay else None
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        value = train_epoch(model, loader, optimizer, args, ema=ema)
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite training loss; no silent fallback")
        scheduler.step()
        history.append(dict(epoch=epoch, loss=value))
        print("fixed_train", history[-1], flush=True)
    if ema is not None and args.epochs:
        model.load_state_dict(ema.state_dict())
    return history


def fixed_refine(model, data, args):
    history = []
    reference = persistence_metrics(subset(data, min(2048, len(data))))
    weights, audit = adaptive.sampling_weights(data, args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.refine_lr, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.refine_epochs + 1):
        ids = adaptive.epoch_indices(weights, args.refine_origins, args.seed, epoch)
        loader = DataLoader(Subset(data, ids), batch_size=args.refine_batch_size)
        torch.manual_seed(args.seed + 5100 + epoch)
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for i, batch in enumerate(loader):
            past, future, target, base = inputs(batch, args)
            samples = model.differentiable_sample(
                past, future, base, args.refine_samples
            )
            mse, crps = fair_sample_losses(samples, target)
            mae = (samples.mean(1) - target).abs().mean()
            loss = (
                (1 - args.refine_mae_weight) * mse / max(reference["mse"], 1e-6)
                + args.refine_mae_weight * mae / max(reference["mae"], 1e-6)
                + args.refine_crps_weight * crps / max(reference["crps"], 1e-6)
                + args.refine_denoise_weight
                * model.distribution_loss(past, future, target, base)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite refinement loss; no silent fallback"
                )
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(past)
        history.append(dict(epoch=epoch, loss=total / len(ids), origins=len(ids)))
        print("fixed_refine", history[-1], flush=True)
    return history, audit


def arguments(job, output, device="cuda:0"):
    args = adaptive.parser().parse_args(["--output-dir", output])
    vars(args).update(multiscale_history=False, channel_response=False)
    vars(args).update(job["arguments"])
    args.output_dir, args.device, args.threads = output, device, 2
    args.evaluate_test, args.validation_fold = True, 2
    return args


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
        adaptive.native.verify_prefix(saved, args, protocol)
        state = dict(saved["state_dict"])
        if job.get("checkpoint"):
            model.load_state_dict(state, strict=True)
        else:
            if (
                saved["args"].get("history_condition") != args.history_condition
                or saved["args"].get("history_period") != args.history_period
            ):
                changes["removed"] = [
                    k
                    for k in state
                    if k.startswith(("history_conditioner.", "exogenous.history_bank."))
                ]
                state = {k: v for k, v in state.items() if k not in changes["removed"]}
            missing, unexpected = model.load_state_dict(state, strict=False)
            allowed = (
                "history_conditioner.",
                "exogenous.history_bank.",
                "exogenous.multiscale.",
                "state_response.reliability.",
                "state_response.attention.gate.",
                "history_trust.",
            )
            if unexpected or any(not k.startswith(allowed) for k in missing):
                raise ValueError(str((missing, unexpected)))
            changes["missing"] = missing
    return model, saved, changes


def run(job, output, device="cuda:0", smoke=False):
    args = arguments(job, output, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    datasets, protocol = load_windows(args)
    model, saved, changes = load_model(job, args, protocol)
    if smoke:
        datasets = {k: subset(v, min(4, len(v))) for k, v in datasets.items()}
        args.epochs = min(1, args.epochs)
        args.refine_epochs = min(1, args.refine_epochs)
        args.refine_origins = 4
        args.refine_samples = 2
        args.samples = 4
        args.refine_batch_size = 1
        args.batch_size = 2
        args.eval_batch_size = 2
        model.config = replace(model.config, diffusion_steps=4)
    history, audit = {}, {}
    if job.get("checkpoint"):
        selection = dict(saved["selection"], exploration_evaluation_only=True)
    else:
        history["training"] = fixed_train(model, datasets["train"], args)
        if args.refine_epochs:
            history["refinement"], audit = fixed_refine(model, datasets["train"], args)
        selection = dict(
            epoch=args.epochs,
            refine_epoch=args.refine_epochs,
            checkpoint_policy="fixed budget final EMA/weights; validation never selects or rolls back",
            warm_start=job.get("init_checkpoint"),
        )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        args=vars(args),
        protocol=protocol,
        model_config=asdict(model.config),
        selection=selection,
        worker="history_explore",
        evaluation_protocol="TEST_TUNED_EXPLORATORY",
        validation_selection=False,
        test_gradients=False,
        future_z_used=False,
        smoke=smoke,
        warm_start_changes=changes,
        sampling_audit=audit,
        parameters=sum(p.numel() for p in model.parameters()),
        posthoc_sample_centering=False,
    )
    if job.get("checkpoint"):
        checkpoint_ref = job["checkpoint"]
    else:
        checkpoint_ref = str(destination / "model.pt")
        torch.save(dict(state_dict=model.state_dict(), **metadata), checkpoint_ref)
    write_json(destination / "history.json", history)
    # Full test, same seed/solver/sample count as the existing benchmark.
    metrics = dict(
        test=score(model, datasets["test"], args, args.seed + 2000, args.samples)
    )
    result = dict(**metadata, checkpoint_ref=checkpoint_ref, metrics=metrics)
    write_json(destination / "summary.json", result)
    print("RESULT", metrics, flush=True)
    return result
