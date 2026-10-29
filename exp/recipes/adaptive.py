import copy

import random

import sys

from dataclasses import asdict

from pathlib import Path

import numpy as np

import torch

from torch.utils.data import DataLoader, Subset

from exp.recipes import native as native

from exp.recipes.core import (
    inputs,
    load_windows,
    persistence_metrics,
    score,
    selection_value,
    subset,
    train,
    write_json,
)

from layers.embedding import TransformerDiffusionConfig

from layers.history import HistoryAdaptiveEDM

from layers.objectives import fair_sample_losses

EXTRA = dict(
    history_trust=False,
    module_warm_epochs=0,
    refine_accumulate=4,
    rare_state_sampling=False,
    recent_sampling=0.0,
)


def parser():
    p = native.parser()
    for name, value in EXTRA.items():
        if isinstance(value, bool):
            p.add_argument("--" + name.replace("_", "-"), action="store_true")
        else:
            p.add_argument(
                "--" + name.replace("_", "-"), type=type(value), default=value
            )
    return p


def make_model(args, protocol, config=None):
    original = native.HistoryNativeEDM
    native.HistoryNativeEDM = lambda *a, **kw: HistoryAdaptiveEDM(
        *a, history_trust=args.history_trust, **kw
    )
    try:
        return native.make_model(args, protocol, config)
    finally:
        native.HistoryNativeEDM = original


def sampling_weights(data, args):
    weights = torch.ones(len(data), dtype=torch.double)
    audit = dict(population=len(data), bins=None)
    if args.rare_state_sampling:
        states = []
        for past, _, _ in DataLoader(data, batch_size=256):
            n = max(2, past.shape[1] // 4)
            scale = past.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
            drift = (
                ((past[:, -n:].mean(1) - past.mean(1)) / scale).square().mean(-1).sqrt()
            )
            states.extend(drift.tolist())
        thresholds = np.quantile(states, [0.9, 0.99])
        groups = np.digitize(states, thresholds)
        # Rare states are genuinely rare (90/9/1), with bounded 1/2/4 weights.
        weights *= torch.tensor([1.0, 2.0, 4.0], dtype=torch.double)[groups]
        audit.update(
            bins=np.bincount(groups, minlength=3).tolist(),
            thresholds=thresholds.tolist(),
        )
    weights *= torch.linspace(
        1.0, 1.0 + args.recent_sampling, len(data), dtype=torch.double
    )
    return weights, audit


def epoch_indices(weights, count, seed, epoch):
    count = len(weights) if count <= 0 else min(count, len(weights))
    generator = torch.Generator().manual_seed(seed + 3300 + epoch)
    indices = torch.multinomial(
        weights, count, replacement=False, generator=generator
    ).tolist()
    if count == len(weights) and weights.max() > weights.min():
        # Small datasets retain EVERY origin; add at most 10% extra rare-state
        # exposures instead of pretending a weighted permutation oversamples.
        extra = torch.multinomial(
            weights - weights.min(),
            max(1, count // 10),
            replacement=True,
            generator=generator,
        ).tolist()
        indices.extend(extra)
    return indices


def refine(model, datasets, args):
    validation = subset(datasets["validation"], args.validation_origins)
    selection_reference = persistence_metrics(validation)
    # Training loss scales use training targets only, not validation labels.
    training_reference = persistence_metrics(subset(datasets["train"], 2048))
    initial = score(model, validation, args, args.seed + 1000, args.validation_samples)
    best = selection_value(initial, args, selection_reference)
    state, chosen, history = copy.deepcopy(model.state_dict()), 0, []
    weights, audit = sampling_weights(datasets["train"], args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.refine_lr, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.refine_epochs + 1):
        indices = epoch_indices(weights, args.refine_origins, args.seed, epoch)
        loader = DataLoader(
            Subset(datasets["train"], indices), batch_size=args.refine_batch_size
        )
        torch.manual_seed(args.seed + 5100 + epoch)
        model.eval()
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            past, future, target, base = inputs(batch, args)
            generated = model.differentiable_sample(
                past, future, base, args.refine_samples
            )
            mean_mse, crps = fair_sample_losses(generated, target)
            mean_mae = (generated.mean(1) - target).abs().mean()
            loss = (
                (1 - args.refine_mae_weight)
                * mean_mse
                / max(training_reference["mse"], 1e-6)
                + args.refine_mae_weight
                * mean_mae
                / max(training_reference["mae"], 1e-6)
                + args.refine_crps_weight * crps / max(training_reference["crps"], 1e-6)
                + args.refine_denoise_weight
                * model.distribution_loss(past, future, target, base)
            )
            group_start = index // args.refine_accumulate * args.refine_accumulate
            group_origins = min(
                args.refine_accumulate * args.refine_batch_size,
                len(indices) - group_start * args.refine_batch_size,
            )
            (loss * len(past) / group_origins).backward()
            if (index + 1) % args.refine_accumulate == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * len(past)
        metrics = score(
            model, validation, args, args.seed + 1000, args.validation_samples
        )
        value = selection_value(metrics, args, selection_reference)
        eligible = (
            max(metrics[m] / max(initial[m], 1e-8) for m in ("mse", "mae")) <= 1.015
            and metrics["crps"] <= initial["crps"] * 1.30
            and metrics["qice"] <= initial["qice"] + 0.06
        )
        history.append(
            dict(
                epoch=epoch,
                loss=total / len(indices),
                eligible=eligible,
                sampled_origins=len(indices),
                first_origins=indices[:12],
                **metrics
            )
        )
        print("dynamic_refine", history[-1], flush=True)
        if eligible and value < best:
            best, chosen, state = value, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return history, dict(
        refine_epoch=chosen,
        refine_initial=initial,
        refine_score=best,
        sampling_audit=audit,
        training_loss_reference=training_reference,
        refinement="Fresh training windows each epoch; full small datasets plus <=10% extra rare exposures; IID Heun sample mean; accumulated gradients; epoch0 eligible",
    )


def run(args):
    saved = (
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if args.checkpoint
        else None
    )
    if saved:
        runtime = {
            k: getattr(args, k)
            for k in (
                "checkpoint",
                "device",
                "threads",
                "evaluate_test",
                "output_dir",
                "validation_blocks",
            )
        }
        vars(args).update(saved["args"])
        vars(args).update(runtime)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    datasets, protocol = load_windows(args)
    config = TransformerDiffusionConfig(**saved["model_config"]) if saved else None
    model = make_model(args, protocol, config).to(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if saved:
        native.verify_prefix(saved, args, protocol)
        model.load_state_dict(saved["state_dict"])
        history, selection = dict(restored_from=args.checkpoint), saved["selection"]
    else:
        if args.init_checkpoint:
            initial = torch.load(
                args.init_checkpoint, map_location="cpu", weights_only=True
            )
            native.verify_prefix(initial, args, protocol)
            missing, unexpected = model.load_state_dict(
                initial["state_dict"], strict=False
            )
            assert not unexpected and all(
                k.startswith(
                    (
                        "exogenous.history_bank.",
                        "state_response.reliability.",
                        "history_trust.",
                    )
                )
                for k in missing
            )
        history, selection = {}, dict(epoch=0, warm_start=args.init_checkpoint)
        if args.module_warm_epochs:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(
                    name.startswith(
                        (
                            "history_trust.",
                            "exogenous.history_bank.",
                            "state_response.reliability.",
                        )
                    )
                )
            assert any(p.requires_grad for p in model.parameters())
            warm = copy.copy(args)
            warm.epochs = args.module_warm_epochs
            warm.eval_every = 1
            warm.lr = max(args.lr, 1e-4)
            history["module_warmup"], _ = train(model, datasets, warm)
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        if args.epochs:
            history["joint"], selection = train(model, datasets, args)
        if args.refine_epochs:
            history["sample_refinement"], update = refine(model, datasets, args)
            selection.update(update)
    metadata = dict(
        args=vars(args),
        protocol=protocol,
        model_config=asdict(model.config),
        parameters=sum(p.numel() for p in model.parameters()),
        posthoc_sample_centering=False,
        metric_space="DAG_train_prefix_standardized",
        worker="history_adaptive",
        selection=selection,
    )
    torch.save(dict(state_dict=model.state_dict(), **metadata), output / "model.pt")
    write_json(output / "history.json", history)
    write_json(output / "config.json", metadata)
    metrics, blocks = {}, []
    for split in ("validation", "test"):
        if split not in datasets:
            continue
        if split == "validation" and args.validation_blocks > 1:
            metrics[split], blocks = score(
                model,
                datasets[split],
                args,
                args.seed + 2000,
                args.samples,
                blocks=args.validation_blocks,
            )
        else:
            metrics[split] = score(
                model, datasets[split], args, args.seed + 2000, args.samples
            )
    result = dict(**metadata, metrics=metrics, validation_blocks=blocks)
    write_json(output / "summary.json", result)
    print("RESULT", metrics, flush=True)
    return result
