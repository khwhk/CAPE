import copy

import random

import sys

from dataclasses import asdict

from pathlib import Path

import numpy as np

import torch

from torch.utils.data import DataLoader, WeightedRandomSampler

from exp.recipes.core import (
    inputs,
    load_windows,
    parser as base_parser,
    persistence_metrics,
    score,
    selection_value,
    subset,
    train,
    write_json,
)

from layers.embedding import TransformerDiffusionConfig

from layers.history import HistoryNativeEDM

from layers.objectives import fair_sample_losses

EXTRA = dict(
    native_lag=False,
    reliable_response=False,
    refine_epochs=0,
    refine_origins=256,
    refine_batch_size=4,
    refine_samples=4,
    refine_lr=1e-5,
    refine_mae_weight=0.5,
    refine_crps_weight=0.1,
    refine_denoise_weight=0.25,
    balanced_history=False,
)


def parser():
    p = base_parser()
    p.add_argument("--init-checkpoint")
    for name, value in EXTRA.items():
        if isinstance(value, bool):
            p.add_argument("--" + name.replace("_", "-"), action="store_true")
        else:
            p.add_argument(
                "--" + name.replace("_", "-"), type=type(value), default=value
            )
    return p


def make_model(args, protocol, config=None):
    if (
        args.future_mode != "history"
        or not args.state_response
        or args.exogenous_mode != "transport"
    ):
        raise ValueError("History-only worker requires wo_future and both innovations")
    if (
        args.location != "zero"
        or args.prediction_type != "edm"
        or args.conditioner != "trajectory"
    ):
        raise ValueError("History-only worker requires direct trajectory EDM")
    config = config or TransformerDiffusionConfig(
        seq_len=args.seq_len,
        horizon=args.horizon,
        past_dim=len(protocol["columns"]),
        future_cond_dim=len(protocol["columns"]) - 1,
        target_dim=1,
        d_model=args.d_model,
        n_heads=4,
        n_layers=args.layers,
        d_ff=2 * args.d_model,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        guidance_weight=0.0,
        center_samples=False,
    )
    names = (
        "window_norm",
        "exogenous_mode",
        "sigma_max",
        "sigma_min",
        "log_sigma_mean",
        "log_sigma_std",
        "norm_floor",
        "denoise_scale_power",
        "edm_backbone",
        "patch_length",
        "target_stats",
        "response_memory",
        "lag_detrend",
        "response_input",
        "response_gain_scale",
        "coordinate_center",
        "response_shift_scale",
        "response_recent_fraction",
        "response_noise_damping",
        "lag_temperature",
        "lag_recent_fraction",
        "lag_uniform_mix",
        "history_condition",
        "history_period",
        "history_decay",
        "native_lag",
        "reliable_response",
    )
    return HistoryNativeEDM(
        config,
        args.location,
        args.fusion,
        args.distribution,
        args.prediction_type,
        args.conditioner,
        args.moment_weight,
        args.denoise_mae,
        lag_candidates=protocol.get("lag_candidates"),
        **{k: getattr(args, k) for k in names}
    )


def refinement_loader(data, args):
    data = subset(data, args.refine_origins)
    sampler, groups = None, None
    if args.balanced_history:
        # Stratify using observed Y/Z changes only, never future targets or Z.
        states = []
        for past, _, _ in DataLoader(data, batch_size=128):
            n = max(2, past.shape[1] // 4)
            scale = past.var(1, unbiased=False).add(1e-5).sqrt().clamp_min(0.1)
            states.extend(
                ((past[:, -n:].mean(1) - past.mean(1)) / scale)
                .square()
                .mean(-1)
                .sqrt()
                .tolist()
            )
        groups = np.digitize(states, np.quantile(states, [1 / 3, 2 / 3]))
        counts = np.bincount(groups, minlength=3)
        weights = torch.as_tensor(1.0 / counts[groups], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights,
            len(data),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed + 3300),
        )
        groups = counts.tolist()
    return (
        DataLoader(
            data,
            batch_size=args.refine_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
        ),
        groups,
    )


def refine(model, datasets, args):
    validation = subset(datasets["validation"], args.validation_origins)
    reference = persistence_metrics(validation)
    initial = score(model, validation, args, args.seed + 1000, args.validation_samples)
    best = selection_value(initial, args, reference)
    state, chosen, history = copy.deepcopy(model.state_dict()), 0, []
    loader, bins = refinement_loader(datasets["train"], args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.refine_lr, weight_decay=args.weight_decay
    )
    torch.manual_seed(args.seed + 5100)
    for epoch in range(1, args.refine_epochs + 1):
        model.eval()  # Inference-consistent dropout, but sampling retains gradients.
        total = 0.0
        for batch in loader:
            past, future, target, base = inputs(batch, args)
            optimizer.zero_grad(set_to_none=True)
            generated = model.differentiable_sample(
                past, future, base, args.refine_samples
            )
            mean_mse, crps = fair_sample_losses(generated, target)
            mean_mae = (generated.mean(1) - target).abs().mean()
            point = (1 - args.refine_mae_weight) * mean_mse / max(
                reference["mse"], 1e-6
            ) + args.refine_mae_weight * mean_mae / max(reference["mae"], 1e-6)
            loss = point + args.refine_crps_weight * crps / max(reference["crps"], 1e-6)
            loss = loss + args.refine_denoise_weight * model.distribution_loss(
                past, future, target, base
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(past)
        metrics = score(
            model, validation, args, args.seed + 1000, args.validation_samples
        )
        value = selection_value(metrics, args, reference)
        eligible = (
            max(metrics[m] / max(initial[m], 1e-8) for m in ("mse", "mae")) <= 1.03
            and metrics["crps"] <= initial["crps"] * 1.25
            and metrics["qice"] <= initial["qice"] + 0.05
        )
        history.append(
            dict(
                epoch=epoch,
                loss=total / len(loader.dataset),
                eligible=eligible,
                **metrics
            )
        )
        print("sample_refine", history[-1], flush=True)
        if eligible and value < best:
            best, chosen, state = value, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return history, dict(
        refine_epoch=chosen,
        refine_initial=initial,
        refine_score=best,
        history_state_bins=bins,
        refine_criterion="validation .5MSE+.5MAE, persistence-normalized; epoch0 allowed",
        sample_objective="IID Heun samples; variance-corrected mean MSE, empirical mean MAE, fair CRPS; no teacher",
    )


def verify_prefix(saved, args, protocol):
    if saved["args"]["future_mode"] != "history" or saved["args"]["seed"] != args.seed:
        raise ValueError("Warm start must be history-only, same seed and same prefix")
    for field in (
        "columns",
        "length",
        "train_end",
        "validation_end",
        "scaler_mean",
        "scaler_scale",
    ):
        if saved["protocol"][field] != protocol[field]:
            raise ValueError("Checkpoint/prefix mismatch: " + field)


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
        verify_prefix(saved, args, protocol)
        model.load_state_dict(saved["state_dict"])
        history, selection = dict(restored_from=args.checkpoint), saved["selection"]
    else:
        if args.init_checkpoint:
            initial = torch.load(
                args.init_checkpoint, map_location="cpu", weights_only=True
            )
            verify_prefix(initial, args, protocol)
            missing, unexpected = model.load_state_dict(
                initial["state_dict"], strict=False
            )
            if unexpected or any(
                not k.startswith(
                    ("exogenous.history_bank.", "state_response.reliability.")
                )
                for k in missing
            ):
                raise ValueError(
                    "Unmatched warm-start model parameters: "
                    + str((missing, unexpected))
                )
        if args.epochs:
            history, selection = train(model, datasets, args)
        else:
            if not args.init_checkpoint:
                raise ValueError("Zero training epochs require a warm start")
            history, selection = {}, dict(epoch=0, warm_start=args.init_checkpoint)
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
        worker="history_native",
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
