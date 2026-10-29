import argparse


import copy


import json


import random


import sys


from dataclasses import asdict


from pathlib import Path


import numpy as np


import torch


from torch.utils.data import DataLoader, Subset


from exp.recipes.core import (
    parser,
    load_windows,
    inputs,
    score,
    subset,
    train_epoch,
    persistence_metrics,
    write_json,
)


from layers.embedding import TransformerDiffusionConfig


from layers.state_response import FocusedFutureEDM


from layers.objectives import fair_sample_losses


DEFAULTS = dict(
    response_temporal="none",
    response_kernel=3,
    response_strength=0.5,
    refine_epochs=0,
    refine_origins=512,
    refine_samples=8,
    refine_batch_size=1,
    refine_accumulate=8,
    refine_lr=1e-5,
    refine_mae_weight=0.25,
    refine_crps_weight=0.02,
    refine_denoise_weight=0.05,
    increment_weight=0.0,
    train_scope="all",
    refine_scope="all",
)


def arguments(job, output, device="cuda:0"):
    a = parser().parse_args(["--output-dir", str(output)])
    vars(a).update(DEFAULTS)
    vars(a).update(job["arguments"])
    a.device, a.output_dir, a.threads = device, str(output), 2
    a.future_mode, a.evaluate_test, a.validation_fold = "realized", True, 2
    if not a.state_response or a.exogenous_mode != "transport" or a.location != "zero":
        raise ValueError("Both LAC/CRD and direct diffusion must remain enabled")
    return a


def make_model(a, protocol, config=None):
    config = config or TransformerDiffusionConfig(
        a.seq_len,
        a.horizon,
        len(protocol["columns"]),
        len(protocol["columns"]) - 1,
        1,
        d_model=a.d_model,
        n_heads=4,
        n_layers=a.layers,
        d_ff=2 * a.d_model,
        dropout=a.dropout,
        diffusion_steps=a.diffusion_steps,
        guidance_weight=0.0,
        center_samples=False,
    )
    keys = (
        "window_norm",
        "exogenous_mode",
        "sigma_max",
        "sigma_min",
        "log_sigma_mean",
        "log_sigma_std",
        "denoise_scale_power",
        "edm_backbone",
        "patch_length",
        "target_stats",
        "response_memory",
        "lag_detrend",
        "response_input",
        "response_gain_scale",
        "coordinate_center",
        "norm_floor",
        "response_shift_scale",
        "response_recent_fraction",
        "response_noise_damping",
        "lag_temperature",
        "lag_recent_fraction",
        "lag_uniform_mix",
        "response_temporal",
        "response_kernel",
        "response_strength",
    )
    return FocusedFutureEDM(
        config,
        a.location,
        a.fusion,
        a.distribution,
        a.prediction_type,
        a.conditioner,
        a.moment_weight,
        a.denoise_mae,
        lag_candidates=protocol.get("lag_candidates"),
        **{k: getattr(a, k) for k in keys}
    )


def load_model(job, a, protocol):
    checkpoint = job.get("init_checkpoint") or job.get("checkpoint")
    saved = (
        torch.load(checkpoint, map_location="cpu", weights_only=True)
        if checkpoint
        else None
    )
    if saved:
        for key in (
            "columns",
            "length",
            "train_end",
            "validation_end",
            "scaler_mean",
            "scaler_scale",
        ):
            if saved["protocol"][key] != protocol[key]:
                raise ValueError("Different training/data protocol: " + key)
    model = make_model(a, protocol).to(a.device)
    changes = dict(missing=[], removed=[])
    if saved:
        state = dict(saved["state_dict"])
        if a.response_temporal == "none":
            changes["removed"] = [
                k for k in state if k.startswith("temporal_response.")
            ]
            state = {k: v for k, v in state.items() if k not in changes["removed"]}
        elif (
            saved["args"].get("response_temporal", "none") != a.response_temporal
            or saved["args"].get("response_kernel", 3) != a.response_kernel
        ):
            state = {
                k: v for k, v in state.items() if not k.startswith("temporal_response.")
            }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("temporal_response.") for k in missing):
            raise ValueError(str((missing, unexpected)))
        changes["missing"] = missing
    return model, changes


def set_scope(model, scope):
    prefixes = dict(
        response=("state_response.", "temporal_response."),
        conditioning=(
            "state_response.",
            "temporal_response.",
            "exogenous.",
            "conditioner.",
        ),
        temporal=("temporal_response.",),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(scope == "all" or name.startswith(prefixes[scope]))
    return [p for p in model.parameters() if p.requires_grad]


def train(model, data, a):
    parameters = set_scope(model, a.train_scope)
    optimizer = torch.optim.AdamW(parameters, lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, a.epochs), eta_min=a.lr * 0.05
    )
    ema = copy.deepcopy(model).eval() if a.ema_decay else None
    history = []
    for epoch in range(1, a.epochs + 1):
        loss = train_epoch(
            model,
            DataLoader(data, batch_size=a.batch_size, shuffle=True),
            optimizer,
            a,
            ema=ema,
        )
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite training objective")
        scheduler.step()
        history.append(dict(epoch=epoch, loss=loss))
        print("TRAIN", history[-1], flush=True)
    if ema is not None and a.epochs:
        model.load_state_dict(ema.state_dict())
    return history


def refine(model, data, a):
    parameters = set_scope(model, a.refine_scope)
    optimizer = torch.optim.AdamW(
        parameters, lr=a.refine_lr, weight_decay=a.weight_decay
    )
    reference = persistence_metrics(subset(data, min(2048, len(data))))
    history = []
    for epoch in range(1, a.refine_epochs + 1):
        rng = np.random.default_rng(a.seed * 10000 + epoch)
        ids = rng.choice(
            len(data), min(a.refine_origins, len(data)), replace=False
        ).tolist()
        loader = DataLoader(Subset(data, ids), batch_size=a.refine_batch_size)
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        torch.manual_seed(a.seed + 5100 + epoch)
        for i, batch in enumerate(loader):
            past, future, target, base = inputs(batch, a)
            samples = model.differentiable_sample(past, future, base, a.refine_samples)
            mse, crps = fair_sample_losses(samples, target)
            # Match the expected MSE of the reported 100-path mean, not a 4-path mean.
            mse = mse + samples.var(1, unbiased=True).mean() / 100
            mae = (samples.mean(1) - target).abs().mean()
            loss = (
                (1 - a.refine_mae_weight) * mse / max(reference["mse"], 1e-6)
                + a.refine_mae_weight * mae / max(reference["mae"], 1e-6)
                + a.refine_crps_weight * crps / max(reference["crps"], 1e-6)
            )
            if a.refine_denoise_weight:
                loss = loss + a.refine_denoise_weight * model.distribution_loss(
                    past, future, target, base
                )
            if a.increment_weight:
                loss = (
                    loss
                    + a.increment_weight
                    * (samples.mean(1).diff(dim=1) - target.diff(dim=1)).abs().mean()
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite sample refinement objective")
            start = (
                (i // a.refine_accumulate) * a.refine_accumulate * a.refine_batch_size
            )
            count = min(a.refine_accumulate * a.refine_batch_size, len(ids) - start)
            (loss * len(past) / count).backward()
            if (i + 1) % a.refine_accumulate == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item() * len(past)
        history.append(dict(epoch=epoch, loss=total / len(ids), origins=len(ids)))
        print("REFINE", history[-1], flush=True)
    return history


def run(job, output, device="cuda:0", smoke=False):
    a = arguments(job, output, device)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    data, protocol = load_windows(a)
    model, changes = load_model(job, a, protocol)
    if smoke:
        data = {k: subset(v, 4) for k, v in data.items()}
        a.epochs = min(1, a.epochs)
        a.refine_epochs = min(1, a.refine_epochs)
        a.batch_size = 2
        a.eval_batch_size = 2
        a.refine_origins = 2
        a.refine_samples = 2
        # Smoke retains the full 20-step solver and 100 evaluation trajectories.
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    history = dict(training=train(model, data["train"], a), refinement=[])
    if a.refine_epochs:
        history["refinement"] = refine(model, data["train"], a)
    metadata = dict(
        args=vars(a),
        protocol=protocol,
        model_config=asdict(model.config),
        selection=dict(
            checkpoint_policy="fixed-budget final weights/EMA; no validation elimination",
            epochs=a.epochs,
            refine_epochs=a.refine_epochs,
        ),
        worker="wfuture_focused",
        evaluation_protocol="TEST_TUNED_EXPLORATORY",
        future_z_used=True,
        test_gradients=False,
        posthoc_sample_centering=False,
        smoke=smoke,
        warm_start=job.get("init_checkpoint"),
        warm_start_changes=changes,
        parameters=sum(p.numel() for p in model.parameters()),
    )
    checkpoint = str(out / "model.pt")
    torch.save(dict(state_dict=model.state_dict(), **metadata), checkpoint)
    write_json(out / "history.json", history)
    metrics = dict(test=score(model, data["test"], a, a.seed + 2000, 100))
    result = dict(
        **metadata,
        checkpoint_ref=checkpoint,
        metrics=metrics,
        peak_allocated_mib=(
            torch.cuda.max_memory_allocated() / 2**20
            if device.startswith("cuda")
            else 0.0
        )
    )
    write_json(out / "summary.json", result)
    print("RESULT", metrics, flush=True)
    return result
