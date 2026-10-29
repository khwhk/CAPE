"""Shared denoising training and validation-selected checkpoint policy."""

import argparse
import copy
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_provider.data_loader import load_windows, subset
from utils.evaluation import (
    inputs,
    score,
    selection_value,
    persistence_metrics,
    write_json,
    aggregate_results,
)
from layers.edm import EXOGENOUS_MODES
from layers.objectives import fair_sample_losses


def point_loss(prediction, target, name):
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "mae":
        return F.l1_loss(prediction, target)
    return 0.5 * (F.mse_loss(prediction, target) + F.l1_loss(prediction, target))


def train_epoch(model, loader, optimizer, args, guide=False, ema=None):
    model.train()
    if not guide and args.location == "learned":
        model.conditioner.eval()
    total = 0.0
    for batch in loader:
        past, future, target, base = inputs(batch, args)
        optimizer.zero_grad(set_to_none=True)
        if guide:
            prediction = model.condition(past, future, base)[1]
            loss = point_loss(prediction, target, args.guide_loss)
        else:
            loss = model.distribution_loss(past, future, target, base)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1 - args.ema_decay)
        total += loss.item() * len(past)
    return total / len(loader.dataset)


def sample_finetune(model, datasets, args, history, selection, validation, reference):
    """Low-rate refinement of actual Heun samples; never recenter predictions."""
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    loader = DataLoader(
        subset(datasets["train"], args.sample_finetune_origins),
        batch_size=args.sample_finetune_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.sample_finetune_lr, weight_decay=args.weight_decay
    )
    model.eval()  # Same dropout behavior as inference, with gradients enabled.
    best_metrics = score(
        model, validation, args, args.seed + 1000, args.validation_samples
    )
    best = selection_value(best_metrics, args, reference)
    state = copy.deepcopy(model.state_dict())
    history["sample_finetune"] = []
    selected_epoch = 0
    for epoch in range(1, args.sample_finetune_epochs + 1):
        total = 0.0
        for batch in loader:
            past, future, target, base = inputs(batch, args)
            optimizer.zero_grad(set_to_none=True)
            generated = model.differentiable_sample(
                past, future, base, args.sample_finetune_samples
            )
            mean_mse, crps = fair_sample_losses(generated, target)
            loss = model.distribution_loss(past, future, target, base)
            loss = loss + args.sample_finetune_weight * (0.5 * mean_mse + 0.5 * crps)
            mean, scale = model.coordinates(past)
            sigma = (
                torch.randn(len(past), 1, 1, device=past.device) * model.log_sigma_std
                + model.log_sigma_mean
            ).exp()
            noisy = (target - mean) / scale + sigma * torch.randn_like(target)
            context, _ = model.condition(past, future, base)
            with torch.no_grad():
                old_context, _ = teacher.condition(past, future, base)
                old = teacher.denoise(noisy, sigma, old_context)
            consistency = F.mse_loss(model.denoise(noisy, sigma, context), old)
            loss = loss + 0.05 * consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(past)
        metrics = score(
            model, validation, args, args.seed + 1000, args.validation_samples
        )
        # Both point metrics and CRPS must preserve the pre-refinement model.
        eligible = all(metrics[m] <= best_metrics[m] for m in ("mse", "mae", "crps"))
        value = selection_value(metrics, args, reference)
        history["sample_finetune"].append(
            {
                "epoch": epoch,
                "loss": total / len(loader.dataset),
                "eligible": eligible,
                **metrics,
            }
        )
        print("sample_finetune", epoch, metrics, "guard", eligible, flush=True)
        if eligible and value < best:
            best, selected_epoch = value, epoch
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    selection["sample_finetune_epoch"] = selected_epoch
    selection["sample_finetune_sampler"] = (
        "same Heun steps/noise schedule as inference; IID draws"
    )
    return history, selection


def train(model, datasets, args):
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    validation = subset(datasets["validation"], args.validation_origins)
    history = {"guide": [], "diffusion": []}
    if args.conditioner_checkpoint:
        state = torch.load(
            args.conditioner_checkpoint, map_location=args.device, weights_only=True
        )["state_dict"]
        model.conditioner.load_state_dict(
            {
                k[len("conditioner.") :]: v
                for k, v in state.items()
                if k.startswith("conditioner.")
            }
        )
    elif args.location == "learned":
        optimizer = torch.optim.AdamW(
            model.conditioner.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.01),
        )
        best, stale = float("inf"), 0
        for epoch in range(1, args.guide_epochs + 1):
            loss = train_epoch(model, loader, optimizer, args, guide=True)
            metrics = score(
                model, validation, args, args.seed + 1000, 1, point_only=True
            )
            value = (
                0.5 * (metrics["mae"] + metrics["mse"])
                if args.guide_loss == "mixed"
                else metrics[args.guide_loss]
            )
            history["guide"].append({"epoch": epoch, "loss": loss, **metrics})
            print(
                "guide",
                epoch,
                "loss",
                round(loss, 5),
                "val",
                round(value, 5),
                flush=True,
            )
            if value < best:
                best, stale = value, 0
                guide_state = copy.deepcopy(model.conditioner.state_dict())
            else:
                stale += 1
            if stale >= args.patience:
                break
        model.conditioner.load_state_dict(guide_state)
    if args.location == "learned":
        for parameter in model.conditioner.parameters():
            parameter.requires_grad_(False)
    point_reference = score(
        model, validation, args, args.seed + 1000, 1, point_only=True
    )
    selection_reference = (
        persistence_metrics(validation)
        if getattr(args, "selection_objective", "crps") in ("balanced", "point")
        else point_reference
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.01),
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.lr * 0.01
        )
        if getattr(args, "lr_schedule", "constant") == "cosine"
        else None
    )
    best_any, best_eligible, stale = float("inf"), float("inf"), 0
    eligible_state = None
    ema = copy.deepcopy(model).eval() if args.ema_decay else None
    evaluation_model = ema if ema is not None else model
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, args, ema=ema)
        if scheduler is not None:
            scheduler.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        metrics = score(
            evaluation_model,
            validation,
            args,
            args.seed + 1000,
            args.validation_samples,
        )
        eligible = args.location == "zero" or all(
            metrics[k] <= point_reference[k] * (1 + args.point_tolerance)
            for k in ("mse", "mae")
        )
        history["diffusion"].append(
            {"epoch": epoch, "loss": loss, "eligible": eligible, **metrics}
        )
        print(
            "distribution",
            epoch,
            "val",
            metrics,
            "point_guard",
            eligible if args.location == "learned" else "not_applied",
            flush=True,
        )
        value = selection_value(metrics, args, selection_reference)
        if value < best_any:
            best_any, stale = value, 0
            any_state, any_epoch = copy.deepcopy(evaluation_model.state_dict()), epoch
        else:
            stale += 1
        if eligible and value < best_eligible:
            best_eligible, stale = value, 0
            eligible_state, eligible_epoch = (
                copy.deepcopy(evaluation_model.state_dict()),
                epoch,
            )
        if stale >= args.patience:
            break
    model.load_state_dict(eligible_state if eligible_state is not None else any_state)
    selection = {
        "epoch": eligible_epoch if eligible_state is not None else any_epoch,
        "point_guard_passed": (
            eligible_state is not None if args.location == "learned" else None
        ),
        "point_guard_applied": args.location == "learned",
        "point_reference": point_reference,
        "selection_origins": len(validation),
        "selection_samples": args.validation_samples,
        "criterion": (
            "minimum validation CRPS subject to MSE/MAE guard; otherwise minimum CRPS"
            if args.location == "learned"
            else "minimum validation CRPS"
        ),
    }
    if getattr(args, "selection_objective", "crps") == "balanced":
        selection.update(
            criterion="validation .45*MSE + .35*MAE + .20*CRPS, persistence-normalized",
            selection_reference=selection_reference,
        )
    if getattr(args, "selection_objective", "crps") == "point":
        selection.update(
            criterion="validation .50*MSE + .50*MAE, persistence-normalized; full-validation probability guards applied by orchestrator",
            selection_reference=selection_reference,
        )
    if getattr(args, "sample_finetune_epochs", 0):
        return sample_finetune(
            model, datasets, args, history, selection, validation, selection_reference
        )
    return history, selection


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="dataset/forecasting/Sdwpfh1.csv")
    p.add_argument("--target", default="Patv")
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--future-mode", choices=("realized", "history"), default="realized")
    p.add_argument("--location", choices=("learned", "zero"), default="learned")
    p.add_argument(
        "--conditioner",
        choices=("full", "patch", "local", "relative", "trajectory"),
        default="full",
    )
    p.add_argument("--fusion", choices=("concat", "cross", "skip"), default="concat")
    p.add_argument(
        "--distribution", choices=("diffusion", "gaussian"), default="diffusion"
    )
    p.add_argument("--guide-loss", choices=("mse", "mae", "mixed"), default="mixed")
    p.add_argument("--guide-epochs", type=int, default=30)
    p.add_argument(
        "--conditioner-checkpoint",
        help="Reuse the same learned conditioner for distribution ablations",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--validation-origins", type=int, default=256)
    p.add_argument("--validation-samples", type=int, default=64)
    p.add_argument("--validation-blocks", type=int, default=1)
    p.add_argument("--point-tolerance", type=float, default=0.02)
    p.add_argument("--diffusion-steps", type=int, default=20)
    p.add_argument(
        "--prediction-type", choices=("epsilon", "v", "x0", "edm"), default="epsilon"
    )
    p.add_argument(
        "--window-norm",
        action="store_true",
        help="Normalize EDM target coordinates with observed history only",
    )
    p.add_argument("--denoise-mae", type=float, default=0.0)
    p.add_argument(
        "--denoise-scale-power", type=float, choices=(0.0, 1.0, 2.0), default=0.0
    )
    p.add_argument("--exogenous-mode", choices=EXOGENOUS_MODES, default="none")
    p.add_argument("--edm-backbone", choices=("global", "dual"), default="global")
    p.add_argument("--patch-length", type=int, default=12)
    p.add_argument("--target-stats", action="store_true")
    p.add_argument("--state-response", action="store_true")
    p.add_argument(
        "--history-condition",
        choices=("provided", "last", "seasonal", "linear", "cycle_mean"),
        default="provided",
    )
    p.add_argument("--history-period", type=int, default=24)
    p.add_argument("--history-decay", type=float, default=0.0)
    p.add_argument(
        "--response-input",
        choices=("aligned", "raw", "aligned_detach"),
        default="aligned",
    )
    p.add_argument("--response-gain-scale", type=float, default=1.0)
    p.add_argument("--response-shift-scale", type=float, default=0.5)
    p.add_argument("--response-recent-fraction", type=float, default=0.25)
    p.add_argument("--response-noise-damping", type=float, default=0.0)
    p.add_argument("--lag-temperature", type=float, default=1.0)
    p.add_argument("--lag-recent-fraction", type=float, default=1.0)
    p.add_argument("--lag-uniform-mix", type=float, default=0.0)
    p.add_argument("--response-memory", action="store_true")
    p.add_argument("--time-aligned", action="store_true")
    p.add_argument("--adaptive-center", action="store_true")
    p.add_argument("--cumulative-exogenous", action="store_true")
    p.add_argument("--lag-detrend", action="store_true")
    p.add_argument("--lag-grid", choices=("fixed", "auto"), default="fixed")
    p.add_argument("--calendar", action="store_true")
    p.add_argument("--validation-fold", type=int, choices=(0, 1, 2), default=2)
    p.add_argument(
        "--selection-objective", choices=("crps", "balanced", "point"), default="crps"
    )
    p.add_argument(
        "--coordinate-center", choices=("baseline", "last"), default="baseline"
    )
    p.add_argument("--norm-floor", type=float, default=0.1)
    p.add_argument("--exogenous-encoder", action="store_true")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--sample-finetune-epochs", type=int, default=0)
    p.add_argument("--sample-finetune-origins", type=int, default=256)
    p.add_argument("--sample-finetune-batch-size", type=int, default=8)
    p.add_argument("--sample-finetune-samples", type=int, default=4)
    p.add_argument("--sample-finetune-lr", type=float, default=5e-5)
    p.add_argument("--sample-finetune-weight", type=float, default=0.1)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--sigma-min", type=float, default=0.002)
    p.add_argument("--log-sigma-mean", type=float, default=-1.2)
    p.add_argument("--log-sigma-std", type=float, default=1.2)
    p.add_argument("--inference-steps", type=int)
    p.add_argument("--inference-sigma-max", type=float)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--moment-weight", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--d-model", type=int, default=48)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--sample-batch-size", type=int, default=128)
    p.add_argument(
        "--sampling-noise-batch-size",
        type=int,
        help="Keep initial RNG chunks fixed while changing the parallel sampling batch",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument(
        "--evaluate-test", action="store_true", help="Use only after configuration lock"
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--checkpoint",
        help="Evaluate saved weights without retraining; model arguments come from the checkpoint",
    )
    p.add_argument("--output-dir", required=True)
    return p
