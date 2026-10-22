import copy
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_provider.data_loader import subset
from utils.evaluation import inputs, score, selection_value, persistence_metrics

def train_epoch(model, loader, optimizer, args, ema=None):
    model.train()
    total = 0.0
    for batch in loader:
        past, future, target, base = inputs(batch, args)
        optimizer.zero_grad(set_to_none=True)
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


def train(model, datasets, args):
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    validation = subset(datasets["validation"], args.validation_origins)
    history = {"guide": [], "diffusion": []}
    point_reference = score(
        model, validation, args, args.seed + 1000, 1, point_only=True
    )
    selection_reference = (persistence_metrics(validation)
                           if getattr(args, "selection_objective", "crps") in ("balanced", "point")
                           else point_reference)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
        weight_decay=getattr(args, "weight_decay", .01)
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)
        if getattr(args, "lr_schedule", "constant") == "cosine" else None
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
        selection.update(criterion="validation .45*MSE + .35*MAE + .20*CRPS, persistence-normalized",
                         selection_reference=selection_reference)
    if getattr(args, "selection_objective", "crps") == "point":
        selection.update(criterion="validation .50*MSE + .50*MAE, persistence-normalized",
                         selection_reference=selection_reference)
    return history, selection
