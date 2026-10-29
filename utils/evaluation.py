import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from .metrics import evaluate_samples


def inputs(batch, args):
    past, future, target = (v.to(args.device) for v in batch)
    if args.future_mode == "history":
        future = torch.zeros_like(future)
    base = torch.zeros_like(target)
    return past, future, target, base


@torch.no_grad()
def score(model, data, args, seed, samples, point_only=False, blocks=1):
    model.eval()
    forecasts, truths = [], []
    for i, batch in enumerate(DataLoader(data, batch_size=args.eval_batch_size)):
        past, future, target, base = inputs(batch, args)
        if point_only:
            prediction = model.condition(past, future, base)[1][:, None]
        else:
            prediction = model.sample(
                past,
                future,
                base,
                n_samples=samples,
                seed=seed + i,
                sample_batch_size=args.sample_batch_size,
                noise_batch_size=getattr(args, "sampling_noise_batch_size", None),
            )
        forecasts.append(prediction.cpu())
        truths.append(target.cpu())
    # Pool all origins before computing QICE; don't average per-batch QICE.
    forecasts, truths = torch.cat(forecasts), torch.cat(truths)
    metrics = evaluate_samples(forecasts, truths)
    if blocks > 1:
        sections = [
            dict(origins=len(i), metrics=evaluate_samples(forecasts[i], truths[i]))
            for i in torch.tensor_split(torch.arange(len(truths)), blocks)
            if len(i)
        ]
        return metrics, sections
    return metrics


def selection_value(metrics, args, reference):
    if getattr(args, "selection_objective", "crps") == "point":
        return sum(
            0.5 * metrics[name] / max(reference[name], 1e-8) for name in ("mse", "mae")
        )
    if getattr(args, "selection_objective", "crps") == "balanced":
        return sum(
            weight * metrics[name] / max(reference[name], 1e-8)
            for name, weight in (("mse", 0.45), ("mae", 0.35), ("crps", 0.20))
        )
    return metrics["crps"]


def persistence_metrics(data):
    forecasts, truths = [], []
    for past, future, target in DataLoader(data, batch_size=256):
        forecasts.append(past[:, -1:, :1].expand_as(target)[:, None])
        truths.append(target)
    return evaluate_samples(torch.cat(forecasts), torch.cat(truths))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def aggregate_results(results):
    metrics = {
        split: {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for key in results[0]["metrics"][split]
            for values in [[r["metrics"][split][key] for r in results]]
        }
        for split in results[0]["metrics"]
    }
    return {
        "seeds": [r["args"]["seed"] for r in results],
        "metrics": metrics,
        "selection": {str(r["args"]["seed"]): r["selection"] for r in results},
    }
