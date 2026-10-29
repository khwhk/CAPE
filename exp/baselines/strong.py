import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from data_provider.data_loader import load_windows, subset, read_data
from utils.evaluation import write_json
from utils.metrics import evaluate_samples
from ts_benchmark.baselines.dag.dag import DAG
from ts_benchmark.baselines.kite.kite import KITE


class RecordValidation:
    def validate(self, *args, **kwargs):
        value = super().validate(*args, **kwargs)
        self.validation_history.append(float(value))
        print(
            "epoch", len(self.validation_history), "validation_loss", value, flush=True
        )
        return value


class RecordedDAG(RecordValidation, DAG):
    pass


class RecordedKITE(RecordValidation, KITE):
    pass


@torch.no_grad()
def score(model, data, args):
    model.eval()
    predictions, truths = [], []
    for index, (past, future, target) in enumerate(
        DataLoader(data, batch_size=args.eval_batch_size)
    ):
        past, future = past.to(args.device), future.to(args.device)
        if getattr(args, "future_mode", "realized") == "history":
            future = None
        torch.manual_seed(args.seed + 2000 + index)
        if args.model == "kite":
            prediction = model.inference(
                past[..., :1],
                past[..., 1:],
                future,
                num_samples=args.samples,
                return_samples=True,
            )
        else:
            prediction = model(past, future)[0]
        predictions.append(prediction.cpu())
        truths.append(target)
    prediction, target = torch.cat(predictions), torch.cat(truths)
    if args.model == "kite":
        return evaluate_samples(prediction, target)
    error = prediction - target
    return {"mse": error.square().mean().item(), "mae": error.abs().mean().item()}


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    parameters = json.loads(Path(args.parameters).read_text())
    if getattr(args, "future_mode", "realized") == "history":
        flag = "use_future_exog" if args.model == "kite" else "infer_use_future"
        if parameters.get(flag, True):
            raise ValueError(
                "Historical-only evaluation requires the official wo_future preset"
            )
    args.seq_len, args.horizon = parameters["seq_len"], parameters["horizon"]
    args.evaluate_test = True
    datasets, protocol = load_windows(args)
    frame = read_data(args.data_path)[protocol["columns"]]
    fit_data = frame.iloc[: protocol["validation_end"]]
    if args.smoke:
        parameters.update(num_epochs=1, batch_size=8)
        fit_data = fit_data.iloc[: args.seq_len + args.horizon + 80]
        datasets = {name: subset(data, 16) for name, data in datasets.items()}
        args.samples = 8
    adapter = (RecordedKITE if args.model == "kite" else RecordedDAG)(**parameters)
    adapter.validation_history = []
    write_json(
        output / "config.json",
        {
            "args": vars(args),
            "preset": parameters,
            "protocol": protocol,
            "training": "official forecast_fit adapter",
            "smoke": args.smoke,
        },
    )
    adapter.forecast_fit(
        fit_data.iloc[:, :1],
        covariates={"exog": fit_data.iloc[:, 1:]},
        train_ratio_in_tv=0.875,
    )
    if not args.smoke:
        actual_mean = np.r_[adapter.scaler1.mean_, adapter.scaler2.mean_]
        actual_scale = np.r_[adapter.scaler1.scale_, adapter.scaler2.scale_]
        np.testing.assert_allclose(actual_mean, protocol["scaler_mean"], rtol=1e-10)
        np.testing.assert_allclose(actual_scale, protocol["scaler_scale"], rtol=1e-10)
    adapter.model.load_state_dict(adapter.check_point["Model"])
    selected = int(np.argmin(adapter.validation_history)) + 1
    selection = {
        "epoch": selected,
        "criterion": (
            "minimum validation flow loss"
            if args.model == "kite"
            else "minimum validation MAE"
        ),
    }
    parameters_count = sum(p.numel() for p in adapter.model.parameters())
    torch.save(
        {
            "state_dict": adapter.model.state_dict(),
            "preset": parameters,
            "args": vars(args),
            "protocol": protocol,
            "selection": selection,
        },
        output / "model.pt",
    )
    write_json(output / "history.json", {"validation_loss": adapter.validation_history})
    metrics = {}
    for name in ("validation", "test"):
        metrics[name] = score(adapter.model, datasets[name], args)
        print(name, metrics[name], flush=True)
    result = {
        "args": vars(args),
        "model": args.model,
        "preset": parameters,
        "protocol": protocol,
        "selection": selection,
        "metrics": metrics,
        "parameters": parameters_count,
        "posthoc_sample_centering": False,
        "metric_space": "DAG_train_prefix_standardized",
        "samples": args.samples if args.model == "kite" else None,
        "cuda_peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "smoke": args.smoke,
    }
    write_json(output / "summary.json", result)
    return result
