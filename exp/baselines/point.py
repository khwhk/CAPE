import argparse
import importlib
import itertools
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from data_provider.data_loader import load_windows, read_data
from utils.evaluation import write_json
from ts_benchmark.baselines.utils import forecasting_data_provider

CLASSES = dict(
    GCGNet="GCGNet.GCGNet.GCGNet",
    TimeXer="timexer.timexer.TimeXer",
    DUET="duet.duet.DUET",
    CrossLinear="crosslinear.crosslinear.CrossLinear",
    Amplifier="amplifier.amplifier.Amplifier",
    TimeKAN="timekan.timekan.TimeKAN",
    xPatch="xpatch.xpatch.xPatch",
)


TRANSFORMERS = dict(
    TFT="TemporalFusionTransformer", TiDE="TiDE", PatchTST="PatchTST", DLinear="DLinear"
)


class CheckpointedMicrobatches(nn.Module):
    """Memory-only adaptation: preserve batch loss and one optimizer step/batch."""

    def __init__(self, core, size=32):
        super().__init__()
        self.core, self.size = core, size

    def forward(self, *args):
        if not self.training or len(args[0]) <= self.size:
            return self.core(*args)
        parts = []
        for start in range(0, len(args[0]), self.size):
            chunk = tuple(
                a[start : start + self.size] if torch.is_tensor(a) else a for a in args
            )
            parts.append(
                checkpoint(
                    self.core, *chunk, use_reentrant=False, preserve_rng_state=True
                )
            )
        return torch.cat(parts)


class Recording:
    def _init_model(self):
        model = super()._init_model()
        if (
            self.model_name == "TemporalFusionTransformer"
            and self.config.horizon == 360
        ):
            model = CheckpointedMicrobatches(model)
        return model

    def validate(self, loader, *args, **kwargs):
        if self.smoke:
            loader = list(itertools.islice(loader, 2))
        value = super().validate(loader, *args, **kwargs)
        self.validation_history.append(float(value))
        print(
            "epoch", len(self.validation_history), "validation_loss", value, flush=True
        )
        return value

    def _init_optimizer(self, *args, **kwargs):
        if self.smoke:
            self.train_data_loader = list(itertools.islice(self.train_data_loader, 2))
        return super()._init_optimizer(*args, **kwargs)


def adapter_for(name, parameters, smoke=False):
    if name in TRANSFORMERS:
        from ts_benchmark.baselines.time_series_library.adapters_for_transformers import (
            TransformerAdapter,
        )

        model = TRANSFORMERS[name]
        cls = getattr(
            importlib.import_module(
                "ts_benchmark.baselines.time_series_library.models." + model
            ),
            model,
        )
        adapter = type("RecordedTransformer", (Recording, TransformerAdapter), {})(
            model, cls, **parameters
        )
    else:
        module, name = CLASSES[name].rsplit(".", 1)
        cls = getattr(importlib.import_module("ts_benchmark.baselines." + module), name)
        adapter = type("Recorded" + name, (Recording, cls), {})(**parameters)
    adapter.validation_history, adapter.smoke = [], smoke
    return adapter


def predict(adapter, past, target, past_mark, target_mark, future):
    # Decoder context contains observed history only. No future target is passed.
    decoder = target.clone()
    decoder[:, -adapter.config.horizon :] = 0
    result = adapter._process(past, decoder, past_mark, target_mark, future)["output"]
    result = result[:, -adapter.config.horizon :, :1]
    if adapter.CovariateFusion is not None:
        result = adapter.CovariateFusion(future, result)
    return adapter._post_process(result, target[:, -adapter.config.horizon :, :1])[0]


@torch.no_grad()
def score(adapter, frame, start, end, args):
    adapter.model.eval()
    if adapter.CovariateFusion is not None:
        adapter.CovariateFusion.eval()
    _, loader = forecasting_data_provider(
        frame.iloc[start - args.seq_len : end],
        adapter.config,
        timeenc=1,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
    )
    square, absolute, count, origins, invariant = 0.0, 0.0, 0, 0, None
    for i, batch in enumerate(loader):
        if args.smoke and i == 2:
            break
        past, target, pm, tm = (x.to(args.device) for x in batch)
        future = target[:, -args.horizon :, 1:]
        if args.future_mode == "history":
            future = torch.zeros_like(future)
        torch.manual_seed(args.seed + 2000 + i)
        result = predict(adapter, past, target, pm, tm, future)
        if i == 0 and args.future_mode == "history":
            torch.manual_seed(args.seed + 2000 + i)
            alternate = predict(adapter, past, target, pm, tm, future + 37.0)
            torch.testing.assert_close(result, alternate, rtol=0, atol=0)
            invariant = True
        error = (result - target[:, -args.horizon :, :1]).double()
        square += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += error.numel()
        origins += len(past)
    return dict(mse=square / count, mae=absolute / count), dict(
        origins=origins, future_invariance=invariant
    )


def run(args):
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    specification = json.loads(Path(args.parameters).read_text())
    parameters = dict(specification["parameters"])
    if args.smoke:
        parameters["num_epochs"] = 1
    args.seq_len, args.horizon = parameters["seq_len"], parameters["horizon"]
    args.evaluate_test = True
    _, protocol = load_windows(args)
    frame = read_data(args.data_path)[protocol["columns"]]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    adapter = adapter_for(args.baseline, parameters, args.smoke)
    write_json(
        output / "config.json",
        dict(args=vars(args), preset=specification, protocol=protocol),
    )
    tv = frame.iloc[: protocol["validation_end"]]
    adapter.forecast_fit(
        tv.iloc[:, :1], covariates={"exog": tv.iloc[:, 1:]}, train_ratio_in_tv=0.875
    )
    np.testing.assert_allclose(
        np.r_[adapter.scaler1.mean_, adapter.scaler2.mean_],
        protocol["scaler_mean"],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.r_[adapter.scaler1.scale_, adapter.scaler2.scale_],
        protocol["scaler_scale"],
        rtol=1e-10,
    )
    adapter.model.load_state_dict(adapter.check_point["Model"])
    if adapter.CovariateFusion is not None:
        adapter.CovariateFusion.load_state_dict(adapter.check_point["CovariateFusion"])
    parameters_count = sum(p.numel() for p in adapter.model.parameters())
    if adapter.CovariateFusion is not None:
        parameters_count += sum(p.numel() for p in adapter.CovariateFusion.parameters())
    standardized = pd.DataFrame(
        ((frame.values - protocol["scaler_mean"]) / protocol["scaler_scale"]).astype(
            np.float32
        ),
        index=frame.index,
        columns=frame.columns,
    )
    metrics, audit = {}, {}
    for split, start, end in [
        ("validation", protocol["train_end"], protocol["validation_end"]),
        ("test", protocol["validation_end"], len(frame)),
    ]:
        metrics[split], audit[split] = score(adapter, standardized, start, end, args)
        if not args.smoke:
            assert audit[split]["origins"] == protocol["origins"][split]
        print(split, metrics[split], audit[split], flush=True)
    selection = dict(
        epoch=int(np.argmin(adapter.validation_history)) + 1,
        criterion="official adapter validation loss",
        loss=adapter.config.loss,
    )
    torch.save(
        dict(
            state_dicts=adapter.check_point,
            preset=specification,
            args=vars(args),
            protocol=protocol,
            selection=selection,
        ),
        output / "model.pt",
    )
    write_json(
        output / "history.json", dict(validation_loss=adapter.validation_history)
    )
    result = dict(
        args=vars(args),
        model=args.baseline,
        preset=specification,
        protocol=protocol,
        selection=selection,
        metrics=metrics,
        evaluation_audit=audit,
        parameters=parameters_count,
        elapsed_seconds=time.time() - started,
        metric_space="DAG_train_prefix_standardized",
        probability_metrics="not applicable: these repository adapters produce point forecasts",
        memory_adaptation=(
            "TFT H360: checkpointed microbatches32, unchanged effective batch and optimizer steps"
            if args.baseline == "TFT" and args.horizon == 360
            else None
        ),
        cuda_peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        smoke=args.smoke,
    )
    write_json(output / "summary.json", result)
    return result
