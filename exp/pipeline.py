"""Execute each selected recipe with portable, seed-matched stage checkpoints."""

import copy
import importlib
import json
from pathlib import Path

from data_provider.data_loader import load_windows, subset
from utils.config import ROOT
from .forecast import run as run_forecast


WORKERS = {
    None: "legacy",
    "wfuture_focused": "future",
    **{
        "history_" + name: name
        for name in (
            "native",
            "adaptive",
            "explore",
            "gap",
            "local",
            "guarded",
            "rescue",
            "rank",
            "horizon",
            "switches",
            "broad",
            "search",
        )
    },
}


def evaluate(args):
    import torch

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (
        saved["args"]["seed"] != args.seed
        or saved["args"]["future_mode"] != args.future_mode
    ):
        raise ValueError("Checkpoint seed and task must match the requested run")
    args.recipe_worker = saved["args"].get(
        "recipe_worker",
        getattr(args, "recipe_worker", WORKERS.get(saved.get("worker"), "legacy")),
    )
    # Preserve the factory identity when the original checkpoint predates this release.
    saved_args = saved["args"]
    if "recipe_worker" not in saved_args:
        from models.CAPE import build_model
        from layers.embedding import TransformerDiffusionConfig
        from utils.evaluation import score, write_json
        import random
        import numpy as np

        runtime = {
            k: getattr(args, k)
            for k in (
                "data_path",
                "device",
                "threads",
                "evaluate_test",
                "output_dir",
                "checkpoint",
                "smoke",
                "recipe_worker",
            )
        }
        vars(args).update(saved_args)
        vars(args).update(runtime)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.threads)
        data, protocol = load_windows(args)
        for key in (
            "columns",
            "length",
            "train_end",
            "validation_end",
            "scaler_mean",
            "scaler_scale",
        ):
            if protocol[key] != saved["protocol"][key]:
                raise ValueError("Checkpoint/data mismatch: " + key)
        model = build_model(
            args, protocol, TransformerDiffusionConfig(**saved["model_config"])
        ).to(args.device)
        model.load_state_dict(saved["state_dict"], strict=True)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        metrics = {
            split: score(model, windows, args, args.seed + 2000, args.samples)
            for split, windows in data.items()
            if split != "train"
        }
        result = dict(
            args=vars(args),
            protocol=protocol,
            metrics=metrics,
            selection=saved["selection"],
        )
        write_json(out / "summary.json", result)
        print(metrics, flush=True)
        return result
    return run_forecast(args)


def run(args, overrides=None):
    recipes = json.loads((ROOT / "scripts" / args.mode / "recipes.json").read_text())
    recipe = recipes[args.dataset][args.length]
    if "by_seed" in recipe:
        recipe = recipe["by_seed"][str(args.seed)]
    checkpoints = {}
    result = None
    for stage in recipe["stages"]:
        worker = stage["worker"]
        output = str(Path(args.output_dir) / stage["id"])
        values = dict(stage["arguments"], **(overrides or {}))
        values.update(
            seed=args.seed,
            data_path=args.data_path,
            device=args.device,
            threads=args.threads,
            output_dir=output,
            checkpoint=None,
            evaluate_test=args.evaluate_test,
            smoke=args.smoke,
            recipe_worker=worker,
        )
        job = dict(arguments=values)
        for field in ("init", "guard", "blend"):
            if field + "_stage" in stage:
                checkpoint = checkpoints[stage[field + "_stage"]]
                values[field + "_checkpoint"] = checkpoint
                job[field + "_checkpoint"] = checkpoint
        if args.smoke:
            values.update(
                epochs=min(1, values.get("epochs", 0)),
                refine_epochs=min(1, values.get("refine_epochs", 0)),
                sample_finetune_epochs=min(1, values.get("sample_finetune_epochs", 0)),
                module_warm_epochs=min(1, values.get("module_warm_epochs", 0)),
                diffusion_steps=4,
                validation_samples=4,
                samples=4,
                validation_origins=4,
                batch_size=2,
                eval_batch_size=2,
                refine_origins=4,
                refine_samples=2,
                refine_batch_size=1,
                sample_finetune_origins=4,
                sample_finetune_batch_size=1,
            )
        print(
            "CAPE",
            args.mode,
            args.dataset,
            args.length,
            args.seed,
            stage["id"],
            flush=True,
        )
        if worker == "legacy":
            child = copy.deepcopy(args)
            vars(child).update(values)
            result = run_forecast(child)
        else:
            module = importlib.import_module("exp.recipes." + worker)
            if worker in ("native", "adaptive"):
                child = module.parser().parse_args(["--output-dir", output])
                vars(child).update(values)
                old_loader = module.load_windows
                if args.smoke:
                    module.load_windows = lambda a: tiny_windows(a)
                try:
                    result = module.run(child)
                finally:
                    module.load_windows = old_loader
            else:
                result = module.run(job, output, args.device, args.smoke)
        checkpoints[stage["id"]] = result.get(
            "checkpoint_ref", str(Path(output) / "model.pt")
        )
    Path(args.output_dir, "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def tiny_windows(args):
    data, protocol = load_windows(args)
    return {key: subset(windows, 4) for key, windows in data.items()}, protocol
