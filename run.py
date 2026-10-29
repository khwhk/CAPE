"""One entry point for both exogenous forecasting settings."""

import argparse
import copy
import json
from pathlib import Path

from exp.pipeline import run, evaluate
from utils.config import load_settings
from utils.evaluation import aggregate_results, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("w_future", "wo_future"), default="w_future")
    parser.add_argument("--dataset", default="Sdwpfh1")
    parser.add_argument("--length", choices=("short", "long"), default="short")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="dataset/forecasting")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--checkpoint", help="Evaluate a checkpoint produced by this release"
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny train/validation pipeline check"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the selected run without training"
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override preset values (JSON or string)",
    )
    options = parser.parse_args()
    args = load_settings(options.mode, options.dataset, options.length)
    overrides = {}
    for assignment in options.set:
        key, value = assignment.split("=", 1)
        if key not in vars(args):
            parser.error("Unknown setting: " + key)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        setattr(args, key, value)
        overrides[key] = value
    args.mode, args.dataset, args.length = options.mode, options.dataset, options.length
    args.future_mode = "realized" if options.mode == "w_future" else "history"
    args.data_path = str(Path(options.data_root) / (options.dataset + ".csv"))
    args.device, args.checkpoint = options.device, options.checkpoint
    args.evaluate_test, args.smoke = options.evaluate_test, options.smoke
    root = Path(options.output_dir) / options.mode / options.dataset / options.length
    results = []
    if options.checkpoint and len(options.seeds) != 1:
        parser.error(
            "Checkpoint evaluation requires --seeds with the checkpoint seed only"
        )
    for seed in options.seeds:
        child = copy.deepcopy(args)
        child.seed, child.output_dir = seed, str(root / ("seed_" + str(seed)))
        if options.dry_run:
            print(
                json.dumps(
                    dict(
                        mode=child.mode,
                        dataset=child.dataset,
                        length=child.length,
                        seed=seed,
                        output_dir=child.output_dir,
                    ),
                    indent=2,
                )
            )
        else:
            results.append(
                evaluate(child) if options.checkpoint else run(child, overrides)
            )
    if options.dry_run:
        return
    write_json(root / "aggregate.json", aggregate_results(results))


if __name__ == "__main__":
    main()
