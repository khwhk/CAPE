"""Run pinned upstream baselines with the saved dataset-specific parameters."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISIONS = dict(
    DAG="0758990e2c73bb54138ea3e7b11a35cbc5476bcc",
    KITE="3140ee824cbd80c5ec7fdf2b54666210519d0b39",
)


def setup(repository):
    path = ROOT / "external" / repository
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/decisionintelligence/" + repository + ".git",
                str(path),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", REVISIONS[repository]],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISIONS[repository]:
        raise ValueError("Baseline checkout must be at " + REVISIONS[repository])
    if repository == "KITE":
        patch = str(ROOT / "scripts/baselines/kite_samples.patch")
        reverse = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "apply",
                "--unidiff-zero",
                "--reverse",
                "--check",
                patch,
            ],
            capture_output=True,
        )
        if reverse.returncode:
            subprocess.run(
                ["git", "-C", str(path), "apply", "--unidiff-zero", "--check", patch],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(path), "apply", "--unidiff-zero", patch], check=True
            )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("DAG", "KITE", "GCGNet", "TimeXer", "PatchTST", "DLinear"),
        required=True,
    )
    parser.add_argument("--mode", choices=("w_future", "wo_future"), default="w_future")
    parser.add_argument("--dataset", default="Sdwpfh1")
    parser.add_argument("--length", choices=("short", "long"), default="short")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root", default=str(ROOT / "dataset/forecasting"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/baselines"))
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    record = json.loads((Path(__file__).parent / "presets.json").read_text())[
        args.mode
    ][args.model][args.dataset][args.length]
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    model=args.model,
                    mode=args.mode,
                    dataset=args.dataset,
                    length=args.length,
                    seeds=args.seeds,
                    **record
                ),
                indent=2,
            )
        )
        return
    setup("DAG")
    if args.model in ("DAG", "KITE"):
        setup("KITE")
    if args.setup_only:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path[:0] = [str(ROOT), str(ROOT / "external/DAG")]
    import ts_benchmark.baselines

    ts_benchmark.baselines.__path__.append(
        str(ROOT / "external/KITE/ts_benchmark/baselines")
    )
    strong = args.model in ("DAG", "KITE")
    worker = importlib.import_module(
        "exp.baselines." + ("strong" if strong else "point")
    )
    from data_provider.data_loader import read_data
    from utils.evaluation import aggregate_results, write_json

    data = str(Path(args.data_root).resolve() / (args.dataset + ".csv"))
    target = read_data(data).columns[-1]
    root = (
        Path(args.output_dir).resolve()
        / args.model
        / args.mode
        / args.dataset
        / args.length
    )
    root.mkdir(parents=True, exist_ok=True)
    parameter_path = root / "parameters.json"
    write_json(parameter_path, record["parameters"] if strong else record)
    results = []
    for seed in args.seeds:
        child = argparse.Namespace(
            model=args.model.lower() if strong else args.model,
            baseline=args.model,
            parameters=str(parameter_path),
            data_path=data,
            target=target,
            seed=seed,
            device="cuda:0",
            samples=100,
            eval_batch_size=4 if strong else 16,
            threads=2,
            future_mode="realized" if args.mode == "w_future" else "history",
            smoke=args.smoke,
            output_dir=str(root / ("seed_" + str(seed))),
        )
        results.append(worker.run(child))
    write_json(root / "aggregate.json", aggregate_results(results))


if __name__ == "__main__":
    main()
