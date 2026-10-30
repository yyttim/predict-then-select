"""Train the declared models; test data are opened only by evaluate_study.py."""

import argparse
from pathlib import Path

from cellselect.data import load_data
from cellselect.study import lock_spec, run_name
from cellselect.training import train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/study.json"))
    parser.add_argument("--output", type=Path, default=Path("results/study"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, help="Run only one declared seed")
    parser.add_argument("--method", choices=("mse", "bce", "joint"))
    parser.add_argument("--fraction", type=float, choices=(0.25, 1.0))
    args = parser.parse_args()
    data = load_data(args.data)
    spec = lock_spec(data, args.config, args.output)
    runs = [
        config
        for config in spec["runs"]
        if (args.seed is None or config["seed"] == args.seed)
        and (args.method is None or config["method"] == args.method)
        and (args.fraction is None or config["fraction"] == args.fraction)
    ]
    for i, config in enumerate(runs, 1):
        name = run_name(config)
        print(f"[{i}/{len(runs)}] {name}", flush=True)
        train_run(data, {**config, "device": args.device}, args.output / "runs" / name)


if __name__ == "__main__":
    main()
