"""Freeze validation choices and evaluate the selected core models."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cellselect.evaluation import freeze_protocol, evaluate_frozen
from cellselect.training import PreparedData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "test"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", default="results/core")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment", help="Declared training grid (required for freeze)")
    args = parser.parse_args()
    if args.action == "freeze":
        if not args.experiment:
            parser.error("freeze requires --experiment to verify the complete planned grid")
        specification = json.loads(Path(args.experiment).read_text())
        settings = specification["common"]
        data = PreparedData(
            args.data,
            marker=settings.get("marker", "CD16"),
            train_fraction=settings.get("train_fraction", 1.0),
            subset_seed=settings.get("subset_seed", 20261021),
        )
        result = freeze_protocol(data, args.runs, args.protocol, expected_spec=specification)
        print(
            {
                k: result[k]
                for k in [
                    "seeds",
                    "weights",
                    "checkpoint_epochs",
                    "reference_epoch",
                    "selected_by_rmse",
                    "selected_by_utility",
                    "regression_baseline",
                    "classifier_baseline",
                ]
            }
        )
    else:
        settings = json.loads(Path(args.protocol).read_text())
        data = PreparedData(
            args.data,
            marker=settings["marker"],
            train_fraction=settings.get("train_fraction", 1.0),
            subset_seed=settings.get("subset_seed", 20261021),
        )
        result = evaluate_frozen(
            data, args.runs, args.protocol, args.output, args.bootstrap, args.device
        )
        print(
            f"Evaluated {len(result['metrics'])} model/budget combinations on {result['n_test']} cells"
        )


if __name__ == "__main__":
    main()
