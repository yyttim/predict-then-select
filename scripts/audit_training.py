"""Audit completed training runs and validation-only optimization diagnostics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cellselect.training import file_sha256, write_json


def audit(spec_file, run_root, require_complete=False):
    spec = json.loads(Path(spec_file).read_text())
    runs, pending, reproduction, histories = [], [], [], {}
    for override in spec["runs"]:
        config = {**spec["common"], **override}
        name = config.pop("name")
        directory = Path(run_root) / name
        if not (directory / "complete.json").exists():
            pending.append(name)
            continue
        saved = json.loads((directory / "config.json").read_text())
        if config != saved:
            raise ValueError(f"Declared configuration changed: {name}")
        complete = json.loads((directory / "complete.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        history = [
            json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()
        ]
        histories[name] = (config, history)
        if [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
            raise ValueError(f"Missing, duplicate or reordered training epochs: {name}")
        if complete["test_evaluated"] or provenance["test_evaluated"]:
            raise ValueError("Training must not evaluate the test partition")
        for filename, key in [
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
            ("final.pt", "checkpoint_sha256"),
            ("validation_predictions.npz", "validation_predictions_sha256"),
        ]:
            if file_sha256(directory / filename) != complete[key]:
                raise ValueError(f"Changed completed artifact: {name}/{filename}")
        for source in ["model.py", "training.py", "data.py", "metrics.py"] + (
            ["losses.py"] if config.get("objective", "mse") != "mse" else []
        ):
            if file_sha256(ROOT / "src/cellselect" / source) != provenance["source_sha256"][source]:
                raise ValueError(f"Training source changed: {name}/{source}")
        for row in history:
            if not all(
                np.isfinite(row[key])
                for key in ["train_loss", "train_mse_standardized", "train_ce"]
            ):
                raise ValueError(f"Nonfinite training objective: {name}")
        candidates = complete["candidates"]
        if [row["epoch"] for row in candidates] != config["checkpoint_epochs"]:
            raise ValueError(f"Incomplete candidate checkpoints: {name}")
        for candidate in candidates:
            for path_key, hash_key in [
                ("checkpoint_file", "checkpoint_sha256"),
                ("predictions_file", "predictions_sha256"),
            ]:
                if file_sha256(directory / candidate[path_key]) != candidate[hash_key]:
                    raise ValueError(f"Changed checkpoint candidate: {name}")
        best_utility = max(candidates, key=lambda row: row["validation"]["utility"])
        best_rmse = min(candidates, key=lambda row: row["validation"]["rmse"])
        postwarm = [
            row for row in history if row["epoch"] > config.get("decision_warmup_epochs", 0)
        ]
        active = [
            row["decision_active_fraction"] for row in postwarm if "decision_active_fraction" in row
        ]
        n_train = provenance["n_train"]
        runs.append(
            {
                "name": name,
                "marker": config["marker"],
                "train_fraction": config["train_fraction"],
                "seed": config["seed"],
                "objective": config.get("objective", "mse"),
                "mode": config["mode"],
                "lambda_class": config["lambda_class"],
                "pg_h": config.get("pg_h"),
                "n_train": n_train,
                "epochs": config["epochs"],
                "optimizer_updates": config["epochs"]
                * int(np.ceil(n_train / config["batch_size"])),
                "elapsed_seconds": complete["elapsed_seconds"],
                "peak_gpu_memory_bytes": complete["peak_gpu_memory_bytes"],
                "best_validation_utility_epoch": best_utility["epoch"],
                "best_validation_utility": best_utility["validation"]["utility"],
                "best_validation_rmse_epoch": best_rmse["epoch"],
                "final_validation_utility": history[-1]["validation"]["utility"],
                "utility_optimum_at_final_candidate": best_utility["epoch"] == config["epochs"],
                "minimum_active_fraction": min(active) if active else None,
                "final_active_fraction": active[-1] if active else None,
                "history_sha256": file_sha256(directory / "history.jsonl"),
            }
        )
        if (
            config["marker"] == "CD16"
            and config["train_fraction"] == 1
            and config.get("objective", "mse") == "mse"
        ):
            original_name = (
                f"class_means_seed_{config['seed']}"
                if config["mode"] == "classifier"
                else f"lambda_{config['lambda_class']:g}_seed_{config['seed']}"
            )
            original = ROOT / "results/runs/core" / original_name
            if original.exists():
                differences = []
                for candidate in candidates:
                    old_file = original / candidate["predictions_file"]
                    if not old_file.exists():
                        continue
                    with (
                        np.load(old_file) as old,
                        np.load(directory / candidate["predictions_file"]) as new,
                    ):
                        if not np.array_equal(old["global_index"], new["global_index"]):
                            raise ValueError("Original/new validation rows differ")
                        differences.append(float(np.max(np.abs(old["score"] - new["score"]))))
                reproduction.append(
                    {
                        "run": name,
                        "original_run": original_name,
                        "matched_checkpoints": len(differences),
                        "maximum_absolute_score_difference": max(differences),
                    }
                )
    if require_complete and pending:
        raise RuntimeError(f"Study incomplete: {len(pending)} of {len(spec['runs'])} runs pending")
    warmups = []
    for name, (config, history) in histories.items():
        if config.get("objective") != "pg_central":
            continue
        matching = [
            (other_name, other_history)
            for other_name, (other, other_history) in histories.items()
            if other.get("objective") == "spo_plus"
            and all(
                other[key] == config[key]
                for key in ["marker", "train_fraction", "subset_seed", "seed"]
            )
        ]
        if len(matching) != 1:
            raise ValueError("Missing or duplicate matched SPO+ warmup comparator")
        other_name, other_history = matching[0]
        differences = []
        for a, b in zip(history[: config["decision_warmup_epochs"]], other_history):
            for key in [
                "train_loss",
                "train_mse_standardized",
                "train_ce",
                "learning_rate",
                "decision_active_fraction",
            ]:
                differences.append(abs(a[key] - b[key]))
            if "validation" in a:
                for key in ["rmse", "utility", "score_std_in_target_units"]:
                    differences.append(abs(a["validation"][key] - b["validation"][key]))
        difference = max(differences)
        if difference > 1e-8:
            raise ValueError(f"PG and SPO+ warmup trajectories differ: {name}")
        warmups.append(
            {
                "pg_run": name,
                "spo_run": other_name,
                "warmup_epochs": config["decision_warmup_epochs"],
                "maximum_absolute_logged_difference": difference,
            }
        )
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": file_sha256(spec_file),
        "declared_runs": len(spec["runs"]),
        "completed_runs": len(runs),
        "pending": pending,
        "training_wall_seconds": sum(row["elapsed_seconds"] for row in runs),
        "runs": runs,
        "original_baseline_reproduction": reproduction,
        "paired_spo_pg_warmup_checks": warmups,
        "scope": "Training and validation only. A final-candidate optimum is a review flag, not automatic evidence of undertraining. Direct-decision score RMSE is not an optimization failure criterion.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/extended.json")
    parser.add_argument("--runs", default="results/runs/extended")
    parser.add_argument("--output", default="results/extended_training_audit.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.spec, args.runs, args.require_complete)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "completed": result["completed_runs"],
                "declared": result["declared_runs"],
                "completed_wall_minutes": result["training_wall_seconds"] / 60,
                "final_candidate_flags": [
                    r["name"] for r in result["runs"] if r["utility_optimum_at_final_candidate"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
