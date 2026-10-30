"""Independently check a completed frozen evaluation without rerunning inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(root, relative, digest):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and sha(path) == digest, f"Changed artifact: {relative}")
    return path


def independent_metrics(y, predictions, score, ids, threshold, fraction):
    k = math.ceil(fraction * len(y))
    chosen = sorted(range(len(y)), key=lambda i: (-float(predictions[score][i]), int(ids[i])))[:k]
    high = y >= threshold
    result = {
        "n_cells": len(y),
        "n_selected": k,
        "n_high": int(high.sum()),
        "mean_marker": float(np.mean(y[chosen])),
        "purity": float(np.mean(high[chosen])),
        "random_mean": float(np.mean(y)),
        "random_purity": float(np.mean(high)),
        "oracle_mean": float(np.mean(sorted(y)[-k:])),
        "oracle_purity": min(int(high.sum()), k) / k,
    }
    if "mean" in predictions:
        result["rmse"] = float(np.sqrt(np.mean(np.square(predictions["mean"] - y))))
    if "hit_logit" in predictions:
        z = predictions["hit_logit"]
        probability = np.exp(-np.logaddexp(0.0, -z))
        result["brier"] = float(np.mean(np.square(probability - high)))
        result["log_loss"] = float(
            np.mean(np.where(high, np.logaddexp(0.0, -z), np.logaddexp(0.0, z)))
        )
    return result


def verify(study, pool="test"):
    require(pool in {"test", "external"}, "Unknown evaluation pool")
    root, output = Path(study), Path(study) / pool
    spec, selection, evaluation = (
        read(root / "study.json"),
        read(root / "selection.json"),
        read(output / "evaluation.json"),
    )
    require(evaluation["status"] == "complete", "Evaluation is not complete")
    require(
        selection["study_sha256"] == sha(root / "study.json")
        and evaluation["selection_sha256"] == sha(root / "selection.json"),
        "Frozen selection binding differs",
    )
    require(
        spec["data_manifest_sha256"]
        == selection["data_manifest_sha256"]
        == evaluation["data_manifest_sha256"],
        "Data binding differs",
    )
    seeds = spec["settings"]["seeds"]
    require(
        seeds
        and len(seeds) == len(set(seeds))
        and selection["primary_fraction"] == spec["settings"]["primary_fraction"],
        "Seed or primary-budget declaration differs",
    )
    threshold = selection["threshold"]
    require(np.isfinite(threshold) and threshold == spec["threshold"]["value"], "Threshold changed")
    cells = pd.read_csv(
        checked(output, "cells.csv", evaluation["cells_sha256"]), float_precision="round_trip"
    )
    metrics = pd.read_csv(
        checked(output, "metrics.csv", evaluation["metrics_sha256"]), float_precision="round_trip"
    )
    ids, y = cells.global_index.to_numpy(), cells.target.to_numpy(dtype=float)
    require(
        len(ids) == evaluation["n_cells"]
        and len(set(ids)) == len(ids)
        and np.issubdtype(ids.dtype, np.integer)
        and len(ids) > 0
        and np.isfinite(y).all()
        and cells.fov_id.nunique() == evaluation["n_fields"],
        "Invalid evaluation cells",
    )
    choices = selection["selections"]
    keys = [(c["train_fraction"], c["method"], c["selected_for"]) for c in choices]
    require(
        len(keys) == len(set(keys))
        and set(keys)
        == {(c["fraction"], c["method"], p) for c in spec["runs"] for p in ("mean", "purity")},
        "Selection grid differs",
    )
    configs = {(c["fraction"], c["method"], c["lambda_binary"], c["seed"]): c for c in spec["runs"]}
    require(len(configs) == len(spec["runs"]), "Duplicate declared run")
    files = {p["checkpoint"]: p for p in evaluation["prediction_files"]}
    required_states = {s["checkpoint"] for c in choices for s in c["states"]}
    require(
        len(files) == len(evaluation["prediction_files"]) and set(files) == required_states,
        "Prediction state grid differs",
    )
    cache, expected = {}, {}
    identity = [
        "train_fraction",
        "selected_for",
        "method",
        "weight",
        "step",
        "seed",
        "score",
        "selection_fraction",
    ]
    for choice in choices:
        require([s["seed"] for s in choice["states"]] == seeds, "Selected seeds are not paired")
        method = choice["method"]
        active = (
            ["mean"]
            if method == "mse"
            else ["hit_logit"]
            if method == "bce"
            else ["mean", "hit_logit"]
        )
        intended = (
            "mean"
            if method == "mse" or (method == "joint" and choice["selected_for"] == "mean")
            else "hit_logit"
        )
        require(choice["score"] == intended, "Selection purpose uses the wrong score")
        for state in choice["states"]:
            relative = state["checkpoint"]
            path = checked(root, relative, state["checkpoint_sha256"])
            if relative not in cache:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                record = files[relative]
                with np.load(
                    checked(output, record["file"], record["sha256"]), allow_pickle=False
                ) as saved:
                    require(
                        np.array_equal(saved["global_index"], ids)
                        and set(saved.files) == {"global_index", *active},
                        "Prediction IDs or active heads differ",
                    )
                    arrays = {key: saved[key].copy() for key in active}
                require(
                    all(v.shape == y.shape and np.isfinite(v).all() for v in arrays.values()),
                    "Invalid score arrays",
                )
                cache[relative] = checkpoint, arrays
            checkpoint, arrays = cache[relative]
            config_key = choice["train_fraction"], method, choice["weight"], state["seed"]
            require(config_key in configs, "Selected configuration is absent from study")
            config = configs[config_key]
            require(
                {k: v for k, v in checkpoint["config"].items() if k != "device"} == config
                and checkpoint["step"] == choice["step"]
                and choice["step"] in config["checkpoint_steps"]
                and checkpoint["threshold"] == threshold
                and checkpoint["manifest_sha256"] == selection["data_manifest_sha256"],
                "Checkpoint configuration or data binding differs",
            )
            for score in active:
                for fraction in spec["settings"]["selection_fractions"]:
                    key = (
                        choice["train_fraction"],
                        choice["selected_for"],
                        method,
                        choice["weight"],
                        choice["step"],
                        state["seed"],
                        score,
                        fraction,
                    )
                    require(
                        key not in expected and 0 < fraction <= 1,
                        "Duplicate row or invalid decision budget",
                    )
                    expected[key] = (
                        score == intended,
                        independent_metrics(y, arrays, score, ids, threshold, fraction),
                    )
    keys = [tuple(row[name] for name in identity) for _, row in metrics.iterrows()]
    require(
        len(keys) == len(set(keys)) and set(keys) == set(expected), "Evaluation metric grid differs"
    )
    max_error = 0.0
    for key, (_, row) in zip(keys, metrics.iterrows()):
        primary, numbers = expected[key]
        require(row.primary_score == primary, "Primary score label differs")
        for name in {"rmse", "brier", "log_loss"} - numbers.keys():
            require(
                name not in row or pd.isna(row[name]),
                f"Metric reported for an inactive head: {name}",
            )
        for name, value in numbers.items():
            require(name in row and np.isfinite(row[name]), f"Missing or invalid metric: {name}")
            error = abs(float(row[name]) - value)
            require(error <= 1e-8, f"Metric mismatch: {key}/{name}")
            max_error = max(max_error, error)
    return {
        "status": "passed",
        "pool": pool,
        "metric_rows": len(metrics),
        "prediction_files": len(files),
        "max_absolute_error": max_error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/study"))
    parser.add_argument("--pool", choices=["test", "external"], default="test")
    args = parser.parse_args()
    print(json.dumps(verify(args.study, args.pool), allow_nan=False))
