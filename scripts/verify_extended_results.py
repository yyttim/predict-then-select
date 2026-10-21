"""Recompute frozen validation choices and outcome metrics from saved arrays.

Use complete candidate pools and independent NumPy metric calculations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cellselect.bootstrap import paired_topk_bootstrap
from cellselect.metrics import paired_seed_field_bootstrap

CONTRASTS = {
    "joint_minus_regression": ("joint", "regression"),
    "spo_plus_minus_regression": ("spo_plus", "regression"),
    "pg_central_minus_regression": ("pg_central", "regression"),
    "utility_minus_predictive_selector": ("utility_selector", "predictive_selector"),
    "regression_minus_class_means": ("regression", "class_means"),
}
TYPE_FIELDS = {
    "accuracy",
    "macro_f1",
    "within_type_gap",
    "type_allocation_gap",
    "same_counts_oracle_utility",
    "selected_type_counts",
    "type_matched_random_utility",
    "within_type_enrichment",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, tolerance=1e-8):
    if expected is None:
        require(actual is None, f"{name}: expected an undefined value")
    else:
        require(
            np.shape(actual) == np.shape(expected)
            and np.allclose(actual, expected, rtol=0, atol=tolerance),
            f"{name}: numerical mismatch",
        )


def _ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2
    return midranks[inverse]


def numpy_metrics(y, scores, ids, fraction, calibration, classes=None, logits=None):
    y, scores, ids = np.asarray(y, float), np.asarray(scores, float), np.asarray(ids)
    require(
        y.ndim == 1 and len(y) and scores.shape == y.shape == ids.shape, "Misaligned metric inputs"
    )
    require(
        np.isfinite(y).all() and np.isfinite(scores).all() and len(np.unique(ids)) == len(ids),
        "Nonfinite outcomes/scores or duplicate IDs",
    )
    require(0 < fraction <= 1 and calibration["slope"] > 0, "Invalid budget/calibration")
    k = math.ceil(fraction * len(y))
    selected = np.lexsort((ids, -scores))[:k]
    oracle = np.lexsort((ids, -y))[:k]
    utility, best = float(y[selected].mean()), float(y[oracle].mean())
    variable = len(y) > 1 and np.any(y != y[0]) and np.any(scores != scores[0])
    result = {
        "rmse": float(np.sqrt(np.square(y - scores).mean())),
        "mae": float(np.abs(y - scores).mean()),
        "pearson": float(np.corrcoef(y, scores)[0, 1]) if variable else None,
        "spearman": float(np.corrcoef(_ranks(y), _ranks(scores))[0, 1]) if variable else None,
        "calibrated_rmse": float(
            np.sqrt(
                np.square(y - (calibration["slope"] * scores + calibration["intercept"])).mean()
            )
        ),
        "fraction": fraction,
        "k": k,
        "utility": utility,
        "random_expected_utility": float(y.mean()),
        "enrichment_difference": utility - float(y.mean()),
        "oracle_utility": best,
        "oracle_gap": best - utility,
    }
    for name in ("pearson", "spearman"):
        if result[name] is not None and not np.isfinite(result[name]):
            result[name] = None
    if classes is not None:
        classes = np.asarray(classes)
        require(
            classes.shape == y.shape and np.isin(classes, [0, 1, 2]).all(),
            "Invalid true class labels",
        )
        counts = {str(int(c)): int(np.sum(classes[selected] == c)) for c in np.unique(classes)}
        matched = np.concatenate(
            [np.sort(y[classes == int(c)])[-count:] for c, count in counts.items() if count]
        )
        same = float(matched.mean())
        random = sum(count * float(y[classes == int(c)].mean()) for c, count in counts.items()) / k
        result.update(
            same_counts_oracle_utility=same,
            within_type_gap=max(0.0, same - utility),
            type_allocation_gap=max(0.0, best - same),
            selected_type_counts=counts,
            oracle_overlap=float(len(set(selected) & set(oracle)) / k),
            type_matched_random_utility=random,
            within_type_enrichment=utility - random,
        )
        if logits is not None:
            require(logits.shape == (len(y), 3) and np.isfinite(logits).all(), "Invalid logits")
            predicted = logits.argmax(axis=1)
            values = []
            for c in range(3):
                tp = int(np.sum((classes == c) & (predicted == c)))
                fp = int(np.sum((classes != c) & (predicted == c)))
                fn = int(np.sum((classes == c) & (predicted != c)))
                values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            result.update(
                accuracy=float(np.mean(predicted == classes)), macro_f1=float(np.mean(values))
            )
    return result


def independent_choices(rows):
    seeds = sorted({r["seed"] for r in rows})
    candidates = {}
    for row in rows:
        key = row["family"], row["epoch"]
        candidates.setdefault(key, []).append(row)
    means = []
    for (family, epoch), members in sorted(candidates.items()):
        require(
            sorted(r["seed"] for r in members) == seeds,
            "Incomplete/duplicate paired validation seeds",
        )
        means.append(
            {
                "family": family,
                "epoch": epoch,
                "rmse": float(np.mean([r["rmse"] for r in members])),
                "utility": float(np.mean([r["utility"] for r in members])),
            }
        )
    filters = {
        "regression": lambda f: f == "mse_lambda_0",
        "class_means": lambda f: f == "class_means",
        "joint": lambda f: f.startswith("mse_lambda_") and f != "mse_lambda_0",
        "predictive_selector": lambda f: f.startswith("mse_lambda_"),
        "utility_selector": lambda f: f.startswith("mse_lambda_"),
        "spo_plus": lambda f: f == "spo_plus",
        "pg_central": lambda f: f.startswith("pg_central_h_"),
    }
    result = {}
    for policy, allowed in filters.items():
        eligible = [r for r in means if allowed(r["family"])]
        require(bool(eligible), f"Missing method: {policy}")
        criterion = "rmse" if policy == "predictive_selector" else "utility"
        chosen = min(eligible, key=lambda r: r[criterion] if criterion == "rmse" else -r[criterion])
        result[policy] = {
            "family": chosen["family"],
            "epoch": chosen["epoch"],
            "criterion": criterion,
            "validation_rmse": chosen["rmse"],
            "validation_utility": chosen["utility"],
        }
    return result


def _family(config):
    if config.get("mode", "joint") == "classifier":
        return "class_means"
    objective = config.get("objective", "mse")
    if objective == "mse":
        return f"mse_lambda_{float(config.get('lambda_class', 0)):g}"
    if objective == "spo_plus":
        return "spo_plus"
    require(objective == "pg_central", "Unknown declared objective")
    return f"pg_central_h_{float(config['pg_h']):g}"


def _load_cohort(root, manifest_hash):
    root = Path(root)
    require(sha(root / "manifest.json") == manifest_hash, "Cohort manifest changed")
    manifest = read_json(root / "manifest.json")
    for name, expected in manifest["prepared_sha256"].items():
        require(sha(root / name) == expected, f"Prepared artifact changed: {name}")
    metadata = pd.read_csv(root / "metadata.csv")
    images = np.load(root / "images.npy", mmap_mode="r")
    require(
        images.shape == (len(metadata), 4, 28, 28) and metadata.global_index.is_unique,
        "Invalid complete image/ID schema",
    )
    return metadata, images, manifest


def _training_rows(metadata, config):
    rows = np.flatnonzero(metadata.split.to_numpy() == "train")
    fraction = float(config["train_fraction"])
    if fraction < 1:
        key = "split_group" if "split_group" in metadata else "fov_id"
        groups = metadata.iloc[rows][key].to_numpy()
        order = np.random.default_rng(config.get("subset_seed", 20261021)).permutation(
            np.unique(groups)
        )
        rows = rows[np.isin(groups, order[: max(1, math.ceil(fraction * len(order)))])]
    return rows


def _normalization(metadata, images, rows, marker, image_cache):
    ids_hash = hashlib.sha256(metadata.iloc[rows].global_index.to_numpy().tobytes()).hexdigest()
    if ids_hash not in image_cache:
        count, mean, m2 = 0, np.zeros(4), np.zeros(4)
        for start in range(0, len(rows), 512):
            block = np.asarray(images[rows[start : start + 512]], dtype=np.float64)
            require(np.isfinite(block).all(), "Nonfinite training pixels")
            n = len(block) * block.shape[2] * block.shape[3]
            average, variance = block.mean(axis=(0, 2, 3)), block.var(axis=(0, 2, 3))
            delta = average - mean
            m2 += n * variance + delta**2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        image_cache[ids_hash] = mean, np.sqrt(m2 / count)
    y = metadata.iloc[rows][marker].to_numpy(dtype=float)
    classes = metadata.iloc[rows].class_id.to_numpy()
    require(
        np.isfinite(y).all() and y.std() > 0 and set(classes) == {0, 1, 2},
        "Invalid training outcomes/classes",
    )
    mean, std = image_cache[ids_hash]
    return {
        "marker": marker,
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "class_means": [float(y[classes == c].mean()) for c in range(3)],
        "image_mean": mean.tolist(),
        "image_std": std.tolist(),
    }, ids_hash


def _compare_mapping(actual, expected, name):
    require(set(actual) == set(expected), f"{name}: fields differ")
    for key, value in expected.items():
        if isinstance(value, str) or isinstance(value, dict):
            require(actual[key] == value, f"{name}/{key}: mismatch")
        else:
            close(actual[key], value, name + "/" + key)


def _safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Artifact path escapes its declared directory")
    return path


def independent_exchange(y, a, b, ids, fraction):
    k, records = math.ceil(fraction * len(y)), []
    for first, second in zip(a, b):
        left, right = np.zeros(len(y), bool), np.zeros(len(y), bool)
        left[np.lexsort((ids, -first))[:k]] = True
        right[np.lexsort((ids, -second))[:k]] = True
        added, removed = left & ~right, right & ~left
        swapped = int(added.sum())
        require(swapped == int(removed.sum()), "Unequal decision budgets")
        difference = float(y[left].mean() - y[right].mean())
        recovered = float((y[added].sum() - y[removed].sum()) / k)
        close(difference, recovered, "Exchange identity")
        records.append(
            {
                "swapped_cells_per_policy": swapped,
                "selected_overlap_fraction": 1 - swapped / k,
                "mean_added_outcome": float(y[added].mean()) if swapped else None,
                "mean_removed_outcome": float(y[removed].mean()) if swapped else None,
                "utility_difference": difference,
                "difference_from_exchanges": recovered,
            }
        )
    return {
        "k": k,
        "paired_replicates": records,
        "mean_selected_overlap_fraction": float(
            np.mean([r["selected_overlap_fraction"] for r in records])
        ),
    }


def verify_study(data_root, run_root, spec_file, protocol_file, result_root, external_root=None):
    """Fail before writing any audit if a grid, outcome row or interval is missing."""
    data_root, run_root, result_root = Path(data_root), Path(run_root), Path(result_root)
    required = [
        Path(spec_file),
        Path(protocol_file),
        result_root / "results.json",
        result_root / "uncertainty.json",
        result_root / "evaluation_started.json",
    ]
    require(
        all(path.is_file() for path in required),
        "Study is incomplete: freeze, evaluation and uncertainty must all exist",
    )
    spec, protocol, result, uncertainty, started = [read_json(path) for path in required]
    protocol_hash = sha(protocol_file)
    require(
        protocol.get("schema_version") == 1 and protocol["spec_sha256"] == sha(spec_file),
        "Declaration/protocol mismatch",
    )
    require(
        protocol.get("scoring_precision")
        == {
            "network_forward": "float32",
            "regression_denormalization": "float64",
            "classifier_softmax": "float64",
            "classifier_expectation": "float64",
            "classifier_validation_source": "saved_logits",
        },
        "Unspecified/inconsistent scoring precision",
    )
    require(
        result["protocol_sha256"]
        == uncertainty["protocol_sha256"]
        == started["protocol_sha256"]
        == protocol_hash,
        "Frozen protocol hashes differ",
    )
    require(
        uncertainty["results_sha256"] == sha(result_root / "results.json"),
        "Uncertainty belongs to different predictions",
    )
    expected, configurations = {}, {}
    for override in spec["runs"]:
        config = {**spec["common"], **override}
        name = config.pop("name")
        require(
            name not in configurations and not config.get("development_only", False),
            "Duplicate/development run in declared grid",
        )
        configurations[name] = config
        gkey = f"{config['marker']}_fraction_{float(config['train_fraction']):g}"
        for epoch in config["checkpoint_epochs"]:
            expected[name, epoch] = (gkey, _family(config), config)
    record_map = {(r["name"], r["epoch"]): r for r in protocol["records"]}
    require(
        len(record_map) == len(protocol["records"]) and set(record_map) == set(expected),
        "Incomplete/duplicate declared candidate grid",
    )
    require(
        set(protocol["groups"]) == {v[0] for v in expected.values()},
        "Declared group coverage differs",
    )
    for key, (gkey, family, config) in expected.items():
        record = record_map[key]
        require(
            (record["group"], record["family"], record["config"]) == (gkey, family, config),
            "Candidate config/family/group mismatch",
        )
    external_names = protocol.get("external_pools", [])
    require(
        not external_names or external_root is not None,
        "Frozen external pools require their complete data directory",
    )
    expected_predictions, expected_metrics, expected_intervals = {}, set(), set()
    for record in protocol["records"]:
        group = protocol["groups"][record["group"]]
        policies = [
            name
            for name, choice in group["choices"].items()
            if (choice["family"], choice["epoch"]) == (record["family"], record["epoch"])
        ]
        if not policies:
            continue
        pools = ["within_slide_test"] + (
            external_names if record["config"]["train_fraction"] == 1 else []
        )
        for pool in pools:
            key = (record["group"], record["name"], record["epoch"], record["config"]["seed"], pool)
            expected_predictions[key] = record, set(policies)
            for policy in policies:
                for q in protocol["fractions"]:
                    expected_metrics.add(
                        (record["group"], pool, policy, record["config"]["seed"], q)
                    )
    actual_prediction_keys = [
        (r["group"], r["name"], r["epoch"], r["seed"], r["pool"]) for r in result["predictions"]
    ]
    require(
        len(set(actual_prediction_keys)) == len(actual_prediction_keys)
        and set(actual_prediction_keys) == set(expected_predictions),
        "Incomplete/duplicate saved prediction grid",
    )
    actual_metric_keys = [
        (r["group"], r["pool"], r["policy"], r["seed"], r["fraction"]) for r in result["metrics"]
    ]
    require(
        len(set(actual_metric_keys)) == len(actual_metric_keys)
        and set(actual_metric_keys) == expected_metrics,
        "Incomplete/duplicate reported metric grid",
    )
    for gkey, group in protocol["groups"].items():
        for pool in ["within_slide_test"] + (
            external_names if group["train_fraction"] == 1 else []
        ):
            expected_intervals.update(
                (gkey, pool, contrast, q) for contrast in CONTRASTS for q in protocol["fractions"]
            )
    interval_keys = [
        (r["group"], r["pool"], r["comparison"], r["fraction"]) for r in uncertainty["comparisons"]
    ]
    require(
        len(set(interval_keys)) == len(interval_keys) and set(interval_keys) == expected_intervals,
        "Incomplete/duplicate uncertainty grid",
    )
    # No external outcomes have been opened before the complete-grid checks above.
    for name, digest in protocol["inference_source_sha256"].items():
        require(
            sha(_safe_path(ROOT / "src/cellselect", name)) == digest,
            "Inference implementation changed",
        )
    metadata, images, _ = _load_cohort(data_root, protocol["manifest_sha256"])
    validation = metadata.loc[metadata.split == "val"].reset_index(drop=True)
    val_ids = validation.global_index.to_numpy()
    image_cache, norm_cache, validation_rows, hashes = (
        {},
        {},
        {g: [] for g in protocol["groups"]},
        {},
    )

    def check(path, digest):
        path = Path(path)
        if path not in hashes:
            hashes[path] = sha(path)
        require(hashes[path] == digest, f"Artifact hash mismatch: {path.name}")

    paired_data = {}
    for name, config in configurations.items():
        directory = _safe_path(run_root, name)
        complete, provenance = (
            read_json(directory / "complete.json"),
            read_json(directory / "provenance.json"),
        )
        require(
            read_json(directory / "config.json") == config, "Completed run differs from declaration"
        )
        require(
            provenance["source_sha256"]["model.py"] == sha(ROOT / "src/cellselect/model.py"),
            "Trained model code mismatch",
        )
        require(
            provenance["source_sha256"]["training.py"]
            == protocol["inference_source_sha256"]["training.py"],
            "Training implementation differs across the declared study",
        )
        for filename, key in [
            ("final.pt", "checkpoint_sha256"),
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
            ("validation_predictions.npz", "validation_predictions_sha256"),
        ]:
            check(directory / filename, complete[key])
        rows = _training_rows(metadata, config)
        norm, ids_hash = _normalization(metadata, images, rows, config["marker"], image_cache)
        norm_cache[name] = norm
        _compare_mapping(
            read_json(directory / "normalization.json"), norm, "Training-only normalization"
        )
        require(
            provenance["manifest_sha256"] == protocol["manifest_sha256"]
            and provenance["train_ids_sha256"] == ids_hash,
            "Training subset/cohort provenance differs",
        )
        group_name = f"{config['marker']}_fraction_{float(config['train_fraction']):g}"
        group = protocol["groups"][group_name]
        _compare_mapping(
            group["train_counts"][str(config["seed"])],
            {
                "cells": len(rows),
                "fields": int(metadata.iloc[rows].fov_id.nunique()),
                "ids_sha256": ids_hash,
            },
            "Training counts",
        )
        pair = group_name, config["seed"]
        if pair in paired_data:
            require(
                paired_data[pair] == (ids_hash, complete["normalization_sha256"]),
                "Unpaired objective cohorts",
            )
        paired_data[pair] = ids_hash, complete["normalization_sha256"]
        require(
            [c["epoch"] for c in complete["candidates"]] == config["checkpoint_epochs"],
            "Missing checkpoint candidate",
        )
        for candidate in complete["candidates"]:
            record = record_map[name, candidate["epoch"]]
            require(
                record["checkpoint_file"] == candidate["checkpoint_file"]
                and record["predictions_file"] == candidate["predictions_file"],
                "Frozen candidate file mapping changed",
            )
            for field, key in [
                ("checkpoint_file", "checkpoint_sha256"),
                ("predictions_file", "predictions_sha256"),
            ]:
                check(_safe_path(directory, candidate[field]), candidate[key])
                require(
                    record[key] == candidate[key],
                    "Frozen candidate hash differs from completion record",
                )
            checkpoint = torch.load(
                _safe_path(directory, candidate["checkpoint_file"]),
                map_location="cpu",
                weights_only=True,
            )
            require(
                checkpoint["config"] == config
                and checkpoint["epoch"] == candidate["epoch"]
                and checkpoint["manifest_sha256"] == protocol["manifest_sha256"],
                "Checkpoint internal metadata differs",
            )
            _compare_mapping(checkpoint["normalization"], norm, "Checkpoint normalization")
            del checkpoint
            with np.load(
                _safe_path(directory, candidate["predictions_file"]), allow_pickle=False
            ) as saved:
                require(
                    np.array_equal(saved["global_index"], val_ids),
                    "Candidate validation IDs differ",
                )
                score, logits = saved["score"].copy(), saved["logits"].copy()
            y = validation[config["marker"]].to_numpy(dtype=float)
            valid_logits = (
                logits if config["mode"] == "classifier" or config["lambda_class"] > 0 else None
            )
            computed = numpy_metrics(
                y,
                score,
                val_ids,
                protocol["primary_fraction"],
                {"slope": 1.0, "intercept": 0.0},
                validation.class_id.to_numpy(),
                valid_logits,
            )
            training_metrics = {
                key: value
                for key, value in computed.items()
                if key
                not in {"calibrated_rmse", "type_matched_random_utility", "within_type_enrichment"}
            }
            training_metrics["score_std_in_target_units"] = float(np.std(score))
            _compare_mapping(
                candidate["validation"], training_metrics, "Recorded candidate validation metrics"
            )
            if config["mode"] == "classifier":
                # Recompute new freeze scores; the stored score above belongs to
                # the original float32 training-log arithmetic and stays intact.
                probabilities = np.exp(logits.astype(float) - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                score = probabilities @ norm["class_means"]
                computed = numpy_metrics(
                    y,
                    score,
                    val_ids,
                    protocol["primary_fraction"],
                    {"slope": 1.0, "intercept": 0.0},
                    validation.class_id.to_numpy(),
                    valid_logits,
                )
            row = {
                "family": record["family"],
                "epoch": record["epoch"],
                "seed": config["seed"],
                **{key: computed[key] for key in ("rmse", "mae", "pearson", "spearman", "utility")},
            }
            validation_rows[group_name].append(row)
            centered = score - np.mean(score)
            denominator = float(centered @ centered)
            slope = (
                max(0.001, float(centered @ (y - y.mean())) / denominator) if denominator else 1.0
            )
            _compare_mapping(
                record["calibration"],
                {
                    "slope": slope,
                    "intercept": float(y.mean() - slope * score.mean()),
                    "fit_partition": "val",
                },
                "Validation-only calibration",
            )
    for gkey, rows in validation_rows.items():
        group = protocol["groups"][gkey]
        require(
            group["seeds"] == sorted({row["seed"] for row in rows}), "Declared seed set mismatch"
        )
        reported = {
            (row["family"], row["epoch"], row["seed"]): row for row in group["validation_table"]
        }
        require(
            len(reported) == len(rows) == len(group["validation_table"]),
            "Missing/duplicate frozen validation row",
        )
        for row in rows:
            key = row["family"], row["epoch"], row["seed"]
            require(key in reported, "Missing frozen validation metric")
            _compare_mapping(reported[key], row, "Frozen validation table")
        choices = independent_choices(rows)
        require(set(group["choices"]) == set(choices), "Policy set differs")
        for policy, choice in choices.items():
            _compare_mapping(group["choices"][policy], choice, "Validation-only choice")
    pools = {"within_slide_test": metadata.loc[metadata.split == "test"].reset_index(drop=True)}
    if external_names:
        external, external_images, external_manifest = _load_cohort(
            external_root, protocol["external_manifest_sha256"]
        )
        require(
            external_manifest.get("role") == "external_acquisition_evaluation",
            "Unstated external cohort role",
        )
        require(
            (
                (external["batch"] == 1)
                & (external.antibodies == "all")
                & (external.split == "external")
                & (external.class_id == -1)
            ).all(),
            "External outcomes must not acquire type labels or another condition",
        )
        require(
            (external.pool_id == external.slide_id).all()
            and set(external.pool_id) == set(external_names),
            "External pool coverage changed",
        )
        require(
            not set(external.global_index) & set(metadata.global_index),
            "External/main cell IDs overlap",
        )
        pools.update(
            {
                name: external.loc[external.pool_id == name].reset_index(drop=True)
                for name in external_names
            }
        )
        del external_images
    score_map, prediction_lookup = {}, {}
    for prediction in result["predictions"]:
        key = (
            prediction["group"],
            prediction["name"],
            prediction["epoch"],
            prediction["seed"],
            prediction["pool"],
        )
        record, policies = expected_predictions[key]
        require(
            set(prediction["policies"]) == policies
            and len(prediction["policies"]) == len(policies),
            "Prediction policy aliases differ",
        )
        path = _safe_path(result_root, prediction["path"])
        check(path, prediction["sha256"])
        pool = pools[prediction["pool"]]
        with np.load(path, allow_pickle=False) as saved:
            require(
                np.array_equal(saved["global_index"], pool.global_index.to_numpy()),
                "Prediction file is not the entire frozen pool",
            )
            scores, logits = saved["score"].copy(), saved["logits"].copy()
        require(
            scores.shape == (len(pool),)
            and logits.shape == (len(pool), 3)
            and np.isfinite(scores).all()
            and np.isfinite(logits).all(),
            "Invalid saved prediction arrays",
        )
        if record["config"]["mode"] == "classifier":
            normalized = logits.astype(float) - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(normalized)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            expected_score = probabilities @ norm_cache[record["name"]]["class_means"]
            close(scores, expected_score, "Float64 class-mean score from classifier outputs")
        for policy in policies:
            map_key = prediction["group"], prediction["pool"], policy, prediction["seed"]
            require(map_key not in score_map, "Duplicate policy/seed predictions")
            score_map[map_key] = scores
            prediction_lookup[map_key] = record, logits
    for row in result["metrics"]:
        pool = pools[row["pool"]]
        key = row["group"], row["pool"], row["policy"], row["seed"]
        record, logits = prediction_lookup[key]
        config = record["config"]
        labeled = row["pool"] == "within_slide_test"
        if not labeled:
            require(
                not TYPE_FIELDS & set(row), "External result invents unavailable class diagnostics"
            )
        expected_row = {
            "group": record["group"],
            "marker": config["marker"],
            "train_fraction": config["train_fraction"],
            "policy": row["policy"],
            "family": record["family"],
            "epoch": record["epoch"],
            "seed": config["seed"],
            "pool": row["pool"],
            "n_cells": len(pool),
            "n_fields": int(pool.fov_id.nunique()),
        }
        valid_logits = (
            logits
            if labeled and (config["mode"] == "classifier" or config["lambda_class"] > 0)
            else None
        )
        expected_row.update(
            numpy_metrics(
                pool[config["marker"]].to_numpy(dtype=float),
                score_map[key],
                pool.global_index.to_numpy(),
                row["fraction"],
                record["calibration"],
                pool.class_id.to_numpy() if labeled else None,
                valid_logits,
            )
        )
        _compare_mapping(row, expected_row, "Reported outcome row")
    checked_intervals, explicit_checks = [], []
    first_group = (
        "CD16_fraction_1"
        if "CD16_fraction_1" in protocol["groups"]
        else sorted(protocol["groups"])[0]
    )
    for row in uncertainty["comparisons"]:
        group, pool = protocol["groups"][row["group"]], pools[row["pool"]]
        a, b = CONTRASTS[row["comparison"]]
        first = np.stack([score_map[row["group"], row["pool"], a, seed] for seed in group["seeds"]])
        second = np.stack(
            [score_map[row["group"], row["pool"], b, seed] for seed in group["seeds"]]
        )
        y, ids, fields = (
            pool[group["marker"]].to_numpy(dtype=float),
            pool.global_index.to_numpy(),
            pool.fov_id.to_numpy(),
        )
        interval = paired_topk_bootstrap(
            y,
            first,
            second,
            fields,
            row["fraction"],
            n_bootstrap=row["n_bootstrap"],
            seed=row["bootstrap_seed"],
            ids=ids,
        )
        for field in (
            "difference_a_minus_b",
            "paired_seed_sd",
            "paired_seed_differences",
            "n_seeds",
            "ci_low",
            "ci_high",
            "n_bootstrap",
            "n_fields",
        ):
            close(row[field], interval[field], "Full-pool paired interval/" + field)
        exchange = independent_exchange(y, first, second, ids, row["fraction"])
        actual = row["exchange_diagnostics"]
        require(
            actual["k"] == exchange["k"]
            and len(actual["paired_replicates"]) == len(group["seeds"]),
            "Exchange pairing/budget mismatch",
        )
        close(
            actual["mean_selected_overlap_fraction"],
            exchange["mean_selected_overlap_fraction"],
            "Mean overlap",
        )
        for observed, expected_exchange in zip(
            actual["paired_replicates"], exchange["paired_replicates"]
        ):
            _compare_mapping(observed, expected_exchange, "Exchange diagnostic")
        if (
            row["group"] == first_group
            and row["pool"] == "within_slide_test"
            and row["comparison"] == "joint_minus_regression"
            and row["fraction"] == protocol["primary_fraction"]
        ):
            explicit = paired_seed_field_bootstrap(
                y,
                first,
                second,
                fields,
                row["fraction"],
                n_bootstrap=row["n_bootstrap"],
                seed=row["bootstrap_seed"],
                ids=ids,
            )
            for field in (
                "difference_a_minus_b",
                "paired_seed_sd",
                "paired_seed_differences",
                "ci_low",
                "ci_high",
            ):
                close(row[field], explicit[field], "Explicit reranked-bootstrap check/" + field)
            explicit_checks.append(
                {
                    "group": row["group"],
                    "pool": row["pool"],
                    "comparison": row["comparison"],
                    "fraction": row["fraction"],
                    "draws": row["n_bootstrap"],
                    "cells": len(y),
                }
            )
        checked_intervals.append((row["group"], row["pool"], row["comparison"], row["fraction"]))
    require(len(explicit_checks) == 1, "Prespecified explicit bootstrap check was not completed")
    return {
        "status": "passed",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "declared_training_runs": len(configurations),
        "candidate_states": len(expected),
        "frozen_validation_rows_recomputed": sum(map(len, validation_rows.values())),
        "prediction_files": len(result["predictions"]),
        "reported_metric_rows_recomputed": len(result["metrics"]),
        "paired_intervals_recomputed_full_pool": len(checked_intervals),
        "exchange_diagnostics_recomputed": len(checked_intervals),
        "explicit_reranked_bootstrap_checks": explicit_checks,
        "pool_sizes": {
            name: {
                "cells": len(pool),
                "fields": int(pool.fov_id.nunique()),
                "type_labels_used": name == "within_slide_test",
            }
            for name, pool in pools.items()
        },
        "external_subsampling": False,
        "absolute_tolerance": 1e-8,
        "hashes": {
            "verifier": sha(__file__),
            "protocol": protocol_hash,
            "spec": sha(spec_file),
            "results": sha(result_root / "results.json"),
            "uncertainty": sha(result_root / "uncertainty.json"),
            "fast_bootstrap_source": sha(ROOT / "src/cellselect/bootstrap.py"),
        },
        "scope": "Numerical/provenance audit, not additional model selection or evidence of causal mechanisms/generalization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = {
        "data": "data/prepared/bsccmnist_cd16_v1",
        "runs": "results/runs/extended",
        "spec": "configs/extended.json",
        "protocol": "results/extended_protocol.json",
        "results": "results/extended",
        "external": "data/prepared/bsccmnist_batch1_all_v1",
    }
    for name, default in defaults.items():
        parser.add_argument("--" + name, type=Path, default=Path(default))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "independent_verification.json"
    require(not output.exists(), "Refusing to replace an existing independent audit")
    verified = verify_study(
        args.data, args.runs, args.spec, args.protocol, args.results, args.external
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verified, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(verified, indent=2), flush=True)


if __name__ == "__main__":
    main()
