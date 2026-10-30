"""Validation-based model selection and frozen within-slide/external evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import ConstantInputWarning

from .evaluation import fit_positive_affine
from .metrics import prediction_metrics, selection_metrics, selection_size, topk_indices
from .model import build_model
from .training import PreparedData, file_sha256, predict, write_json

SCORING_PRECISION = {
    "network_forward": "float32",
    "regression_denormalization": "float64",
    "classifier_softmax": "float64",
    "classifier_expectation": "float64",
    "classifier_validation_source": "saved_logits",
}


def study_prediction_metrics(y, scores, classes=None, logits=None):
    """Keep undefined correlations null, including exactly constant floats."""
    # Reduction roundoff can make std(constant) positive; compare values directly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = prediction_metrics(y, scores, classes, logits)
    y, scores = np.asarray(y), np.asarray(scores)
    variable = len(y) > 1 and np.any(y != y[0]) and np.any(scores != scores[0])
    for name in ("pearson", "spearman"):
        if not variable or result[name] is None or not np.isfinite(result[name]):
            result[name] = None
    return result


def class_mean_scores(logits, class_means):
    """Probability-weighted training class means, evaluated in float64."""
    logits, means = np.asarray(logits, dtype=np.float64), np.asarray(class_means, dtype=np.float64)
    if (
        logits.ndim != 2
        or means.ndim != 1
        or logits.shape[1] != len(means)
        or not len(means)
        or not np.isfinite(logits).all()
        or not np.isfinite(means).all()
    ):
        raise ValueError("Invalid class logits or training class means")
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    # Equal class means must produce a constant score, not rank roundoff in
    # probability-row sums. Centering also avoids multiplying a common offset.
    return means[0] + probabilities @ (means - means[0])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def family(config):
    if config.get("mode", "joint") == "classifier":
        return "class_means"
    objective = config.get("objective", "mse")
    if objective == "mse":
        return f"mse_lambda_{float(config.get('lambda_class', 0)):g}"
    if objective == "pg_central":
        return f"pg_central_h_{float(config['pg_h']):g}"
    if objective == "spo_plus":
        return "spo_plus"
    raise ValueError("Unknown objective family")


def group_key(marker, fraction):
    return f"{marker}_fraction_{float(fraction):g}"


def choose_policies(table):
    """One setting per method, averaged over paired runs, without ensembling."""
    frame = pd.DataFrame(table)
    required = {"family", "epoch", "seed", "rmse", "utility"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("Incomplete validation table")
    seeds = sorted(frame.seed.unique().tolist())
    if frame.duplicated(["family", "epoch", "seed"]).any():
        raise ValueError("Duplicate candidate/seed")
    for _, rows in frame.groupby(["family", "epoch"]):
        if sorted(rows.seed.tolist()) != seeds:
            raise ValueError("Unpaired candidate seeds")
    means = frame.groupby(["family", "epoch"], sort=True)[["rmse", "utility"]].mean().reset_index()
    mse = means.family.str.startswith("mse_lambda_")
    conditions = {
        "regression": (means.family == "mse_lambda_0", "utility"),
        "class_means": (means.family == "class_means", "utility"),
        "joint": (mse & (means.family != "mse_lambda_0"), "utility"),
        "predictive_selector": (mse, "rmse"),
        "utility_selector": (mse, "utility"),
        "spo_plus": (means.family == "spo_plus", "utility"),
        "pg_central": (means.family.str.startswith("pg_central_h_"), "utility"),
    }
    choices = {}
    for name, (mask, criterion) in conditions.items():
        eligible = means.loc[mask]
        if eligible.empty:
            raise ValueError(f"Missing required comparator: {name}")
        # Stable family/epoch order gives an outcome-independent exact-tie rule.
        row = eligible.sort_values(criterion, ascending=criterion == "rmse", kind="stable").iloc[0]
        choices[name] = {
            "family": str(row.family),
            "epoch": int(row.epoch),
            "criterion": criterion,
            "validation_rmse": float(row.rmse),
            "validation_utility": float(row.utility),
        }
    return choices


def freeze_study(data_root, run_root, spec_file, output, external_root=None):
    output, run_root = Path(output), Path(run_root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite frozen study choices")
    spec = read_json(spec_file)
    fractions = spec["fractions"]
    if (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(not np.isfinite(q) or not 0 < q <= 1 for q in fractions)
        or spec["primary_fraction"] not in fractions
    ):
        raise ValueError("Invalid selection fractions")
    data_cache, groups, records = {}, {}, []
    names = set()
    pairing, settings = {}, {}
    for override in spec["runs"]:
        config = {**spec["common"], **override}
        name = config.pop("name")
        if name in names or config.get("development_only", False):
            raise ValueError("Duplicate run name or development run in final grid")
        if config.get("primary_fraction") != spec["primary_fraction"]:
            raise ValueError("Training and declared primary decision budgets differ")
        if config.get("objective") == "pg_central" and any(
            e <= config.get("decision_warmup_epochs", 0) for e in config["checkpoint_epochs"]
        ):
            raise ValueError("Warmup checkpoint cannot be selected as a PG model")
        names.add(name)
        directory = run_root / name
        if read_json(directory / "config.json") != config:
            raise ValueError(f"Run configuration differs from declaration: {name}")
        complete, provenance = (
            read_json(directory / f) for f in ("complete.json", "provenance.json")
        )
        if provenance["source_sha256"]["model.py"] != file_sha256(
            Path(__file__).parent / "model.py"
        ):
            raise ValueError("Current inference model implementation differs from training")
        for filename, key in [
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
            ("final.pt", "checkpoint_sha256"),
            ("validation_predictions.npz", "validation_predictions_sha256"),
        ]:
            if file_sha256(directory / filename) != complete[key]:
                raise ValueError(f"Completed artifact changed: {name}/{filename}")
        data_key = (
            config["marker"],
            float(config["train_fraction"]),
            config.get("subset_seed", 20261021),
        )
        if data_key not in data_cache:
            data_cache[data_key] = PreparedData(data_root, *data_key)
        data = data_cache[data_key]
        train = data.metadata.iloc[data.indices["train"]]
        train_hash = hashlib.sha256(train.global_index.to_numpy().tobytes()).hexdigest()
        if (
            provenance["manifest_sha256"] != data.manifest_hash
            or provenance["train_ids_sha256"] != train_hash
            or read_json(directory / "normalization.json") != data.normalization()
        ):
            raise ValueError("Run cohort or training-only normalization mismatch")
        pair_key = (data_key[0], data_key[1], config["seed"])
        pair_value = (train_hash, complete["normalization_sha256"])
        if pair_key in pairing and pairing[pair_key] != pair_value:
            raise ValueError("Objectives do not use paired training cells/normalization")
        pairing[pair_key] = pair_value
        matched_keys = [
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "warmup_epochs",
            "min_lr_ratio",
            "feature_dim",
            "amp",
            "loss_scale",
            "clip_grad_norm",
            "primary_fraction",
            "train_fraction",
            "subset_seed",
        ]
        matched = {key: config.get(key) for key in matched_keys}
        if pair_key in settings and settings[pair_key] != matched:
            raise ValueError("Objectives have unmatched optimization or label-budget settings")
        settings[pair_key] = matched
        validation = data.metadata.iloc[data.indices["val"]]
        ids, y = validation.global_index.to_numpy(), validation[data.marker].to_numpy()
        gkey = group_key(data.marker, data.train_fraction)
        group = groups.setdefault(
            gkey,
            {
                "marker": data.marker,
                "train_fraction": data.train_fraction,
                "validation_table": [],
                "train_counts": {},
            },
        )
        group["train_counts"][str(config["seed"])] = {
            "cells": len(train),
            "fields": int(train.fov_id.nunique()),
            "ids_sha256": train_hash,
        }
        candidates = complete["candidates"]
        if [x["epoch"] for x in candidates] != config["checkpoint_epochs"]:
            raise ValueError("Incomplete checkpoint grid")
        for candidate in candidates:
            for path_key, hash_key in [
                ("checkpoint_file", "checkpoint_sha256"),
                ("predictions_file", "predictions_sha256"),
            ]:
                if file_sha256(directory / candidate[path_key]) != candidate[hash_key]:
                    raise ValueError("Checkpoint candidate hash mismatch")
            with np.load(
                directory / candidate["predictions_file"], allow_pickle=False
            ) as predictions:
                if not np.array_equal(ids, predictions["global_index"]):
                    raise ValueError("Validation cell IDs differ")
                # Recompute from logits to match float64 evaluation arithmetic.
                score = (
                    class_mean_scores(predictions["logits"], data.class_means)
                    if config["mode"] == "classifier"
                    else predictions["score"].copy()
                )
            k = selection_size(len(ids), spec["primary_fraction"])
            row = {
                "family": family(config),
                "epoch": candidate["epoch"],
                "seed": config["seed"],
                **study_prediction_metrics(y, score),
                "utility": float(y[topk_indices(score, k, ids)].mean()),
            }
            group["validation_table"].append(row)
            records.append(
                {
                    "group": gkey,
                    "name": name,
                    "config": config,
                    "family": family(config),
                    "epoch": candidate["epoch"],
                    "checkpoint_file": candidate["checkpoint_file"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "predictions_file": candidate["predictions_file"],
                    "predictions_sha256": candidate["predictions_sha256"],
                    "config_sha256": complete["config_sha256"],
                    "normalization_sha256": complete["normalization_sha256"],
                    "calibration": fit_positive_affine(y, score),
                }
            )
    for group in groups.values():
        group["choices"] = choose_policies(group["validation_table"])
        group["seeds"] = sorted(int(s) for s in group["train_counts"])
        group["replicate_variation"] = (
            "training-subset and initialization variation"
            if group["train_fraction"] < 1
            else "initialization/training-order variation on the same full cohort"
        )
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": file_sha256(spec_file),
        "manifest_sha256": file_sha256(Path(data_root) / "manifest.json"),
        "inference_source_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in ("model.py", "training.py", "study.py")
        },
        "scoring_precision": dict(SCORING_PRECISION),
        "primary_fraction": spec["primary_fraction"],
        "fractions": spec["fractions"],
        "groups": groups,
        "records": records,
        "test_status": "Exploratory follow-up to an already evaluated within-slide test; choices use validation only.",
        "external_status": "No external performance used for any model or checkpoint choice.",
        "selector": "Average each validation metric over paired seeds; no prediction ensembling.",
    }
    if external_root is not None:
        from .external_data import load_external

        external = load_external(external_root)
        for marker in {g["marker"] for g in groups.values()}:
            if not np.isfinite(external.metadata[marker].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"External {marker} outcomes require a separately frozen valid pool"
                )
        result["external_manifest_sha256"] = file_sha256(Path(external_root) / "manifest.json")
        result["external_pools"] = sorted(external.metadata.pool_id.unique().tolist())
    write_json(output, result)
    return result


def outcome_metrics(y, score, ids, fraction, classes=None):
    """Selection metrics, with type diagnostics only for labelled cohorts."""
    if classes is not None:
        result = selection_metrics(y, score, classes, fraction, ids)
        expected = (
            sum(
                count * float(np.mean(y[classes == int(c)]))
                for c, count in result["selected_type_counts"].items()
            )
            / result["k"]
        )
        result["type_matched_random_utility"] = expected
        result["within_type_enrichment"] = result["utility"] - expected
        return result
    k = selection_size(len(y), fraction)
    u = float(np.mean(y[topk_indices(score, k, ids)]))
    oracle = float(np.mean(y[topk_indices(y, k, ids)]))
    return {
        "fraction": fraction,
        "k": k,
        "utility": u,
        "random_expected_utility": float(np.mean(y)),
        "enrichment_difference": u - float(np.mean(y)),
        "oracle_utility": oracle,
        "oracle_gap": oracle - u,
    }


def evaluate_study(data_root, run_root, protocol_file, output, external_root=None, device="cuda"):
    """Save predictions and metrics for frozen choices; bootstrap is a separate step."""
    output, run_root = Path(output), Path(run_root)
    protocol = read_json(protocol_file)
    protocol_hash = file_sha256(protocol_file)
    if protocol.get("scoring_precision") != SCORING_PRECISION:
        raise ValueError("Frozen scoring precision differs from this implementation")
    for name, expected in protocol["inference_source_sha256"].items():
        if file_sha256(Path(__file__).parent / name) != expected:
            raise ValueError(f"Inference implementation changed after freeze: {name}")
    if file_sha256(Path(data_root) / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Main data manifest changed")
    for name, expected in read_json(Path(data_root) / "manifest.json")["prepared_sha256"].items():
        if file_sha256(Path(data_root) / name) != expected:
            raise ValueError(f"Main prepared data changed: {name}")
    if output.exists() and (output / "evaluation_started.json").exists():
        if read_json(output / "evaluation_started.json")["protocol_sha256"] != protocol_hash:
            raise ValueError("Output belongs to different frozen choices")
    output.mkdir(parents=True, exist_ok=True)
    external = None
    if external_root is not None:
        from .external_data import load_external

        if file_sha256(Path(external_root) / "manifest.json") != protocol.get(
            "external_manifest_sha256"
        ):
            raise ValueError("External data was not frozen or changed")
        external = load_external(external_root)
    selected = []
    for record in protocol["records"]:
        choices = protocol["groups"][record["group"]]["choices"]
        policies = [
            name
            for name, choice in choices.items()
            if (record["family"], record["epoch"]) == (choice["family"], choice["epoch"])
        ]
        if policies:
            selected.append((record, policies))
    # Verify every selected state before any new outcome evaluation.
    for record, _ in selected:
        root = run_root / record["name"]
        for filename, key in [
            (record["checkpoint_file"], "checkpoint_sha256"),
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
            (record["predictions_file"], "predictions_sha256"),
        ]:
            if file_sha256(root / filename) != record[key]:
                raise ValueError(f"Frozen model artifact changed: {filename}")
    if not (output / "evaluation_started.json").exists():
        write_json(
            output / "evaluation_started.json",
            {
                "protocol_sha256": protocol_hash,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    metadata = pd.read_csv(Path(data_root) / "metadata.csv")
    images = np.load(Path(data_root) / "images.npy", mmap_mode="r")
    test_rows = np.flatnonzero(metadata.split.to_numpy() == "test")
    test_images = torch.from_numpy(np.asarray(images[test_rows], dtype=np.float32).copy())
    test_meta = metadata.iloc[test_rows].reset_index(drop=True)
    metrics, predictions = [], []
    for record, policies in selected:
        config = record["config"]
        checkpoint = torch.load(
            run_root / record["name"] / record["checkpoint_file"],
            map_location="cpu",
            weights_only=True,
        )
        norm = checkpoint["normalization"]
        model = build_model(feature_dim=config["feature_dim"]).to(device)
        model.load_state_dict(checkpoint["model"])
        mean = torch.tensor(norm["image_mean"], dtype=torch.float32, device=device)[
            None, :, None, None
        ]
        std = torch.tensor(norm["image_std"], dtype=torch.float32, device=device)[
            None, :, None, None
        ]
        pools = [("within_slide_test", test_images, test_meta, True)]
        if external is not None and config["train_fraction"] == 1.0:
            for pool_id in protocol["external_pools"]:
                rows = external.pool_rows(pool_id)
                x = torch.from_numpy(np.asarray(external.images[rows], dtype=np.float32).copy())
                pools.append(
                    (pool_id, x, external.metadata.iloc[rows].reset_index(drop=True), False)
                )
        for pool_id, x, meta, labeled in pools:
            logits, marker = predict(model, x, mean, std, device, 512, amp=False)
            score = (
                class_mean_scores(logits, norm["class_means"])
                if config["mode"] == "classifier"
                else marker.astype(float) * norm["target_std"] + norm["target_mean"]
            )
            ids, y = meta.global_index.to_numpy(), meta[config["marker"]].to_numpy(dtype=float)
            classes = meta.class_id.to_numpy() if labeled else None
            has_logits = labeled and (config["mode"] == "classifier" or config["lambda_class"] > 0)
            values = study_prediction_metrics(y, score, classes, logits if has_logits else None)
            calibration = record["calibration"]
            values["calibrated_rmse"] = study_prediction_metrics(
                y, calibration["slope"] * score + calibration["intercept"]
            )["rmse"]
            pool_hash = hashlib.sha256(pool_id.encode()).hexdigest()[:12]
            filename = f"{record['name']}_epoch_{record['epoch']:03d}_{pool_hash}.npz"
            prediction_path = output / "predictions" / filename
            prediction_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(prediction_path, global_index=ids, score=score, logits=logits)
            predictions.append(
                {
                    "group": record["group"],
                    "name": record["name"],
                    "epoch": record["epoch"],
                    "seed": config["seed"],
                    "pool": pool_id,
                    "policies": policies,
                    "path": "predictions/" + filename,
                    "sha256": file_sha256(prediction_path),
                }
            )
            for q in protocol["fractions"]:
                for policy in policies:
                    metrics.append(
                        {
                            "group": record["group"],
                            "marker": config["marker"],
                            "train_fraction": config["train_fraction"],
                            "policy": policy,
                            "family": record["family"],
                            "epoch": record["epoch"],
                            "seed": config["seed"],
                            "pool": pool_id,
                            "n_cells": len(meta),
                            "n_fields": int(meta.fov_id.nunique()),
                            **values,
                            **outcome_metrics(y, score, ids, q, classes),
                        }
                    )
        del model
    result = {
        "protocol_sha256": protocol_hash,
        "metrics": metrics,
        "predictions": predictions,
        "uncertainty_status": "Not yet computed; run the paired-field uncertainty step.",
        "scope": "Within-slide follow-up plus frozen-model external acquisition evaluation; no external fitting.",
    }
    write_json(output / "results.json", result)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    return result
