"""Freeze validation decisions, then evaluate a fixed test candidate set."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .metrics import (
    paired_field_bootstrap,
    paired_seed_field_bootstrap,
    prediction_metrics,
    quota_counts,
    quota_selection,
    selection_metrics,
    selection_size,
)
from .model import build_model
from .training import PreparedData, file_sha256, predict, write_json


def fit_positive_affine(y, scores, minimum_slope=0.001):
    """Validation least squares subject to a strictly positive slope."""
    y, scores = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    if (
        y.ndim != 1
        or y.shape != scores.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(scores).all()
        or not 0 < minimum_slope <= 1
    ):
        raise ValueError("Invalid calibration data")
    centered = scores - scores.mean()
    denominator = float(centered @ centered)
    slope = (
        max(minimum_slope, float(centered @ (y - y.mean())) / denominator) if denominator else 1.0
    )
    intercept = float(y.mean() - slope * scores.mean())
    return {"slope": slope, "intercept": intercept, "fit_partition": "val"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_protocol(
    data: PreparedData,
    run_root,
    output,
    primary_fraction=0.1,
    fractions=(0.05, 0.1, 0.2),
    expected_spec=None,
):
    """Choose weights and checkpoints from validation predictions."""
    output, run_root = Path(output), Path(run_root)
    if (
        not fractions
        or any(not np.isfinite(q) or not 0 < q <= 1 for q in fractions)
        or not np.isfinite(primary_fraction)
        or not 0 < primary_fraction <= 1
        or primary_fraction not in fractions
        or len(set(fractions)) != len(fractions)
    ):
        raise ValueError(
            "Selection fractions must be unique, in (0,1], and include the primary fraction"
        )
    if output.exists():
        raise FileExistsError("Refusing to replace a frozen protocol")
    validation = data.metadata.iloc[data.indices["val"]]
    ids, y = validation.global_index.to_numpy(), validation[data.marker].to_numpy()
    classes = validation.class_id.to_numpy()
    runs, table = [], []
    for directory in sorted(run_root.iterdir()):
        if not directory.is_dir() or not (directory / "complete.json").exists():
            continue
        config, complete = _read(directory / "config.json"), _read(directory / "complete.json")
        provenance = _read(directory / "provenance.json")
        if config.get("development_only", False):
            continue
        if provenance["manifest_sha256"] != data.manifest_hash:
            raise ValueError("Cohort mismatch between run and current data")
        if file_sha256(directory / "final.pt") != complete["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        for filename, key in [
            ("validation_predictions.npz", "validation_predictions_sha256"),
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
        ]:
            if file_sha256(directory / filename) != complete.get(key):
                raise ValueError(f"Completed run artifact checksum mismatch: {filename}")
        current_train = data.metadata.iloc[data.indices["train"]].global_index.to_numpy()
        import hashlib

        if provenance["train_ids_sha256"] != hashlib.sha256(current_train.tobytes()).hexdigest():
            raise ValueError("Training IDs differ between compared runs")
        candidates = complete.get(
            "candidates",
            [
                {
                    "epoch": int(config["epochs"]),
                    "checkpoint_file": "final.pt",
                    "checkpoint_sha256": complete["checkpoint_sha256"],
                    "predictions_file": "validation_predictions.npz",
                    "predictions_sha256": complete["validation_predictions_sha256"],
                }
            ],
        )
        declared_epochs = config.get("checkpoint_epochs", [config["epochs"]])
        if sorted(c["epoch"] for c in candidates) != sorted(declared_epochs):
            raise ValueError("Incomplete declared checkpoint grid")
        for candidate in candidates:
            if (
                file_sha256(directory / candidate["checkpoint_file"])
                != candidate["checkpoint_sha256"]
                or file_sha256(directory / candidate["predictions_file"])
                != candidate["predictions_sha256"]
            ):
                raise ValueError("Candidate artifact checksum mismatch")
            predictions = np.load(directory / candidate["predictions_file"])
            if not np.array_equal(predictions["global_index"], ids):
                raise ValueError("Validation IDs mismatch")
            scores = predictions["score"]
            row = {
                "name": directory.name,
                "epoch": int(candidate["epoch"]),
                "seed": int(config["seed"]),
                "mode": config.get("mode", "joint"),
                "lambda_class": float(config.get("lambda_class", 0)),
                **prediction_metrics(y, scores),
                **selection_metrics(y, scores, classes, primary_fraction, ids),
            }
            table.append(row)
            runs.append(
                {
                    "name": directory.name,
                    "epoch": int(candidate["epoch"]),
                    "config": config,
                    "checkpoint_file": candidate["checkpoint_file"],
                    "predictions_file": candidate["predictions_file"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                    "config_sha256": file_sha256(directory / "config.json"),
                    "normalization_sha256": file_sha256(directory / "normalization.json"),
                    "validation_predictions_sha256": candidate["predictions_sha256"],
                    "calibration": fit_positive_affine(y, scores),
                }
            )
    if not runs:
        raise ValueError("No completed non-development runs")
    if expected_spec is not None:
        expected = {}
        for run in expected_spec["runs"]:
            configuration = {**expected_spec["common"], **run}
            name = configuration.pop("name")
            if not configuration.get("development_only", False):
                expected[name] = configuration
        if {r["name"]: r["config"] for r in runs} != expected:
            raise ValueError("Completed runs do not match the declared experiment grid")
    tab = pd.DataFrame(table)
    joint = tab[tab["mode"] == "joint"]
    if joint.empty or 0.0 not in joint.lambda_class.values:
        raise ValueError("A regression-only baseline is required")
    seeds = sorted(int(x) for x in joint.seed.unique())
    weights = sorted(float(x) for x in joint.lambda_class.unique())
    epochs = sorted(int(x) for x in joint.epoch.unique())
    for weight in weights:
        for epoch in epochs:
            members = joint[(joint.lambda_class == weight) & (joint.epoch == epoch)]
            if sorted(members.seed.tolist()) != seeds:
                raise ValueError("Every weight/checkpoint must have the same paired seeds")
    classifiers = tab[tab["mode"] == "classifier"]
    for epoch in epochs:
        if sorted(classifiers[classifiers.epoch == epoch].seed.tolist()) != seeds:
            raise ValueError("Classifier baseline must have the same paired seeds/checkpoints")
    common_keys = [
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "warmup_epochs",
        "min_lr_ratio",
        "feature_dim",
        "amp",
        "train_fraction",
        "marker",
        "loss_scale",
        "checkpoint_epochs",
        "reference_epoch",
        "subset_seed",
    ]
    for key in common_keys:
        if len({str(r["config"].get(key)) for r in runs}) > 1:
            raise ValueError(f"Unmatched experimental setting: {key}")
    if any(
        r["config"].get("mode") == "classifier" and r["config"].get("lambda_class", 0) != 0
        for r in runs
    ):
        raise ValueError("Classifier baseline must use the canonical zero lambda key")
    grid_means = joint.groupby(["lambda_class", "epoch"]).agg(
        rmse=("rmse", "mean"),
        utility=("utility", "mean"),
        within=("within_type_gap", "mean"),
        allocation=("type_allocation_gap", "mean"),
    )
    # Group order and stable sort break ties by lower weight, then earlier epoch.
    predictive_pair = grid_means.sort_values("rmse", kind="stable").index[0]
    decision_pair = grid_means.sort_values("utility", ascending=False, kind="stable").index[0]
    predictive, decision = float(predictive_pair[0]), float(decision_pair[0])
    reference_epoch = int(runs[0]["config"].get("reference_epoch", epochs[-1]))
    if reference_epoch not in epochs:
        raise ValueError("Reference epoch must belong to candidate grid")
    means = grid_means.xs(reference_epoch, level="epoch")
    reference_joint = joint[joint.epoch == reference_epoch]
    regression_epoch = int(
        grid_means.xs(0.0, level="lambda_class")
        .sort_values("utility", ascending=False, kind="stable")
        .index[0]
    )
    classifier_epoch = int(
        classifiers.groupby("epoch")
        .utility.mean()
        .sort_values(ascending=False, kind="stable")
        .index[0]
    )
    possible_pairs = []
    for a in weights:
        for b in weights:
            if (
                means.loc[a, "allocation"] < means.loc[b, "allocation"]
                and means.loc[a, "within"] > means.loc[b, "within"]
                and means.loc[a, "utility"] > means.loc[b, "utility"]
            ):
                possible_pairs.append(
                    (float(means.loc[a, "utility"] - means.loc[b, "utility"]), a, b)
                )
    quota = None
    if possible_pairs:
        _, a, b = sorted(possible_pairs, key=lambda t: (-t[0], t[1], t[2]))[0]
        counts = np.zeros(3)
        selected_rows = reference_joint[reference_joint.lambda_class.isin([a, b])]
        for row in selected_rows.itertuples():
            for c, count in row.selected_type_counts.items():
                counts[int(c)] += count
        quota = {
            "lambda_a": a,
            "lambda_b": b,
            "epoch": reference_epoch,
            "proportions": (counts / counts.sum()).tolist(),
            "rule": "mean validation selected type composition of both paired model families; capacity-constrained apportionment on test",
            "selection": "largest validation utility gain among improved-allocation/worse-within pairs",
        }
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": data.manifest_hash,
        "marker": data.marker,
        "train_fraction": runs[0]["config"].get("train_fraction", 1.0),
        "subset_seed": runs[0]["config"].get("subset_seed", 20261021),
        "primary_fraction": primary_fraction,
        "fractions": list(fractions),
        "seeds": seeds,
        "weights": weights,
        "runs": runs,
        "checkpoint_epochs": epochs,
        "reference_epoch": reference_epoch,
        "schema_version": 2,
        "selector": "mean validation metric across paired seeds; identical weight/checkpoint grid",
        "lambda_by_rmse": predictive,
        "lambda_by_utility": decision,
        "selected_by_rmse": {"lambda_class": predictive, "epoch": int(predictive_pair[1])},
        "selected_by_utility": {"lambda_class": decision, "epoch": int(decision_pair[1])},
        "regression_baseline": {
            "lambda_class": 0.0,
            "epoch": regression_epoch,
            "criterion": "validation utility",
        },
        "classifier_baseline": {
            "lambda_class": 0.0,
            "epoch": classifier_epoch,
            "criterion": "validation utility",
        },
        "validation_stability_planned": True,
        "quota_diagnostic": quota,
        "validation_table": table,
        "expected_experiment_verified": expected_spec is not None,
        "test_used_to_choose_protocol": False,
    }
    write_json(output, protocol)
    return protocol


def evaluate_frozen(
    data: PreparedData, run_root, protocol_file, output, n_bootstrap=2000, device="cuda"
):
    """Evaluate locked models. Re-runs must use the identical frozen protocol."""
    run_root, output = Path(run_root), Path(output)
    protocol = _read(protocol_file)
    protocol_hash = file_sha256(protocol_file)
    if protocol["manifest_sha256"] != data.manifest_hash or protocol["marker"] != data.marker:
        raise ValueError("Protocol cohort mismatch")
    output.mkdir(parents=True, exist_ok=True)
    started_file = output / "test_started.json"
    if started_file.exists() and _read(started_file)["protocol_sha256"] != protocol_hash:
        raise ValueError("A different protocol already opened this test evaluation")
    # Check every artifact before first test prediction.
    for run in protocol["runs"]:
        directory = run_root / run["name"]
        for filename, key in [
            (run["checkpoint_file"], "checkpoint_sha256"),
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
            (run["predictions_file"], "validation_predictions_sha256"),
        ]:
            if file_sha256(directory / filename) != run[key]:
                raise ValueError(f"Frozen artifact changed: {run['name']}/{filename}")
    if not started_file.exists():
        write_json(
            started_file,
            {
                "protocol_sha256": protocol_hash,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    device = torch.device(device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x, _, _, metadata = data.partition("test")
    y, classes = metadata[data.marker].to_numpy(), metadata.class_id.to_numpy()
    ids, groups = metadata.global_index.to_numpy(), metadata.fov_id.to_numpy()
    all_rows, score_map, quota_rows = [], {}, []
    reference_epoch = protocol["reference_epoch"]
    selections = {
        "utility_selector": ("joint", protocol["selected_by_utility"]),
        "rmse_selector": ("joint", protocol["selected_by_rmse"]),
        "regression_baseline": ("joint", protocol["regression_baseline"]),
        "class_means_baseline": ("classifier", protocol["classifier_baseline"]),
    }
    selected_keys = {
        (mode, choice["lambda_class"], choice["epoch"]) for mode, choice in selections.values()
    }
    for run in protocol["runs"]:
        directory = run_root / run["name"]
        config = run["config"]
        mode, lam, seed = (
            config.get("mode", "joint"),
            float(config.get("lambda_class", 0)),
            int(config["seed"]),
        )
        epoch = int(run["epoch"])
        # Open only the controlled frontier and the preselected policies/baselines.
        if epoch != reference_epoch and (mode, lam, epoch) not in selected_keys:
            continue
        checkpoint = torch.load(
            directory / run["checkpoint_file"], map_location="cpu", weights_only=True
        )
        config, norm = run["config"], checkpoint["normalization"]
        model = build_model(feature_dim=int(config.get("feature_dim", 256)))
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        mean = torch.tensor(norm["image_mean"], dtype=torch.float32, device=device)[
            None, :, None, None
        ]
        std = torch.tensor(norm["image_std"], dtype=torch.float32, device=device)[
            None, :, None, None
        ]
        logits, marker = predict(model, x, mean, std, device, int(config["batch_size"]), amp=False)
        score = (
            (torch.softmax(torch.from_numpy(logits), dim=1).numpy() @ norm["class_means"])
            if mode == "classifier"
            else marker.astype(float) * norm["target_std"] + norm["target_mean"]
        )
        key = (mode, lam, epoch, seed)
        score_map[key] = score
        np.savez_compressed(
            output / f"{run['name']}_epoch_{epoch:03d}_test_predictions.npz",
            global_index=ids,
            score=score,
            logits=logits,
        )
        metrics = prediction_metrics(
            y, score, classes, logits if mode == "classifier" or lam > 0 else None
        )
        calibration = run["calibration"]
        calibrated = calibration["slope"] * score + calibration["intercept"]
        metrics["calibrated_rmse"] = prediction_metrics(y, calibrated)["rmse"]
        metrics["calibration_slope"] = calibration["slope"]
        metrics["calibration_intercept"] = calibration["intercept"]
        for fraction in protocol["fractions"]:
            all_rows.append(
                {
                    "run": run["name"],
                    "epoch": epoch,
                    "seed": seed,
                    "mode": mode,
                    "lambda_class": lam,
                    **metrics,
                    **selection_metrics(y, score, classes, fraction, ids),
                }
            )
        del model
    comparisons = []
    fraction = protocol["primary_fraction"]

    def policy_scores(policy, seed):
        mode, choice = selections[policy]
        return score_map[(mode, choice["lambda_class"], choice["epoch"], seed)]

    for seed in protocol["seeds"]:
        a = policy_scores("utility_selector", seed)
        for name, b in [
            ("utility_selector_minus_rmse_selector", policy_scores("rmse_selector", seed)),
            ("utility_selector_minus_regression", policy_scores("regression_baseline", seed)),
            ("utility_selector_minus_class_means", policy_scores("class_means_baseline", seed)),
        ]:
            interval = paired_field_bootstrap(
                y, a, b, groups, fraction, n_bootstrap=n_bootstrap, ids=ids
            )
            bootstrap_seed = interval.pop("seed")
            comparisons.append(
                {
                    "comparison": name,
                    "seed": seed,
                    "bootstrap_seed": bootstrap_seed,
                    "fraction": fraction,
                    **interval,
                }
            )
    quota = protocol["quota_diagnostic"]
    if quota:
        counts = quota_counts(
            quota["proportions"],
            np.bincount(classes, minlength=3),
            selection_size(len(y), fraction),
        )
        for seed in protocol["seeds"]:
            for arm in ["a", "b"]:
                lam = quota[f"lambda_{arm}"]
                score = score_map[("joint", lam, quota["epoch"], seed)]
                quota_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "lambda_class": lam,
                        **quota_selection(y, score, classes, counts, ids),
                    }
                )
    aggregate_comparisons = []
    a = np.stack([policy_scores("utility_selector", s) for s in protocol["seeds"]])
    for name, policy in [
        ("utility_selector_minus_rmse_selector", "rmse_selector"),
        ("utility_selector_minus_regression", "regression_baseline"),
        ("utility_selector_minus_class_means", "class_means_baseline"),
    ]:
        b = np.stack([policy_scores(policy, s) for s in protocol["seeds"]])
        interval = paired_seed_field_bootstrap(
            y, a, b, groups, fraction, n_bootstrap=n_bootstrap, ids=ids
        )
        bootstrap_seed = interval.pop("seed")
        aggregate_comparisons.append(
            {"comparison": name, "fraction": fraction, "bootstrap_seed": bootstrap_seed, **interval}
        )
    frontier_rows = [r for r in all_rows if r["epoch"] == reference_epoch]
    selected_rows = []
    for policy, (mode, choice) in selections.items():
        selected_rows.extend(
            {"policy": policy, **r}
            for r in all_rows
            if r["mode"] == mode
            and r["lambda_class"] == choice["lambda_class"]
            and r["epoch"] == choice["epoch"]
        )
    result = {
        "protocol_sha256": protocol_hash,
        "n_test": len(y),
        "n_test_fields": len(np.unique(groups)),
        "metrics": frontier_rows,
        "reference_epoch": reference_epoch,
        "selected_metrics": selected_rows,
        "paired_comparisons": comparisons,
        "quota_diagnostic": quota_rows,
        "aggregate_comparisons": aggregate_comparisons,
        "interpretation": "Field intervals condition on each trained pair; seed variability is reported separately. Hindsight gaps are not fully avoidable regret.",
    }
    write_json(output / "results.json", result)
    table = pd.DataFrame(frontier_rows)
    table.drop(columns="selected_type_counts").to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(aggregate_comparisons).to_csv(output / "aggregate_comparisons.csv", index=False)
    pd.DataFrame(selected_rows).drop(columns="selected_type_counts").to_csv(
        output / "selected_policies.csv", index=False
    )
    return result
