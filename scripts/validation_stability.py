"""Measure validation-selection stability conditional on frozen models.

Resample validation fields without retraining or selecting a new final model.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cellselect.metrics import selection_size, topk_indices
from cellselect.training import file_sha256, write_json


def choice_stability(y, scores, groups, ids, candidates, fraction, repetitions=1000, seed=20261021):
    """scores has shape (candidate pairs, paired seeds, cells); pairs are (lambda, epoch)."""
    y, scores = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    groups, ids, pairs = np.asarray(groups), np.asarray(ids), np.asarray(candidates, dtype=float)
    if (
        y.ndim != 1
        or not len(y)
        or scores.ndim != 3
        or pairs.ndim != 2
        or pairs.shape != (scores.shape[0], 2)
        or not len(pairs)
        or scores.shape[1] < 1
        or scores.shape[2] != len(y)
        or groups.shape != y.shape
        or ids.shape != y.shape
        or not np.isfinite(y).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(pairs).all()
        or (pairs[:, 0] < 0).any()
        or (pairs[:, 1] < 1).any()
        or not np.equal(pairs[:, 1], np.floor(pairs[:, 1])).all()
        or len(np.unique(ids)) != len(ids)
        or len(np.unique(pairs, axis=0)) != len(pairs)
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, (int, np.integer))
        or repetitions < 2
        or pd.isna(groups).any()
    ):
        raise ValueError("Invalid aligned validation scores, weights, IDs, fields, or repetitions")
    selection_size(len(y), fraction)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, scores = pairs[order], scores[order]
    weights = np.unique(pairs[:, 0])
    records = [{"lambda_class": float(w), "epoch": int(e)} for w, e in pairs]
    fields = np.unique(groups)
    if len(fields) < 2:
        raise ValueError("At least two validation fields are required")
    members = [np.flatnonzero(groups == group) for group in fields]

    def assess(index):
        target, prediction = y[index], scores[:, :, index]
        rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=2)).mean(axis=1)
        k = selection_size(len(index), fraction)
        utility = np.array(
            [
                [target[topk_indices(s, k, ids[index])].mean() for s in family]
                for family in prediction
            ]
        ).mean(axis=1)
        return int(np.argmin(rmse)), int(np.argmax(utility)), rmse, utility

    rmse_choice, utility_choice, rmse, utility = assess(np.arange(len(y)))
    counts = np.zeros((2, len(pairs)), dtype=int)
    agreement = lambda_agreement = 0
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        index = np.concatenate([members[g] for g in rng.integers(len(fields), size=len(fields))])
        a, b, _, _ = assess(index)  # One common draw for every model and seed.
        counts[0, a] += 1
        counts[1, b] += 1
        agreement += a == b
        lambda_agreement += pairs[a, 0] == pairs[b, 0]
    marginal = np.array([counts[:, pairs[:, 0] == w].sum(axis=1) for w in weights]).T
    return {
        "schema_version": 2,
        "candidate_pairs": records,
        "partition": "val",
        "repetitions": int(repetitions),
        "bootstrap_seed": int(seed),
        "n_cells": len(y),
        "n_fields": len(fields),
        "n_training_seeds": scores.shape[1],
        "primary_fraction": float(fraction),
        "weights": weights.tolist(),
        "observed_selected_by_rmse": records[rmse_choice],
        "observed_selected_by_utility": records[utility_choice],
        "observed_mean_rmse": rmse.tolist(),
        "observed_mean_utility": utility.tolist(),
        "rmse_choice_counts": counts[0].tolist(),
        "utility_choice_counts": counts[1].tolist(),
        "rmse_choice_frequencies": (counts[0] / repetitions).tolist(),
        "utility_choice_frequencies": (counts[1] / repetitions).tolist(),
        "rmse_lambda_counts": marginal[0].tolist(),
        "utility_lambda_counts": marginal[1].tolist(),
        "rmse_lambda_frequencies": (marginal[0] / repetitions).tolist(),
        "utility_lambda_frequencies": (marginal[1] / repetitions).tolist(),
        "selectors_agreement_frequency": agreement / repetitions,
        "selectors_lambda_agreement_frequency": lambda_agreement / repetitions,
        "interpretation": "Descriptive sensitivity to resampling observed validation fields, conditional on trained models; not probabilities of true optimality or evidence about training-data efficiency.",
        "reference": "https://www.jmlr.org/papers/v11/cawley10a.html",
    }


def load_frozen_validation(data_root, run_root, protocol_file):
    data_root, run_root = Path(data_root), Path(run_root)
    protocol = json.loads(Path(protocol_file).read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != 2
        or not protocol.get("expected_experiment_verified")
        or protocol.get("test_used_to_choose_protocol") is not False
    ):
        raise ValueError("A verified, validation-only schema-2 frozen protocol is required")
    if file_sha256(data_root / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Prepared manifest changed after freeze")
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if file_sha256(data_root / "metadata.csv") != manifest["prepared_sha256"]["metadata.csv"]:
        raise ValueError("Prepared metadata changed")
    # Convert only validation outcomes to numbers; train/test outcomes are unused.
    metadata = pd.read_csv(data_root / "metadata.csv", dtype={protocol["marker"]: str})
    val = metadata.loc[metadata["split"] == "val"]
    ids, y = val.global_index.to_numpy(), val[protocol["marker"]].to_numpy(dtype=float)
    fields = val.fov_id.to_numpy()
    weights, seeds, epochs = (
        sorted(protocol[k]) for k in ("weights", "seeds", "checkpoint_epochs")
    )
    if any(not x or len(set(x)) != len(x) for x in (weights, seeds, epochs)):
        raise ValueError("Empty or duplicate protocol weight, epoch, or seed")
    candidates = [(w, e) for w in weights for e in epochs]
    predictions = {}
    for run in protocol["runs"]:
        directory, config = run_root / run["name"], run["config"]
        for filename, key in [
            (run["checkpoint_file"], "checkpoint_sha256"),
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
            (run["predictions_file"], "validation_predictions_sha256"),
        ]:
            if file_sha256(directory / filename) != run[key]:
                raise ValueError(f"Frozen artifact changed: {run['name']}/{filename}")
        if config.get("mode", "joint") != "joint":
            continue
        key = (float(config["lambda_class"]), int(run["epoch"]), int(config["seed"]))
        if key in predictions:
            raise ValueError("Duplicate weight/epoch/seed candidate")
        with np.load(directory / run["predictions_file"], allow_pickle=False) as saved:
            if not np.array_equal(saved["global_index"], ids):
                raise ValueError("Validation prediction IDs do not match prepared rows")
            predictions[key] = saved["score"].copy()
    if set(predictions) != {(w, e, s) for w, e in candidates for s in seeds}:
        raise ValueError("Incomplete paired weight/epoch/seed grid")
    scores = np.array([[predictions[w, e, s] for s in seeds] for w, e in candidates])
    return protocol, (y, scores, fields, ids, candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "protocol"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output", default="results/validation_stability.json")
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Refusing to overwrite existing validation stability results")
    protocol, arrays = load_frozen_validation(args.data, args.runs, args.protocol)
    result = choice_stability(
        *arrays, fraction=protocol["primary_fraction"], repetitions=args.repetitions, seed=args.seed
    )
    for name in ("selected_by_rmse", "selected_by_utility"):
        if result["observed_" + name] != protocol[name]:
            raise ValueError("Validation scores disagree with the frozen selector")
    result["protocol_sha256"] = file_sha256(args.protocol)
    result["training_seeds"] = protocol["seeds"]
    write_json(args.output, result)
    print(f"Saved validation choice frequencies over {args.repetitions} common field draws.")


if __name__ == "__main__":
    main()
