"""Recompute validation-budget choices and assessment utilities."""

import argparse
import hashlib
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cellselect.training import file_sha256, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared/bsccmnist_cd16_v1"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/core_protocol.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results/validation_budget.json.gz")
    parser.add_argument("--runs", type=Path, default=ROOT / "results/runs/core")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/validation_budget_verification.json"
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    results_path = args.results
    with gzip.open(results_path, "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    data_root = args.data
    manifest = json.loads((data_root / "manifest.json").read_text())
    require(
        file_sha256(data_root / "manifest.json") == protocol["manifest_sha256"],
        "Data manifest changed",
    )
    require(
        file_sha256(data_root / "metadata.csv") == manifest["prepared_sha256"]["metadata.csv"],
        "Evaluation metadata changed",
    )
    meta = pd.read_csv(data_root / "metadata.csv")
    val = meta[meta.split == "val"]
    ids, y, fields = val.global_index.to_numpy(), val.CD16.to_numpy(), val.fov_id.to_numpy()
    # Input rows are ID-sorted, so a stable score sort implements the ID tie rule.
    require(
        np.array_equal(ids, np.sort(ids)), "Validation IDs must be sorted for the declared tie rule"
    )
    pairs = [(x["lambda_class"], x["epoch"]) for x in saved["candidate_pairs"]]
    bank = {}
    for record in protocol["runs"]:
        if record["config"]["mode"] != "joint":
            continue
        path = args.runs / record["name"] / record["predictions_file"]
        require(
            file_sha256(path) == record["validation_predictions_sha256"],
            "Validation prediction checksum changed",
        )
        with np.load(path) as values:
            require(np.array_equal(ids, values["global_index"]), "Validation prediction IDs differ")
            bank[record["config"]["lambda_class"], record["epoch"], record["config"]["seed"]] = (
                values["score"].copy()
            )
    scores = np.array([[bank[w, e, seed] for seed in protocol["seeds"]] for w, e in pairs])
    rows = {
        (r["repeat"], r["budget_fraction"], r["criterion"], r["fraction"]): r for r in saved["rows"]
    }
    verified = 0
    for split in saved["field_splits"]:
        assessment = np.flatnonzero(np.isin(fields, split["assessment_fields"]))
        order = split["selection_field_order"]
        require(
            set(fields[assessment]).isdisjoint(order), "Selection and assessment fields overlap"
        )
        for budget in saved["budgets"]:
            selected = np.flatnonzero(np.isin(fields, order[: math.ceil(budget * len(order))]))
            prediction = scores[:, :, selected]
            k = math.ceil(saved["selection_fraction"] * len(selected))
            rank = np.argsort(-prediction, axis=2, kind="stable")[:, :, :k]
            utilities = np.mean(y[selected][rank], axis=(1, 2))
            rmse = np.sqrt(np.mean((prediction - y[selected]) ** 2, axis=2)).mean(axis=1)
            for criterion, index in [
                ("rmse", int(rmse.argmin())),
                ("utility", int(utilities.argmax())),
            ]:
                for q in saved["fractions"]:
                    row = rows[split["repeat"], budget, criterion, q]
                    require(
                        (row["selected_lambda"], row["selected_epoch"]) == pairs[index],
                        "Saved model choice differs",
                    )
                    k_assess = math.ceil(q * len(assessment))
                    rank = np.argsort(-scores[index][:, assessment], axis=1, kind="stable")[
                        :, :k_assess
                    ]
                    actual = y[assessment][rank].mean(axis=1)
                    np.testing.assert_allclose(
                        actual, row["assessment_utility_by_seed"], atol=1e-8, rtol=0
                    )
                    require(
                        abs(actual.mean() - row["assessment_utility"]) < 1e-8,
                        "Assessment utility differs",
                    )
                    for index_set, key in [
                        (selected, "selection_ids_sha256"),
                        (assessment, "assessment_ids_sha256"),
                    ]:
                        encoded = json.dumps(
                            ids[index_set].tolist(), separators=(",", ":")
                        ).encode()
                        require(
                            hashlib.sha256(encoded).hexdigest() == row[key],
                            "Selection or assessment ID hash differs",
                        )
                    verified += 1
    result = {
        "verified_rows": verified,
        "repeated_splits": len(saved["field_splits"]),
        "recomputed_all_candidate_choices": True,
        "assessment_utility_recomputed_per_seed": True,
        "disjoint_fields_verified": True,
        "calibration_used": False,
        "absolute_tolerance": 1e-8,
        "results_file": str(results_path),
        "results_sha256": file_sha256(results_path),
        "protocol_sha256": file_sha256(args.protocol),
    }
    write_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
