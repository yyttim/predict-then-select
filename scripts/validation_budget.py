"""Validation-size study with disjoint field selection and assessment pools.

This is an exploratory follow-up using frozen models. Repeated overlapping
splits describe conditional sensitivity, not independent experimental replicates.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validation_stability import load_frozen_validation
from cellselect.metrics import topk_indices
from cellselect.training import file_sha256, write_json


def _fingerprint(values):
    return hashlib.sha256(
        json.dumps(np.asarray(values).tolist(), separators=(",", ":")).encode()
    ).hexdigest()


def _measure(y, scores, ids, index, fractions):
    target, prediction = y[index], scores[:, :, index]
    rmse = np.sqrt(np.mean((prediction - target) ** 2, axis=2)).mean(axis=1)
    utilities = []
    for fraction in fractions:
        k = math.ceil(fraction * len(index))
        utilities.append(
            np.array(
                [
                    [target[topk_indices(s, k, ids[index])].mean() for s in family]
                    for family in prediction
                ]
            )
        )
    return rmse, np.asarray(utilities)


def assess_budgets(
    y,
    scores,
    groups,
    ids,
    candidates,
    *,
    repetitions=500,
    seed=20261022,
    budgets=(0.1, 0.25, 0.5, 1.0),
    fractions=(0.05, 0.1, 0.2),
    selection_fraction=0.1,
):
    y, scores, pairs = (
        np.asarray(y, float),
        np.asarray(scores, float),
        np.asarray(candidates, float),
    )
    groups, ids = np.asarray(groups), np.asarray(ids)
    if (
        y.ndim != 1
        or not len(y)
        or scores.ndim != 3
        or scores.shape[1] < 1
        or scores.shape[2] != len(y)
        or pairs.shape != (scores.shape[0], 2)
        or groups.shape != y.shape
        or ids.shape != y.shape
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(y).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(pairs).all()
        or len(np.unique(pairs, axis=0)) != len(pairs)
        or (pairs[:, 0] < 0).any()
        or (pairs[:, 1] < 1).any()
        or not np.equal(pairs[:, 1], np.floor(pairs[:, 1])).all()
    ):
        raise ValueError("Invalid aligned candidate/seed/cell inputs")
    if any(
        g is None or (isinstance(g, (float, np.floating)) and not np.isfinite(g)) for g in groups
    ):
        raise ValueError("Missing field IDs")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or not budgets
        or list(budgets) != sorted(set(budgets))
        or any(not 0 < b <= 1 for b in budgets)
        or not fractions
        or len(set(fractions)) != len(fractions)
        or any(not 0 < q <= 1 for q in fractions)
        or not 0 < selection_fraction <= 1
    ):
        raise ValueError("Invalid repetitions, nested budgets, or selection fractions")
    groups = groups.astype(str)
    fields = np.unique(groups)
    if len(fields) < 4:
        raise ValueError("At least four fields are required")
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, scores = pairs[order], scores[order]
    rng = np.random.default_rng(seed)
    rows, splits = [], []
    for repeat in range(repetitions):
        field_order = rng.permutation(fields)
        selection_fields, assessment_fields = np.split(field_order, [len(fields) // 2])
        assessment_index = np.flatnonzero(np.isin(groups, assessment_fields))
        assessment_rmse, assessment_utilities = _measure(
            y, scores, ids, assessment_index, fractions
        )
        splits.append(
            {
                "repeat": repeat,
                "selection_field_order": selection_fields.tolist(),
                "assessment_fields": assessment_fields.tolist(),
            }
        )
        for budget in budgets:
            selected_fields = selection_fields[: max(1, math.ceil(budget * len(selection_fields)))]
            index = np.flatnonzero(np.isin(groups, selected_fields))
            rmse, utility = _measure(y, scores, ids, index, [selection_fraction])
            mean_utility = utility[0].mean(axis=1)
            for criterion, chosen in [
                ("rmse", int(np.argmin(rmse))),
                ("utility", int(np.argmax(mean_utility))),
            ]:
                for j, fraction in enumerate(fractions):
                    values = assessment_utilities[j, chosen]
                    rows.append(
                        {
                            "repeat": repeat,
                            "budget_fraction": float(budget),
                            "criterion": criterion,
                            "selected_lambda": float(pairs[chosen, 0]),
                            "selected_epoch": int(pairs[chosen, 1]),
                            "selection_fields": len(selected_fields),
                            "selection_cells": len(index),
                            "assessment_fields": len(assessment_fields),
                            "assessment_cells": len(assessment_index),
                            "selection_ids_sha256": _fingerprint(ids[index]),
                            "assessment_ids_sha256": _fingerprint(ids[assessment_index]),
                            "selection_rmse": float(rmse[chosen]),
                            "selection_utility": float(mean_utility[chosen]),
                            "assessment_rmse": float(assessment_rmse[chosen]),
                            "fraction": float(fraction),
                            "assessment_k": math.ceil(fraction * len(assessment_index)),
                            "assessment_utility": float(values.mean()),
                            "assessment_utility_by_seed": values.tolist(),
                        }
                    )
    summaries, comparisons = [], []

    def describe(values):
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "split_quantiles_025_975": np.quantile(values, [0.025, 0.975]).tolist(),
        }

    for budget in budgets:
        for fraction in fractions:
            series = {}
            for criterion in ("rmse", "utility"):
                selected = [
                    r
                    for r in rows
                    if r["budget_fraction"] == budget
                    and r["fraction"] == fraction
                    and r["criterion"] == criterion
                ]
                series[criterion] = np.array([r["assessment_utility"] for r in selected])
                summaries.append(
                    {
                        "budget_fraction": budget,
                        "fraction": fraction,
                        "criterion": criterion,
                        "selection_cells_mean": float(
                            np.mean([r["selection_cells"] for r in selected])
                        ),
                        **describe(series[criterion]),
                    }
                )
            differences = series["utility"] - series["rmse"]
            comparisons.append(
                {
                    "budget_fraction": budget,
                    "fraction": fraction,
                    "contrast": "utility-selected minus RMSE-selected assessment utility",
                    **describe(differences),
                    "fraction_positive": float(np.mean(differences > 0)),
                    "fraction_zero": float(np.mean(differences == 0)),
                }
            )
    return {
        "analysis": "exploratory validation-field selection/assessment cross-fitting",
        "partition": "val",
        "seed": seed,
        "repetitions": repetitions,
        "split": "50/50 whole fields, disjoint within each repeat; nested selection-half prefixes",
        "budgets": list(budgets),
        "fractions": list(fractions),
        "selection_fraction": selection_fraction,
        "candidate_pairs": [{"lambda_class": float(w), "epoch": int(e)} for w, e in pairs],
        "n_training_seeds": scores.shape[1],
        "n_cells": len(y),
        "n_fields": len(fields),
        "calibration": "raw predictions; no globally fitted calibration is used",
        "uncertainty": "Quantiles describe overlapping random splits, not confidence intervals. No independent-replicate standard error is computed. Models are fixed; no retraining, test selection, or slide variation is represented.",
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "rows": rows,
        "field_splits": splits,
    }


def save_analysis(result, output, summary_output):
    """Compression changes storage only; all raw records remain available."""
    output, summary_output = Path(output), Path(summary_output)
    if output.suffix == ".gz":
        payload = json.dumps(result, separators=(",", ":"), allow_nan=False).encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
        temporary.replace(output)
    else:
        write_json(output, result)
    summary = {key: value for key, value in result.items() if key not in {"rows", "field_splits"}}
    summary["raw_records"] = {
        "path": output.name,
        "sha256": file_sha256(output),
        "compression": "gzip" if output.suffix == ".gz" else "none",
        "rows": len(result["rows"]),
        "field_splits": len(result["field_splits"]),
    }
    write_json(summary_output, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "protocol"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "validation_budget.json.gz",
    )
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    summary_output = args.summary_output or args.output.parent / "validation_budget_summary.json"
    if args.output.exists() or summary_output.exists():
        raise FileExistsError("Use a new output path instead of replacing completed analysis")
    protocol, arrays = load_frozen_validation(args.data, args.runs, args.protocol)
    result = assess_budgets(
        *arrays,
        repetitions=args.repetitions,
        seed=args.seed,
        fractions=protocol["fractions"],
        selection_fraction=protocol["primary_fraction"],
    )
    result.update(
        protocol_sha256=file_sha256(args.protocol),
        training_seeds=protocol["seeds"],
        source_sha256=file_sha256(__file__),
        existing_main_test_previously_evaluated=True,
    )
    save_analysis(result, args.output, summary_output)
    print(
        f"Saved {len(result['rows'])} raw rows from {args.repetitions} disjoint selection/assessment splits."
    )


if __name__ == "__main__":
    main()
