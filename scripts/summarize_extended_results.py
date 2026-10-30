"""Compact tables from the independently verified frozen study; no model selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_verified_inputs(protocol_path, result_root):
    """Require the current audited outputs and paired replicates before reporting."""
    root = Path(result_root)
    protocol = read(protocol_path)
    results = read(root / "results.json")
    uncertainty = read(root / "uncertainty.json")
    audit = read(root / "independent_verification.json")
    expected_hashes = {
        "protocol": sha(protocol_path),
        "results": sha(root / "results.json"),
        "uncertainty": sha(root / "uncertainty.json"),
    }
    if audit["status"] != "passed" or any(
        audit["hashes"][key] != value for key, value in expected_hashes.items()
    ):
        raise ValueError("The complete current study must pass independent verification first")
    if (
        results["protocol_sha256"] != uncertainty["protocol_sha256"]
        or results["protocol_sha256"] != expected_hashes["protocol"]
        or uncertainty["results_sha256"] != expected_hashes["results"]
    ):
        raise ValueError("Mismatched frozen results")
    frame = pd.DataFrame(results["metrics"])
    for (group, _, _, policy), paired in frame.groupby(
        ["group", "pool", "fraction", "policy"], sort=True
    ):
        settings = protocol["groups"][group]
        if paired.sort_values("seed").seed.tolist() != settings["seeds"]:
            raise ValueError("Missing or duplicate paired replicate in report")
        chosen = settings["choices"][policy]
        if set(zip(paired.family, paired.epoch)) != {(chosen["family"], chosen["epoch"])}:
            raise ValueError("Policy does not match its frozen choice")
    return protocol, results, uncertainty, expected_hashes


def summarize(protocol_path, result_root):
    root = Path(result_root)
    protocol, results, uncertainty, expected_hashes = load_verified_inputs(protocol_path, root)
    frame = pd.DataFrame(results["metrics"])
    keys = ["group", "pool", "fraction", "policy"]
    metric_names = [
        "rmse",
        "calibrated_rmse",
        "mae",
        "spearman",
        "utility",
        "enrichment_difference",
        "macro_f1",
        "type_mix_enrichment",
        "within_type_enrichment",
    ]
    mandatory = {"rmse", "calibrated_rmse", "mae", "utility", "enrichment_difference"}
    if "type_matched_random_utility" in frame:
        frame["type_mix_enrichment"] = (
            frame.type_matched_random_utility - frame.random_expected_utility
        )
    rows = []
    for values, paired in frame.groupby(keys, sort=True):
        group, pool, fraction, policy = values
        settings = protocol["groups"][group]
        paired = paired.sort_values("seed")
        chosen = settings["choices"][policy]
        row = dict(zip(keys, values))
        row.update(
            marker=settings["marker"],
            train_fraction=settings["train_fraction"],
            family=chosen["family"],
            epoch=chosen["epoch"],
            n_seeds=len(paired),
            n_cells=int(paired.n_cells.iloc[0]),
            n_fields=int(paired.n_fields.iloc[0]),
            selection_size=int(paired.k.iloc[0]),
            random_utility=float(paired.random_expected_utility.iloc[0]),
            hindsight_utility=float(paired.oracle_utility.iloc[0]),
        )
        for metric in metric_names:
            if metric not in paired:
                if metric in mandatory:
                    raise ValueError(f"Missing mandatory metric: {metric}")
                continue
            numbers = paired[metric].to_numpy(dtype=float)
            if metric in {"pearson", "spearman"}:
                if np.isinf(numbers).any():
                    raise ValueError(f"Invalid correlation: {metric}")
                row[metric + "_n_defined"] = int(np.isfinite(numbers).sum())
                if not np.isfinite(numbers).all():
                    row[metric + "_mean"] = row[metric + "_sd"] = None
                    continue
            elif paired[metric].isna().all() and metric not in mandatory:
                continue
            if not np.isfinite(numbers).all():
                raise ValueError(f"Partially missing or invalid metric: {metric}")
            row[metric + "_mean"] = float(numbers.mean())
            row[metric + "_sd"] = float(numbers.std(ddof=1))
        rows.append(row)
    contrasts = []
    for row in uncertainty["comparisons"]:
        contrasts.append(
            {
                key: row[key]
                for key in [
                    "group",
                    "pool",
                    "marker",
                    "train_fraction",
                    "fraction",
                    "comparison",
                    "difference_a_minus_b",
                    "paired_seed_sd",
                    "ci_low",
                    "ci_high",
                ]
            }
            | {
                "selected_overlap_fraction": row["exchange_diagnostics"][
                    "mean_selected_overlap_fraction"
                ]
            }
        )
    summary = {
        "input_sha256": expected_hashes
        | {
            "independent_verification": sha(root / "independent_verification.json"),
            "summary_script": sha(__file__),
        },
        "primary_fraction": protocol["primary_fraction"],
        "policy_summaries": rows,
        "paired_comparisons": contrasts,
        "interpretation": {
            "means_and_sd": "Three paired model replicates, no prediction ensemble. Low-data runs vary subset and initialization.",
            "intervals": uncertainty["scope"],
            "within_slide": protocol["test_status"],
            "external": "Each physical slide is ranked separately without fitting. Acquisition and population-selection shift; donor independence unverified.",
            "calibration": "Positive affine map fitted on validation predictions only; all selections use the original scores.",
            "correlation": "Undefined correlations remain null; no aggregate correlation is reported unless all declared replicates are defined.",
            "enrichment": "Type-mix and within-type terms sum to achieved enrichment, using true types only for retrospective accounting.",
            "hindsight": "The measured-outcome optimum includes unpredictable variation; its full gap is not necessarily learnable.",
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(root / "policy_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(root / "paired_summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("results/extended_protocol.json"))
    parser.add_argument("--results", type=Path, default=Path("results/extended"))
    args = parser.parse_args()
    summary = summarize(args.protocol, args.results)
    print(
        json.dumps(
            {
                "policy_rows": len(summary["policy_summaries"]),
                "paired_comparisons": len(summary["paired_comparisons"]),
            }
        )
    )


if __name__ == "__main__":
    main()
