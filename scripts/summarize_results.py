"""Render descriptive tables from a frozen protocol and its completed evaluation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

POLICIES = {
    "utility_selector": "Utility selector",
    "rmse_selector": "RMSE selector",
    "regression_baseline": "Regression baseline",
    "class_means_baseline": "Class-means baseline",
}
UNCERTAINTY = (
    "Means and sample SDs use one value per paired training seed, not cells times seeds. "
    "The 95% paired FOV percentile intervals use common test-field draws across fixed trained models "
    "and frozen policy choices. They exclude retraining, model-selection, split and slide/donor uncertainty."
)


def _paired(rows, seeds):
    if len(rows) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
        raise ValueError("Expected exactly one observation per declared training seed")
    return [next(row for row in rows if row["seed"] == seed) for seed in seeds]


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot summarize missing or nonfinite metrics")
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def summarize(protocol, results, stability=None):
    if (
        not protocol.get("expected_experiment_verified")
        or protocol.get("test_used_to_choose_protocol") is not False
    ):
        raise ValueError("A verified validation-only frozen protocol is required")
    seeds, fraction, reference = (
        protocol["seeds"],
        protocol["primary_fraction"],
        protocol["reference_epoch"],
    )
    if not seeds or len(set(seeds)) != len(seeds) or results["reference_epoch"] != reference:
        raise ValueError("Invalid seed list or mismatched reference epoch")
    choices = {
        "utility_selector": ("joint", protocol["selected_by_utility"]),
        "rmse_selector": ("joint", protocol["selected_by_rmse"]),
        "regression_baseline": ("joint", protocol["regression_baseline"]),
        "class_means_baseline": ("classifier", protocol["classifier_baseline"]),
    }
    policies, utilities = [], {}
    for policy, (mode, choice) in choices.items():
        rows = _paired(
            [
                r
                for r in results["selected_metrics"]
                if r["policy"] == policy and r["fraction"] == fraction
            ],
            seeds,
        )
        if any(
            (r["mode"], r["lambda_class"], r["epoch"])
            != (mode, choice["lambda_class"], choice["epoch"])
            for r in rows
        ):
            raise ValueError("Reported policy differs from the frozen choice")
        metrics = {
            name: _stats([r[name] for r in rows]) for name in ("rmse", "calibrated_rmse", "utility")
        }
        metrics["macro_f1"] = (
            _stats([r["macro_f1"] for r in rows])
            if mode == "classifier" or choice["lambda_class"] > 0
            else None
        )
        policies.append(
            {
                "policy": policy,
                "mode": mode,
                "lambda_class": choice["lambda_class"],
                "epoch": choice["epoch"],
                "n_seeds": len(seeds),
                "metrics": metrics,
            }
        )
        utilities[policy] = np.array([r["utility"] for r in rows])
    comparisons = []
    for suffix, baseline in [
        ("rmse_selector", "rmse_selector"),
        ("regression", "regression_baseline"),
        ("class_means", "class_means_baseline"),
    ]:
        name = "utility_selector_minus_" + suffix
        found = [
            c
            for c in results["aggregate_comparisons"]
            if c["comparison"] == name and c["fraction"] == fraction
        ]
        if (
            len(found) != 1
            or found[0]["n_seeds"] != len(seeds)
            or found[0]["n_fields"] != results["n_test_fields"]
        ):
            raise ValueError("Missing, duplicate, or incomplete paired comparison")
        item, difference = found[0], utilities["utility_selector"] - utilities[baseline]
        if (
            np.shape(item["paired_seed_differences"]) != difference.shape
            or not np.allclose(item["paired_seed_differences"], difference, rtol=0, atol=1e-8)
            or not np.isclose(item["difference_a_minus_b"], difference.mean(), rtol=0, atol=1e-8)
        ):
            raise ValueError("Comparison sign or pairing disagrees with selected policy utilities")
        sd = _stats(difference)["sd"]
        if (sd is None and item["paired_seed_sd"] is not None) or (
            sd is not None and not np.isclose(item["paired_seed_sd"], sd)
        ):
            raise ValueError("Paired training-seed SD is inconsistent")
        if (
            not np.isfinite([item["ci_low"], item["ci_high"]]).all()
            or item["ci_low"] > item["ci_high"]
        ):
            raise ValueError("Invalid paired FOV interval")
        comparisons.append(
            {
                "comparison": name,
                "baseline": baseline,
                **{
                    k: item[k]
                    for k in (
                        "difference_a_minus_b",
                        "paired_seed_sd",
                        "ci_low",
                        "ci_high",
                        "n_seeds",
                        "n_fields",
                    )
                },
            }
        )
    frontier = []
    for mode, weight in [("joint", w) for w in protocol["weights"]] + [("classifier", 0.0)]:
        rows = _paired(
            [
                r
                for r in results["metrics"]
                if r["mode"] == mode
                and r["lambda_class"] == weight
                and r["epoch"] == reference
                and r["fraction"] == fraction
            ],
            seeds,
        )
        fractions = []
        for row in rows:
            counts, k = row["selected_type_counts"], row["k"]
            if k <= 0 or any(v < 0 for v in counts.values()) or sum(counts.values()) != k:
                raise ValueError("Selected type counts do not match selection size")
            fractions.append(counts.get("1", counts.get(1, 0)) / k)
        metrics = {
            name: _stats([r[name] for r in rows])
            for name in ("rmse", "utility", "within_type_gap", "type_allocation_gap")
        }
        metrics["macro_f1"] = (
            _stats([r["macro_f1"] for r in rows]) if mode == "classifier" or weight > 0 else None
        )
        metrics["selected_granulocyte_fraction"] = _stats(fractions)
        frontier.append(
            {
                "mode": mode,
                "lambda_class": weight,
                "epoch": reference,
                "n_seeds": len(seeds),
                "metrics": metrics,
            }
        )
    return {
        "protocol_sha256": results["protocol_sha256"],
        "primary_fraction": fraction,
        "reference_epoch": reference,
        "training_seeds": seeds,
        "n_training_seeds": len(seeds),
        "n_test_cells": results["n_test"],
        "n_test_fields": results["n_test_fields"],
        "uncertainty": UNCERTAINTY,
        "policies": policies,
        "paired_comparisons": comparisons,
        "reference_frontier": frontier,
        "validation_stability": stability,
    }


def markdown(summary, links):
    def estimate(value):
        if value is None:
            return "—"
        return f"{value['mean']:.3f}" + (f" ± {value['sd']:.3f}" if value["sd"] is not None else "")

    lines = [
        "# Core result tables",
        "",
        " · ".join(f"[{name}](<{path}>)" for name, path in links.items()),
        "",
        f"Primary selection fraction: {summary['primary_fraction']:.0%}. Training seeds: "
        + ", ".join(map(str, summary["training_seeds"]))
        + ".",
        "",
        summary["uncertainty"],
        "",
        "Calibration is fitted on validation data. Macro-F1 is omitted for an untrained classification head; SD is unavailable with one seed. The class-means baseline is trained with CE only, so λ is not applicable.",
        "",
        "| Policy | λ | Epoch | n seeds | RMSE | Calibrated RMSE | Utility | Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["policies"]:
        values = " | ".join(
            estimate(row["metrics"][key])
            for key in ("rmse", "calibrated_rmse", "utility", "macro_f1")
        )
        weight = "—" if row["mode"] == "classifier" else f"{row['lambda_class']:g}"
        lines.append(
            f"| {POLICIES[row['policy']]} | {weight} | {row['epoch']} | {row['n_seeds']} | {values} |"
        )
    lines += [
        "",
        "Differences below are utility selector minus the named comparison policy.",
        "",
        "| Comparison policy | Mean utility difference | Paired seed SD | Conditional 95% FOV interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["paired_comparisons"]:
        sd = "—" if row["paired_seed_sd"] is None else f"{row['paired_seed_sd']:.3f}"
        lines.append(
            f"| {POLICIES[row['baseline']]} | {row['difference_a_minus_b']:.3f} | {sd} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |"
        )
    lines += [
        "",
        f"## Reference frontier at epoch {summary['reference_epoch']}",
        "",
        "Stopping time is held fixed here; this comparison does not isolate representation changes causally.",
        "",
        "| Model | λ | n seeds | Macro-F1 | RMSE | Utility | Within-type gap | Type-allocation gap | Selected granulocyte fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["reference_frontier"]:
        label = "Shared model" if row["mode"] == "joint" else "Class-means baseline"
        values = " | ".join(
            estimate(row["metrics"][key])
            for key in (
                "macro_f1",
                "rmse",
                "utility",
                "within_type_gap",
                "type_allocation_gap",
                "selected_granulocyte_fraction",
            )
        )
        weight = "—" if row["mode"] == "classifier" else f"{row['lambda_class']:g}"
        lines.append(f"| {label} | {weight} | {row['n_seeds']} | {values} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--stability", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    args = parser.parse_args()
    protocol, results = [
        json.loads(path.read_text(encoding="utf-8")) for path in (args.protocol, args.results)
    ]
    digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    stability = json.loads(args.stability.read_text(encoding="utf-8")) if args.stability else None
    if results["protocol_sha256"] != digest or (
        stability is not None and stability["protocol_sha256"] != digest
    ):
        raise ValueError("Results or stability belong to a different frozen protocol")
    summary = summarize(protocol, results, stability)
    sources = {"Frozen protocol": args.protocol, "Evaluation results": args.results}
    if args.stability:
        sources["Validation stability"] = args.stability
    summary["sources"] = {name: str(path.resolve()) for name, path in sources.items()}
    text = markdown(
        summary,
        {
            name: Path(os.path.relpath(path.resolve(), args.output.resolve())).as_posix()
            for name, path in sources.items()
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "core_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output / "core_tables.md").write_text(text, encoding="utf-8")
    print(
        f"Summarized {summary['n_training_seeds']} paired seeds; wrote core_summary.json and core_tables.md"
    )


if __name__ == "__main__":
    main()
