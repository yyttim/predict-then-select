"""Report aggregation and provenance tests using synthetic results."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from summarize_extended_results import load_verified_inputs, sha, summarize


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def bind_audit(root, protocol):
    results = json.loads((root / "results.json").read_text())
    uncertainty = json.loads((root / "uncertainty.json").read_text())
    uncertainty["results_sha256"] = sha(root / "results.json")
    write(root / "uncertainty.json", uncertainty)
    write(
        root / "independent_verification.json",
        {
            "status": "passed",
            "hashes": {
                "protocol": sha(protocol),
                "results": sha(root / "results.json"),
                "uncertainty": sha(root / "uncertainty.json"),
            },
        },
    )


@pytest.fixture
def synthetic_report(tmp_path):
    root = tmp_path / "results" / "extended"
    root.mkdir(parents=True)
    protocol = root.parent / "extended_protocol.json"
    choice = {"family": "mse_lambda_0", "epoch": 40}
    write(
        protocol,
        {
            "primary_fraction": 0.1,
            "test_status": "synthetic",
            "groups": {
                "CD16_fraction_1": {
                    "marker": "CD16",
                    "train_fraction": 1.0,
                    "seeds": [16, 26, 36],
                    "choices": {"regression": choice},
                }
            },
        },
    )
    rows = []
    for seed, utility in [(36, 14.0), (16, 10.0), (26, 12.0)]:
        rows.append(
            {
                "group": "CD16_fraction_1",
                "pool": "within_slide_test",
                "fraction": 0.1,
                "policy": "regression",
                "seed": seed,
                **choice,
                "n_cells": 100,
                "n_fields": 30,
                "k": 10,
                "random_expected_utility": 4.0,
                "oracle_utility": 20.0,
                "utility": utility,
                "rmse": 3.0,
                "calibrated_rmse": 2.0,
                "mae": 2.5,
                "type_matched_random_utility": 9.0,
                "within_type_enrichment": utility - 9.0,
                "enrichment_difference": utility - 4.0,
            }
        )
    write(root / "results.json", {"protocol_sha256": sha(protocol), "metrics": rows})
    write(
        root / "uncertainty.json",
        {
            "protocol_sha256": sha(protocol),
            "scope": "conditional synthetic interval",
            "comparisons": [
                {
                    "group": "CD16_fraction_1",
                    "pool": "within_slide_test",
                    "marker": "CD16",
                    "train_fraction": 1.0,
                    "fraction": 0.1,
                    "comparison": "joint_minus_regression",
                    "difference_a_minus_b": -2.0,
                    "paired_seed_sd": 1.0,
                    "ci_low": -4.0,
                    "ci_high": 1.0,
                    "exchange_diagnostics": {"mean_selected_overlap_fraction": 0.8},
                }
            ],
        },
    )
    bind_audit(root, protocol)
    return protocol, root


def test_report_uses_seed_mean_sample_sd_and_preserves_contrast_direction(synthetic_report):
    result = summarize(*synthetic_report)
    row = result["policy_summaries"][0]
    assert row["n_seeds"] == 3 and row["utility_mean"] == 12 and row["utility_sd"] == 2
    assert row["type_mix_enrichment_mean"] == 5 and row["within_type_enrichment_mean"] == 3
    assert row["enrichment_difference_mean"] == 8
    assert "macro_f1_mean" not in row
    contrast = result["paired_comparisons"][0]
    assert (
        contrast["difference_a_minus_b"] == -2
        and contrast["ci_low"] == -4
        and contrast["ci_high"] == 1
    )


@pytest.mark.parametrize("correlations,n_defined", [([0.4, None, 0.6], 2), ([None, None, None], 0)])
def test_undefined_correlations_are_counted_without_averaging_fewer_seeds(
    synthetic_report, correlations, n_defined
):
    protocol, root = synthetic_report
    result = json.loads((root / "results.json").read_text())
    for row, correlation in zip(result["metrics"], correlations):
        row["spearman"] = correlation
    write(root / "results.json", result)
    bind_audit(root, protocol)
    row = summarize(protocol, root)["policy_summaries"][0]
    assert row["spearman_n_defined"] == n_defined
    assert row["spearman_mean"] is None and row["spearman_sd"] is None
    assert row["utility_mean"] == 12.0


def test_missing_mandatory_utility_is_never_treated_as_undefined_correlation(synthetic_report):
    protocol, root = synthetic_report
    result = json.loads((root / "results.json").read_text())
    for row in result["metrics"]:
        row["utility"] = None
    write(root / "results.json", result)
    bind_audit(root, protocol)
    with pytest.raises(ValueError, match="invalid metric: utility"):
        summarize(protocol, root)


def test_reporting_rejects_duplicate_or_unmatched_paired_seed(synthetic_report):
    protocol, root = synthetic_report
    result = json.loads((root / "results.json").read_text())
    result["metrics"][0]["seed"] = 26
    write(root / "results.json", result)
    bind_audit(root, protocol)
    with pytest.raises(ValueError, match="paired replicate"):
        load_verified_inputs(protocol, root)


def test_final_plots_require_current_complete_independent_audit(synthetic_report, monkeypatch):
    protocol, root = synthetic_report
    spec = importlib.util.spec_from_file_location(
        "plot_extended_results", SCRIPTS / "plot_extended_results.py"
    )
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    monkeypatch.setattr(plot, "ROOT", root.parent.parent)
    monkeypatch.setattr(
        plot, "save", lambda *_: pytest.fail("Unverified final figure was generated")
    )
    result = json.loads((root / "results.json").read_text())
    result["metrics"][0]["utility"] += 1
    write(root / "results.json", result)
    with pytest.raises(ValueError, match="independent verification"):
        plot.completed_study()
