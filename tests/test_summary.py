"""Summary tests using synthetic result records."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("cellselect_summary", SCRIPT)
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


@pytest.fixture
def synthetic_results():
    seeds = [16, 26, 36]
    protocol = {
        "expected_experiment_verified": True,
        "test_used_to_choose_protocol": False,
        "seeds": seeds,
        "weights": [0.0, 1.0],
        "primary_fraction": 0.1,
        "reference_epoch": 40,
        "selected_by_utility": {"lambda_class": 1.0, "epoch": 20},
        "selected_by_rmse": {"lambda_class": 0.0, "epoch": 40},
        "regression_baseline": {"lambda_class": 0.0, "epoch": 20},
        "classifier_baseline": {"lambda_class": 0.0, "epoch": 40},
    }
    configurations = [
        ("utility_selector", "joint", 1.0, 20, [10.0, 12.0, 14.0]),
        ("rmse_selector", "joint", 0.0, 40, [13.0, 14.0, 15.0]),
        ("regression_baseline", "joint", 0.0, 20, [8.0, 10.0, 12.0]),
        ("class_means_baseline", "classifier", 0.0, 40, [11.0, 11.0, 11.0]),
    ]
    selected = []
    for policy, mode, weight, epoch, utilities in configurations:
        for j, seed in enumerate(seeds):
            row = {
                "policy": policy,
                "mode": mode,
                "lambda_class": weight,
                "epoch": epoch,
                "seed": seed,
                "fraction": 0.1,
                "rmse": 3.0 + j,
                "calibrated_rmse": 2.0 + j,
                "utility": utilities[j],
                "macro_f1": 0.7 + 0.1 * j,
                "within_type_gap": 4.0 + 2 * j,
                "type_allocation_gap": 1.0 + j,
                "k": 10,
                "selected_type_counts": {"0": 4 - j, "1": 6 + j, "2": 0},
            }
            selected.extend([row, {**row, "fraction": 0.2, "utility": 999.0}])
    frontier = []
    for mode, weight in [("joint", 0.0), ("joint", 1.0), ("classifier", 0.0)]:
        for j, seed in enumerate(seeds):
            base = selected[2 * j]
            frontier.append({**base, "mode": mode, "lambda_class": weight, "epoch": 40})
    comparisons = []
    for suffix, baseline in [
        ("rmse_selector", [13.0, 14.0, 15.0]),
        ("regression", [8.0, 10.0, 12.0]),
        ("class_means", [11.0, 11.0, 11.0]),
    ]:
        differences = np.array([10.0, 12.0, 14.0]) - baseline
        mean = float(differences.mean())
        comparisons.append(
            {
                "comparison": "utility_selector_minus_" + suffix,
                "fraction": 0.1,
                "difference_a_minus_b": mean,
                "paired_seed_differences": differences.tolist(),
                "paired_seed_sd": float(differences.std(ddof=1)),
                "ci_low": mean - 1.5,
                "ci_high": mean + 1.5,
                "n_seeds": 3,
                "n_fields": 20,
            }
        )
    results = {
        "protocol_sha256": "synthetic",
        "reference_epoch": 40,
        "n_test": 100,
        "n_test_fields": 20,
        "selected_metrics": selected,
        "metrics": frontier,
        "aggregate_comparisons": comparisons,
    }
    return protocol, results


def test_known_means_sample_sd_and_regression_f1_are_correct(synthetic_results):
    summary = summary_module.summarize(*synthetic_results)
    assert summary["n_training_seeds"] == 3 and summary["n_test_cells"] == 100
    rows = {r["policy"]: r for r in summary["policies"]}
    assert rows["utility_selector"]["metrics"]["utility"] == {"mean": 12.0, "sd": 2.0}
    assert rows["utility_selector"]["metrics"]["rmse"] == {"mean": 4.0, "sd": 1.0}
    assert rows["utility_selector"]["metrics"]["calibrated_rmse"] == {"mean": 3.0, "sd": 1.0}
    assert rows["utility_selector"]["metrics"]["macro_f1"]["mean"] == pytest.approx(0.8)
    # Ignore even an accidentally supplied F1 for an untrained classification head.
    assert rows["rmse_selector"]["metrics"]["macro_f1"] is None
    assert rows["regression_baseline"]["metrics"]["macro_f1"] is None
    assert rows["class_means_baseline"]["metrics"]["macro_f1"] is not None
    frontier = summary["reference_frontier"][0]
    assert frontier["epoch"] == 40 and frontier["n_seeds"] == 3
    assert frontier["metrics"]["within_type_gap"] == {"mean": 6.0, "sd": 2.0}
    assert frontier["metrics"]["type_allocation_gap"] == {"mean": 2.0, "sd": 1.0}
    assert frontier["metrics"]["selected_granulocyte_fraction"]["mean"] == pytest.approx(0.7)
    assert frontier["metrics"]["selected_granulocyte_fraction"]["sd"] == pytest.approx(0.1)


def test_comparison_sign_and_uncertainty_definitions_are_preserved(synthetic_results):
    summary = summary_module.summarize(*synthetic_results)
    comparison = summary["paired_comparisons"][0]
    assert comparison["difference_a_minus_b"] == -2.0
    assert comparison["paired_seed_sd"] == 1.0
    assert comparison["ci_low"] == -3.5 and comparison["ci_high"] == -0.5
    text = summary_module.markdown(summary, {"Frozen protocol": "protocol.json"})
    assert "-2.000" in text and "[-3.500, -0.500]" in text
    assert "one value per paired training seed" in text
    assert "exclude retraining, model-selection, split and slide/donor uncertainty" in text
    assert "does not isolate representation changes causally" in text
    assert "| Class-means baseline | — | 40 | 3 |" in text
    assert "| Class-means baseline | — | 3 |" in text
    assert "trained with CE only, so λ is not applicable" in text
    assert summary["policies"][-1]["lambda_class"] == 0.0


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_epoch", "missing_frontier"])
def test_duplicate_incomplete_or_unfrozen_policy_rows_are_rejected(synthetic_results, change):
    protocol, results = synthetic_results
    if change == "duplicate":
        results["selected_metrics"].append(copy.deepcopy(results["selected_metrics"][0]))
    elif change == "missing":
        results["selected_metrics"].pop(0)
    elif change == "wrong_epoch":
        results["selected_metrics"][0]["epoch"] = 200
    else:
        results["metrics"].pop(0)
    with pytest.raises(ValueError):
        summary_module.summarize(protocol, results)


@pytest.mark.parametrize(
    "change", ["wrong_sign", "short_pair_vector", "wrong_sd", "wrong_fields", "reversed_ci"]
)
def test_inconsistent_paired_comparison_is_rejected(synthetic_results, change):
    protocol, results = synthetic_results
    comparison = results["aggregate_comparisons"][1]
    if change == "wrong_sign":
        comparison["difference_a_minus_b"] *= -1
    if change == "short_pair_vector":
        comparison["paired_seed_differences"] = [2.0]
    if change == "wrong_sd":
        comparison["paired_seed_sd"] = 10.0
    if change == "wrong_fields":
        comparison["n_fields"] = 100
    if change == "reversed_ci":
        comparison["ci_low"], comparison["ci_high"] = 10.0, -10.0
    with pytest.raises(ValueError):
        summary_module.summarize(protocol, results)


def test_cli_writes_json_markdown_and_real_source_links(synthetic_results, tmp_path):
    protocol, results = synthetic_results
    protocol_file, results_file, stability_file = [
        tmp_path / name for name in ("protocol.json", "results.json", "stability.json")
    ]
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    digest = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    results["protocol_sha256"] = digest
    results_file.write_text(json.dumps(results), encoding="utf-8")
    stability = {
        "protocol_sha256": digest,
        "partition": "val",
        "schema_specific_payload": [0.3, 0.7],
    }
    stability_file.write_text(json.dumps(stability), encoding="utf-8")
    output = tmp_path / "summary"
    command = [
        sys.executable,
        str(SCRIPT),
        "--protocol",
        str(protocol_file),
        "--results",
        str(results_file),
        "--stability",
        str(stability_file),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "3 paired seeds" in completed.stdout
    written = json.loads((output / "core_summary.json").read_text())
    assert written["validation_stability"] == stability
    assert all(Path(path).is_file() for path in written["sources"].values())
    text = (output / "core_tables.md").read_text(encoding="utf-8")
    assert "[Frozen protocol](<../protocol.json>)" in text
    assert "[Evaluation results](<../results.json>)" in text
    assert "[Validation stability](<../stability.json>)" in text
    stability["protocol_sha256"] = "wrong protocol"
    stability_file.write_text(json.dumps(stability), encoding="utf-8")
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and "different frozen protocol" in failed.stderr
