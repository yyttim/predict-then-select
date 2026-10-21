"""Synthetic checks for independent validation selection and assessment."""

import importlib.util
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts" / "validation_budget.py"
spec = importlib.util.spec_from_file_location("validation_budget", path)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def fixture_arrays():
    y = np.arange(24, dtype=float)
    groups = np.repeat([f"field_{i}" for i in range(8)], 3)
    scores = np.array([[y + 20, y + 10], [-y + 10, -y + 30]])
    return y, scores, groups, np.arange(24), [(0, 20), (1, 40)]


def test_selection_and_assessment_are_disjoint_with_nested_budgets():
    arrays = fixture_arrays()
    result = budget.assess_budgets(*arrays, repetitions=3, budgets=(0.25, 0.5, 1.0))
    for split in result["field_splits"]:
        selection, assessment = split["selection_field_order"], split["assessment_fields"]
        assert len(selection) == len(assessment) == 4
        assert set(selection).isdisjoint(assessment)
        rows = [row for row in result["rows"] if row["repeat"] == split["repeat"]]
        assert len({row["assessment_ids_sha256"] for row in rows}) == 1
        assert {row["selection_fields"] for row in rows} == {1, 2, 4}
        for row in rows:
            chosen_groups = selection[: row["selection_fields"]]
            ids = arrays[3][np.isin(arrays[2], chosen_groups)]
            assert row["selection_ids_sha256"] == budget._fingerprint(ids)


def test_assessment_labels_cannot_change_selected_candidate():
    arrays = fixture_arrays()
    first = budget.assess_budgets(*arrays, repetitions=1, budgets=(0.25, 1.0))
    heldout = first["field_splits"][0]["assessment_fields"]
    changed_y = arrays[0].copy()
    changed_y[np.isin(arrays[2], heldout)] += 1000
    second = budget.assess_budgets(changed_y, *arrays[1:], repetitions=1, budgets=(0.25, 1.0))
    for before, after in zip(first["rows"], second["rows"]):
        assert (before["selected_lambda"], before["selected_epoch"]) == (
            after["selected_lambda"],
            after["selected_epoch"],
        )
        assert before["selection_rmse"] == after["selection_rmse"]
        assert after["assessment_utility"] - before["assessment_utility"] == pytest.approx(1000)


def test_rmse_selection_averages_root_errors_across_paired_seeds():
    y = np.zeros(24)
    scores = np.array([[np.zeros(24), np.full(24, 4.0)], [np.full(24, 2.5), np.full(24, 2.5)]])
    result = budget.assess_budgets(
        y, scores, fixture_arrays()[2], np.arange(24), [(0, 20), (1, 40)], repetitions=2
    )
    assert all(row["selected_lambda"] == 0 for row in result["rows"] if row["criterion"] == "rmse")
    assert all(row["selection_rmse"] == 2 for row in result["rows"] if row["criterion"] == "rmse")


def test_assessment_utility_is_recomputed_on_heldout_pool_and_seed_averaged():
    arrays = fixture_arrays()
    result = budget.assess_budgets(*arrays, repetitions=2)
    for row in result["rows"]:
        index = np.flatnonzero(
            np.isin(arrays[2], result["field_splits"][row["repeat"]]["assessment_fields"])
        )
        chosen = int(row["selected_lambda"])
        expected = []
        for score in arrays[1][chosen]:
            selected = np.lexsort((arrays[3][index], -score[index]))[: row["assessment_k"]]
            expected.append(arrays[0][index[selected]].mean())
        np.testing.assert_allclose(row["assessment_utility_by_seed"], expected)
        assert row["assessment_utility"] == np.mean(expected)
    assert "not confidence intervals" in result["uncertainty"]
    assert all("standard_error" not in row for row in result["summaries"])


def test_repeatability_and_rejection_of_missing_fields():
    arrays = fixture_arrays()
    assert budget.assess_budgets(*arrays, repetitions=2) == budget.assess_budgets(
        *arrays, repetitions=2
    )
    groups = arrays[2].astype(object)
    groups[0] = None
    with pytest.raises(ValueError):
        budget.assess_budgets(arrays[0], arrays[1], groups, arrays[3], arrays[4], repetitions=2)


def test_gzip_storage_preserves_all_records_and_is_deterministic(tmp_path):
    result = budget.assess_budgets(*fixture_arrays(), repetitions=2)
    first, second = tmp_path / "first.json.gz", tmp_path / "second.json.gz"
    budget.save_analysis(result, first, tmp_path / "first_summary.json")
    budget.save_analysis(result, second, tmp_path / "second_summary.json")
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"
    assert json.loads(gzip.decompress(first.read_bytes())) == result
    summary = json.loads((tmp_path / "first_summary.json").read_text())
    assert "rows" not in summary and "field_splits" not in summary
    assert summary["summaries"] == result["summaries"]
    assert summary["raw_records"]["rows"] == len(result["rows"])
    assert summary["raw_records"]["sha256"] == budget.file_sha256(first)
