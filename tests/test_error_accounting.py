"""Known finite pools check the accounting direction, denominator, and ties."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "error_accounting", Path(__file__).resolve().parents[1] / "scripts/error_accounting.py"
)
accounting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accounting)


def test_tied_scores_use_ids_and_loss_contributions_keep_full_pool_denominator():
    groups, totals = accounting.account(
        np.array([0.0, 6.0, 10.0, 4.0, 2.0]),
        np.array([2.0, 2.0, 1.0, 0.0, -1.0]),
        np.array([0.0, 2.0, 2.0, 0.0, -1.0]),
        np.array([4, 1, 3, 2, 5]),
        6.0,
        0.4,
    )
    groups = {g["group"]: g for g in groups}
    assert [
        groups[g]["group_cells"] for g in ("selected_by_both", "exchanged", "selected_by_neither")
    ] == [1, 2, 2]
    assert totals["old_high_selected"] == 1 and totals["new_high_selected"] == 2
    assert (
        totals["exchanged_per_policy"] == 1
        and totals["mean_change"] == 5.0
        and totals["purity_change"] == 0.5
    )
    assert groups["selected_by_both"]["log_loss_gain_per_pool_cell"] == 0.0
    assert groups["selected_by_neither"]["log_loss_gain_per_pool_cell"] == 0.0
    # Two changed losses are averaged over all five cells, not the exchanged pair.
    expected = (
        np.log1p(np.exp(2.0)) - np.log(2.0) + np.log1p(np.exp(-1.0)) - np.log1p(np.exp(-2.0))
    ) / 5
    assert groups["exchanged"]["log_loss_gain_per_pool_cell"] == pytest.approx(expected)


def test_identical_scores_have_zero_exchanges_and_zero_gains():
    scores = np.ones(4)
    groups, totals = accounting.account(
        np.array([1.0, 9.0, 4.0, 7.0]), scores, scores, np.array([4, 1, 3, 2]), 7.0, 0.5
    )
    assert totals["old_high_selected"] == totals["new_high_selected"] == 2
    assert totals["exchanged_per_policy"] == 0
    for key in ("mean_change", "purity_change", "log_loss_gain", "brier_gain"):
        assert totals[key] == 0.0
    assert next(g for g in groups if g["group"] == "exchanged")["group_cells"] == 0
