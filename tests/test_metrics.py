import numpy as np
import pytest

from cellselect.metrics import (
    paired_field_bootstrap,
    prediction_metrics,
    selection_metrics,
    top_indices,
)


def test_ties_use_ids_and_never_outcomes():
    ids = np.array([30, 10, 20])
    assert top_indices([1, 1, 1], ids, 0.5).tolist() == [1, 2]
    a = selection_metrics([100, 2, 3], [1, 1, 1], ids, 3, 0.5)
    assert a["mean_marker"] == 2.5 and a["purity"] == 0.5


def test_threshold_includes_equality_and_ceiling_budget():
    result = selection_metrics([1, 2, 3], [3, 2, 1], [0, 1, 2], 2, 0.5)
    assert result["n_selected"] == 2
    assert result["n_high"] == 2
    assert result["purity"] == 0.5
    assert result["oracle_purity"] == 1
    result = selection_metrics([1, 2, 3], [3, 2, 1], [0, 1, 2], 4, 1)
    assert result["oracle_purity"] == 0


def test_extreme_logits_keep_order_and_finite_diagnostics():
    assert top_indices([1000, 1001], [0, 1], 0.5).tolist() == [1]
    m = prediction_metrics([0, 2], 1, hit_logit=[-1000, 1000])
    assert m["brier"] == 0 and m["log_loss"] == 0
    assert "rmse" not in m


def test_mean_enrichment_and_hit_purity_can_prefer_different_sets():
    # A has mean20 and hit probability.2; B has mean15 and hit probability1.
    target = np.array([100, 0, 0, 0, 0, 15, 15, 15, 15, 15])
    mean_score = [20] * 5 + [15] * 5
    hit_score = [0.2] * 5 + [1] * 5
    a = selection_metrics(target, mean_score, np.arange(10), 10, 0.5)
    b = selection_metrics(target, hit_score, np.arange(10), 10, 0.5)
    assert a["mean_marker"] > b["mean_marker"]
    assert a["purity"] < b["purity"]


def test_paired_resampling_cancels_identical_policies():
    y = np.array([1, 8, 4, 3, 5, 9], dtype=float)
    a = np.stack([y, -y])
    rows = paired_field_bootstrap(
        y, a, a, np.arange(6), [0, 0, 1, 1, 2, 2], threshold=5, fractions=(0.5,), draws=30
    )
    assert all(r["difference"] == r["ci_low"] == r["ci_high"] == 0 for r in rows)
    assert all(r["selected_overlap"] == 1 for r in rows)


def test_paired_resampling_reproducible_and_preserves_sign():
    y = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    args = (y, y[None], -y[None], np.arange(6), [0, 0, 1, 1, 2, 2], 3)
    a = paired_field_bootstrap(*args, fractions=(0.5,), draws=40)
    b = paired_field_bootstrap(*args, fractions=(0.5,), draws=40)
    assert a == b
    assert all(r["difference"] > 0 and r["ci_low"] >= 0 for r in a)


def test_identical_budget_contrasts_have_zero_interaction():
    y = np.arange(6, dtype=float)
    a, b = y[None], -y[None]
    result = paired_field_bootstrap(
        y,
        a,
        b,
        np.arange(6),
        [0, 0, 1, 1, 2, 2],
        threshold=3,
        fractions=(0.5,),
        draws=30,
        subtract_pair=(a, b),
    )
    assert all(r["difference"] == r["ci_low"] == r["ci_high"] == 0 for r in result)


@pytest.mark.parametrize(
    "scores,ids,q", [([], [], 0.1), ([np.nan], [1], 1), ([1], [1], 0), ([1], [1, 2], 1)]
)
def test_invalid_rank_inputs(scores, ids, q):
    with pytest.raises(ValueError):
        top_indices(scores, ids, q)
