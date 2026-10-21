import itertools
import numpy as np
import pytest
from cellselect.metrics import (
    selection_metrics,
    topk_indices,
    quota_counts,
    quota_selection,
    paired_field_bootstrap,
)


def test_oracle_decomposition_matches_exhaustive_subsets():
    y = np.array([9.0, 1.0, 8.0, 7.0, 2.0, 0.0])
    score = np.array([2.0, 8.0, 1.0, 3.0, 7.0, 4.0])
    c = np.array([0, 0, 1, 1, 2, 2])
    for k in range(1, len(y) + 1):
        m = selection_metrics(y, score, c, k / len(y))
        selected = topk_indices(score, k)
        counts = np.bincount(c[selected], minlength=3)
        sets = list(itertools.combinations(range(len(y)), k))
        oracle = max(y[list(s)].mean() for s in sets)
        constrained = max(
            y[list(s)].mean()
            for s in sets
            if np.array_equal(np.bincount(c[list(s)], minlength=3), counts)
        )
        assert m["oracle_utility"] == pytest.approx(oracle)
        assert m["same_counts_oracle_utility"] == pytest.approx(constrained)
        assert m["oracle_gap"] == pytest.approx(m["within_type_gap"] + m["type_allocation_gap"])


def test_ties_and_monotone_transform_do_not_change_selection():
    score = np.array([1.0, 3.0, 3.0, 2.0])
    ids = np.array([20, 40, 10, 30])
    selected = ids[topk_indices(score, 2, ids)]
    assert np.array_equal(selected, [10, 40])
    p = np.array([2, 0, 3, 1])
    assert np.array_equal(selected, ids[p][topk_indices(score[p], 2, ids[p])])
    assert np.array_equal(topk_indices(score, 3), topk_indices(8 * score + 91, 3))


def test_quota_capacity_is_common_and_exact():
    counts = quota_counts([0.8, 0.1, 0.1], [1, 5, 4], 7)
    assert counts.sum() == 7 and counts[0] == 1
    y = np.arange(10.0)
    classes = np.array([0] + [1] * 5 + [2] * 4)
    assert quota_selection(y, y, classes, counts)["k"] == 7


def test_paired_bootstrap_identity_and_reproducibility():
    y = np.arange(12.0)
    groups = np.repeat(np.arange(4), 3)
    m = paired_field_bootstrap(y, y, y, groups, 0.25, 20)
    assert m["ci_low"] == m["ci_high"] == m["difference_a_minus_b"] == 0
    a = paired_field_bootstrap(y, y, -y, groups, 0.25, 20)
    b = paired_field_bootstrap(y, y, -y, groups, 0.25, 20)
    assert a == b and a["ci_low"] > 0


def test_bad_inputs_fail_instead_of_silent_filtering():
    with pytest.raises(ValueError):
        selection_metrics([1, np.nan], [1, 2], [0, 0], 0.5)
    with pytest.raises(ValueError):
        quota_counts([1, 1], [1, 1], 3)
