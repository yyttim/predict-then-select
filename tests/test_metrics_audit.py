"""Independent edge-case checks for prediction and selection metrics."""

import itertools

import numpy as np
import pytest

from cellselect.metrics import (
    paired_field_bootstrap,
    prediction_metrics,
    quota_counts,
    quota_selection,
    selection_metrics,
    selection_size,
    topk_indices,
)


def test_unsigned_scores_are_sorted_by_numeric_value_not_wrapped_negation():
    scores = np.array([0, 255, 20], dtype=np.uint8)
    assert topk_indices(scores, 2).tolist() == [1, 2]


def test_integer_prediction_arithmetic_is_not_subject_to_overflow():
    y = np.array([0, 255], dtype=np.uint8)
    score = np.array([255, 0], dtype=np.uint8)
    measured = prediction_metrics(y, score)
    assert measured["rmse"] == pytest.approx(255.0)
    assert measured["mae"] == pytest.approx(255.0)


@pytest.mark.parametrize("n", [2.5, True])
def test_selection_size_requires_integer_pool_size(n):
    with pytest.raises((TypeError, ValueError)):
        selection_size(n, 0.5)


@pytest.mark.parametrize("capacities,k", [([1.9, 3], 2), ([1, 3], 1.5), ([1, 3], True)])
def test_quota_apportionment_does_not_silently_truncate(capacities, k):
    with pytest.raises((TypeError, ValueError)):
        quota_counts([0.6, 0.4], capacities, k)


@pytest.mark.parametrize("counts", [[1.9, 1], [-0.5, 1]])
def test_quota_selection_does_not_truncate_counts(counts):
    with pytest.raises((TypeError, ValueError)):
        quota_selection([1, 2, 3, 4], [2, 1, 4, 3], [0, 0, 1, 1], counts)


def test_zero_quota_for_absent_class_is_feasible():
    result = quota_selection([1, 4, 2, 3], [0.1, 0.4, 0.2, 0.3], [0, 0, 2, 2], [1, 0, 1])
    assert result["k"] == 2
    assert result["utility"] == pytest.approx(3.5)


def test_noninteger_cell_types_cannot_collapse_to_same_dictionary_key():
    with pytest.raises((TypeError, ValueError)):
        selection_metrics([1, 2, 3, 4], [4, 3, 2, 1], [0.1, 0.2, 1.0, 1.0], 0.5)


@pytest.mark.parametrize("classes", [[0, 1, 4], [0, -1, 2], [0, 1]])
def test_classification_metrics_require_valid_aligned_class_ids(classes):
    with pytest.raises((TypeError, ValueError)):
        prediction_metrics([1, 2, 3], [1, 2, 3], classes, np.eye(3))


def test_quota_apportionment_minimizes_squared_allocation_deviation():
    proportions = np.array([0.75, 0.15, 0.10])
    capacities = np.array([1, 4, 3])
    for k in range(1, capacities.sum() + 1):
        actual = quota_counts(proportions, capacities, k)
        desired = k * proportions / proportions.sum()
        candidates = [
            np.asarray(c)
            for c in itertools.product(*(range(n + 1) for n in capacities))
            if sum(c) == k
        ]
        best = min(np.square(candidate - desired).sum() for candidate in candidates)
        assert actual.sum() == k
        assert np.all(actual <= capacities)
        assert np.square(actual - desired).sum() == pytest.approx(best)


def test_oracle_decomposition_with_negative_values_ties_and_absent_class():
    y = np.array([-9.0, 4.0, 4.0, -3.0, 7.0])
    scores = np.array([3.0, 2.0, 2.0, 4.0, 0.0])
    classes = np.array([0, 0, 2, 2, 2])
    ids = np.array([50, 40, 30, 20, 10])
    for k in range(1, len(y) + 1):
        result = selection_metrics(y, scores, classes, k / len(y), ids)
        chosen = topk_indices(scores, k, ids)
        composition = np.bincount(classes[chosen], minlength=3)
        candidates = [np.asarray(s) for s in itertools.combinations(range(len(y)), k)]
        oracle = max(y[s].mean() for s in candidates)
        same_counts = max(
            y[s].mean()
            for s in candidates
            if np.array_equal(np.bincount(classes[s], minlength=3), composition)
        )
        assert result["oracle_utility"] == pytest.approx(oracle)
        assert result["same_counts_oracle_utility"] == pytest.approx(same_counts)
        assert result["within_type_gap"] >= 0
        assert result["type_allocation_gap"] >= 0
        assert result["oracle_gap"] == pytest.approx(
            result["within_type_gap"] + result["type_allocation_gap"]
        )


def test_positive_affine_calibration_changes_rmse_but_no_selection_statistic():
    y = np.array([8.0, -1.0, 2.0, 11.0, 7.0, 3.0])
    scores = np.array([1.0, 3.0, 3.0, -1.0, 0.0, 5.0])
    classes = np.array([0, 0, 1, 1, 2, 2])
    ids = np.array([50, 60, 10, 30, 40, 20])
    calibrated = 2.0 * scores + 9.0
    assert prediction_metrics(y, scores)["rmse"] != prediction_metrics(y, calibrated)["rmse"]
    for fraction in (0.1, 0.5, 1.0):
        assert selection_metrics(y, scores, classes, fraction, ids) == selection_metrics(
            y, calibrated, classes, fraction, ids
        )


def test_bootstrap_uses_same_id_tie_rule_as_primary_utility_and_row_permutation():
    y = np.array([100.0, 0.0, 50.0, 20.0])
    a = np.ones(4)
    b = np.array([0.0, 1.0, 0.0, 0.0])
    ids = np.array([30, 20, 10, 40])
    groups = np.array(["a", "b", "c", "d"])
    first = paired_field_bootstrap(y, a, b, groups, 0.25, n_bootstrap=80, ids=ids)
    assert first["difference_a_minus_b"] == pytest.approx(50.0)
    p = np.array([3, 1, 0, 2])
    second = paired_field_bootstrap(y[p], a[p], b[p], groups[p], 0.25, n_bootstrap=80, ids=ids[p])
    assert first == second


def test_paired_bootstrap_is_paired_and_reranks_resampled_pools():
    # Unequal field sizes exercise changing pool sizes and ceil(q*N).
    y = np.array([0.0, 10.0, 3.0, 8.0, 1.0, 7.0])
    a = np.array([2.0, 5.0, 3.0, 0.0, 1.0, 4.0])
    b = -a
    groups = np.array([0, 1, 1, 2, 2, 2])
    repetitions, seed, fraction = 80, 13, 0.4
    actual = paired_field_bootstrap(y, a, b, groups, fraction, n_bootstrap=repetitions, seed=seed)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    random = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        indices = np.concatenate([members[g] for g in random.integers(3, size=3)])
        k = int(np.ceil(fraction * len(indices)))
        # Duplicate copies of a cell have identical measured outcomes.
        ranked_a = np.argsort(-a[indices], kind="stable")[:k]
        ranked_b = np.argsort(-b[indices], kind="stable")[:k]
        differences.append(y[indices[ranked_a]].mean() - y[indices[ranked_b]].mean())
    lower, upper = np.quantile(differences, [0.025, 0.975])
    assert actual["ci_low"] == pytest.approx(lower)
    assert actual["ci_high"] == pytest.approx(upper)


@pytest.mark.parametrize("groups", [[0.0, 1.0, np.nan], ["a", "b", None]])
def test_bootstrap_rejects_missing_field_identifiers(groups):
    with pytest.raises((TypeError, ValueError)):
        paired_field_bootstrap([1, 2, 3], [1, 2, 3], [3, 2, 1], groups, 0.5, n_bootstrap=5)
