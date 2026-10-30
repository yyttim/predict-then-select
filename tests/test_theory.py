"""Check top-k identities by exhaustively enumerating small candidate pools."""

import itertools

import numpy as np
import pytest

from cellselect.metrics import selection_metrics, topk_indices


def test_exchange_error_bound_for_exhaustive_three_cell_pools():
    ids = np.array([30, 10, 20])
    classes = np.array([0, 1, 1])
    for values in itertools.product([-2.0, 0.0, 3.0], repeat=3):
        y = np.asarray(values)
        for scores in itertools.product([-1.0, 0.0, 1.0], repeat=3):
            score = np.asarray(scores)
            error = score - y
            for k in (1, 2, 3):
                # Enumerating subsets is independent of the evaluator's oracle.
                choices = list(itertools.combinations(range(3), k))
                oracle = max(choices, key=lambda group: y[list(group)].sum())
                selected = topk_indices(score, k, ids)
                changed = np.array(sorted(set(oracle) ^ set(selected)), dtype=int)
                r = len(changed) // 2
                regret = y[list(oracle)].mean() - y[selected].mean()
                reported = selection_metrics(y, score, classes, k / 3, ids)
                assert reported["oracle_gap"] == pytest.approx(regret)
                assert regret >= -1e-12
                exchange_bound = np.abs(error[changed]).sum() / k
                l2_bound = np.sqrt(2 * r) * np.linalg.norm(error[changed]) / k
                rmse_bound = np.sqrt(2 * len(y) / k) * np.sqrt(np.mean(error**2))
                infinity_bound = 2 * r * np.max(np.abs(error)) / k
                assert regret <= exchange_bound + 1e-12
                assert exchange_bound <= l2_bound + 1e-12
                assert l2_bound <= rmse_bound + 1e-12
                assert regret <= infinity_bound + 1e-12


def test_strict_half_boundary_margin_preserves_topk():
    y = np.array([-1.0, 2.0, 6.0, 11.0])
    ids = np.array([40, 10, 30, 20])
    descending = np.sort(y)[::-1]
    for k in range(1, len(y)):
        gap = descending[k - 1] - descending[k]
        epsilon = 0.499 * gap
        expected = set(topk_indices(y, k, ids))
        # Corners include the most adversarial opposite perturbations.
        for signs in itertools.product([-1.0, 1.0], repeat=len(y)):
            score = y + epsilon * np.asarray(signs)
            assert set(topk_indices(score, k, ids)) == expected


def test_bayes_optimal_selection_can_have_positive_hindsight_gap():
    # Independent, symmetric outcome noise with known means (0.5, 0).
    # Selecting the first cell is optimal without observing realized outcomes.
    score = np.array([0.5, 0.0])
    utilities, oracle_utilities, hindsight_gaps = [], [], []
    for noise in itertools.product([-1.0, 1.0], repeat=2):
        y = score + np.asarray(noise)
        metrics = selection_metrics(y, score, [0, 1], 0.5)
        utilities.append(metrics["utility"])
        oracle_utilities.append(metrics["oracle_utility"])
        hindsight_gaps.append(metrics["oracle_gap"])
    assert np.mean(utilities) == pytest.approx(score.max())
    assert np.mean(hindsight_gaps) > 0
    assert np.mean(hindsight_gaps) == pytest.approx(np.mean(oracle_utilities) - score.max())
