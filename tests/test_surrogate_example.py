"""Exact expected risks checked by enumerating outcomes and feasible actions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "verify_surrogate_example",
    Path(__file__).parents[1] / "scripts" / "verify_surrogate_example.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize(
    "a,b,p", [(100.0, 1.0, 0.1), (100.0, 99.0, 0.9), (10.0, 2.0, 0.2), (10.0, 2.0, 0.5)]
)
def test_curves_match_independent_cost_minimization_enumeration(a, b, p):
    gaps = np.array([-a, -b / 2, 0.0, (a - b) / 2, a])
    result = example.exact_risks(gaps, a, b, p)
    actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    mean = np.array([p * a, b])
    for i, gap in enumerate(gaps):
        cost_prediction = -np.array([gap, 0.0])
        chosen = min(actions, key=lambda w: cost_prediction @ w)
        surrogate, hindsight = 0.0, 0.0
        for probability, cost in [(p, -np.array([a, b])), (1 - p, -np.array([0.0, b]))]:
            oracle = min(actions, key=lambda w: cost @ w)
            surrogate += probability * (
                max((cost - 2 * cost_prediction) @ w for w in actions)
                + (2 * cost_prediction - cost) @ oracle
            )
            hindsight += probability * (cost @ chosen - cost @ oracle)
        assert result["expected_spo_plus"][i] == pytest.approx(surrogate)
        assert result["expected_hindsight_regret"][i] == pytest.approx(hindsight)
        assert result["expected_utility"][i] == pytest.approx(mean @ chosen)
        assert result["expected_avoidable_regret"][i] == pytest.approx(max(mean) - mean @ chosen)
        assert surrogate >= hindsight - 1e-12


def test_default_counterexample_and_reverse_failure():
    summary = example.exact_summary()
    assert summary["mean_optimal_choices"] == ["A"]
    assert summary["spo_minimizer_choices"] == ["B"]
    assert summary["spo_minimizer_score_gap_interval"] == [-0.5, -0.5]
    assert summary["minimum_expected_spo_plus"] == pytest.approx(10.0)
    assert summary["avoidable_regret_at_spo_minimizers"] == {"B": 9.0}
    assert summary["irreducible_hindsight_gap"] == pytest.approx(0.9)
    reverse = example.exact_summary(100.0, 99.0, 0.9)
    assert reverse["mean_optimal_choices"] == ["B"]
    assert reverse["spo_minimizer_choices"] == ["A"]
    assert reverse["avoidable_regret_at_spo_minimizers"] == {"A": 9.0}


def test_half_probability_has_interval_of_minimizers():
    summary = example.exact_summary(10.0, 2.0, 0.5)
    assert summary["spo_minimizer_score_gap_interval"] == [-1.0, 4.0]
    assert summary["spo_minimizer_choices"] == ["A", "B"]
    risks = example.exact_risks(np.linspace(-1.0, 4.0, 21), 10.0, 2.0, 0.5)
    np.testing.assert_allclose(risks["expected_spo_plus"], 5.0)


def test_minimizers_and_hindsight_decomposition_across_parameter_grid():
    for a in [1.0, 7.0]:
        for b in [0.1 * a, 0.8 * a]:
            for p in [0.1, 0.49, 0.5, 0.51, 0.9]:
                summary = example.exact_summary(a, b, p)
                optimum = summary["spo_minimizer_score_gap_interval"][0]
                gaps = np.r_[np.linspace(-a, a, 101), optimum]
                risks = example.exact_risks(gaps, a, b, p)
                assert risks["expected_spo_plus"][-1] == pytest.approx(
                    summary["minimum_expected_spo_plus"]
                )
                assert (
                    np.min(risks["expected_spo_plus"])
                    >= summary["minimum_expected_spo_plus"] - 1e-12
                )
                np.testing.assert_allclose(
                    risks["expected_hindsight_regret"],
                    risks["expected_avoidable_regret"] + summary["irreducible_hindsight_gap"],
                )


@pytest.mark.parametrize(
    "a,b,p", [(1, 1, 0.1), (1, 0, 0.1), (1, 0.5, 0), (1, 0.5, 1), (1, 0.5, float("nan"))]
)
def test_invalid_parameters_fail(a, b, p):
    with pytest.raises(ValueError):
        example.exact_summary(a, b, p)
