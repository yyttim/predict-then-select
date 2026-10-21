"""Independent exhaustive decisions and numerical derivatives check the losses."""

import itertools

import numpy as np
import pytest
import torch

from cellselect.losses import topk_pg_central, topk_spo_plus


def _decisions(n, k):
    result = []
    for subset in itertools.combinations(range(n), k):
        decision = np.zeros(n)
        decision[list(subset)] = 1
        result.append(decision)
    return result


def _source_spo(scores, labels, k):
    # Use the paper's cost-minimization convention as an independent reference.
    cost, prediction = -np.asarray(labels), -np.asarray(scores)
    decisions = _decisions(len(cost), k)
    true_decision = min(decisions, key=lambda w: cost @ w)
    return (
        max((cost - 2 * prediction) @ w for w in decisions)
        + (2 * prediction - cost) @ true_decision
    ) / k


def _source_pg(scores, labels, k, h):
    cost, prediction = -np.asarray(labels), -np.asarray(scores)
    decisions = _decisions(len(cost), k)
    plus = min((prediction + h * cost) @ w for w in decisions)
    minus = min((prediction - h * cost) @ w for w in decisions)
    return (plus - minus) / (2 * h * k)


def _gradient(loss, scores, labels, **kwargs):
    s = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(labels, dtype=torch.float64)
    value = loss(s, y, **kwargs)
    (gradient,) = torch.autograd.grad(value, s)
    return float(value.detach()), gradient.detach().numpy()


def test_exhaustive_spo_value_and_realized_regret_upper_bound():
    for label_tuple in itertools.product([-1.0, 0.0, 2.0], repeat=3):
        y = np.array(label_tuple)
        for score_tuple in itertools.product([-1.0, 0.0, 1.0], repeat=3):
            s = np.array(score_tuple)
            for k in (1, 2, 3):
                decisions = _decisions(3, k)
                chosen = max(decisions, key=lambda w: s @ w)
                regret = (max(y @ w for w in decisions) - y @ chosen) / k
                result = topk_spo_plus(torch.tensor(s), torch.tensor(y), k / 3).item()
                assert result == pytest.approx(_source_spo(s, y, k))
                assert result >= regret - 1e-12
    assert topk_spo_plus(torch.tensor([1.0, 3.0, -1.0]), torch.tensor([1.0, 3.0, -1.0]), 0.5) == 0


@pytest.mark.parametrize("h", [0.1, 1.0])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_pg_value_and_gradients_match_enumeration_and_numerical_derivatives(h, k):
    s, y = np.array([-0.7, 0.35, 0.9, -0.1]), np.array([1.3, -0.4, 0.25, 0.8])
    value, gradient = _gradient(topk_pg_central, s, y, fraction=k / 4, h=h)
    assert value == pytest.approx(_source_pg(s, y, k, h))
    epsilon = 1e-6
    numerical = np.array(
        [
            (_source_pg(s + epsilon * d, y, k, h) - _source_pg(s - epsilon * d, y, k, h))
            / (2 * epsilon)
            for d in np.eye(4)
        ]
    )
    np.testing.assert_allclose(gradient, numerical, atol=1e-8)


def test_spo_subgradient_matches_independent_finite_differences():
    s, y = np.array([-0.7, 0.35, 0.9, -0.1]), np.array([1.3, -0.4, 0.25, 0.8])
    for k in (1, 2, 4):
        _, gradient = _gradient(topk_spo_plus, s, y, fraction=k / 4)
        eps = 1e-6
        numerical = np.array(
            [
                (_source_spo(s + eps * d, y, k) - _source_spo(s - eps * d, y, k)) / (2 * eps)
                for d in np.eye(4)
            ]
        )
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)


def test_spo_gradient_at_tie_is_a_valid_convex_subgradient():
    s, y = np.zeros(4), np.array([2.0, 1.0, 1.0, 0.0])
    value, gradient = _gradient(topk_spo_plus, s, y, fraction=0.5)
    for trial in itertools.product([-1.0, 0.0, 1.0], repeat=4):
        z = np.asarray(trial)
        assert _source_spo(z, y, 2) >= value + gradient @ (z - s) - 1e-12
    _, repeated = _gradient(topk_spo_plus, s, y, fraction=0.5)
    np.testing.assert_array_equal(gradient, repeated)


def test_pg_small_perturbations_equal_actual_negative_utility_off_boundaries():
    s, y = np.array([3.0, 0.0, -2.0, 1.0]), np.array([2.0, -1.0, 0.5, 4.0])
    chosen = max(_decisions(4, 2), key=lambda w: s @ w)
    for h in (0.1, 0.01, 1e-5, 1e-9):
        value, gradient = _gradient(topk_pg_central, s, y, fraction=0.5, h=h)
        assert value == pytest.approx(-y @ chosen / 2)
        np.testing.assert_array_equal(gradient, np.zeros(4))


def test_translation_and_all_selected_cases():
    s, y, shift = np.array([0.1, -0.3, 0.9, 1.7]), np.array([2.0, 0.5, 1.2, -0.7]), 7.0
    for loss, kwargs, value_shift in [
        (topk_spo_plus, {}, 0.0),
        (topk_pg_central, {"h": 0.3}, -shift),
    ]:
        value, gradient = _gradient(loss, s, y, fraction=0.5, **kwargs)
        moved, moved_gradient = _gradient(loss, s, y + shift, fraction=0.5, **kwargs)
        assert moved == pytest.approx(value + value_shift)
        np.testing.assert_allclose(moved_gradient, gradient)
        score_shifted, shifted_gradient = _gradient(loss, s + shift, y, fraction=0.5, **kwargs)
        assert score_shifted == pytest.approx(value)
        np.testing.assert_allclose(shifted_gradient, gradient)
        whole, whole_gradient = _gradient(loss, s, y, fraction=1.0, **kwargs)
        assert whole == pytest.approx(0.0 if loss is topk_spo_plus else -y.mean())
        np.testing.assert_array_equal(whole_gradient, np.zeros(4))


def test_asymmetric_two_candidate_spo_minimizer_disagrees_with_mean_utility():
    # A yields 100 with probability 0.1, otherwise 0; B always yields 1.
    # Means (10, 1) favor A, but SPO+ risk is minimized at score gap -0.5.
    # This discrete distribution lies outside the consistency assumptions.
    outcomes = [np.array([100.0, 1.0]), np.array([0.0, 1.0])]
    probabilities = [0.1, 0.9]
    gaps = np.array([-2.0, -1.0, -0.5, 0.0, 1.0, 9.0, 49.5, 51.0])
    risks = []
    for gap in gaps:
        score = np.array([gap, 0.0])
        risk = sum(
            p * topk_spo_plus(torch.tensor(score), torch.tensor(y), 0.5).item()
            for p, y in zip(probabilities, outcomes)
        )
        # Compare with the independently derived piecewise expected risk.
        exact = (
            9.9 - 0.2 * gap
            if gap <= -0.5
            else (10.8 + 1.6 * gap if gap <= 49.5 else 0.9 + 1.8 * gap)
        )
        assert risk == pytest.approx(exact)
        risks.append(risk)
    assert gaps[np.argmin(risks)] == -0.5 and min(risks) == pytest.approx(10.0)
    means = sum(p * y for p, y in zip(probabilities, outcomes))
    np.testing.assert_allclose(means, [10.0, 1.0])
    assert means[0] - means[1] == pytest.approx(9.0)


@pytest.mark.parametrize("loss", [topk_spo_plus, topk_pg_central])
def test_invalid_inputs_fail_and_labels_receive_no_gradient(loss):
    good = torch.tensor([1.0, 2.0], requires_grad=True)
    labels = torch.tensor([2.0, 1.0], requires_grad=True)
    loss(good, labels).backward()
    assert labels.grad is None
    for fraction in (0.0, -1.0, float("nan"), 1.1, True):
        with pytest.raises(ValueError):
            loss(good, labels, fraction)
    with pytest.raises(ValueError):
        loss(torch.tensor([float("inf"), 1.0]), labels)
    with pytest.raises(ValueError):
        loss(good[:, None], labels)


@pytest.mark.parametrize("h", [0.0, -1.0, float("nan"), float("inf"), True])
def test_pg_requires_positive_finite_step(h):
    with pytest.raises(ValueError):
        topk_pg_central(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]), h=h)
