import numpy as np
import pytest

from cellselect.bootstrap import _weighted_topk_means, paired_topk_bootstrap
from cellselect.metrics import paired_seed_field_bootstrap, selection_size


@pytest.mark.parametrize("fraction", [0.01, 0.25, 0.6, 1.0])
@pytest.mark.parametrize("draw", [[0, 0, 0], [2, 1, 1], [0, 2, 1]])
def test_weighted_topk_equals_explicit_replicated_pool(fraction, draw):
    y = np.array([3.0, -1.0, 9.0, 2.0, 7.0, 5.0])
    groups = np.array(["c", "a", "b", "b", "c", "b"])
    # Non-monotonic IDs exercise tie breaking independently of row order.
    ids = np.array([20, 80, 10, 40, 30, 60])
    scores = np.array([[1.0, 1.0, 2.0, 1.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    unique, inverse = np.unique(groups, return_inverse=True)
    orders = np.stack([np.lexsort((ids, -s)) for s in scores])
    members = [np.flatnonzero(groups == group) for group in unique]
    expanded = np.concatenate([members[group] for group in draw])
    k = selection_size(len(expanded), fraction)
    counts = np.bincount(draw, minlength=len(unique))
    observed = _weighted_topk_means(y[orders], inverse[orders], counts, k)
    expected = []
    for score in scores:
        ranked = np.lexsort((ids[expanded], -score[expanded]))[:k]
        expected.append(y[expanded[ranked]].mean())
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)


def test_paired_seed_bootstrap_matches_previous_algorithm_and_rng():
    rng = np.random.default_rng(44)
    n = 29
    y = rng.normal(size=n)
    groups = np.repeat(np.arange(7), [1, 2, 3, 4, 5, 6, 8])
    ids = rng.permutation(n) + 100
    a, b = np.round(rng.normal(size=(3, n)), 1), np.round(rng.normal(size=(3, n)), 1)
    old = paired_seed_field_bootstrap(y, a, b, groups, 0.3, n_bootstrap=257, seed=901, ids=ids)
    new = paired_topk_bootstrap(y, a, b, groups, 0.3, n_bootstrap=257, seed=901, ids=ids)
    assert old.keys() == new.keys()
    for key in old:
        if key != "method":
            np.testing.assert_allclose(new[key], old[key], rtol=1e-12, atol=1e-12)


def test_single_pair_all_selected_is_exactly_zero():
    result = paired_topk_bootstrap(
        [1.0, 3.0, 2.0], [2.0, 0.0, 1.0], [0.0, 1.0, 2.0], ["a", "b", "b"], 1.0, n_bootstrap=10
    )
    assert result["difference_a_minus_b"] == result["ci_low"] == result["ci_high"] == 0
    assert result["paired_seed_sd"] is None and result["n_seeds"] == 1


def test_duplicate_original_ids_are_rejected_to_keep_resampled_ties_unambiguous():
    with pytest.raises(ValueError, match="unique"):
        paired_topk_bootstrap(
            [1.0, 2.0], [0.0, 0.0], [1.0, 1.0], [0, 1], 0.5, n_bootstrap=4, ids=[4, 4]
        )
