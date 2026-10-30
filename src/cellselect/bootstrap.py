"""Exact multiplicity-based paired bootstrap for fixed-budget selection.

Sorting is performed once per trained score vector. Bootstrap draws replicate
whole fields, so their multiplicities can be accumulated along each fixed score
order without explicitly constructing or repeatedly sorting replicated cells.
"""

from __future__ import annotations

import numpy as np

from .metrics import selection_size


def _weighted_topk_means(ordered_y, ordered_groups, field_counts, k):
    """Top-k means of a replicated pool, including partial boundary-cell copies."""
    multiplicities = field_counts[ordered_groups]
    cumulative = np.cumsum(multiplicities, axis=1, dtype=np.int64)
    means = np.empty(len(ordered_y), dtype=np.float64)
    for row in range(len(ordered_y)):
        boundary = int(np.searchsorted(cumulative[row], k, side="left"))
        before = int(cumulative[row, boundary - 1]) if boundary else 0
        total = np.dot(multiplicities[row, :boundary], ordered_y[row, :boundary])
        total += (k - before) * ordered_y[row, boundary]
        means[row] = total / k
    return means


def paired_topk_bootstrap(
    y, scores_a, scores_b, groups, fraction, n_bootstrap=2000, seed=20261021, ids=None
):
    """Mean paired-seed utility difference with common whole-FOV resampling.

    ``scores_a`` and ``scores_b`` may be seed-by-cell matrices or one-dimensional
    score vectors. The interval is conditional on these trained models; it does
    not account for retraining, validation model selection, or new donor cohorts.
    Original cell IDs must be unique so the primary tie rule also remains exact
    after field resampling. Copies of a cell have identical outcomes and scores.
    """
    y = np.asarray(y, dtype=np.float64)
    a, b = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    groups = np.asarray(groups)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Outcomes must be a finite, nonempty vector")
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(y) or not a.shape[0]:
        raise ValueError("Expected aligned seed-by-cell score matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or groups.shape != y.shape:
        raise ValueError("Nonfinite scores or invalid field IDs")
    if any(
        g is None or (isinstance(g, (float, np.floating)) and not np.isfinite(g)) for g in groups
    ):
        raise ValueError("Missing field IDs")
    if (
        isinstance(n_bootstrap, (bool, np.bool_))
        or not isinstance(n_bootstrap, (int, np.integer))
        or n_bootstrap < 2
    ):
        raise ValueError("At least two bootstrap draws required")
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    if ids.shape != y.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Original cell IDs must be aligned and unique")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two fields required")
    n_seeds = len(a)
    scores = np.concatenate([a, b], axis=0)
    orders = np.stack([np.lexsort((ids, -score)) for score in scores])
    ordered_y, ordered_groups = y[orders], inverse[orders]
    sizes = np.bincount(inverse, minlength=len(unique))
    k = selection_size(len(y), fraction)
    means = _weighted_topk_means(ordered_y, ordered_groups, np.ones(len(unique), dtype=np.int64), k)
    observed = means[:n_seeds] - means[n_seeds:]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        # Same RNG draws as the explicit expanded-pool implementation.
        sampled_fields = rng.integers(len(unique), size=len(unique))
        counts = np.bincount(sampled_fields, minlength=len(unique))
        k_draw = selection_size(int(counts @ sizes), fraction)
        utilities = _weighted_topk_means(ordered_y, ordered_groups, counts, k_draw)
        draws[draw] = (utilities[:n_seeds] - utilities[n_seeds:]).mean()
    if fraction == 1:
        # Every model selects the complete pool; remove summation-order roundoff.
        observed.fill(0)
        draws.fill(0)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "difference_a_minus_b": float(observed.mean()),
        "paired_seed_sd": float(observed.std(ddof=1)) if n_seeds > 1 else None,
        "paired_seed_differences": observed.tolist(),
        "n_seeds": n_seeds,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_bootstrap": int(n_bootstrap),
        "n_fields": len(unique),
        "seed": seed,
        "method": "common paired FOV draws across fixed trained seeds; exact weighted top-k in precomputed score order",
    }
