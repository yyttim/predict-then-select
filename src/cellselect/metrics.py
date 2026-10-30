"""Prediction and fixed-budget selection metrics.

Measured outcomes and class labels enter evaluation only. The primary policy
depends solely on image-based scores, a fixed budget, and label-free tie IDs.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score


def _vector(value, name, allow_empty=False):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or (not len(value) and not allow_empty):
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite entries")
    return value


def _integer_vector(value, name):
    value = _vector(value, name)
    if (value < 0).any() or not np.equal(value, np.floor(value)).all():
        raise ValueError(f"{name} must contain nonnegative integers")
    return value.astype(np.int64)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def selection_size(n: int, fraction: float) -> int:
    if not _integer(n) or n < 1 or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Need n >= 1 and 0 < fraction <= 1")
    return min(n, math.ceil(fraction * n))


def topk_indices(scores, k: int, ids=None) -> np.ndarray:
    scores = _vector(scores, "scores", allow_empty=True)
    if not _integer(k) or not 0 <= k <= len(scores):
        raise ValueError("k must be an integer between zero and pool size")
    ids = np.arange(len(scores)) if ids is None else np.asarray(ids)
    if ids.shape != scores.shape:
        raise ValueError("IDs and scores must have matching shapes")
    # Stable, measured-outcome-independent tie breaking.
    return np.lexsort((ids, -scores))[:k]


def selection_metrics(y, scores, classes, fraction: float, ids=None) -> dict:
    y, scores, classes = (
        _vector(v, n) for v, n in [(y, "y"), (scores, "scores"), (classes, "classes")]
    )
    classes = _integer_vector(classes, "classes")
    if not (y.shape == scores.shape == classes.shape):
        raise ValueError("Outcome, score, and class shapes must match")
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    k = selection_size(len(y), fraction)
    selected = topk_indices(scores, k, ids)
    oracle = topk_indices(y, k, ids)
    matched = []
    counts = {}
    for c in np.unique(classes):
        count = int(np.sum(classes[selected] == c))
        counts[str(int(c))] = count
        group = np.flatnonzero(classes == c)
        matched.extend(group[topk_indices(y[group], count, ids[group])].tolist())
    utility = float(y[selected].mean())
    same = float(y[matched].mean())
    global_best = float(y[oracle].mean())
    within, allocation = same - utility, global_best - same
    tolerance = 1e-10 * max(1.0, float(np.abs(y).max()))
    if within < -tolerance or allocation < -tolerance:
        raise AssertionError("Invalid oracle gap")
    return {
        "fraction": float(fraction),
        "k": k,
        "utility": utility,
        "random_expected_utility": float(y.mean()),
        "enrichment_difference": float(utility - y.mean()),
        "oracle_utility": global_best,
        "same_counts_oracle_utility": same,
        "oracle_gap": global_best - utility,
        "within_type_gap": max(0.0, within),
        "type_allocation_gap": max(0.0, allocation),
        "selected_type_counts": counts,
        "oracle_overlap": float(len(np.intersect1d(selected, oracle)) / k),
    }


def prediction_metrics(y, scores, classes=None, logits=None) -> dict:
    y, scores = _vector(y, "y"), _vector(scores, "scores")
    if y.shape != scores.shape:
        raise ValueError("Outcome and score shapes must match")
    result = {
        "rmse": float(np.sqrt(np.mean((y - scores) ** 2))),
        "mae": float(np.mean(np.abs(y - scores))),
    }
    variable = len(y) > 1 and np.std(y) > 0 and np.std(scores) > 0
    result["pearson"] = float(pearsonr(y, scores).statistic) if variable else None
    result["spearman"] = float(spearmanr(y, scores).statistic) if variable else None
    if logits is not None:
        logits = np.asarray(logits)
        classes = _integer_vector(classes, "classes")
        if logits.ndim != 2 or logits.shape[0] != len(y) or not np.isfinite(logits).all():
            raise ValueError("Invalid logits")
        if len(classes) != len(y) or np.any(classes >= logits.shape[1]):
            raise ValueError("Class IDs must match rows and logits columns")
        pred = logits.argmax(axis=1)
        result.update(
            macro_f1=float(
                f1_score(
                    classes,
                    pred,
                    labels=np.arange(logits.shape[1]),
                    average="macro",
                    zero_division=0,
                )
            ),
            accuracy=float(accuracy_score(classes, pred)),
        )
    return result


def quota_counts(proportions, capacities, k: int) -> np.ndarray:
    """Apportion a fixed validation-derived rule with test-capacity constraints.

    Capacities are used only in the label-dependent quota diagnostic. The
    same integer allocation is used for every compared model. No marker
    outcomes are consulted. Largest-deficit allocation handles exhaustion.
    """
    proportions = np.asarray(proportions, dtype=float)
    capacities = _integer_vector(capacities, "capacities")
    if (
        proportions.ndim != 1
        or capacities.shape != proportions.shape
        or not np.isfinite(proportions).all()
        or np.any(proportions < 0)
        or np.any(capacities < 0)
        or proportions.sum() <= 0
        or not _integer(k)
        or not 0 <= k <= capacities.sum()
    ):
        raise ValueError("Invalid proportions, capacities, or k")
    desired = k * proportions / proportions.sum()
    counts = np.minimum(np.floor(desired).astype(int), capacities)
    while counts.sum() < k:
        deficit = np.where(counts < capacities, desired - counts, -np.inf)
        counts[np.argmax(deficit)] += 1
    return counts


def quota_selection(y, scores, classes, counts, ids=None) -> dict:
    y, scores, classes = (
        _vector(v, n) for v, n in [(y, "y"), (scores, "scores"), (classes, "classes")]
    )
    classes = _integer_vector(classes, "classes")
    counts = _integer_vector(counts, "counts")
    if not (y.shape == scores.shape == classes.shape):
        raise ValueError("Shapes must match")
    if np.any(classes >= len(counts)):
        raise ValueError("Quota vector does not cover all class IDs")
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    selected = []
    for c, count in enumerate(counts):
        group = np.flatnonzero(classes == c)
        selected.extend(group[topk_indices(scores[group], int(count), ids[group])])
    if not selected:
        raise ValueError("Quota must select at least one cell")
    return {
        "utility": float(y[selected].mean()),
        "k": len(selected),
        "selected_type_counts": [int(x) for x in counts],
    }


def paired_field_bootstrap(
    y, scores_a, scores_b, groups, fraction, n_bootstrap=2000, seed=20261021, ids=None
) -> dict:
    """Paired field bootstrap that reranks each resampled candidate pool.

    Conditional on two trained models and the observed acquisition cohort;
    this is not a donor-generalization or training-seed confidence interval.
    """
    y, a, b = (_vector(v, n) for v, n in [(y, "y"), (scores_a, "scores_a"), (scores_b, "scores_b")])
    groups = np.asarray(groups)
    if not y.shape == a.shape == b.shape == groups.shape:
        raise ValueError("All arrays must align")
    if any(
        g is None or (isinstance(g, (float, np.floating)) and not np.isfinite(g)) for g in groups
    ):
        raise ValueError("Field IDs must not be missing")
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    if ids.shape != y.shape:
        raise ValueError("IDs must align with outcomes")
    unique = np.unique(groups)
    if len(unique) < 2 or not _integer(n_bootstrap) or n_bootstrap < 2:
        raise ValueError("Need at least two fields and bootstrap replicates")
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    differences = np.empty(n_bootstrap)
    for j in range(n_bootstrap):
        index = np.concatenate([members[t] for t in rng.integers(len(unique), size=len(unique))])
        k = selection_size(len(index), fraction)
        # Carry original IDs through resampling so ties follow the primary rule.
        # Copies of the same cell have identical score and measured outcome.
        sa, sb = topk_indices(a[index], k, ids[index]), topk_indices(b[index], k, ids[index])
        differences[j] = y[index[sa]].mean() - y[index[sb]].mean()
    k = selection_size(len(y), fraction)
    estimate = float(y[topk_indices(a, k, ids)].mean() - y[topk_indices(b, k, ids)].mean())
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "difference_a_minus_b": estimate,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_bootstrap": n_bootstrap,
        "n_fields": len(unique),
        "seed": seed,
        "method": "paired FOV percentile bootstrap; rerank resampled pools",
    }


def paired_seed_field_bootstrap(
    y, scores_a, scores_b, groups, fraction, n_bootstrap=2000, seed=20261021, ids=None
):
    """Conditional FOV uncertainty of a mean paired-seed difference.

    Each bootstrap uses one common field draw across all fixed trained seeds.
    Training-seed SD is separate; this is not a new-donor confidence interval.
    """
    y = _vector(y, "y")
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    groups = np.asarray(groups)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(y) or not a.shape[0]:
        raise ValueError("Expected aligned seed-by-cell score matrices")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or groups.shape != y.shape:
        raise ValueError("Nonfinite predictions or invalid groups")
    if any(
        g is None or (isinstance(g, (float, np.floating)) and not np.isfinite(g)) for g in groups
    ):
        raise ValueError("Missing field IDs")
    if not _integer(n_bootstrap) or n_bootstrap < 2:
        raise ValueError("At least two bootstrap replicates required")
    ids = np.arange(len(y)) if ids is None else np.asarray(ids)
    if ids.shape != y.shape:
        raise ValueError("IDs must align")
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    if len(members) < 2:
        raise ValueError("At least two fields required")

    def difference(index):
        k = selection_size(len(index), fraction)
        return np.array(
            [
                y[index[topk_indices(sa[index], k, ids[index])]].mean()
                - y[index[topk_indices(sb[index], k, ids[index])]].mean()
                for sa, sb in zip(a, b)
            ]
        )

    observed = difference(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for j in range(n_bootstrap):
        index = np.concatenate([members[g] for g in rng.integers(len(members), size=len(members))])
        draws[j] = difference(index).mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "difference_a_minus_b": float(observed.mean()),
        "paired_seed_sd": float(observed.std(ddof=1)) if len(observed) > 1 else None,
        "paired_seed_differences": observed.tolist(),
        "n_seeds": len(observed),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_bootstrap": n_bootstrap,
        "n_fields": len(members),
        "seed": seed,
        "method": "common paired FOV draws across fixed trained seeds; rerank every pool",
    }
