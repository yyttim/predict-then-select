"""Prediction diagnostics and fixed-budget cell selection."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def top_indices(scores, ids, fraction):
    scores = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(ids)
    if scores.ndim != 1 or ids.shape != scores.shape or not len(scores):
        raise ValueError("Scores and cell IDs must be nonempty, matching vectors")
    if not np.isfinite(scores).all() or not 0 < fraction <= 1:
        raise ValueError("Finite scores and a fraction in (0, 1] are required")
    k = int(np.ceil(fraction * len(scores)))
    return np.lexsort((np.arange(len(scores)), ids, -scores))[:k]


def selection_metrics(target, scores, ids, threshold, fraction):
    target = np.asarray(target, dtype=np.float64)
    if (
        target.shape != np.asarray(scores).shape
        or not np.isfinite(target).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("Targets must be finite and match the scores")
    chosen = top_indices(scores, ids, fraction)
    high = target >= threshold
    k = len(chosen)
    return {
        "n_cells": len(target),
        "n_selected": k,
        "n_high": int(high.sum()),
        "mean_marker": float(target[chosen].mean()),
        "purity": float(high[chosen].mean()),
        "random_mean": float(target.mean()),
        "random_purity": float(high.mean()),
        "oracle_mean": float(np.sort(target)[-k:].mean()),
        "oracle_purity": min(int(high.sum()), k) / k,
    }


def prediction_metrics(target, threshold, mean=None, hit_logit=None):
    target = np.asarray(target, dtype=np.float64)
    if (
        target.ndim != 1
        or not len(target)
        or not np.isfinite(target).all()
        or not np.isfinite(threshold)
    ):
        raise ValueError("Finite nonempty targets and threshold are required")
    result = {}
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float64)
        if mean.shape != target.shape or not np.isfinite(mean).all():
            raise ValueError("Invalid marker predictions")
        result["rmse"] = float(np.sqrt(np.mean((mean - target) ** 2)))
    if hit_logit is not None:
        logit = np.asarray(hit_logit, dtype=np.float64)
        if logit.shape != target.shape or not np.isfinite(logit).all():
            raise ValueError("Invalid event logits")
        high = (target >= threshold).astype(np.float64)
        result["brier"] = float(np.mean((expit(logit) - high) ** 2))
        result["log_loss"] = float(np.mean(np.logaddexp(0, logit) - high * logit))
    return result


def paired_field_bootstrap(
    target,
    scores_a,
    scores_b,
    ids,
    fields,
    threshold,
    fractions=(0.05, 0.1, 0.2),
    draws=2000,
    seed=20261030,
    subtract_pair=None,
):
    """Rerank common resampled fields, averaging utilities across fitted seeds."""
    target = np.asarray(target, dtype=np.float64)
    ids, fields = np.asarray(ids), np.asarray(fields)
    a, b = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(target):
        raise ValueError("Paired predictions must have shape [seeds, cells]")
    if len(ids) != len(target) or len(fields) != len(target) or draws < 2:
        raise ValueError("Invalid field bootstrap inputs")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite paired predictions")
    contrasts = [(a, b, 1)]
    if subtract_pair is not None:
        c, d = (np.asarray(x, dtype=np.float64) for x in subtract_pair)
        if (
            c.shape != a.shape
            or d.shape != a.shape
            or not np.isfinite(c).all()
            or not np.isfinite(d).all()
        ):
            raise ValueError("The second contrast must use the same paired seeds and cells")
        contrasts.append((c, d, -1))
    unique_fields, field_index = np.unique(fields, return_inverse=True)
    n_fields = len(unique_fields)
    if n_fields < 2:
        raise ValueError("At least two fields are required")
    fractions = tuple(fractions)
    if not fractions or not all(0 < q <= 1 for q in fractions):
        raise ValueError("Selection fractions must be in (0, 1]")
    orders = [
        ([(top_indices(sa, ids, 1), top_indices(sb, ids, 1)) for sa, sb in zip(left, right)], sign)
        for left, right, sign in contrasts
    ]
    high = (target >= threshold).astype(float)

    def differences(weights):
        k = np.ceil(np.asarray(fractions) * weights.sum()).astype(int)
        values = np.zeros((len(fractions), 2), dtype=np.float64)

        # Repeated copies of a cell are adjacent in score/ID order.
        def sums(order):
            w = weights[order]
            cumulative = np.cumsum(w)
            boundary = np.searchsorted(cumulative, k, side="left")
            excess = cumulative[boundary] - k
            y, z = target[order], high[order]
            return np.column_stack(
                (
                    np.cumsum(w * y)[boundary] - excess * y[boundary],
                    np.cumsum(w * z)[boundary] - excess * z[boundary],
                )
            )

        for paired_orders, sign in orders:
            for oa, ob in paired_orders:
                values += sign * (sums(oa) - sums(ob)) / k[:, None]
        return values / len(a)

    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, len(fractions), 2))
    for draw in range(draws):
        chosen = rng.integers(0, n_fields, n_fields)
        weights = np.bincount(chosen, minlength=n_fields)[field_index]
        estimates[draw] = differences(weights)
    observed = differences(np.ones(len(target), dtype=int))
    low, high = np.quantile(estimates, [0.025, 0.975], axis=0)
    rows = []
    for j, fraction in enumerate(fractions):
        overlap = (
            None
            if subtract_pair is not None
            else float(
                np.mean(
                    [
                        len(
                            set(top_indices(sa, ids, fraction))
                            & set(top_indices(sb, ids, fraction))
                        )
                        / len(top_indices(sa, ids, fraction))
                        for sa, sb in zip(a, b)
                    ]
                )
            )
        )
        for m, metric in enumerate(("mean_marker", "purity")):
            rows.append(
                {
                    "selection_fraction": fraction,
                    "metric": metric,
                    "difference": float(observed[j, m]),
                    "ci_low": float(low[j, m]),
                    "ci_high": float(high[j, m]),
                    "selected_overlap": overlap,
                    "draws": draws,
                    "n_fields": n_fields,
                    "n_seeds": len(a),
                }
            )
    return rows
