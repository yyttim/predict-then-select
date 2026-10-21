"""Decision-focused losses for maximizing mean marker value in a top-k pool.

Scores and labels must already use the same training-derived scale. Each call
treats its input vector as one decision pool; minibatch pools and a full held-out
pool need not define identical training and evaluation objectives.

Sources:
Elmachtoub and Grigas (2022), https://doi.org/10.1287/mnsc.2020.3922
Gupta and Huang (2024), https://doi.org/10.52202/079017-2514
"""

from __future__ import annotations

import math

import torch


def _prepare(scores: torch.Tensor, targets: torch.Tensor, fraction: float):
    if (
        not isinstance(scores, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or scores.ndim != 1
        or not scores.numel()
        or targets.shape != scores.shape
        or not scores.is_floating_point()
        or not targets.is_floating_point()
        or scores.device != targets.device
    ):
        raise ValueError("Scores and targets must be aligned floating vectors on one device")
    if (
        isinstance(fraction, bool)
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
        or not torch.isfinite(scores).all()
        or not torch.isfinite(targets).all()
    ):
        raise ValueError("Finite inputs and a selection fraction in (0, 1] are required")
    # Outcomes are labels, not an optimization variable. Preserve score precision.
    targets = targets.detach().to(dtype=scores.dtype)
    if not torch.isfinite(targets).all():
        raise ValueError("Targets are not representable in the score dtype")
    return targets, min(scores.numel(), math.ceil(fraction * scores.numel()))


def _topk_mask(values: torch.Tensor, k: int):
    # Stable input-row order resolves ties independently of measured outcomes.
    # The discrete solver is held fixed when differentiating its objective value.
    indices = torch.argsort(values.detach(), descending=True, stable=True)[:k]
    return torch.zeros_like(values).scatter_(0, indices, 1.0)


def topk_spo_plus(scores: torch.Tensor, targets: torch.Tensor, fraction: float = 0.1):
    """SPO+ for maximization: [F(2s-y) - (2s-y)^T topk(y)] / k.

    F is the sum of the k largest entries. The score subgradient is
    2 * (topk(2s-y) - topk(y)) / k. This upper-bounds realized top-k regret;
    population consistency requires additional assumptions on the outcome law.
    """
    targets, k = _prepare(scores, targets, fraction)
    adjusted = 2 * scores - targets
    difference = _topk_mask(adjusted, k) - _topk_mask(targets, k)
    # Cancel shared selected entries before summation for numerical stability.
    return torch.sum(adjusted * difference) / k


def topk_pg_central(
    scores: torch.Tensor, targets: torch.Tensor, fraction: float = 0.1, h: float = 0.1
):
    """Central PG approximation to negative selection utility.

    Loss = -[F(s+h*y) - F(s-h*y)] / (2*h*k), with score gradient
    -[topk(s+h*y) - topk(s-h*y)] / (2*h*k). For a fixed pool away from
    a score boundary, sufficiently small h gives exactly negative utility.
    The loss is not necessarily convex, positive, or a regret upper bound.
    Adding b to every label shifts the loss by -b but leaves gradients unchanged.
    """
    if isinstance(h, bool) or not math.isfinite(h) or h <= 0:
        raise ValueError("The perturbation h must be finite and positive")
    targets, k = _prepare(scores, targets, fraction)
    plus, minus = _topk_mask(scores + h * targets, k), _topk_mask(scores - h * targets, k)
    # Algebraically equal to the finite difference, without subtracting two
    # nearly equal large sums when both perturbations select the same cells.
    difference = torch.sum(scores * (plus - minus)) / (2 * h)
    outcome = torch.sum(targets * (plus + minus)) / 2
    return -(difference + outcome) / k
