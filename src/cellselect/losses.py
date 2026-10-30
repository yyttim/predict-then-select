"""Proper losses for conditional marker mean and high-marker probability."""

from __future__ import annotations

import math

from torch.nn import functional as F


def objective_loss(outputs, target_z, high, method, lambda_binary=1.0):
    """Return total, standardized MSE, and unweighted binary cross-entropy."""
    mean, logit = outputs["mean_z"], outputs["hit_logit"]
    if (
        mean.ndim != 1
        or not mean.numel()
        or mean.shape != logit.shape
        or mean.shape != target_z.shape
        or mean.shape != high.shape
    ):
        raise ValueError("Predictions and targets must be aligned nonempty vectors")
    if method not in {"mse", "bce", "joint"}:
        raise ValueError("Unknown marker objective")
    if (
        not math.isfinite(lambda_binary)
        or lambda_binary < 0
        or (method == "joint" and lambda_binary == 0)
    ):
        raise ValueError("lambda_binary must be nonnegative, and positive for joint training")
    mse = F.mse_loss(mean, target_z)
    bce = F.binary_cross_entropy_with_logits(logit, high)
    # Unused heads must have grad=None, so AdamW does not decay their parameters.
    total = mse if method == "mse" else bce if method == "bce" else mse + lambda_binary * bce
    return total, mse, bce
