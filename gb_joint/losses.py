from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch


def calibrate_lambda_reconstruction(
    reconstruction_to_clustering_gradient_ratios: Sequence[float],
    *,
    target_gradient_ratio: float,
) -> float:
    """Calibrate the reconstruction weight from the median encoder gradient ratio."""
    ratios = np.asarray(reconstruction_to_clustering_gradient_ratios, dtype=float)
    if ratios.ndim != 1 or ratios.size == 0:
        raise ValueError("At least one gradient ratio is required.")
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError("Gradient ratios must be finite and positive.")
    if not math.isfinite(target_gradient_ratio) or target_gradient_ratio <= 0:
        raise ValueError("target_gradient_ratio must be finite and positive.")
    median_ratio = float(np.median(ratios))
    value = float(target_gradient_ratio / median_ratio)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("Calibrated lambda_R is not finite and positive.")
    return value


def full_data_total_estimate(batch_sum: torch.Tensor, *, n: int, batch_size: int) -> torch.Tensor:
    """Return the ``n / B`` stochastic estimate of a raw full-data sum."""

    if n < 1 or batch_size < 1:
        raise ValueError("n and batch_size must be positive.")
    return (n / float(batch_size)) * batch_sum


def total_loss_neural_objective(
    *,
    eta_mean: float,
    lambda_reconstruction: float,
    reconstruction_total: torch.Tensor,
    clustering_total: torch.Tensor,
    network_prior: torch.Tensor,
) -> torch.Tensor:
    """Network objective at the supplied effective eta.

    The caller applies any sensitivity multiplier once, before this function.
    The network prior is not multiplied by eta.
    """

    if not math.isfinite(eta_mean) or eta_mean <= 0:
        raise ValueError("eta_mean must be finite and positive.")
    if not math.isfinite(lambda_reconstruction) or lambda_reconstruction <= 0:
        raise ValueError("lambda_reconstruction must be finite and positive.")
    return float(eta_mean) * (
        float(lambda_reconstruction) * reconstruction_total + clustering_total
    ) + network_prior
