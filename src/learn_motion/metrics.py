"""Metrics matching the magnitude-based evaluation in the original code."""

from __future__ import annotations

import torch

from .data.complex import complex_magnitude


def magnitude_snr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = complex_magnitude(prediction)
    target = complex_magnitude(target)
    error = torch.linalg.vector_norm((target - prediction).flatten(1), dim=1)
    signal = torch.linalg.vector_norm(target.flatten(1), dim=1)
    return (20 * torch.log10(signal / error.clamp_min(1e-12))).mean()


def magnitude_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = complex_magnitude(prediction)
    target = complex_magnitude(target)
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    return (10 * torch.log10(1.0 / mse.clamp_min(1e-12))).mean()

