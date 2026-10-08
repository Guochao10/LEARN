"""Losses shared by training and diagnostic scripts."""

from __future__ import annotations

import torch


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    mask = mask.float()
    squared_error = (prediction - target).square() * mask
    denominator = mask.expand_as(squared_error).sum().clamp_min(1.0)
    return squared_error.sum() / denominator
