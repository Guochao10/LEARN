"""Conversions between complex arrays and LEARN-IMG tensors."""

from __future__ import annotations

import numpy as np
import torch


def complex_to_channels(array: np.ndarray) -> np.ndarray:
    """Convert ``[H,W,E]`` complex data to ``[2,E,H,W]`` float32."""
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected [H,W,E], got {array.shape}")
    if not np.iscomplexobj(array):
        raise ValueError("input must be complex-valued")
    channels = np.stack((array.real, array.imag), axis=0)
    return np.ascontiguousarray(channels.transpose(0, 3, 1, 2), dtype=np.float32)


def channels_to_complex(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert ``[2,E,H,W]`` or ``[B,2,E,H,W]`` to a complex array."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim not in (4, 5):
        raise ValueError(f"expected 4-D or 5-D channels-first data, got {array.shape}")
    channel_axis = 0 if array.ndim == 4 else 1
    if array.shape[channel_axis] != 2:
        raise ValueError("complex channel dimension must have length two")
    real = np.take(array, 0, axis=channel_axis)
    imag = np.take(array, 1, axis=channel_axis)
    return real + 1j * imag


def complex_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """Return magnitude from ``[B,2,E,H,W]`` data."""
    if tensor.ndim != 5 or tensor.shape[1] != 2:
        raise ValueError(f"expected [B,2,E,H,W], got {tuple(tensor.shape)}")
    return torch.sqrt(tensor[:, 0].square() + tensor[:, 1].square() + 1e-12)

