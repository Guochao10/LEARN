"""LEARN-IMG U-Net implemented with PyTorch.

The original TensorFlow model receives tensors in ``[B, H, W, E, 2]``
order. ``E`` is the echo axis and the final axis stores real and imaginary
components. PyTorch Conv3d uses ``[B, C, D, H, W]``, so this implementation
uses ``[B, 2, E, H, W]``. Pooling is applied only in the two image-plane
directions; the number of echoes is preserved at every level.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, value, value
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise ValueError("kernel_size must contain three elements")
    return result


class _ConvStack3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_count: int,
        kernel_size: int | Sequence[int],
    ) -> None:
        super().__init__()
        if conv_count < 1:
            raise ValueError("conv_count must be positive")
        kernel = _triple(kernel_size)
        padding = tuple(item // 2 for item in kernel)
        layers: list[nn.Module] = []
        current_channels = in_channels
        for _ in range(conv_count):
            layers.extend(
                [
                    nn.Conv3d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=padding,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class LearnImgUNet(nn.Module):
    """Original LEARN-IMG anisotropic 3D U-Net.

    Parameters reproduce ``codes/method/model.py::unet_3d`` by default.
    Despite using Conv3d, this is a slice-wise network: depth represents the
    echo dimension rather than the through-plane anatomical dimension.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        convs_per_level: int = 3,
        num_levels: int = 4,
        kernel_size: int | Sequence[int] = 3,
        output_relu: bool = False,
        upconv_relu: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if base_channels < 1 or num_levels < 1:
            raise ValueError("base_channels and num_levels must be positive")
        if residual and in_channels != out_channels:
            raise ValueError("residual mode requires equal input/output channels")

        self.num_levels = num_levels
        self.output_relu = output_relu
        self.upconv_relu = upconv_relu
        self.residual = residual

        kernel = _triple(kernel_size)
        padding = tuple(item // 2 for item in kernel)
        self.input_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=kernel,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_channels = base_channels
        skip_channels: list[int] = []
        for level in range(num_levels):
            level_channels = base_channels * (2**level)
            self.encoder_blocks.append(
                _ConvStack3d(
                    current_channels,
                    level_channels,
                    convs_per_level,
                    kernel,
                )
            )
            self.pools.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            skip_channels.append(level_channels)
            current_channels = level_channels

        bottom_channels = base_channels * (2**num_levels)
        self.bottom = _ConvStack3d(
            current_channels,
            bottom_channels,
            convs_per_level,
            kernel,
        )
        current_channels = bottom_channels

        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for level in range(num_levels - 1, -1, -1):
            level_channels = skip_channels[level]
            self.upconvs.append(
                nn.ConvTranspose3d(
                    current_channels,
                    level_channels,
                    kernel_size=kernel,
                    stride=(1, 2, 2),
                    padding=padding,
                    output_padding=(0, 1, 1),
                    bias=True,
                )
            )
            self.decoder_blocks.append(
                _ConvStack3d(
                    level_channels * 2,
                    level_channels,
                    convs_per_level,
                    kernel,
                )
            )
            current_channels = level_channels

        self.output_conv = nn.Conv3d(current_channels, out_channels, kernel_size=1)
        self.output_activation = nn.ReLU(inplace=False) if output_relu else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the Glorot initialization employed by Keras Conv layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.residual:
            nn.init.zeros_(self.output_conv.weight)
            nn.init.zeros_(self.output_conv.bias)

    def _validate_input(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 5:
            raise ValueError(
                "LEARN-IMG expects [batch, complex_channel, echo, height, width], "
                f"got {tuple(tensor.shape)}"
            )
        divisor = 2**self.num_levels
        if tensor.shape[-2] % divisor or tensor.shape[-1] % divisor:
            raise ValueError(
                f"height and width must be divisible by {divisor}, got "
                f"{tensor.shape[-2:]}"
            )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self._validate_input(tensor)
        original = tensor
        tensor = self.input_conv(tensor)
        skip_tensors: list[torch.Tensor] = []

        for block, pool in zip(self.encoder_blocks, self.pools, strict=True):
            tensor = block(tensor)
            skip_tensors.append(tensor)
            tensor = pool(tensor)

        tensor = self.bottom(tensor)
        for upconv, block, skip in zip(
            self.upconvs,
            self.decoder_blocks,
            reversed(skip_tensors),
            strict=True,
        ):
            tensor = upconv(tensor)
            if self.upconv_relu:
                tensor = torch.relu(tensor)
            if tensor.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    "decoder/encoder shape mismatch: "
                    f"{tuple(tensor.shape)} versus {tuple(skip.shape)}"
                )
            tensor = torch.cat((tensor, skip), dim=1)
            tensor = block(tensor)

        tensor = self.output_activation(self.output_conv(tensor))
        if self.residual:
            tensor = original - tensor
        return tensor
