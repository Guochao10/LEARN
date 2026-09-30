"""Factories shared by training and inference commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .data.h5_dataset import LearnImgH5Dataset
from .data.learn_img_dataset import LearnImgMatDataset
from .models.learn_img import LearnImgUNet


def build_learn_img_model(config: dict[str, Any]) -> LearnImgUNet:
    model_config = config["model"]
    return LearnImgUNet(
        in_channels=int(model_config.get("in_channels", 2)),
        out_channels=int(model_config.get("out_channels", 2)),
        base_channels=int(model_config.get("base_channels", 64)),
        convs_per_level=int(model_config.get("convs_per_level", 3)),
        num_levels=int(model_config.get("num_levels", 4)),
        kernel_size=model_config.get("kernel_size", 3),
        output_relu=bool(model_config.get("output_relu", False)),
        upconv_relu=bool(model_config.get("upconv_relu", False)),
        residual=bool(model_config.get("residual", False)),
    )


def build_dataset(config: dict[str, Any], split: str) -> LearnImgMatDataset | LearnImgH5Dataset:
    data_config = config["data"]
    data_format = data_config.get("format", "mat")
    if data_format == "h5":
        return LearnImgH5Dataset(
            root=Path(data_config["root"]),
            split=split,
            slice_range=tuple(data_config.get("slice_range", (112, 176))),
            normalization=str(data_config.get("normalization", "middle_slice_all_echoes")),
            shared_input_scale=bool(data_config.get("shared_input_scale", False)),
            use_brain_mask=bool(data_config.get("use_brain_mask", False)),
        )
    if data_format != "mat":
        raise ValueError(f"unknown data format: {data_format}")
    split_config = data_config["splits"][split]
    return LearnImgMatDataset(
        root=Path(data_config["root"]),
        subjects=list(split_config["subjects"]),
        motion_pattern=str(split_config["motion_pattern"]),
        truth_pattern=str(split_config.get("truth_pattern", "truth/{subject}/ima_comb_{subject}.mat")),
        mat_key=str(data_config.get("mat_key", "ima_comb")),
        slice_ranges=data_config.get("slice_ranges"),
        normalization=str(data_config.get("normalization", "middle_slice_all_echoes")),
        pad_width_190_to_192=bool(data_config.get("pad_width_190_to_192", True)),
        cache_size=int(data_config.get("cache_size", 2)),
    )


def build_loader(
    dataset: LearnImgMatDataset | LearnImgH5Dataset,
    config: dict[str, Any],
    split: str,
) -> DataLoader:
    loader_config = config.get("loader", {})
    training = split == "train"
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 1)),
        shuffle=training,
        num_workers=int(loader_config.get("num_workers", 0)),
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=bool(loader_config.get("persistent_workers", False))
        and int(loader_config.get("num_workers", 0)) > 0,
        drop_last=training and bool(loader_config.get("drop_last", False)),
    )
