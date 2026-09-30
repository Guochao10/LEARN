"""Paired LEARN-IMG slices from the Radial HDF5 dataset manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class LearnImgH5Dataset(Dataset[dict[str, Any]]):
    """Read ``/image [Z,2,E,H,W]`` as single-slice ``[2,E,H,W]`` samples.

    ``slice_range`` uses zero-based Python indices and an exclusive stop. The
    planned anatomical slices 113 through 176 are therefore ``(112, 176)``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        slice_range: tuple[int, int] = (112, 176),
        normalization: str = "middle_slice_all_echoes",
        shared_input_scale: bool = False,
        use_brain_mask: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = "validation" if split == "valid" else split
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {split}")
        if normalization not in {"middle_slice_all_echoes", "middle_slice_first_echo", "none"}:
            raise ValueError(f"unknown normalization: {normalization}")
        if len(slice_range) != 2:
            raise ValueError("slice_range must contain (start, stop)")
        self.start, self.stop = (int(value) for value in slice_range)
        if not 0 <= self.start < self.stop:
            raise ValueError(f"invalid slice_range: {slice_range}")
        self.normalization = normalization
        self.shared_input_scale = shared_input_scale
        self.use_brain_mask = use_brain_mask
        self._scales: dict[Path, float] = {}

        with (self.root / "splits.json").open("r", encoding="utf-8") as stream:
            splits = json.load(stream)
        if splits.get("split_unit") != "subject":
            raise ValueError("splits.json must use subject-level splits")
        expected_subjects = set(splits[self.split])
        with (self.root / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == self.split]
        if not rows or {row["subject"] for row in rows} != expected_subjects:
            raise ValueError(f"manifest does not cover all {self.split} subjects")

        self.pairs: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        seen_motion: set[Path] = set()
        for row in rows:
            if row["subject"] not in expected_subjects or row["status"] != "completed" or row["reference_status"] != "completed":
                raise ValueError(f"incomplete or mismatched pair: {row['pair_id']}")
            motion_path = self._resolve_path(row["corrupted_path"])
            truth_path = self._resolve_path(row["reference_path"])
            if motion_path in seen_motion:
                raise ValueError(f"duplicate corrupted file: {motion_path}")
            seen_motion.add(motion_path)
            with h5py.File(motion_path, "r") as motion, h5py.File(truth_path, "r") as truth:
                motion_shape = self._image_shape(motion, motion_path)
                truth_shape = self._image_shape(truth, truth_path)
                if motion_shape != truth_shape:
                    raise ValueError(f"paired images differ: {motion_path} and {truth_path}")
                if self.stop > motion_shape[0]:
                    raise ValueError(f"slice_range exceeds {motion_shape[0]} slices: {slice_range}")
                if use_brain_mask and tuple(truth["mask"].shape) != (motion_shape[0], *motion_shape[3:]):
                    raise ValueError(f"invalid /mask shape in {truth_path}")
            pair_index = len(self.pairs)
            self.pairs.append({
                "subject": row["subject"],
                "pair_id": row["pair_id"],
                "motion_case": row.get("pair_id", "").split("_")[2] if len(row.get("pair_id", "").split("_")) > 2 else "unknown",
                "motion_path": motion_path,
                "truth_path": truth_path,
                "shape": motion_shape,
            })
            self.index.extend((pair_index, z) for z in range(self.start, self.stop))

    def _resolve_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise FileNotFoundError(f"dataset file not found under {self.root}: {relative_path}")
        return path

    @staticmethod
    def _image_shape(handle: h5py.File, path: Path) -> tuple[int, ...]:
        if "image" not in handle:
            raise ValueError(f"missing /image in {path}")
        image = handle["image"]
        if image.ndim != 5 or image.shape[1] != 2 or image.shape[2] != 4 or image.dtype.kind != "f":
            raise ValueError(f"expected float /image [Z,2,4,H,W] in {path}, got {image.shape} {image.dtype}")
        return tuple(image.shape)

    def _scale(self, handle: h5py.File, path: Path) -> float:
        if path not in self._scales:
            if self.normalization == "none":
                factor = 1.0
            else:
                middle = handle["image"].shape[0] // 2
                raw = np.asarray(handle["image"][middle], dtype=np.float32)
                if self.normalization == "middle_slice_first_echo":
                    raw = raw[:, :1]
                factor = float(np.mean(np.hypot(raw[0], raw[1])))
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f"invalid normalization factor {factor} in {path}")
            self._scales[path] = factor
        return self._scales[path]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        pair_index, slice_index = self.index[item]
        pair = self.pairs[pair_index]
        with h5py.File(pair["motion_path"], "r") as motion, h5py.File(pair["truth_path"], "r") as truth:
            input_scale = self._scale(motion, pair["motion_path"])
            target_scale = input_scale if self.shared_input_scale else self._scale(truth, pair["truth_path"])
            input_array = np.asarray(motion["image"][slice_index], dtype=np.float32) / input_scale
            target_array = np.asarray(truth["image"][slice_index], dtype=np.float32) / target_scale
            if self.use_brain_mask:
                mask = np.asarray(truth["mask"][slice_index], dtype=np.float32)[None, None]
            else:
                mask = np.ones((1, 1, *input_array.shape[-2:]), dtype=np.float32)

        if not np.isfinite(input_array).all() or not np.isfinite(target_array).all():
            raise ValueError(f"non-finite image values in {pair['pair_id']} slice {slice_index}")
        return {
            "input": torch.from_numpy(np.ascontiguousarray(input_array)),
            "target": torch.from_numpy(np.ascontiguousarray(target_array)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "subject": pair["subject"],
            "pair_id": pair["pair_id"],
            "motion_case": pair["motion_case"],
            "motion_path": str(pair["motion_path"]),
            "truth_path": str(pair["truth_path"]),
            "slice_index": slice_index,
            "input_scale": input_scale,
            "target_scale": target_scale,
            "original_width": input_array.shape[-1],
        }
