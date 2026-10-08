"""MAT-backed dataset for paired motion-corrupted/motion-free mGRE slices."""

from __future__ import annotations

from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .complex import complex_to_channels
from .mat_io import load_mat_array


class LearnImgMatDataset(Dataset[dict[str, Any]]):
    """Load paired complex mGRE volumes using the original repository layout.

    Input MAT arrays must use MATLAB ``[X,Y,Z,E]`` order. Each returned sample
    is one anatomical slice in PyTorch ``[2,E,H,W]`` order.
    """

    def __init__(
        self,
        root: str | Path,
        subjects: list[str],
        motion_pattern: str,
        truth_pattern: str = "truth/{subject}/ima_comb_{subject}.mat",
        mat_key: str = "ima_comb",
        slice_ranges: dict[int | str, list[int]] | None = None,
        normalization: str = "middle_slice_all_echoes",
        pad_width_190_to_192: bool = True,
        cache_size: int = 2,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.mat_key = mat_key
        self.slice_ranges = slice_ranges or {}
        self.normalization = normalization
        self.pad_width_190_to_192 = pad_width_190_to_192
        self.cache_size = max(int(cache_size), 0)
        self._volume_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        self.pairs: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        for subject in subjects:
            truth_path = self.root / truth_pattern.format(subject=subject)
            motion_glob = str(self.root / motion_pattern.format(subject=subject))
            motion_paths = [Path(item) for item in sorted(glob(motion_glob))]
            if not truth_path.is_file():
                raise FileNotFoundError(f"truth file not found: {truth_path}")
            if not motion_paths:
                raise FileNotFoundError(f"no motion files match: {motion_glob}")

            truth = self._load_volume(str(truth_path))
            num_slices = truth.shape[2]
            start, stop = self._slice_range(num_slices)
            for motion_path in motion_paths:
                motion = self._load_volume(str(motion_path))
                if motion.shape != truth.shape:
                    raise ValueError(
                        f"paired volumes differ: {motion_path} {motion.shape} versus "
                        f"{truth_path} {truth.shape}"
                    )
                pair_index = len(self.pairs)
                self.pairs.append(
                    {
                        "subject": subject,
                        "motion_path": motion_path,
                        "truth_path": truth_path,
                        "shape": truth.shape,
                    }
                )
                self.index.extend((pair_index, slice_index) for slice_index in range(start, stop))

    def _slice_range(self, num_slices: int) -> tuple[int, int]:
        value = self.slice_ranges.get(num_slices, self.slice_ranges.get(str(num_slices)))
        if value is None:
            return 0, num_slices
        if len(value) != 2:
            raise ValueError("slice range must contain [start, stop]")
        start, stop = (int(item) for item in value)
        if not 0 <= start < stop <= num_slices:
            raise ValueError(f"invalid slice range {value} for {num_slices} slices")
        return start, stop

    def _load_volume_uncached(self, path: str) -> np.ndarray:
        volume = np.asarray(load_mat_array(path, self.mat_key))
        if volume.ndim != 4:
            raise ValueError(f"expected [X,Y,Z,E] in {path}, got {volume.shape}")
        if not np.iscomplexobj(volume):
            raise ValueError(f"LEARN-IMG requires complex data: {path}")
        if not np.all(np.isfinite(volume)):
            raise ValueError(f"non-finite values found in {path}")
        return volume

    def _load_volume(self, path: str) -> np.ndarray:
        if path in self._volume_cache:
            volume = self._volume_cache.pop(path)
            self._volume_cache[path] = volume
            return volume
        volume = self._load_volume_uncached(path)
        if self.cache_size > 0:
            self._volume_cache[path] = volume
            while len(self._volume_cache) > self.cache_size:
                self._volume_cache.popitem(last=False)
        return volume

    def __getstate__(self) -> dict[str, Any]:
        """Avoid copying cached volumes into spawned DataLoader workers."""
        state = self.__dict__.copy()
        state["_volume_cache"] = OrderedDict()
        return state

    def _normalization_factor(self, volume: np.ndarray) -> float:
        middle = volume.shape[2] // 2
        if self.normalization == "middle_slice_all_echoes":
            factor = float(np.mean(np.abs(volume[:, :, middle, :])))
        elif self.normalization == "middle_slice_first_echo":
            factor = float(np.mean(np.abs(volume[:, :, middle, 0])))
        elif self.normalization == "none":
            factor = 1.0
        else:
            raise ValueError(f"unknown normalization: {self.normalization}")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"invalid normalization factor {factor}")
        return factor

    def _prepare_slice(self, volume: np.ndarray, slice_index: int) -> tuple[np.ndarray, float, int]:
        factor = self._normalization_factor(volume)
        image = volume[:, :, slice_index, :] / factor
        original_width = image.shape[1]
        if self.pad_width_190_to_192 and original_width == 190:
            image = np.concatenate((image, image[:, -2:, :]), axis=1)
        return complex_to_channels(image), factor, original_width

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        pair_index, slice_index = self.index[item]
        pair = self.pairs[pair_index]
        motion = self._load_volume(str(pair["motion_path"]))
        truth = self._load_volume(str(pair["truth_path"]))
        input_array, input_scale, original_width = self._prepare_slice(motion, slice_index)
        target_array, target_scale, _ = self._prepare_slice(truth, slice_index)
        mask = np.ones((1, input_array.shape[1], input_array.shape[2], input_array.shape[3]), dtype=np.float32)
        return {
            "input": torch.from_numpy(input_array),
            "target": torch.from_numpy(target_array),
            "mask": torch.from_numpy(mask),
            "subject": pair["subject"],
            "motion_path": str(pair["motion_path"]),
            "truth_path": str(pair["truth_path"]),
            "slice_index": slice_index,
            "input_scale": input_scale,
            "target_scale": target_scale,
            "original_width": original_width,
        }
