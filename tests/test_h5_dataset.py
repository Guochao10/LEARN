import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from learn_motion.data.h5_dataset import LearnImgH5Dataset
from learn_motion.factory import build_dataset, build_loader


def test_manifest_h5_pairs_and_slice_selection(tmp_path: Path) -> None:
    (tmp_path / "references").mkdir()
    (tmp_path / "corrupted").mkdir()
    (tmp_path / "splits.json").write_text(
        json.dumps({"split_unit": "subject", "train": ["sub-01"], "validation": [], "test": []}),
        encoding="utf-8",
    )
    shape = (6, 2, 4, 8, 8)
    truth = np.zeros(shape, dtype=np.float32)
    motion = np.zeros(shape, dtype=np.float32)
    truth[:, 0] = 2.0
    motion[:, 0] = 4.0
    truth[2, 0] = 10.0
    motion[2, 0] = 20.0
    truth[3, 0] = 6.0
    motion[3, 0] = 12.0
    motion[3, 1] = 8.0
    mask = np.ones((6, 8, 8), dtype=np.uint8)
    mask[3, 0, 0] = 0

    for relative, image in (("references/ref.h5", truth), ("corrupted/motion.h5", motion)):
        with h5py.File(tmp_path / relative, "w") as handle:
            handle.create_dataset("image", data=image)
            handle.create_dataset("mask", data=mask)

    fields = ["pair_id", "subject", "split", "reference_path", "corrupted_path", "status", "reference_status"]
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "pair_id": "sub-01_m01", "subject": "sub-01", "split": "train",
            "reference_path": "references/ref.h5", "corrupted_path": "corrupted/motion.h5",
            "status": "completed", "reference_status": "completed",
        })

    dataset = LearnImgH5Dataset(tmp_path, "train", slice_range=(2, 4))
    assert len(dataset) == 2
    sample = dataset[1]
    assert sample["slice_index"] == 3
    assert sample["pair_id"] == "sub-01_m01"
    assert sample["motion_case"] == "unknown"
    assert tuple(sample["input"].shape) == (2, 4, 8, 8)
    assert tuple(sample["mask"].shape) == (1, 1, 8, 8)
    assert sample["mask"][0, 0, 0, 0] == 0
    # Middle slice is z=3 in the full six-slice volume, before selection.
    np.testing.assert_allclose(sample["input"][0].numpy(), 12.0 / np.hypot(12.0, 8.0), atol=1e-6)
    np.testing.assert_allclose(sample["input"][1].numpy(), 8.0 / np.hypot(12.0, 8.0), atol=1e-6)
    np.testing.assert_allclose(sample["target"][0].numpy(), 1.0)

    shared = LearnImgH5Dataset(tmp_path, "train", slice_range=(2, 4), shared_input_scale=True)[1]
    assert shared["target_scale"] == shared["input_scale"]
    np.testing.assert_allclose(shared["target"][0].numpy(), 6.0 / np.hypot(12.0, 8.0), atol=1e-6)

    with pytest.raises(ValueError, match="slice_range exceeds"):
        LearnImgH5Dataset(tmp_path, "train", slice_range=(2, 7))

    config = {
        "data": {"format": "h5", "root": str(tmp_path), "slice_range": [2, 4]},
        "loader": {"batch_size": 1, "num_workers": 0},
    }
    with pytest.raises(ValueError, match="data.format is required"):
        build_dataset({"data": {"root": str(tmp_path)}}, "train")
    assert len(build_dataset(config, "train")) == 2
    batch = next(iter(build_loader(build_dataset(config, "train"), config, "train")))
    assert tuple(batch["input"].shape) == (1, 2, 4, 8, 8)
