"""Verify paired test evaluation without storing prediction volumes."""

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

from learn_motion.checkpoint import save_checkpoint
from learn_motion.factory import build_learn_img_model
from scripts import evaluate_learn_img


def test_evaluate_test_split_in_one_pass(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    (root / "references").mkdir(parents=True)
    (root / "corrupted").mkdir()
    (root / "splits.json").write_text(json.dumps({
        "split_unit": "subject", "train": [], "validation": [], "test": ["sub-09"],
    }), encoding="utf-8")
    image = np.ones((2, 2, 4, 16, 16), dtype=np.float32)
    image[:, 1] = 0
    reference = "references/sub-09.h5"
    with h5py.File(root / reference, "w") as handle:
        handle.create_dataset("image", data=image)
    rows = []
    for case, factor in (("m01", 1.2), ("m02", 1.5)):
        path = f"corrupted/sub-09_{case}.h5"
        with h5py.File(root / path, "w") as handle:
            handle.create_dataset("image", data=image * factor)
        rows.append({"pair_id": f"sub-09_ses-01_{case}_demo", "subject": "sub-09",
                     "split": "test", "reference_path": reference,
                     "corrupted_path": path, "status": "completed",
                     "reference_status": "completed"})
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "output_dir": str(tmp_path / "training"),
        "model": {"base_channels": 2, "num_levels": 1, "convs_per_level": 1,
                  "upconv_relu": True},
        "data": {"format": "h5", "root": str(root), "slice_range": [0, 2],
                 "normalization": "middle_slice_all_echoes", "shared_input_scale": True,
                 "use_brain_mask": False},
        "loader": {"batch_size": 2, "num_workers": 0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint_path = tmp_path / "best-loss.pt"
    save_checkpoint(checkpoint_path, build_learn_img_model(config), None, 11, config)
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate_learn_img.py", "--config", str(config_path),
                                   "--checkpoint", str(checkpoint_path), "--output-dir", str(output_dir),
                                   "--preview-slice", "1", "--device", "cpu"])
    evaluate_learn_img.main()

    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["checkpoint_epoch"] == 12
    assert report["overall"]["pairs"] == 2
    assert report["overall"]["slices"] == 4
    assert {row["motion_case"] for row in report["by_motion_case"]} == {"m01", "m02"}
    assert report["by_subject"][0]["slices"] == 4
    assert report["overall"]["input_complex_mse"] > 0
    with (output_dir / "per_slice.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert len(list((output_dir / "previews").glob("*.png"))) == 2
    assert not list(output_dir.rglob("*.mat"))
    assert not list(output_dir.rglob("*.h5"))
