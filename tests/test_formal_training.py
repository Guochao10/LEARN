"""Exercise the formal HDF5 training path with tiny paired volumes."""

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from scripts import train_learn_img


def test_formal_epoch_and_resume(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "data"
    (root / "references").mkdir(parents=True)
    (root / "corrupted").mkdir()
    (root / "splits.json").write_text(json.dumps({
        "split_unit": "subject", "train": ["sub-01"],
        "validation": ["sub-02"], "test": [],
    }), encoding="utf-8")
    image = np.ones((2, 2, 4, 16, 16), dtype=np.float32)
    image[:, 1] = 0
    rows = []
    for subject, split in (("sub-01", "train"), ("sub-02", "validation")):
        reference = f"references/{subject}.h5"
        corrupted = f"corrupted/{subject}.h5"
        for name, factor in ((reference, 1.0), (corrupted, 1.2)):
            with h5py.File(root / name, "w") as handle:
                handle.create_dataset("image", data=image * factor)
        rows.append({"pair_id": f"{subject}_ses-01_m01_demo", "subject": subject,
                     "split": split, "reference_path": reference,
                     "corrupted_path": corrupted, "status": "completed",
                     "reference_status": "completed"})
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "results"
    config = {
        "seed": 0, "output_dir": str(output),
        "model": {"base_channels": 2, "num_levels": 1, "convs_per_level": 1,
                  "upconv_relu": True},
        "data": {"format": "h5", "root": str(root), "slice_range": [0, 1],
                 "shared_input_scale": True, "use_brain_mask": False},
        "loader": {"batch_size": 1, "num_workers": 0},
        "training": {"epochs": 1, "learning_rate": 1e-4, "monitor": "loss",
                     "scheduler": {"patience": 1}},
        "validation_preview": {"pair_id": "sub-02_ses-01_m01_demo",
                               "slice_index": 0, "echo_index": 0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_learn_img.py", "--config", str(config_path), "--device", "cpu"])
    train_learn_img.main()
    assert (output / "latest.pt").is_file()
    assert (output / "best-loss.pt").is_file()
    assert (output / "best-validation.png").is_file()
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert history[0]["validation"]["m01"]["input_loss"] > 0
    checkpoint = torch.load(output / "latest.pt", weights_only=False)
    assert checkpoint["scheduler"] is not None

    config["training"]["epochs"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_learn_img.py", "--config", str(config_path),
                                   "--device", "cpu", "--resume", str(output / "latest.pt")])
    train_learn_img.main()
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1, 2]
