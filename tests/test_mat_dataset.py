from pathlib import Path

import numpy as np
from scipy.io import savemat

from learn_motion.data.mat_dataset import LearnImgMatDataset
from learn_motion.factory import build_dataset


def test_dataset_reads_matlab_volume_and_preserves_axes(tmp_path: Path) -> None:
    subject = "subject-01"
    truth_dir = tmp_path / "truth" / subject
    motion_dir = tmp_path / "motion" / subject
    truth_dir.mkdir(parents=True)
    motion_dir.mkdir(parents=True)

    generator = np.random.default_rng(3)
    truth = generator.standard_normal((8, 6, 3, 4)) + 1j * generator.standard_normal((8, 6, 3, 4))
    motion = truth + (0.25 + 0.5j)
    savemat(truth_dir / f"ima_comb_{subject}.mat", {"ima_comb": truth})
    savemat(motion_dir / f"ima_comb_{subject}_motion.mat", {"ima_comb": motion})

    dataset = LearnImgMatDataset(
        root=tmp_path,
        subjects=[subject],
        motion_pattern="motion/{subject}/ima_comb_{subject}_*.mat",
        normalization="none",
        pad_width_190_to_192=False,
    )

    assert len(dataset) == 3
    sample = dataset[1]
    assert tuple(sample["input"].shape) == (2, 4, 8, 6)
    recovered = sample["input"][0].numpy() + 1j * sample["input"][1].numpy()
    np.testing.assert_allclose(recovered.transpose(1, 2, 0), motion[:, :, 1, :], atol=1e-6)
    assert sample["slice_index"] == 1
    assert sample["original_width"] == 6

    config = {
        "data": {
            "format": "mat",
            "root": str(tmp_path),
            "splits": {"train": {"subjects": [subject],
                                  "motion_pattern": "motion/{subject}/ima_comb_{subject}_*.mat"}},
            "normalization": "none",
            "pad_width_190_to_192": False,
        },
    }
    assert isinstance(build_dataset(config, "train"), LearnImgMatDataset)
