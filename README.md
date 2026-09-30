# LEARN-IMG PyTorch

This repository contains a PyTorch implementation of LEARN-IMG for complex
multi-echo GRE motion correction. The original TensorFlow 1.x research code,
environment file, and examples are archived together in [`legacy_tf/`](legacy_tf/).
The PyTorch model follows the original LEARN-IMG network architecture; the
training code and data interfaces are maintained independently.

## Layout

- `src/learn_motion/`: PyTorch model, datasets, metrics, and checkpoints.
- `scripts/`: training, inference, and diagnostic entry points.
- `configs/`: PyTorch experiment configurations.
- `tests/`: PyTorch tests.
- Optional `data/`: MATLAB example data for the PyTorch MAT loader, if downloaded.
- `results/`: local checkpoints and inference outputs (ignored by Git).
- `legacy_tf/`: original TensorFlow implementation and its historical README.

See [README_PYTORCH.md](README_PYTORCH.md) for tensor conventions and setup.

## HN Radial Data

The current experiment reads paired HDF5 files directly from
`/data1/gcjiang3/generated/HNRepeatability_RadialMotion_v1` using its
`manifest.csv`. The training configuration selects anatomical slices 113-176
with Python slice `[112:176]`.

```bash
python -m pip install -e ".[train,test]"
python -m pytest
```

The completed HN Radial training run uses
`configs/hn_radial_learn_img_formal.yaml`. See
[`docs/正式训练.md`](docs/正式训练.md) for the tmux launch command and output files.
`configs/hn_radial_learn_img.yaml` remains as a small diagnostic configuration.
