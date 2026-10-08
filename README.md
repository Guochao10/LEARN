# LEARN-IMG PyTorch

This repository contains a PyTorch implementation of LEARN-IMG for complex
multi-echo GRE motion correction. The original TensorFlow 1.x implementation,
environment, and examples are archived in [`legacy_tf/`](legacy_tf/). The
PyTorch model follows the LEARN-IMG architecture, while its training and data
interfaces are maintained separately. LEARN-BIO is not implemented in PyTorch.

## Repository layout

- `src/learn_motion/`: model, datasets, losses, metrics, and checkpoints.
- `scripts/`: training, test evaluation, smoke test, and diagnostic entry points.
- `configs/`: separate YAML configurations for each experiment.
- `tests/`: automated checks; see [`tests/README.md`](tests/README.md).
- `docs/`: training instructions, experiment records, and the paper.
- `results/`: local checkpoints and evaluation artifacts. Model weights are ignored by Git.
- `legacy_tf/`: archived TensorFlow implementation and its historical README.
- Optional `data/`: MATLAB example data, if downloaded for the MAT loader.

## Tensor and data conventions

The original Keras model uses `[B,H,W,E,2]`, with echoes in `E` and real and
imaginary channels last. PyTorch uses `[B,2,E,H,W]`. Its `Conv3d` operates over
the image plane and echoes. Pooling and upsampling use `(1,2,2)`, preserving
the echo count. The network processes anatomical slices independently, not
whole spatial 3-D volumes.

The current HN Radial experiments read paired HDF5 files directly from
`/data1/gcjiang3/generated/HNRepeatability_RadialMotion_v1`. The dataset uses
`manifest.csv` and subject-level splits. Each image has four echoes; the
configured Python slice range `[112,176]` selects anatomical slices 113-176.
The source HDF5 files are read without copying or cropping them into the repo.

## Environment and checks

Install a PyTorch build suited to the local CUDA driver in a dedicated Python
environment, then install this package and its training and test dependencies:

```bash
python -m pip install -e ".[train,test]"
python scripts/smoke_test_learn_img.py
python -m pytest
```

The smoke test checks a forward pass, backward pass, optimizer step, and
checkpoint round trip with random data. Pytest also exercises the HDF5 training
and evaluation entry points with small temporary datasets. The archived
`legacy_tf/learn_env.yml` targets Python 3.6 and TensorFlow 1.13; it is not the
environment for this PyTorch implementation.
For an initial correctness run, use one GPU and FP32 (`bf16: false`); enable
BF16 only after the smaller diagnostic and validation checks pass.

## HN Radial workflow

`configs/hn_radial_learn_img.yaml` is a small workflow check.
`configs/hn_radial_learn_img_formal.yaml` is the non-residual formal baseline;
the residual and brain-mask experiments have separate configs and output
directories. See [`docs/正式训练.md`](docs/正式训练.md) for the tmux training procedure.
Start a fresh experiment with an empty output directory; use `--resume` with a
checkpoint when continuing an existing run. The training script writes
`history.jsonl`, `latest.pt`, the best checkpoint selected by the configured
validation metric, and an optional `best-validation.png`.

For paired HDF5 test data, inference and evaluation run together:

```bash
python scripts/evaluate_learn_img.py \
  --config configs/hn_radial_learn_img_formal.yaml \
  --checkpoint results/hn-radial-learn-img-formal/best-loss.pt \
  --output-dir results/hn-radial-learn-img-formal/test-evaluation-new
```

This writes `summary.json`, `per_pair.csv`, `per_slice.csv`, and optional
four-echo preview images directly from the predictions. It does not save or
reload intermediate `.mat` volumes. Choose a new `--output-dir` for each run;
the script rejects an existing nonempty directory. The test report's complex
MSE and the training loss use different mask normalization denominators, so
their absolute values should not be compared directly.

## Optional MATLAB example data

The MAT loader remains available for the original author's example data.
Place downloaded files under the optional `data/` directory:

```text
data/
  motion/subj_sim/ima_comb_*.mat
  motion/subj_exp/ima_comb_subj_exp.mat
  truth/subj_sim/ima_comb_subj_sim.mat
  truth/subj_exp/ima_comb_subj_exp.mat
```

MAT arrays must contain complex `ima_comb` in `[X,Y,Z,E]` order. The
`configs/learn_img_original.yaml` configuration uses `data.format: mat`; check
its filename patterns against the downloaded archive before use. This path is
separate from the current HDF5 experiments.

## Scope

LEARN-BIO and its signal-model loss, conversion of the original Keras `.h5`
weights, multi-GPU launch scripts, and SSIM parity with the TensorFlow code
are not implemented in the PyTorch branch.
