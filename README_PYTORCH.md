# LEARN-IMG PyTorch migration

This branch keeps the original TensorFlow implementation under `codes/` and
adds a modern PyTorch implementation under `src/learn_motion/`.

## Scientific tensor convention

The original Keras model uses `[B,H,W,E,2]`, where `E` is echo and the final
dimension is `[real,imaginary]`. PyTorch uses:

```text
[B,2,E,H,W]
```

`Conv3d` operates jointly over in-plane space and echo. Pooling and upsampling
use `(1,2,2)`, so echo count is never downsampled. This is a slice-wise
multi-echo network, not a spatial 3-D whole-volume network.

## Environment

Create a dedicated environment and install a PyTorch build appropriate for the
local CUDA driver using the official PyTorch installation selector. Then run:

```powershell
python -m pip install -e ".[train,test]"
```

Do not install the original `learn_env.yml` for this implementation; it pins
Python 3.6 and TensorFlow 1.13.

## Data-independent verification

```powershell
python scripts/smoke_test_learn_img.py
pytest
```

The smoke test performs forward propagation, backpropagation, an optimizer
step, checkpoint save/load, and output equality checking.

## Author data layout

After downloading the exemplar data from the author, retain this layout:

```text
data/
  motion/subj_sim/ima_comb_*.mat
  motion/subj_exp/ima_comb_subj_exp.mat
  truth/subj_sim/ima_comb_subj_sim.mat
  truth/subj_exp/ima_comb_subj_exp.mat
```

MAT arrays are expected to contain complex `ima_comb` in `[X,Y,Z,E]` order.
Review `configs/learn_img_original.yaml` and adjust the filename glob if the
downloaded archive uses a different suffix.

## Training

```powershell
python scripts/train_learn_img.py --config configs/learn_img_original.yaml
```

Resume with:

```powershell
python scripts/train_learn_img.py `
  --config configs/learn_img_original.yaml `
  --resume results/pytorch-learn-img/latest.pt
```

The first correctness run should use one GPU, FP32, and `bf16: false`. Enable
BF16 only after the one-batch overfitting and validation checks pass.

## Inference

```powershell
python scripts/infer_learn_img.py `
  --config configs/learn_img_original.yaml `
  --checkpoint results/pytorch-learn-img/best-psnr.pt `
  --split test
```

Predictions are saved as complex MATLAB `[X,Y,Z,E]` arrays. Only configured
slices are reconstructed, so use a full slice range when producing complete
volumes for downstream QSM processing.

## Deliberately deferred work

- conversion of the original Keras `.h5` weights;
- LEARN-BIO and its R2* signal-model loss;
- DistributedDataParallel and multi-GPU launch scripts;
- four-echo Radial/ViewMotionQSM data adapters;
- SSIM parity with the TensorFlow implementation.

These should be added after the PyTorch LEARN-IMG data and training path is
validated with the author's exemplar dataset.

