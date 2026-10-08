"""Check the single-slice diagnostic measurements."""

import torch

from learn_motion.losses import masked_mse
from scripts.overfit_one_h5_slice import masked_magnitude_gradient_mae


def test_masked_loss_and_gradient_use_only_brain_region() -> None:
    target = torch.zeros(1, 2, 2, 2, 3)
    image = target.clone()
    image[:, 0, :, 0, 0] = 1
    image[:, 0, :, 0, 2] = 100
    mask = torch.tensor([[[[[1, 1, 0], [1, 1, 0]]]]], dtype=torch.float32)

    assert torch.isclose(masked_mse(image, target, mask), torch.tensor(0.125))
    assert torch.isclose(masked_magnitude_gradient_mae(target, target, mask), torch.tensor(0.0))
    assert masked_magnitude_gradient_mae(image, target, mask) > 0
