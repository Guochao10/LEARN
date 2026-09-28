import pytest
import torch

from learn_motion.models.learn_img import LearnImgUNet


def test_shape_and_backward() -> None:
    model = LearnImgUNet(base_channels=4, convs_per_level=1, num_levels=2)
    source = torch.randn(1, 2, 4, 32, 48, requires_grad=True)
    result = model(source)
    assert result.shape == source.shape
    result.square().mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_echo_dimension_is_preserved() -> None:
    model = LearnImgUNet(base_channels=2, convs_per_level=1, num_levels=2)
    for echo_count in (4, 10):
        source = torch.randn(1, 2, echo_count, 32, 32)
        assert model(source).shape == source.shape


def test_invalid_in_plane_size_is_rejected() -> None:
    model = LearnImgUNet(base_channels=2, convs_per_level=1, num_levels=3)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 2, 4, 30, 32))

