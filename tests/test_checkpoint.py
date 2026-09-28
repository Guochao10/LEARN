from pathlib import Path

import torch

from learn_motion.checkpoint import load_checkpoint, save_checkpoint
from learn_motion.models.learn_img import LearnImgUNet


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = LearnImgUNet(base_channels=2, convs_per_level=1, num_levels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    source = torch.randn(1, 2, 4, 16, 16)
    model.eval()
    with torch.inference_mode():
        expected = model(source)

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, epoch=3, config={"name": "test"}, best_metric=12.0)
    restored = LearnImgUNet(base_channels=2, convs_per_level=1, num_levels=1)
    payload = load_checkpoint(path, restored)
    restored.eval()
    with torch.inference_mode():
        actual = restored(source)

    torch.testing.assert_close(actual, expected)
    assert payload["epoch"] == 3
    assert payload["best_metric"] == 12.0

