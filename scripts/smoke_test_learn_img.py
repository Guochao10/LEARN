"""Run a data-independent LEARN-IMG forward/backward/checkpoint test."""

# 用随机数据快速检查模型前向, 反向更新和checkpoint保存和读取.

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from learn_motion.checkpoint import load_checkpoint, save_checkpoint
from learn_motion.models.learn_img import LearnImgUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base-channels", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = LearnImgUNet(base_channels=args.base_channels, num_levels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    inputs = torch.randn(1, 2, 4, 64, 64, device=device)
    targets = torch.randn_like(inputs)

    prediction = model(inputs)
    loss = torch.nn.functional.mse_loss(prediction, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "smoke.pt"
        save_checkpoint(checkpoint_path, model, optimizer, 0, {"smoke_test": True})
        restored = LearnImgUNet(base_channels=args.base_channels, num_levels=2).to(device)
        load_checkpoint(checkpoint_path, restored, map_location=device)
        restored.eval()
        model.eval()
        with torch.inference_mode():
            reference = model(inputs)
            recovered = restored(inputs)
        torch.testing.assert_close(reference, recovered)

    print(f"LEARN-IMG smoke test passed on {device}; loss={loss.item():.6g}")


if __name__ == "__main__":
    main()

