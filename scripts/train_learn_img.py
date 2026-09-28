"""Train the PyTorch LEARN-IMG model on paired complex mGRE data."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from learn_motion.checkpoint import load_checkpoint, save_checkpoint
from learn_motion.config import load_config
from learn_motion.factory import build_dataset, build_learn_img_model, build_loader
from learn_motion.metrics import magnitude_psnr, magnitude_snr
from learn_motion.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction * mask, target * mask)


def evaluate(model, loader, device, use_bf16: bool) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "snr": 0.0}
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                predictions = model(inputs)
                loss = masked_mse(predictions, targets, masks)
            totals["loss"] += loss.detach().item()
            totals["psnr"] += magnitude_psnr(predictions.float(), targets).item()
            totals["snr"] += magnitude_snr(predictions.float(), targets).item()
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_config = config["training"]
    seed_everything(
        int(config.get("seed", 0)),
        deterministic=bool(config.get("deterministic", False)),
    )
    device = torch.device(args.device)
    train_dataset = build_dataset(config, "train")
    valid_dataset = build_dataset(config, "valid")
    train_loader = build_loader(train_dataset, config, "train")
    valid_loader = build_loader(valid_dataset, config, "valid")

    model = build_learn_img_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("learning_rate", 1e-4)))
    start_epoch = 0
    best_psnr = float("-inf")
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(payload["epoch"]) + 1
        if payload.get("best_metric") is not None:
            best_psnr = float(payload["best_metric"])

    output_dir = Path(config.get("output_dir", "results/pytorch-learn-img"))
    output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = bool(train_config.get("bf16", False))
    epochs = int(train_config.get("epochs", 200))

    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{epochs}")
        for batch in progress:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                predictions = model(inputs)
                loss = masked_mse(predictions, targets, masks)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.detach().item():.5g}")

        metrics = evaluate(model, valid_loader, device, use_bf16)
        print(
            f"epoch={epoch + 1} valid_loss={metrics['loss']:.6g} "
            f"valid_psnr={metrics['psnr']:.4f} valid_snr={metrics['snr']:.4f}"
        )
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            save_checkpoint(output_dir / "best-psnr.pt", model, optimizer, epoch, config, best_psnr)
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, config, best_psnr)


if __name__ == "__main__":
    main()
