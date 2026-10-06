"""Train LEARN-IMG on paired complex mGRE slices."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
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
    prediction = prediction.float()
    target = target.float()
    mask = mask.float()
    squared_error = (prediction - target).square() * mask
    denominator = mask.expand_as(squared_error).sum().clamp_min(1.0)
    return squared_error.sum() / denominator


def amp_context(device: torch.device, use_bf16: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and device.type == "cuda" else nullcontext()


def evaluate(model, loader, device, use_bf16: bool) -> dict:
    model.eval()
    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with amp_context(device, use_bf16):
                predictions = model(inputs)
            for i in range(len(inputs)):
                values = {
                    "loss": masked_mse(predictions[i], targets[i], masks[i]).item(),
                    "input_loss": masked_mse(inputs[i], targets[i], masks[i]).item(),
                    "psnr": magnitude_psnr(predictions[i:i + 1].float(), targets[i:i + 1]).item(),
                    "snr": magnitude_snr(predictions[i:i + 1].float(), targets[i:i + 1]).item(),
                }
                for key in {"all", batch.get("motion_case", ["all"] * len(inputs))[i]}:
                    counts[key] += 1
                    for metric, value in values.items():
                        totals[key][metric] += value
    if not counts:
        raise ValueError("validation dataset is empty")
    return {key: {metric: value / counts[key] for metric, value in totals[key].items()} for key in sorted(counts)}


def save_preview(model, dataset, device, settings: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_id = str(settings["pair_id"])
    slice_index = int(settings["slice_index"])
    echo_index = int(settings.get("echo_index", 0))
    index = next((i for i, (pair, z) in enumerate(dataset.index)
                  if dataset.pairs[pair]["pair_id"] == pair_id and z == slice_index), None)
    if index is None:
        raise ValueError(f"preview pair/slice is absent from validation: {pair_id}, {slice_index}")
    sample = dataset[index]
    with torch.inference_mode():
        prediction = model(sample["input"].unsqueeze(0).to(device)).float().cpu()[0]

    def magnitude(tensor):
        array = tensor[:, echo_index].numpy()
        return np.hypot(array[0], array[1])

    images = [magnitude(sample["input"]), magnitude(prediction), magnitude(sample["target"])]
    limit = float(np.percentile(images[2], 99.5))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, title, image in zip(axes, ("Corrupted", "Prediction", "Reference"), images, strict=True):
        ax.imshow(np.rot90(image), cmap="gray", vmin=0, vmax=limit)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_config = config["training"]
    output_dir = Path(config.get("output_dir", "results/pytorch-learn-img"))
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}; choose a new directory or --resume")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"resume output directory does not exist: {output_dir}")
    seed_everything(int(config.get("seed", 0)), deterministic=bool(config.get("deterministic", False)))
    device = torch.device(args.device)
    train_dataset = build_dataset(config, "train")
    valid_dataset = build_dataset(config, "valid")
    train_loader = build_loader(train_dataset, config, "train")
    valid_loader = build_loader(valid_dataset, config, "valid")
    if not len(train_loader) or not len(valid_loader):
        raise ValueError("training and validation loaders must be nonempty")

    model = build_learn_img_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("learning_rate", 1e-4)))
    scheduler_config = train_config.get("scheduler", {})
    scheduler = None
    if scheduler_config:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 8)), min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
    monitor = train_config.get("monitor", "psnr")
    if monitor not in {"loss", "psnr"}:
        raise ValueError("training.monitor must be 'loss' or 'psnr'")
    start_epoch = 0
    best_metric = float("inf") if monitor == "loss" else float("-inf")
    stale_epochs = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        previous = payload["config"]
        for section in ("model", "data"):
            if previous.get(section) != config.get(section):
                raise ValueError(f"resume config differs in {section}")
        if previous.get("output_dir") != config.get("output_dir") or previous.get("training", {}).get("monitor", "psnr") != monitor:
            raise ValueError("resume output directory or monitor differs from checkpoint")
        start_epoch = int(payload["epoch"]) + 1
        if payload.get("best_metric") is not None:
            best_metric = float(payload["best_metric"])
        stale_epochs = int(payload.get("stale_epochs", 0))
        if scheduler is not None:
            if payload.get("scheduler") is None:
                raise ValueError("resume checkpoint has no scheduler state")
            scheduler.load_state_dict(payload["scheduler"])

    output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = bool(train_config.get("bf16", False))
    epochs = int(train_config.get("epochs", 200))
    early_patience = train_config.get("early_stopping_patience")
    log_path = output_dir / "history.jsonl"
    print(f"device={device} train={len(train_dataset)} valid={len(valid_dataset)} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{epochs}")
        for batch in progress:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, use_bf16):
                predictions = model(inputs)
                loss = masked_mse(predictions, targets, masks)
            loss.backward()
            optimizer.step()
            train_total += loss.detach().item() * len(inputs)
            train_count += len(inputs)
            progress.set_postfix(loss=f"{loss.detach().item():.5g}")

        metrics = evaluate(model, valid_loader, device, use_bf16)
        selected = metrics["all"][monitor]
        improved = selected < best_metric if monitor == "loss" else selected > best_metric
        if improved:
            best_metric = selected
            stale_epochs = 0
        else:
            stale_epochs += 1
        current_lr = optimizer.param_groups[0]["lr"]
        record = {"epoch": epoch + 1, "train_loss": train_total / train_count,
                  "learning_rate": current_lr, "validation": metrics,
                  "best_metric": best_metric, "best_epoch": improved}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"epoch={epoch + 1} train_loss={record['train_loss']:.6g} valid_loss={metrics['all']['loss']:.6g} "
              f"input_loss={metrics['all']['input_loss']:.6g} psnr={metrics['all']['psnr']:.4f} "
              f"lr={current_lr:.3g} best={best_metric:.6g}", flush=True)
        if scheduler is not None:
            scheduler.step(metrics["all"]["loss"])
        extra = {"scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "stale_epochs": stale_epochs}
        if improved:
            save_checkpoint(output_dir / f"best-{monitor}.pt", model, optimizer, epoch, config, best_metric, extra)
            if config.get("validation_preview"):
                save_preview(model, valid_dataset, device, config["validation_preview"], output_dir / "best-validation.png")
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, config, best_metric, extra)
        if early_patience is not None and stale_epochs >= int(early_patience):
            print(f"early stopping after {stale_epochs} epochs without improvement", flush=True)
            break


if __name__ == "__main__":
    main()
