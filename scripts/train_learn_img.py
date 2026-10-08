"""Train LEARN-IMG on paired complex mGRE slices."""

# 正式训练入口.

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from learn_motion.checkpoint import load_checkpoint, save_checkpoint
from learn_motion.config import load_config
from learn_motion.factory import build_dataset, build_learn_img_model, build_loader
from learn_motion.losses import masked_mse
from learn_motion.metrics import magnitude_psnr, magnitude_snr
from learn_motion.reproducibility import seed_everything


# 读取命令行参数.
# --resume: 指定一个 checkpoint 从已有训练状态继续.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# 是否开启 CUDA 的 bf16 混合精度训练.
def amp_context(device: torch.device, use_bf16: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 and device.type == "cuda" else nullcontext()


# 计算返回: loss, input_loss, psnr, snr. 验证集配对样本各层指标的平均.
# loss:       预测结果的掩膜复数 MSE.
# input_loss: 未校正输入的掩膜复数 MSE.
# psnr:       预测幅值图的 PSNR.
# snr:        参考与预测的幅值范数比.
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


# 生成一张固定切片的可视化对比图: best-validation.png, 验证指标刷新历史最佳值时同步刷新图像.
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


# 串联数据准备, 模型训练, 验证选优和结果保存.
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_config = config["training"]
    output_dir = Path(config.get("output_dir", "results/pytorch-learn-img"))

    # 新训练不能覆盖已有结果; 续训必须使用已存在的输出目录.
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}; choose a new directory or --resume")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"resume output directory does not exist: {output_dir}")

    # 固定随机种子, 构建训练集和验证集及对应的数据加载器.
    seed_everything(int(config.get("seed", 0)), deterministic=bool(config.get("deterministic", False)))
    device = torch.device(args.device)
    train_dataset = build_dataset(config, "train")
    valid_dataset = build_dataset(config, "valid")
    train_loader = build_loader(train_dataset, config, "train")
    valid_loader = build_loader(valid_dataset, config, "valid")
    if not len(train_loader) or not len(valid_loader):
        raise ValueError("training and validation loaders must be nonempty")

    # 根据配置创建模型和 Adam 优化器; 可选调度器根据验证损失降低学习率.
    model = build_learn_img_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.get("learning_rate", 1e-4)))
    scheduler_config = train_config.get("scheduler", {})
    scheduler = None
    if scheduler_config:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 8)), min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
    
    # 最佳权重和早停依据 monitor: loss 越低越好, psnr 越高越好.
    monitor = train_config.get("monitor", "psnr")
    if monitor not in {"loss", "psnr"}:
        raise ValueError("training.monitor must be 'loss' or 'psnr'")
    start_epoch = 0
    best_metric = float("inf") if monitor == "loss" else float("-inf")
    stale_epochs = 0

    # 续训时恢复模型, 优化器, 轮数和选优状态, 并核对关键配置.
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

    # 准备每轮训练所需的设置和追加写入的历史记录文件.
    output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = bool(train_config.get("bf16", False))
    epochs = int(train_config.get("epochs", 200))
    early_patience = train_config.get("early_stopping_patience")
    log_path = output_dir / "history.jsonl"
    print(f"device={device} train={len(train_dataset)} valid={len(valid_dataset)} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)

    for epoch in range(start_epoch, epochs):
        # 逐批前向计算掩膜 MSE, 反向传播并更新参数.
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

        # 每轮结束后在验证集上计算指标, 判断是否刷新历史最佳值.
        metrics = evaluate(model, valid_loader, device, use_bf16)
        selected = metrics["all"][monitor]
        improved = selected < best_metric if monitor == "loss" else selected > best_metric
        if improved:
            best_metric = selected
            stale_epochs = 0
        else:
            stale_epochs += 1
        
        # 保存完整验证指标和训练损失; 命令行只显示主要的总体指标.
        current_lr = optimizer.param_groups[0]["lr"]
        record = {"epoch": epoch + 1, "train_loss": train_total / train_count,
                  "learning_rate": current_lr, "validation": metrics,
                  "best_metric": best_metric, "best_epoch": improved}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"epoch={epoch + 1} train_loss={record['train_loss']:.6g} valid_loss={metrics['all']['loss']:.6g} "
              f"input_loss={metrics['all']['input_loss']:.6g} psnr={metrics['all']['psnr']:.4f} "
              f"lr={current_lr:.3g} best={best_metric:.6g}", flush=True)
        
        # 调度器始终观察验证 loss, 即使最佳权重由 psnr 选出.
        if scheduler is not None:
            scheduler.step(metrics["all"]["loss"])
        extra = {"scheduler": scheduler.state_dict() if scheduler is not None else None,
                 "stale_epochs": stale_epochs}
        
        # 最佳权重和预览图只在指标改善时更新; latest.pt 每轮更新以供续训.
        if improved:
            save_checkpoint(output_dir / f"best-{monitor}.pt", model, optimizer, epoch, config, best_metric, extra)
            if config.get("validation_preview"):
                save_preview(model, valid_dataset, device, config["validation_preview"], output_dir / "best-validation.png")
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, config, best_metric, extra)
        
        # 连续未改善的轮数达到耐心值后提前结束训练.
        if early_patience is not None and stale_epochs >= int(early_patience):
            print(f"early stopping after {stale_epochs} epochs without improvement", flush=True)
            break


if __name__ == "__main__":
    main()
