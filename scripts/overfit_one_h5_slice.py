"""Check whether LEARN-IMG can fit one real Radial motion-corrupted slice.

Run from the LEARN repository root with a masked residual config and a training pair.
"""

# 单切片过拟合诊断脚本. 判断当前模型和优化过程是否能拟合一张已知切片.

# 一个例子:
# cd /data/home/gcjiang3/qsm/LEARN
# CUDA_VISIBLE_DEVICES=0 /data1/gcjiang3/envs/LEARN/bin/python \
#   scripts/overfit_one_h5_slice.py \
#   --config configs/hn_radial_learn_img_residual_masked.yaml \
#   --pair-id sub-01_ses-01_m01_rl_step_moderate_seed10101 \
#   --slice-number 141 \
#   --steps 1000 \
#   --device cuda:0 \
#   --output-dir results/hn-radial-learn-img-residual-masked/overfit-check-sub01-m01-slice141

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from learn_motion.config import load_config
from learn_motion.factory import build_dataset, build_learn_img_model
from learn_motion.losses import masked_mse
from learn_motion.reproducibility import seed_everything


# 选择训练集中的一个运动配对和解剖切片, 并设置优化步数及输出位置.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hn_radial_learn_img_residual_masked.yaml")
    pair = parser.add_mutually_exclusive_group()
    pair.add_argument("--pair-id", help="Exact pair_id in the training manifest")
    pair.add_argument("--pair-index", type=int, help="Zero-based pair index within train")
    parser.add_argument("--slice-number", type=int, default=141, help="One-based anatomical slice number")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


# 将四回波复数图像的实部和虚部转换为幅值图, 用于绘制对比图.
def magnitude(tensor: torch.Tensor) -> np.ndarray:
    channels = tensor.detach().float().cpu().numpy()[0]
    return np.hypot(channels[0], channels[1])


# 只比较掩膜内相邻像素的幅值梯度, 作为观察图像细节的诊断指标.
def masked_magnitude_gradient_mae(image: torch.Tensor, target: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
    """Mean magnitude-gradient error over neighboring pixels inside the mask."""
    image_mag = torch.linalg.vector_norm(image.float(), dim=1)
    target_mag = torch.linalg.vector_norm(target.float(), dim=1)
    region = mask[:, 0, 0].bool()
    # 一对相邻像素都在脑掩膜内时, 才计入水平或垂直梯度误差.
    dx_mask = (region[..., 1:] & region[..., :-1])[:, None]
    dy_mask = (region[..., 1:, :] & region[..., :-1, :])[:, None]
    dx_error = ((image_mag[..., 1:] - image_mag[..., :-1]) -
                (target_mag[..., 1:] - target_mag[..., :-1])).abs()
    dy_error = ((image_mag[..., 1:, :] - image_mag[..., :-1, :]) -
                (target_mag[..., 1:, :] - target_mag[..., :-1, :])).abs()
    numerator = (dx_error * dx_mask).sum() + (dy_error * dy_mask).sum()
    echoes = image_mag.shape[1]
    denominator = (dx_mask.sum() + dy_mask.sum()).clamp_min(1) * echoes
    return numerator / denominator


# 每个回波绘制一行, 对比污染输入, 训练前预测, 训练后预测和参考图像.
def save_comparison(
    path: Path,
    source: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
    target: torch.Tensor,
) -> None:
    columns = [magnitude(tensor) for tensor in (source, before, after, target)]
    names = ("Corrupted input", "Before training", "After training", "Reference")
    num_echoes = columns[0].shape[0]
    fig, axes = plt.subplots(num_echoes, 4, figsize=(12, 3 * num_echoes), squeeze=False)
    for echo in range(num_echoes):
        vmax = float(np.percentile(columns[-1][echo], 99.5))
        for col, (name, volume) in enumerate(zip(names, columns, strict=True)):
            ax = axes[echo, col]
            ax.imshow(np.rot90(volume[echo]), cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(f"Echo {echo + 1} | {name}", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 绘制同一切片在每个优化步骤的掩膜复数 MSE.
def save_loss_curve(path: Path, losses: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set(xlabel="Optimization step", ylabel="Brain-mask complex MSE")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# 从训练集抽取一张切片反复拟合, 并保存数值和图像诊断结果.
def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    config = load_config(args.config)

    # 该诊断依赖 HDF5 配对数据和参考图像的脑掩膜.
    if config["data"].get("format") != "h5" or not config["data"].get("use_brain_mask"):
        raise ValueError("diagnostic requires HDF5 data with use_brain_mask: true")

    # 已有结果目录非空时直接报错, 防止覆盖先前的诊断结果.
    output_dir = args.output_dir or Path(config["output_dir"]) / "overfit-check"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"diagnostic output directory is not empty: {output_dir}")
    seed_everything(int(config.get("seed", 0)), deterministic=bool(config.get("deterministic", False)))
    dataset = build_dataset(config, "train")

    # 优先按完整 pair_id 查找; 否则使用训练集中的案例索引, 默认第 0 个.
    if args.pair_id is not None:
        pair_index = next((i for i, pair in enumerate(dataset.pairs)
                           if pair["pair_id"] == args.pair_id), None)
        if pair_index is None:
            raise ValueError(f"pair_id is absent from training split: {args.pair_id}")
    else:
        pair_index = args.pair_index if args.pair_index is not None else 0
        if not 0 <= pair_index < len(dataset.pairs):
            raise ValueError(f"--pair-index must be between 0 and {len(dataset.pairs) - 1}")

    # 命令行使用从 1 开始的解剖层号, 数据集使用从 0 开始的索引.
    z = args.slice_number - 1
    if not dataset.start <= z < dataset.stop:
        raise ValueError(f"--slice-number must be between {dataset.start + 1} and {dataset.stop}")

    # 只取所选运动配对的同一层污染图像, 参考图像和脑掩膜.
    item = pair_index * (dataset.stop - dataset.start) + z - dataset.start
    sample = dataset[item]
    device = torch.device(args.device)
    source = sample["input"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    model = build_learn_img_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    # 记录原始输入和随机初始化模型的误差, 作为训练后的比较基线.
    model.eval()
    with torch.inference_mode():
        before = model(source)
        input_loss = masked_mse(source, target, mask).item()
        initial_loss = masked_mse(before, target, mask).item()
        input_gradient_error = masked_magnitude_gradient_mae(source, target, mask).item()
        initial_gradient_error = masked_magnitude_gradient_mae(before, target, mask).item()

    # 每一步重复使用同一张切片; 仅用掩膜复数 MSE 更新模型参数.
    losses: list[float] = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(source)
        loss = masked_mse(prediction, target, mask)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step {step:3d}/{args.steps}: loss={losses[-1]:.6g}", flush=True)

    # 训练结束后重新推理, 同时检查复数 MSE 和幅值梯度误差.
    model.eval()
    with torch.inference_mode():
        after = model(source)
        final_loss = masked_mse(after, target, mask).item()
        final_gradient_error = masked_magnitude_gradient_mae(after, target, mask).item()

    # 保存四回波对比图, 损失曲线及每一步的数值记录.
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.pairs[pair_index]['pair_id']}_slice{args.slice_number}"
    image_path = output_dir / f"{stem}.png"
    curve_path = output_dir / f"{stem}_loss.png"
    history_path = output_dir / f"{stem}_loss.json"
    save_comparison(image_path, source, before, after, target)
    save_loss_curve(curve_path, losses)
    history_path.write_text(json.dumps({
        "pair_id": dataset.pairs[pair_index]["pair_id"],
        "slice_number": args.slice_number,
        "slice_index": z,
        "steps": args.steps,
        "input_loss": input_loss,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "input_gradient_mae": input_gradient_error,
        "initial_gradient_mae": initial_gradient_error,
        "final_gradient_mae": final_gradient_error,
        "losses": losses,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"input_loss={input_loss:.6g} initial_loss={initial_loss:.6g} final_loss={final_loss:.6g}")
    print(f"magnitude_gradient_mae: input={input_gradient_error:.6g} "
          f"initial={initial_gradient_error:.6g} final={final_gradient_error:.6g}")
    print(f"saved {image_path}")
    print(f"saved {curve_path}")
    print(f"saved {history_path}")


if __name__ == "__main__":
    main()
