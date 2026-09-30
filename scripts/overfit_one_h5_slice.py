"""Check whether LEARN-IMG can fit one real Radial motion-corrupted slice.

Run from the LEARN repository root:
    python scripts/overfit_one_h5_slice.py --steps 100 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from learn_motion.config import load_config
from learn_motion.factory import build_dataset, build_learn_img_model
from learn_motion.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hn_radial_learn_img.yaml")
    parser.add_argument("--pair-index", type=int, default=0, help="Zero-based manifest pair index within train")
    parser.add_argument("--slice-number", type=int, default=141, help="One-based anatomical slice number")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def magnitude(tensor: torch.Tensor) -> np.ndarray:
    channels = tensor.detach().float().cpu().numpy()[0]
    return np.hypot(channels[0], channels[1])


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


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 0)), deterministic=bool(config.get("deterministic", False)))
    dataset = build_dataset(config, "train")
    if not 0 <= args.pair_index < len(dataset.pairs):
        raise ValueError(f"--pair-index must be between 0 and {len(dataset.pairs) - 1}")
    z = args.slice_number - 1
    if not dataset.start <= z < dataset.stop:
        raise ValueError(f"--slice-number must be between {dataset.start + 1} and {dataset.stop}")

    item = args.pair_index * (dataset.stop - dataset.start) + z - dataset.start
    sample = dataset[item]
    device = torch.device(args.device)
    source = sample["input"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    model = build_learn_img_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))

    model.eval()
    with torch.inference_mode():
        before = model(source)
        initial_loss = F.mse_loss(before * mask, target * mask).item()

    losses: list[float] = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(source)
        loss = F.mse_loss(prediction * mask, target * mask)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step {step:3d}/{args.steps}: loss={losses[-1]:.6g}", flush=True)

    model.eval()
    with torch.inference_mode():
        after = model(source)
        final_loss = F.mse_loss(after * mask, target * mask).item()

    output_dir = args.output_dir or Path(config["output_dir"]) / "overfit-check"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.pairs[args.pair_index]['pair_id']}_slice{args.slice_number}"
    image_path = output_dir / f"{stem}.png"
    history_path = output_dir / f"{stem}_loss.json"
    save_comparison(image_path, source, before, after, target)
    history_path.write_text(json.dumps({
        "pair_id": dataset.pairs[args.pair_index]["pair_id"],
        "slice_number": args.slice_number,
        "slice_index": z,
        "steps": args.steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "losses": losses,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"initial_loss={initial_loss:.6g} final_loss={final_loss:.6g}")
    print(f"saved {image_path}")
    print(f"saved {history_path}")


if __name__ == "__main__":
    main()
