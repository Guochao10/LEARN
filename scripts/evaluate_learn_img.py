"""Evaluate a LEARN-IMG checkpoint on paired HDF5 test data in one pass."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from learn_motion.checkpoint import load_checkpoint
from learn_motion.config import load_config
from learn_motion.factory import build_dataset, build_learn_img_model, build_loader
from learn_motion.metrics import magnitude_psnr, magnitude_snr


METRICS = (
    "input_complex_mse", "prediction_complex_mse",
    "input_psnr_db", "prediction_psnr_db",
    "input_snr_db", "prediction_snr_db",
)
IDENTIFIERS = ("pair_id", "subject", "motion_case")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preview-slice", type=int, default=141, help="One-based anatomical slice number")
    parser.add_argument("--no-previews", action="store_true")
    return parser.parse_args()


def sample_metrics(input_image: torch.Tensor, prediction: torch.Tensor,
                   reference: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    def complex_mse(image: torch.Tensor) -> float:
        return F.mse_loss(image.float() * mask, reference * mask).item()

    return {
        "input_complex_mse": complex_mse(input_image),
        "prediction_complex_mse": complex_mse(prediction),
        "input_psnr_db": magnitude_psnr(input_image[None], reference[None]).item(),
        "prediction_psnr_db": magnitude_psnr(prediction[None], reference[None]).item(),
        "input_snr_db": magnitude_snr(input_image[None], reference[None]).item(),
        "prediction_snr_db": magnitude_snr(prediction[None], reference[None]).item(),
    }


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {name: float(np.mean([row[name] for row in rows])) for name in METRICS}


def summarize(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [{key: value, "pairs": len(group), "slices": sum(row["slices"] for row in group),
             **mean_metrics(group)} for value, group in sorted(groups.items())]


def save_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_preview(path: Path, images: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                 pair_id: str, slice_number: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = [image.detach().float().cpu().numpy() for image in images]
    echoes = arrays[0].shape[1]
    fig, axes = plt.subplots(echoes, 3, figsize=(10, 3 * echoes), squeeze=False,
                             constrained_layout=True)
    for echo in range(echoes):
        reference_magnitude = np.hypot(arrays[2][0, echo], arrays[2][1, echo])
        limit = float(np.percentile(reference_magnitude, 99.5))
        for col, title in enumerate(("Corrupted", "Prediction", "Reference")):
            magnitude = np.hypot(arrays[col][0, echo], arrays[col][1, echo])
            axes[echo, col].imshow(np.rot90(magnitude), cmap="gray", vmin=0,
                                   vmax=max(limit, 1e-12))
            axes[echo, col].set_title(f"Echo {echo + 1} | {title}", fontsize=10)
            axes[echo, col].axis("off")
    fig.suptitle(f"{pair_id} | Slice {slice_number}", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config["data"].get("format") != "h5" or not config["data"].get("shared_input_scale"):
        raise ValueError("evaluation requires paired HDF5 data with shared_input_scale: true")
    dataset = build_dataset(config, "test")
    if not dataset.start < args.preview_slice <= dataset.stop and not args.no_previews:
        raise ValueError(f"preview slice must be between {dataset.start + 1} and {dataset.stop}")
    loader = build_loader(dataset, config, "test")
    device = torch.device(args.device)
    model = build_learn_img_model(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, map_location=device)
    for section in ("model", "data"):
        if checkpoint["config"].get(section) != config.get(section):
            raise ValueError(f"checkpoint and evaluation config differ in {section}")
    model.eval()

    output_dir = args.output_dir or Path(config["output_dir"]) / "test-evaluation"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if not args.no_previews:
        preview_dir.mkdir()

    rows: list[dict] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate test"):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            predictions = model(inputs).float()
            for i in range(len(inputs)):
                identifiers = {key: batch[key][i] for key in IDENTIFIERS}
                slice_index = int(batch["slice_index"][i])
                row = {**identifiers, "slice_number": slice_index + 1,
                       **sample_metrics(inputs[i], predictions[i], targets[i], masks[i])}
                rows.append(row)
                if not args.no_previews and slice_index + 1 == args.preview_slice:
                    save_preview(preview_dir / f"{identifiers['pair_id']}_slice{args.preview_slice}.png",
                                 (inputs[i], predictions[i], targets[i]),
                                 identifiers["pair_id"], args.preview_slice)

    if not rows:
        raise ValueError("test dataset is empty")
    per_pair = [{**{key: group[0][key] for key in IDENTIFIERS}, "slices": len(group),
                 **mean_metrics(group)}
                for _, group in sorted(_group_rows(rows, "pair_id").items())]
    per_subject = summarize(per_pair, "subject")
    per_motion = summarize(per_pair, "motion_case")
    overall = mean_metrics(per_pair)
    overall["pairs"] = len(per_pair)
    overall["slices"] = len(rows)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "split": "test",
        "normalization": config["data"]["normalization"],
        "metric_note": "Complex MSE uses the training mask; magnitude PSNR uses fixed peak 1 on normalized data; SNR is target/error norm.",
        "overall": overall,
        "by_subject": per_subject,
        "by_motion_case": per_motion,
    }
    save_csv(output_dir / "per_slice.csv", rows,
             (*IDENTIFIERS, "slice_number", *METRICS))
    save_csv(output_dir / "per_pair.csv", per_pair,
             (*IDENTIFIERS, "slices", *METRICS))
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    before = overall["input_complex_mse"]
    after = overall["prediction_complex_mse"]
    print(f"checkpoint epoch={report['checkpoint_epoch']} test pairs={len(per_pair)} slices={len(rows)}")
    reduction = (before - after) / before * 100 if before > 0 else float("nan")
    print(f"complex MSE: input={before:.6g} prediction={after:.6g} reduction={reduction:.2f}%")
    print(f"saved {output_dir / 'summary.json'} and CSV reports")


def _group_rows(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return groups


if __name__ == "__main__":
    main()
