"""Run slice-wise LEARN-IMG inference and save complex MATLAB volumes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm

from learn_motion.checkpoint import load_checkpoint
from learn_motion.config import load_config
from learn_motion.data.complex import channels_to_complex
from learn_motion.factory import build_dataset, build_learn_img_model, build_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--output-dir", default="results/pytorch-learn-img/inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    dataset = build_dataset(config, args.split)
    loader = build_loader(dataset, config, args.split)
    model = build_learn_img_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    predictions: dict[str, dict[int, tuple[np.ndarray, float, int]]] = defaultdict(dict)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="infer"):
            output = model(batch["input"].to(device, non_blocking=True))
            complex_batch = channels_to_complex(output)
            for index in range(complex_batch.shape[0]):
                path = batch["motion_path"][index]
                slice_index = int(batch["slice_index"][index])
                scale = float(batch["input_scale"][index])
                original_width = int(batch["original_width"][index])
                predictions[path][slice_index] = (complex_batch[index], scale, original_width)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for motion_path, slices in predictions.items():
        ordered_indices = sorted(slices)
        restored = []
        for slice_index in ordered_indices:
            echo_height_width, scale, original_width = slices[slice_index]
            image = echo_height_width.transpose(1, 2, 0) * scale
            restored.append(image[:, :original_width, :])
        volume = np.stack(restored, axis=2)
        destination = output_dir / f"{Path(motion_path).stem}_learn_img.mat"
        savemat(
            destination,
            {
                "ima_comb": volume,
                "slice_indices_zero_based": np.asarray(ordered_indices, dtype=np.int32),
            },
            do_compression=True,
        )
        print(f"saved {destination} {volume.shape}")


if __name__ == "__main__":
    main()
