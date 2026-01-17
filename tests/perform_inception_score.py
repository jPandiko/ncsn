#!/usr/bin/env python3
"""
compute_is.py

Compute Inception Score (IS) for a folder of images that are already prepared
(e.g., RGB PNGs sized to 299x299 by your earlier script).

Usage:
  python perform_inception_score.py --folder ./samples_299 --batch_size 64 --splits 10

Notes:
- IS is typically computed on >= 5k images for a stable estimate.
- This script expects images readable by PIL (png/jpg/jpeg/webp/bmp/tiff).
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

try:
    from torchmetrics.image.inception import InceptionScore
except ImportError as e:
    raise SystemExit(
        "Missing dependency: torchmetrics\n"
        "Install with: pip install torchmetrics torchvision"
    ) from e


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Inception Score (IS) for a folder of images.")
    parser.add_argument("--folder", "-f", required=True, help="Folder containing images (prepared RGB 299x299 recommended).")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size for metric updates (default: 64).")
    parser.add_argument("--splits", "-s", type=int, default=10, help="Number of splits for IS (default: 10).")
    parser.add_argument("--device", "-d", default=None, help='Device, e.g. "cuda", "cpu". Default: auto.')
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {folder}")

    img_paths = list_images(folder)
    if not img_paths:
        raise SystemExit(f"No images found in: {folder}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CHANGE: we still convert to RGB and ToTensor for safety, even if already prepared
    tfm = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),  # no-op for ToTensor output, safe
    ])

    # PIL -> Tensor in [0,1], shape (C,H,W)
    to_tensor = transforms.ToTensor()

    # Torchmetrics InceptionScore expects images in [0,1] float, RGB, and internally normalizes if normalize=True
    is_metric = InceptionScore(splits=args.splits, normalize=True).to(device)

    batch = []
    for p in img_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")  # safety: ensure 3 channels
            x = to_tensor(im)       # float in [0,1], (3,H,W)
            x = tfm(x)
            batch.append(x)

        if len(batch) == args.batch_size:
            x_batch = torch.stack(batch, dim=0).to(device)  # (B,3,H,W)
            is_metric.update(x_batch)
            batch.clear()

    if batch:
        x_batch = torch.stack(batch, dim=0).to(device)
        is_metric.update(x_batch)

    mean, std = is_metric.compute()
    print(f"Inception Score (IS): {float(mean):.4f} ± {float(std):.4f}")
    print(f"Images: {len(img_paths)} | Splits: {args.splits} | Device: {device}")


if __name__ == "__main__":
    main()
