#!/usr/bin/env python3
"""
compute_fid.py

Compute FID (Fréchet Inception Distance) between two folders of images:
- --real: folder with real images
- --fake: folder with generated images

Assumes images are already prepared for scoring (typically RGB, 299x299).
The script still converts to RGB + tensor in [0,1] for safety.

Usage:
  python compute_fid.py --real ./real_299 --fake ./fake_299 --batch_size 64

Notes:
- FID is most stable with >= 10k images per set.
- Requires: pip install torchmetrics torchvision
"""

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError as e:
    raise SystemExit(
        "Missing dependency: torchmetrics\n"
        "Install with: pip install torchmetrics torchvision"
    ) from e


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def update_from_folder(fid_metric: FrechetInceptionDistance, folder: Path, *, real: bool, batch_size: int, device: torch.device):
    img_paths = list_images(folder)
    if not img_paths:
        raise SystemExit(f"No images found in: {folder}")

    to_tensor = transforms.ToTensor()
    batch = []

    for p in img_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")      # safety: ensure 3 channels
            x = to_tensor(im)           # (3,H,W) in [0,1]
            batch.append(x)

        if len(batch) == batch_size:
            x_batch = torch.stack(batch, dim=0).to(device)  # (B,3,H,W)
            fid_metric.update(x_batch, real=real)
            batch.clear()

    if batch:
        x_batch = torch.stack(batch, dim=0).to(device)
        fid_metric.update(x_batch, real=real)

    return len(img_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID between two image folders.")
    parser.add_argument("--real", required=True, help="Folder containing real images (prepared RGB 299x299 recommended).")
    parser.add_argument("--fake", required=True, help="Folder containing generated images (prepared RGB 299x299 recommended).")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size for metric updates (default: 64).")
    parser.add_argument("--feature", type=int, default=2048, help="Inception feature dim (default: 2048).")
    parser.add_argument("--device", "-d", default=None, help='Device, e.g. "cuda", "cpu". Default: auto.')
    args = parser.parse_args()

    real_dir = Path(args.real)
    fake_dir = Path(args.fake)

    if not real_dir.exists() or not real_dir.is_dir():
        raise SystemExit(f"--real is not a directory: {real_dir}")
    if not fake_dir.exists() or not fake_dir.is_dir():
        raise SystemExit(f"--fake is not a directory: {fake_dir}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fid = FrechetInceptionDistance(feature=args.feature, normalize=True).to(device)

    n_real = update_from_folder(fid, real_dir, real=True, batch_size=args.batch_size, device=device)
    n_fake = update_from_folder(fid, fake_dir, real=False, batch_size=args.batch_size, device=device)

    score = fid.compute()
    print(f"FID: {float(score):.4f}")
    print(f"Real images: {n_real} | Fake images: {n_fake} | Feature: {args.feature} | Device: {device}")


if __name__ == "__main__":
    main()
