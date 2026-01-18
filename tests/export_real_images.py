#!/usr/bin/env python3
"""
export_real_images.py

Download MNIST and/or CIFAR-10 via torchvision and save N images to disk as PNGs.

Default behavior:
- Uses the TEST split (recommended for FID real set)
- Saves exactly 10,000 images (or fewer if dataset is smaller)
- Writes sequential filenames: img_000000.png, img_000001.png, ...

Examples:
  # Save 10k MNIST test images
  python export_real_images.py --dataset mnist --out ./real_mnist --count 10000

  # Save 10k CIFAR-10 test images
  python export_real_images.py --dataset cifar10 --out ./real_cifar10 --count 10000

  # Save both
  python export_real_images.py --dataset both --out ./real --count 10000

Notes:
- MNIST test has 10,000 images -> exact.
- CIFAR-10 test has 10,000 images -> exact.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


def export_dataset(dataset_name: str,out_dir: Path,count: int,train: bool,root: Path,seed: int) -> Tuple[int, int]:
    """
    Returns (saved_count, total_available).
    Saves images as PNG, with values expected in [0,1] (ToTensor does that).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use ToTensor to ensure tensors in [0,1], shape (C,H,W)
    tfm = ToTensor()

    if dataset_name == "mnist":
        ds = MNIST(root=str(root / "mnist"), train=train, download=True, transform=tfm)
    elif dataset_name == "cifar10":
        ds = CIFAR10(root=str(root / "cifar10"), train=train, download=True, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    total = len(ds)
    n = min(count, total)

    # Deterministic selection (shuffle indices with seed)
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(total, generator=g)[:n].tolist()

    for i, idx in enumerate(idxs):
        x, _ = ds[idx]  # x is Tensor (C,H,W) in [0,1]
        save_image(x, out_dir / f"img_{i:06d}.png")

    return n, total


def main() -> None:
    ap = argparse.ArgumentParser(description="Export MNIST/CIFAR10 images to PNG files.")
    ap.add_argument("--dataset", choices=["mnist", "cifar10", "both"], default="both",
                    help="Which dataset to export (default: both).")
    ap.add_argument("--out", required=True,
                    help="Output directory. If --dataset=both, creates subfolders mnist/ and cifar10/.")
    ap.add_argument("--count", type=int, default=10000,
                    help="Number of images to save (default: 10000).")
    ap.add_argument("--train", action="store_true",
                    help="Use TRAIN split instead of TEST split (default: test).")
    ap.add_argument("--root", default="./datasets",
                    help="Where to download/store torchvision datasets (default: ./datasets).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed used to choose which images to export (default: 0).")
    args = ap.parse_args()

    out_base = Path(args.out)
    root = Path(args.root)

    if args.dataset in ("mnist", "both"):
        out_dir = out_base / ("mnist" if args.dataset == "both" else "")
        saved, total = export_dataset(
            "mnist",
            out_dir if out_dir.name else out_base,
            count=args.count,
            train=args.train,
            root=root,
            seed=args.seed,
        )
        split = "train" if args.train else "test"
        print(f"[OK] MNIST {split}: saved {saved}/{total} images to {out_dir if out_dir.name else out_base}")

    if args.dataset in ("cifar10", "both"):
        out_dir = out_base / ("cifar10" if args.dataset == "both" else "")
        saved, total = export_dataset(
            "cifar10",
            out_dir if out_dir.name else out_base,
            count=args.count,
            train=args.train,
            root=root,
            seed=args.seed,
        )
        split = "train" if args.train else "test"
        print(f"[OK] CIFAR-10 {split}: saved {saved}/{total} images to {out_dir if out_dir.name else out_base}")


if __name__ == "__main__":
    main()
