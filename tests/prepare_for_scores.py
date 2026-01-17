#!/usr/bin/env python3
"""
prepare_for_scores.py

Convert all images in a folder to an "Inception-ready" format for IS/FID:
- RGB (3 channels)
- 299x299
- PNG output
- sequential filenames

Example:
  python prepare_for_scores.py --input ./samples --output ./samples_299 --size 299
"""

import argparse
import os
from pathlib import Path
from PIL import Image


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


def convert_folder(input_dir: Path, output_dir: Path, size: int, overwrite: bool) -> None:
    input_files = sorted([p for p in input_dir.iterdir() if is_image_file(p)])

    if not input_files:
        raise SystemExit(f"No supported images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: We keep conversion deterministic by sorting filenames and writing sequential outputs.
    out_ext = ".png"

    for idx, src in enumerate(input_files):
        dst = output_dir / f"img_{idx:06d}{out_ext}"

        if dst.exists() and not overwrite:
            continue

        try:
            with Image.open(src) as im:
                # CHANGE: Ensure 3-channel RGB (Inception expects RGB)
                im = im.convert("RGB")

                # CHANGE: Resize to requested size (default 299x299)
                # Using BICUBIC is typical for resizing images for Inception/FID.
                im = im.resize((size, size), resample=Image.BICUBIC)

                # CHANGE: Save as PNG for lossless, consistent decoding
                im.save(dst, format="PNG", optimize=True)

        except Exception as e:
            print(f"[WARN] Skipping {src.name}: {e}")

    print(f"[OK] Converted images written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert images in a folder to RGB + fixed size for IS/FID.")
    parser.add_argument("--input", "-i", required=True, help="Input folder containing images.")
    parser.add_argument("--output", "-o", required=True, help="Output folder to write converted PNGs.")
    parser.add_argument("--size", "-s", type=int, default=299, help="Output image size (default: 299).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input path is not a directory: {input_dir}")

    convert_folder(input_dir=input_dir, output_dir=output_dir, size=args.size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
