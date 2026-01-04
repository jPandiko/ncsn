from pathlib import Path
from PIL import Image
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]

def load_image(path: Path):
    # Robust load; fails fast on corruption
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im

def test_gen_dir_exists_and_nonempty(gen_dir):
    assert gen_dir.exists(), f"Generated dir not found: {gen_dir}"
    imgs = list_images(gen_dir)
    assert len(imgs) > 0, f"No images found in {gen_dir}"

def test_real_dir_exists_and_nonempty(real_dir):
    assert real_dir.exists(), f"Real dir not found: {real_dir}"
    imgs = list_images(real_dir)
    assert len(imgs) > 0, f"No images found in {real_dir}"

def test_gen_has_enough_images(gen_dir, min_images):
    imgs = list_images(gen_dir)
    assert len(imgs) >= min_images, f"Too few generated images: {len(imgs)} < {min_images}"

def test_real_has_enough_images(real_dir, min_images):
    imgs = list_images(real_dir)
    assert len(imgs) >= min_images, f"Too few real images: {len(imgs)} < {min_images}"

def test_gen_images_readable(gen_dir):
    imgs = list_images(gen_dir)[:200]  # sample to keep test quick
    bad = []
    for p in imgs:
        try:
            _ = load_image(p)
        except Exception as e:
            bad.append((p, str(e)))
    assert not bad, f"Unreadable/corrupt generated images (showing up to 5): {bad[:5]}"

def test_real_images_readable(real_dir):
    imgs = list_images(real_dir)[:200]
    bad = []
    for p in imgs:
        try:
            _ = load_image(p)
        except Exception as e:
            bad.append((p, str(e)))
    assert not bad, f"Unreadable/corrupt real images (showing up to 5): {bad[:5]}"

def test_sizes_match_expected(gen_dir, expected_size):
    imgs = list_images(gen_dir)[:200]
    mismatches = []
    for p in imgs:
        im = load_image(p)
        if im.size != expected_size:
            mismatches.append((p.name, im.size))
    assert not mismatches, f"Generated image size mismatches (up to 10): {mismatches[:10]}"

def test_pixel_range_sane(gen_dir):
    # Catch weird NaN/inf-like artifacts after bad conversions (rare but useful)
    imgs = list_images(gen_dir)[:50]
    for p in imgs:
        im = load_image(p)
        arr = np.asarray(im)
        assert arr.dtype == np.uint8, f"{p.name}: expected uint8 PNG/JPG decode, got {arr.dtype}"
        assert arr.min() >= 0 and arr.max() <= 255, f"{p.name}: pixel range out of bounds"
