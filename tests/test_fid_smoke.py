import random
import shutil
from pathlib import Path

def copy_subset(src: Path, dst: Path, n: int):
    dst.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    pick = random.sample(imgs, min(n, len(imgs)))
    for p in pick:
        shutil.copy2(p, dst / p.name)

def test_fid_smoke(gen_dir, real_dir, expected_size, tmp_path):
    # Skip if clean-fid isn't installed
    try:
        from cleanfid import fid
    except Exception:
        import pytest
        pytest.skip("clean-fid not installed")

    gen_small = tmp_path / "gen_small"
    real_small = tmp_path / "real_small"
    copy_subset(gen_dir, gen_small, n=128)
    copy_subset(real_dir, real_small, n=128)

    # If this runs without error, your FID pipeline is basically wired correctly.
    score = fid.compute_fid(str(real_small), str(gen_small), dataset_res=expected_size[0], batch_size=32, device="cpu")
    assert score >= 0.0
