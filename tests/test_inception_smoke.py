import math
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]

def copy_subset(src: Path, dst: Path, n: int):
    dst.mkdir(parents=True, exist_ok=True)
    imgs = list_images(src)
    pick = random.sample(imgs, min(n, len(imgs)))
    for p in pick:
        shutil.copy2(p, dst / p.name)

def test_inception_score_smoke(gen_dir, tmp_path):
    """
    Smoke test: Inception Score should run without errors and produce a finite score.
    This is NOT the reportable score; it just validates the pipeline.
    """
    try:
        from torch_fidelity import calculate_metrics
    except Exception:
        import pytest
        pytest.skip("torch-fidelity not installed")

    gen_small = tmp_path / "gen_small"
    copy_subset(gen_dir, gen_small, n=128)

    metrics = calculate_metrics(
        input1=str(gen_small),     # generated images folder
        isc=True,                  # Inception Score
        fid=False,
        kid=False,
        prc=False,
        cuda=False                 # keep it robust in CI; set True locally if you want
    )

    # torch-fidelity returns keys like 'inception_score_mean' and 'inception_score_std'
    is_mean = float(metrics.get("inception_score_mean", float("nan")))
    is_std  = float(metrics.get("inception_score_std", float("nan")))

    assert math.isfinite(is_mean), f"IS mean not finite: {is_mean}"
    assert math.isfinite(is_std), f"IS std not finite: {is_std}"
    assert is_mean > 1.0, f"IS mean unexpectedly <= 1.0: {is_mean}"
