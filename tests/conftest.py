import os
import pytest
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption("--gen_dir", action="store", default=os.getenv("GEN_DIR", "gen_images"))
    parser.addoption("--real_dir", action="store", default=os.getenv("REAL_DIR", "real_images"))
    parser.addoption("--min_images", action="store", type=int, default=int(os.getenv("MIN_IMAGES", "1000")))
    parser.addoption("--expected_size", action="store", default=os.getenv("EXPECTED_SIZE", "32,32"))

@pytest.fixture
def gen_dir(request):
    return Path(request.config.getoption("--gen_dir"))

@pytest.fixture
def real_dir(request):
    return Path(request.config.getoption("--real_dir"))

@pytest.fixture
def min_images(request):
    return request.config.getoption("--min_images")

@pytest.fixture
def expected_size(request):
    w, h = request.config.getoption("--expected_size").split(",")
    return (int(w), int(h))
