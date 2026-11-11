import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config
from ddos_tl.data.synthetic import generate


@pytest.fixture(scope="session")
def synthetic_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    src, tgt = root / "src_raw", root / "tgt_raw"
    generate(src, tgt, n_benign=600, n_attack=200, seed=0)
    return src, tgt


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/synthetic.yaml")
