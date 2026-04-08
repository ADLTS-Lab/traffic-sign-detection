from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml
from src.pipeline.train import build_data_yaml


def test_default_config_exists():
    assert Path("configs/default.yaml").exists()


def test_default_config_has_required_keys():
    cfg = load_yaml("configs/default.yaml")
    for key in ["project", "data", "model", "train", "export"]:
        assert key in cfg


def test_build_data_yaml_creates_file():
    cfg = load_yaml("configs/default.yaml")
    out = build_data_yaml(cfg)
    assert out.exists()
    payload = load_yaml(out)
    assert payload["nc"] == len(payload["names"])
