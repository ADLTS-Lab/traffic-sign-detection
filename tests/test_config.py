from pathlib import Path
from src.utils.io import load_yaml


def test_default_config_exists():
    assert Path("configs/default.yaml").exists()


def test_default_config_has_required_keys():
    cfg = load_yaml("configs/default.yaml")
    for key in ["project", "data", "model", "train", "export"]:
        assert key in cfg
