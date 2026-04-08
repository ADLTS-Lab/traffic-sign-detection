from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_default_config_exists():
    assert Path("configs/default.yaml").exists()


def test_default_config_has_required_keys():
    cfg = load_yaml("configs/default.yaml")
    for key in ["project", "data", "model", "train", "export"]:
        assert key in cfg


def test_default_config_class_consistency():
    cfg = load_yaml("configs/default.yaml")
    class_names = cfg["model"]["class_names"]
    assert isinstance(class_names, list)
    assert len(class_names) == 50
