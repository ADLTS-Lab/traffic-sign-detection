from pathlib import Path
from src.pipeline.train import build_data_yaml


def get_best_weights(config: dict) -> Path:
    tr = config["train"]
    return Path(tr["project"]) / tr["name"] / "weights" / "best.pt"


def validate_model(config: dict):
    from ultralytics import YOLO

    best = get_best_weights(config)
    if not best.exists():
        raise FileNotFoundError(f"Best weights not found: {best}")
    data_yaml = build_data_yaml(config)
    model = YOLO(str(best))
    return model.val(data=str(data_yaml), split="val")


def predict_one_image(config: dict, image_path: str, conf: float = 0.25):
    from ultralytics import YOLO

    best = get_best_weights(config)
    if not best.exists():
        raise FileNotFoundError(f"Best weights not found: {best}")
    model = YOLO(str(best))
    return model.predict(source=image_path, conf=conf, imgsz=config["train"]["imgsz"])
