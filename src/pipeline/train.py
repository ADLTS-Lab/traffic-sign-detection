from pathlib import Path
from ultralytics import YOLO
from src.utils.io import save_yaml


def build_data_yaml(config: dict) -> Path:
    root = config["data"]["root"]
    class_names = config["model"]["class_names"]
    payload = {
        "path": root,
        "train": config["data"]["train"],
        "val": config["data"]["val"],
        "test": config["data"]["test"],
        "nc": len(class_names),
        "names": class_names,
    }
    out = Path(config["project"]["output_dir"]) / "TT100K_data.yaml"
    return save_yaml(out, payload)


def train_model(config: dict):
    data_yaml_path = build_data_yaml(config)
    model = YOLO(config["model"]["weights"])
    tr = config["train"]
    return model.train(
        data=str(data_yaml_path),
        epochs=tr["epochs"],
        imgsz=tr["imgsz"],
        batch=tr["batch"],
        device=tr["device"],
        workers=tr["workers"],
        project=tr["project"],
        name=tr["name"],
        exist_ok=tr["exist_ok"],
    )


def export_onnx(config: dict):
    tr = config["train"]
    weights = Path(tr["project"]) / tr["name"] / "weights" / "best.pt"
    model = YOLO(str(weights))
    ex = config["export"]
    return model.export(format=ex["format"], imgsz=ex["imgsz"])
