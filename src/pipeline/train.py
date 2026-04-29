from pathlib import Path
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


def validate_data_layout(config: dict) -> None:
    root = Path(config["data"]["root"])
    required = [
        root / config["data"]["train"],
        root / config["data"]["val"],
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset structure check failed. Missing paths:\n- " + "\n- ".join(missing)
        )


def train_model(config: dict):
    from ultralytics import YOLO

    validate_data_layout(config)
    data_yaml_path = build_data_yaml(config)
    tr = config["train"]
    run_dir = Path(tr["project"]) / tr["name"]
    last_weights = run_dir / "weights" / "last.pt"

    if last_weights.exists():
        model = YOLO(str(last_weights))
        model.train(resume=True)
    else:
        model = YOLO(config["model"]["weights"])
        model.train(
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
    return run_dir / "weights" / "best.pt"


def export_onnx(config: dict):
    from ultralytics import YOLO

    tr = config["train"]
    weights = Path(tr["project"]) / tr["name"] / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Best weights not found: {weights}")
    model = YOLO(str(weights))
    ex = config["export"]
    return model.export(format=ex["format"], imgsz=ex["imgsz"])
