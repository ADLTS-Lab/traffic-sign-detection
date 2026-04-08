import argparse
from src.utils.io import load_yaml
from src.pipeline.train import train_model, export_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="TT100K YOLOv8 pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "export", "all"],
        help="Pipeline step to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.mode in ("train", "all"):
        print("[INFO] Training started...")
        train_model(cfg)
        print("[INFO] Training finished.")

    if args.mode in ("export", "all"):
        print("[INFO] ONNX export started...")
        export_onnx(cfg)
        print("[INFO] ONNX export finished.")


if __name__ == "__main__":
    main()
