import argparse
from src.utils.io import load_yaml
from src.pipeline.train import export_onnx


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    export_onnx(cfg)
    print("ONNX export complete.")


if __name__ == "__main__":
    main()
