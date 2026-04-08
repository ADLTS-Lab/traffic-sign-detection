import argparse
from src.utils.io import load_yaml
from src.pipeline.train import train_model, export_onnx
from src.pipeline.evaluate import validate_model, predict_one_image


def parse_args():
    parser = argparse.ArgumentParser(description="TT100K YOLOv8 pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "validate", "export", "infer", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument("--image", help="Image path for infer mode")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence for infer mode")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.mode in ("train", "all"):
        print("[INFO] Training started...")
        best = train_model(cfg)
        print(f"[INFO] Training finished. Best weights: {best}")

    if args.mode in ("validate", "all"):
        print("[INFO] Validation started...")
        metrics = validate_model(cfg)
        print("[INFO] Validation finished.")
        print(metrics)

    if args.mode in ("export", "all"):
        print("[INFO] ONNX export started...")
        export_onnx(cfg)
        print("[INFO] ONNX export finished.")

    if args.mode == "infer":
        if not args.image:
            raise ValueError("--image is required for infer mode")
        print("[INFO] Inference started...")
        result = predict_one_image(cfg, image_path=args.image, conf=args.conf)
        print("[INFO] Inference finished.")
        print(result)


if __name__ == "__main__":
    main()
