import argparse
from src.utils.io import load_yaml
from src.pipeline.evaluate import predict_one_image


def main():
    parser = argparse.ArgumentParser(description="Run inference on one image")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    results = predict_one_image(cfg, args.image, args.conf)
    print(results)


if __name__ == "__main__":
    main()
