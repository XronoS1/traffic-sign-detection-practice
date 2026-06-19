
import argparse

from src.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run object detection inference.")
    parser.add_argument("--config", default="configs/predict.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    output_dir = ensure_dir(config["prediction"]["output_dir"])

    print("Prediction configuration")
    print(f"Model path: {config['prediction']['model_path']}")
    print(f"Source: {config['prediction']['source']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
