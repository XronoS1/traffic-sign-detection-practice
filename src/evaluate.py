"""Evaluate trained object detection models from a YAML configuration."""

import argparse

from src.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate object detection models.")
    parser.add_argument("--config", default="configs/evaluate.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Load config and print evaluation settings."""
    config = load_config(parse_args().config)
    output_dir = ensure_dir(config["evaluation"]["output_dir"])

    print("Evaluation configuration")
    print(f"Model path: {config['evaluation']['model_path']}")
    print(f"Dataset split: {config['evaluation']['split']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
