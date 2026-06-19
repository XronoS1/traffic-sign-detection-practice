
import argparse

from src.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark object detection models.")
    parser.add_argument("--config", default="configs/benchmark.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    output_dir = ensure_dir(config["benchmark"]["output_dir"])

    print("Benchmark configuration")
    print(f"Source: {config['data']['source']}")
    print(f"Models: {', '.join(config['benchmark']['model_paths'])}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
