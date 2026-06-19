
import argparse
import csv
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO


DATASET_NAME = "Traffic Signs Detection Europe"
DATASET_VERSION = "Roboflow version 14"
DATASET_LICENSE = "CC BY 4.0"

STRONG_AUGMENTATION_PARAMS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model checkpoint or name.")
    parser.add_argument("--data", default="data/traffic-signs/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--project", default="outputs/runs", help="Directory for training runs.")
    parser.add_argument("--name", default="yolo_train", help="Experiment name.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer name passed to Ultralytics.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate scheduler.")

    parser.add_argument("--hsv-h", type=float, default=None, help="Hue augmentation value.")
    parser.add_argument("--hsv-s", type=float, default=None, help="Saturation augmentation value.")
    parser.add_argument("--hsv-v", type=float, default=None, help="Value augmentation value.")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation augmentation in degrees.")
    parser.add_argument("--translate", type=float, default=None, help="Translation augmentation value.")
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation value.")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp augmentation probability.")
    parser.add_argument(
        "--strong-aug",
        action="store_true",
        help="Use the predefined strong augmentation profile.",
    )
    return parser.parse_args()


def get_augmentation_params(args: argparse.Namespace) -> dict[str, float]:
    if args.strong_aug:
        return STRONG_AUGMENTATION_PARAMS.copy()

    return {}


def find_training_artifacts(
    experiment_dir: Path,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None]:
    best_weights = experiment_dir / "weights" / "best.pt"
    last_weights = experiment_dir / "weights" / "last.pt"
    results_csv = experiment_dir / "results.csv"
    args_yaml = experiment_dir / "args.yaml"

    return (
        experiment_dir,
        best_weights if best_weights.exists() else None,
        last_weights if last_weights.exists() else None,
        results_csv if results_csv.exists() else None,
        args_yaml if args_yaml.exists() else None,
    )


def file_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None

    return round(path.stat().st_size / (1024 * 1024), 3)


def path_to_string(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def get_cuda_device_index(device_arg: Any) -> int | None:
    if device_arg is None:
        return None

    if isinstance(device_arg, int):
        return device_arg

    device = str(device_arg).strip().lower()
    if device in {"", "cpu"}:
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        index_text = device.split(":", maxsplit=1)[1]
        return int(index_text) if index_text.isdigit() else None
    if device.isdigit():
        return int(device)

    return None


def get_hardware_info(device: str, gpu_memory_peak_mb: float | None = None) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_device_name = None
    gpu_memory_total_mb = None

    device_index = get_cuda_device_index(device)
    if cuda_available and device_index is not None and 0 <= device_index < torch.cuda.device_count():
        try:
            cuda_device_name = torch.cuda.get_device_name(device_index)
            gpu_memory_total_mb = round(
                torch.cuda.get_device_properties(device_index).total_memory / (1024 * 1024),
                3,
            )
        except RuntimeError as exc:
            print(f"Warning: cannot read CUDA device info for device={device!r}: {exc}")
    elif cuda_available and device_index is not None:
        print(f"Warning: CUDA device index {device_index} is not available; hardware info is partial.")

    return {
        "hardware": platform.platform(),
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "python_version": sys.version.split()[0],
        "gpu_memory_total_mb": gpu_memory_total_mb,
        "gpu_memory_peak_mb": gpu_memory_peak_mb,
    }


def reset_cuda_peak_memory_stats(device_arg: Any) -> None:
    if not torch.cuda.is_available():
        return

    device_index = get_cuda_device_index(device_arg)
    if device_index is None:
        if str(device_arg).strip().lower() != "cpu":
            print(f"Warning: cannot parse CUDA device from --device={device_arg!r}; memory stats disabled.")
        return
    if device_index < 0 or device_index >= torch.cuda.device_count():
        print(f"Warning: CUDA device index {device_index} is not available; memory stats disabled.")
        return

    try:
        torch.cuda.set_device(device_index)
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError as exc:
        print(f"Warning: cannot reset CUDA peak memory stats for device={device_arg!r}: {exc}")


def read_cuda_peak_memory_mb(device_arg: Any) -> float | None:
    if not torch.cuda.is_available():
        return None

    device_index = get_cuda_device_index(device_arg)
    if device_index is None:
        if str(device_arg).strip().lower() != "cpu":
            print(f"Warning: cannot parse CUDA device from --device={device_arg!r}; peak memory unavailable.")
        return None
    if device_index < 0 or device_index >= torch.cuda.device_count():
        print(f"Warning: CUDA device index {device_index} is not available; peak memory unavailable.")
        return None

    try:
        torch.cuda.set_device(device_index)
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    except RuntimeError as exc:
        print(f"Warning: cannot read CUDA peak memory for device={device_arg!r}: {exc}")
        return None


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def find_metric(row: dict[str, str], metric: str) -> float | None:
    normalized = {normalize_column_name(key): value for key, value in row.items()}

    if metric == "precision":
        keys = [key for key in normalized if "precision" in key]
    elif metric == "recall":
        keys = [key for key in normalized if "recall" in key]
    elif metric == "map50":
        keys = [
            key
            for key in normalized
            if ("map50" in key or "map05" in key) and "95" not in key
        ]
    elif metric == "map50_95":
        keys = [
            key
            for key in normalized
            if "map5095" in key or "map50095" in key or ("map50" in key and "95" in key)
        ]
    else:
        keys = []

    for key in keys:
        value = to_float(normalized[key])
        if value is not None:
            return value

    return None


def read_final_metrics(results_csv: Path | None) -> dict[str, float | None]:
    metrics = {
        "final_precision": None,
        "final_recall": None,
        "final_map50": None,
        "final_map50_95": None,
    }
    if results_csv is None or not results_csv.exists():
        return metrics

    with results_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return metrics

    last_row = rows[-1]
    metrics["final_precision"] = find_metric(last_row, "precision")
    metrics["final_recall"] = find_metric(last_row, "recall")
    metrics["final_map50"] = find_metric(last_row, "map50")
    metrics["final_map50_95"] = find_metric(last_row, "map50_95")
    return metrics


def save_experiment_summary(
    summary_path: Path,
    args: argparse.Namespace,
    augmentation_mode: str,
    augmentation_params: dict[str, float],
    best_weights: Path | None,
    last_weights: Path | None,
    results_csv: Path | None,
    args_yaml: Path | None,
    training_start_time: datetime,
    training_end_time: datetime,
    total_training_time_seconds: float,
    gpu_memory_peak_mb: float | None,
) -> None:
    summary: dict[str, Any] = {
        "experiment_name": args.name,
        "model": args.model,
        "pretrained_weights_source": args.model,
        "data": str(Path(args.data)),
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "dataset_license": DATASET_LICENSE,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "learning_rate_initial": args.lr0,
        "learning_rate_final_factor": args.lrf,
        "scheduler": "cosine" if args.cos_lr else "Ultralytics default",
        "cos_lr": args.cos_lr,
        "augmentation_mode": augmentation_mode,
        "augmentation_params": augmentation_params,
        "path_to_best_pt": path_to_string(best_weights),
        "path_to_last_pt": path_to_string(last_weights),
        "path_to_results_csv": path_to_string(results_csv),
        "path_to_args_yaml": path_to_string(args_yaml),
        "best_model_size_mb": file_size_mb(best_weights),
        "training_start_time": training_start_time.isoformat(timespec="seconds"),
        "training_end_time": training_end_time.isoformat(timespec="seconds"),
        "total_training_time_seconds": round(total_training_time_seconds, 3),
    }
    summary.update(get_hardware_info(args.device, gpu_memory_peak_mb))
    summary.update(read_final_metrics(results_csv))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    augmentation_mode = "strong" if args.strong_aug else "default"
    augmentation_params = get_augmentation_params(args)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": True,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": args.cos_lr,
    }
    train_kwargs.update(augmentation_params)

    reset_cuda_peak_memory_stats(args.device)

    model = YOLO(args.model)
    training_start_time = datetime.now()
    timer_start = perf_counter()

    model.train(**train_kwargs)

    total_training_time_seconds = perf_counter() - timer_start
    training_end_time = datetime.now()

    gpu_memory_peak_mb = read_cuda_peak_memory_mb(args.device)

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    experiment_dir_candidate = Path(save_dir) if save_dir is not None else project_dir / args.name
    experiment_dir, best_weights, last_weights, results_csv, args_yaml = find_training_artifacts(
        experiment_dir_candidate
    )
    summary_path = experiment_dir / "experiment_summary.json"

    save_experiment_summary(
        summary_path=summary_path,
        args=args,
        augmentation_mode=augmentation_mode,
        augmentation_params=augmentation_params,
        best_weights=best_weights,
        last_weights=last_weights,
        results_csv=results_csv,
        args_yaml=args_yaml,
        training_start_time=training_start_time,
        training_end_time=training_end_time,
        total_training_time_seconds=total_training_time_seconds,
        gpu_memory_peak_mb=gpu_memory_peak_mb,
    )

    print("\nTraining completed.")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Best weights: {best_weights if best_weights else 'not found'}")
    print(f"Last weights: {last_weights if last_weights else 'not found'}")
    print(f"Results CSV: {results_csv if results_csv else 'not found'}")
    print(f"Args YAML: {args_yaml if args_yaml else 'not found'}")
    print(f"Experiment summary: {summary_path}")


if __name__ == "__main__":
    main()
