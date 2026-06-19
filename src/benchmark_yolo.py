
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WARMUP_RUNS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO inference.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--data", default="data/traffic-signs/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--images", required=True, help="Path to image directory.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-images", type=int, default=200, help="Maximum number of images.")
    parser.add_argument("--output", default="outputs/reports/yolo_benchmark.json", help="Output JSON path.")
    return parser.parse_args()


def list_images(images_dir: Path, max_images: int) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Expected image directory: {images_dir}")

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return image_paths[:max_images]


def file_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 3)


def parse_device_index(device: str) -> int:
    if device.lower().startswith("cuda:"):
        return int(device.split(":", maxsplit=1)[1])
    if device.isdigit():
        return int(device)
    return 0


def cuda_device_name(device: str) -> str | None:
    if not torch.cuda.is_available() or device.lower() == "cpu":
        return None
    return torch.cuda.get_device_name(parse_device_index(device))


def synchronize_if_cuda(device: str) -> None:
    if torch.cuda.is_available() and device.lower() != "cpu":
        torch.cuda.synchronize(parse_device_index(device))


def reset_peak_memory_if_cuda(device: str) -> None:
    if torch.cuda.is_available() and device.lower() != "cpu":
        torch.cuda.reset_peak_memory_stats(parse_device_index(device))


def peak_memory_mb(device: str) -> float | None:
    if not torch.cuda.is_available() or device.lower() == "cpu":
        return None
    return round(torch.cuda.max_memory_allocated(parse_device_index(device)) / (1024 * 1024), 3)


def run_prediction(model: YOLO, image_path: Path, args: argparse.Namespace):
    return model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        verbose=False,
    )


def count_detections(results: list[Any]) -> int:
    if not results:
        return 0
    boxes = getattr(results[0], "boxes", None)
    return len(boxes) if boxes is not None else 0


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    data_yaml = Path(args.data)
    images_dir = Path(args.images)
    output_path = Path(args.output)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    image_paths = list_images(images_dir, args.max_images)
    if not image_paths:
        raise ValueError(f"No images found in: {images_dir}")

    model = YOLO(str(weights))

    warmup_images = image_paths[: min(WARMUP_RUNS, len(image_paths))]
    for image_path in warmup_images:
        run_prediction(model, image_path, args)
    synchronize_if_cuda(args.device)

    reset_peak_memory_if_cuda(args.device)
    benchmark_start_time = datetime.now()
    latencies_ms: list[float] = []
    detection_counts: list[int] = []

    for image_path in image_paths:
        synchronize_if_cuda(args.device)
        start = perf_counter()
        results = run_prediction(model, image_path, args)
        synchronize_if_cuda(args.device)
        elapsed_ms = (perf_counter() - start) * 1000

        latencies_ms.append(elapsed_ms)
        detection_counts.append(count_detections(results))

    benchmark_end_time = datetime.now()
    mean_latency_ms = statistics.fmean(latencies_ms)
    fps = 1000 / mean_latency_ms if mean_latency_ms > 0 else None

    report = {
        "weights": str(weights),
        "data": str(data_yaml),
        "images": str(images_dir),
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "max_images": args.max_images,
        "processed_images": len(image_paths),
        "latencies_ms": [round(value, 3) for value in latencies_ms],
        "mean_latency_ms": round(mean_latency_ms, 3),
        "median_latency_ms": round(statistics.median(latencies_ms), 3),
        "std_latency_ms": round(statistics.stdev(latencies_ms), 3) if len(latencies_ms) > 1 else 0.0,
        "min_latency_ms": round(min(latencies_ms), 3),
        "max_latency_ms": round(max(latencies_ms), 3),
        "fps": round(fps, 3) if fps is not None else None,
        "avg_detections_per_image": round(statistics.fmean(detection_counts), 3),
        "model_size_mb": file_size_mb(weights),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": cuda_device_name(args.device),
        "gpu_memory_peak_mb": peak_memory_mb(args.device),
        "benchmark_start_time": benchmark_start_time.isoformat(timespec="seconds"),
        "benchmark_end_time": benchmark_end_time.isoformat(timespec="seconds"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Benchmark completed.")
    print(f"Processed images: {len(image_paths)}")
    print(f"Mean latency, ms: {report['mean_latency_ms']}")
    print(f"FPS: {report['fps']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
