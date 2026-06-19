
import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "experiment_name",
    "model",
    "augmentation_mode",
    "epochs",
    "imgsz",
    "batch",
    "optimizer",
    "learning_rate_initial",
    "scheduler",
    "final_precision",
    "final_recall",
    "final_map50",
    "final_map50_95",
    "mean_latency_ms",
    "median_latency_ms",
    "std_latency_ms",
    "fps",
    "gpu_memory_peak_mb",
    "best_model_size_mb",
    "total_training_time_seconds",
    "dataset_version",
    "cuda_device_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiment results into a CSV table.")
    parser.add_argument("--runs-dir", default="outputs/runs", help="Directory with training runs.")
    parser.add_argument("--benchmarks-dir", default="outputs/reports", help="Directory with benchmark JSON files.")
    parser.add_argument("--output", default="outputs/reports/experiment_summary.csv", help="Output CSV path.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def find_training_summaries(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("*/experiment_summary.json"))


def find_benchmark_reports(benchmarks_dir: Path) -> list[dict[str, Any]]:
    if not benchmarks_dir.exists():
        return []

    reports = []
    for path in sorted(benchmarks_dir.glob("*.json")):
        try:
            report = load_json(path)
        except json.JSONDecodeError:
            continue
        report["_source_path"] = str(path)
        reports.append(report)
    return reports


def experiment_name_from_weights(weights: str | None) -> str | None:
    if not weights:
        return None
    parts = Path(weights).parts
    if "weights" in parts:
        weights_index = parts.index("weights")
        if weights_index > 0:
            return parts[weights_index - 1]
    return None


def find_matching_benchmark(
    experiment_name: str,
    benchmarks: list[dict[str, Any]],
) -> dict[str, Any]:
    for benchmark in benchmarks:
        if experiment_name_from_weights(benchmark.get("weights")) == experiment_name:
            return benchmark

    for benchmark in benchmarks:
        source_path = str(benchmark.get("_source_path", ""))
        if experiment_name in source_path:
            return benchmark

    return {}


def build_row(summary: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": summary.get("experiment_name"),
        "model": summary.get("model"),
        "augmentation_mode": summary.get("augmentation_mode"),
        "epochs": summary.get("epochs"),
        "imgsz": summary.get("imgsz"),
        "batch": summary.get("batch"),
        "optimizer": summary.get("optimizer"),
        "learning_rate_initial": summary.get("learning_rate_initial"),
        "scheduler": summary.get("scheduler"),
        "final_precision": summary.get("final_precision"),
        "final_recall": summary.get("final_recall"),
        "final_map50": summary.get("final_map50"),
        "final_map50_95": summary.get("final_map50_95"),
        "mean_latency_ms": benchmark.get("mean_latency_ms"),
        "median_latency_ms": benchmark.get("median_latency_ms"),
        "std_latency_ms": benchmark.get("std_latency_ms"),
        "fps": benchmark.get("fps"),
        "gpu_memory_peak_mb": benchmark.get("gpu_memory_peak_mb", summary.get("gpu_memory_peak_mb")),
        "best_model_size_mb": summary.get("best_model_size_mb"),
        "total_training_time_seconds": summary.get("total_training_time_seconds"),
        "dataset_version": summary.get("dataset_version"),
        "cuda_device_name": summary.get("cuda_device_name"),
    }


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    benchmarks_dir = Path(args.benchmarks_dir)
    output_path = Path(args.output)

    summaries = find_training_summaries(runs_dir)
    benchmarks = find_benchmark_reports(benchmarks_dir)
    rows: list[dict[str, Any]] = []

    for summary_path in summaries:
        try:
            summary = load_json(summary_path)
        except json.JSONDecodeError:
            continue

        experiment_name = summary.get("experiment_name") or summary_path.parent.name
        summary["experiment_name"] = experiment_name
        benchmark = find_matching_benchmark(experiment_name, benchmarks)
        rows.append(build_row(summary, benchmark))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("Results collection completed.")
    print(f"Training summaries: {len(summaries)}")
    print(f"Rows written: {len(rows)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
