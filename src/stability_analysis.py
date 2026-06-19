"""Analyze latency and FPS stability from a YOLO benchmark JSON file."""

import argparse
import json
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze benchmark stability with bootstrap CIs.")
    parser.add_argument("--benchmark-json", required=True, help="Path to benchmark JSON.")
    parser.add_argument("--output", default="outputs/reports/stability.json", help="Output JSON path.")
    parser.add_argument("--bootstrap-iters", type=int, default=1000, help="Bootstrap iterations.")
    return parser.parse_args()


def load_latencies(path: Path) -> list[float]:
    """Load latencies_ms from benchmark JSON."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    latencies = data.get("latencies_ms")
    if not isinstance(latencies, list) or not latencies:
        raise ValueError("Benchmark JSON must contain a non-empty 'latencies_ms' list.")

    return [float(value) for value in latencies]


def percentile(values: list[float], q: float) -> float:
    """Return percentile using nearest-rank interpolation."""
    if not values:
        raise ValueError("Cannot compute percentile for an empty list.")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def bootstrap_mean_ci(values: list[float], iterations: int) -> tuple[float, float]:
    """Compute a 95% bootstrap confidence interval for the mean."""
    means = []
    for _ in range(iterations):
        sample = [random.choice(values) for _ in values]
        means.append(statistics.fmean(sample))
    return percentile(means, 0.025), percentile(means, 0.975)


def main() -> None:
    """Run stability analysis and save JSON report."""
    args = parse_args()
    benchmark_path = Path(args.benchmark_json)
    output_path = Path(args.output)
    latencies = load_latencies(benchmark_path)
    fps_values = [1000 / latency for latency in latencies if latency > 0]

    latency_ci_low, latency_ci_high = bootstrap_mean_ci(latencies, args.bootstrap_iters)
    fps_ci_low, fps_ci_high = bootstrap_mean_ci(fps_values, args.bootstrap_iters)

    report = {
        "source_benchmark_json": str(benchmark_path),
        "bootstrap_iters": args.bootstrap_iters,
        "latency_mean_ms": round(statistics.fmean(latencies), 3),
        "latency_std_ms": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0.0,
        "latency_ci95_low_ms": round(latency_ci_low, 3),
        "latency_ci95_high_ms": round(latency_ci_high, 3),
        "fps_mean": round(statistics.fmean(fps_values), 3),
        "fps_std": round(statistics.stdev(fps_values), 3) if len(fps_values) > 1 else 0.0,
        "fps_ci95_low": round(fps_ci_low, 3),
        "fps_ci95_high": round(fps_ci_high, 3),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("Stability analysis completed.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
