"""Train RT-DETR for road sign detection with Ultralytics."""

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RT-DETR training."""
    parser = argparse.ArgumentParser(description="Train an RT-DETR detector.")
    parser.add_argument("--model", default="rtdetr-l.pt", help="RT-DETR model checkpoint or name.")
    parser.add_argument("--data", default="data/traffic-signs/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="Training device, for example 0, cpu, or cuda:0.")
    parser.add_argument("--project", default="outputs/runs", help="Directory for training runs.")
    parser.add_argument("--name", default="rtdetr_l_640", help="Experiment name.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    return parser.parse_args()


def find_training_artifacts(
    experiment_dir: Path,
) -> tuple[Path, Path | None, Path | None, Path | None, Path]:
    """Return expected experiment artifact paths."""
    best_weights = experiment_dir / "weights" / "best.pt"
    last_weights = experiment_dir / "weights" / "last.pt"
    results_csv = experiment_dir / "results.csv"
    summary_json = experiment_dir / "experiment_summary.json"

    return (
        experiment_dir,
        best_weights if best_weights.exists() else None,
        last_weights if last_weights.exists() else None,
        results_csv if results_csv.exists() else None,
        summary_json,
    )


def get_experiment_dir(training_result: Any, model: RTDETR, fallback: Path) -> Path:
    """Resolve the actual Ultralytics experiment directory."""
    result_save_dir = getattr(training_result, "save_dir", None)
    if result_save_dir is not None:
        return Path(result_save_dir)

    trainer_save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if trainer_save_dir is not None:
        return Path(trainer_save_dir)

    return fallback


def save_experiment_summary(
    summary_json: Path,
    args: argparse.Namespace,
    experiment_dir: Path,
    best_weights: Path | None,
    last_weights: Path | None,
    results_csv: Path | None,
) -> None:
    """Save a compact RT-DETR experiment summary."""
    summary = {
        "experiment_name": args.name,
        "model": args.model,
        "data": str(Path(args.data)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "experiment_dir": str(experiment_dir),
        "path_to_best_pt": str(best_weights) if best_weights else None,
        "path_to_last_pt": str(last_weights) if last_weights else None,
        "path_to_results_csv": str(results_csv) if results_csv else None,
        "path_to_experiment_summary_json": str(summary_json),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run RT-DETR training and print key output artifacts."""
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model = RTDETR(args.model)
    training_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        workers=args.workers,
    )

    experiment_dir = get_experiment_dir(training_result, model, project_dir / args.name)
    experiment_dir, best_weights, last_weights, results_csv, summary_json = find_training_artifacts(
        experiment_dir
    )
    save_experiment_summary(
        summary_json=summary_json,
        args=args,
        experiment_dir=experiment_dir,
        best_weights=best_weights,
        last_weights=last_weights,
        results_csv=results_csv,
    )

    print("\nTraining completed.")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Best weights: {best_weights if best_weights else 'not found'}")
    print(f"Last weights: {last_weights if last_weights else 'not found'}")
    print(f"Results CSV: {results_csv if results_csv else 'not found'}")
    print(f"Experiment summary: {summary_json}")


if __name__ == "__main__":
    main()
