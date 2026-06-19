"""Train Faster R-CNN ResNet50-FPN on a YOLO-format road sign dataset."""

import argparse
import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

try:
    from src.datasets.yolo_detection_dataset import YoloDetectionDataset
except ModuleNotFoundError:
    from datasets.yolo_detection_dataset import YoloDetectionDataset


DATASET_NAME = "Traffic Signs Detection Europe"
DATASET_VERSION = "Roboflow version 14"
DATASET_LICENSE = "CC BY 4.0"
DEFAULT_NUM_CLASSES = 56


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load YAML config if it exists; otherwise return an empty config."""
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config or {}


def load_data_config(data_yaml: Path) -> dict[str, Any]:
    """Load data.yaml."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"data.yaml must contain a mapping: {data_yaml}")
    return config


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for road sign detection.")
    parser.add_argument("--config", default="configs/fasterrcnn.yaml", help="Optional YAML config path.")
    parser.add_argument("--data", default=None, help="Path to dataset data.yaml.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--device", default=None, help="Training device, for example cuda or cpu.")
    parser.add_argument("--output", default=None, help="Output directory for the experiment.")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers.")
    parser.add_argument("--backbone", default=None, help="Backbone name.")
    return parser.parse_args()


def value_from_cli_or_config(
    cli_value: Any,
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any,
) -> Any:
    """Resolve a setting from CLI, config, or default."""
    if cli_value is not None:
        return cli_value
    return config.get(section, {}).get(key, default)


def collate_fn(batch):
    """Collate detection batch as lists of images and targets."""
    return tuple(zip(*batch))


def get_device(device_arg: str) -> torch.device:
    """Resolve training device safely."""
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_cuda_device_name(device: torch.device) -> str | None:
    """Return CUDA device name if available."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(cuda_device_index(device))


def get_gpu_total_memory_mb(device: torch.device) -> float | None:
    """Return total GPU memory in MB."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return round(
        torch.cuda.get_device_properties(cuda_device_index(device)).total_memory / (1024 * 1024),
        3,
    )


def cuda_device_index(device: torch.device) -> int:
    """Return an explicit CUDA device index."""
    return device.index if device.index is not None else 0


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak memory stats if possible."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(cuda_device_index(device))
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: torch.device) -> float | None:
    """Return peak GPU memory in MB."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.set_device(cuda_device_index(device))
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)


def model_size_mb(path: Path) -> float | None:
    """Return model checkpoint size in MB."""
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 3)


def create_model(num_classes: int):
    """Create Faster R-CNN ResNet50-FPN and replace the classification head."""
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        print("Loaded Faster R-CNN with torchvision COCO pretrained weights.")
    except Exception as exc:
        print(f"Warning: pretrained weights unavailable, using random initialization: {exc}")
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(
    model,
    data_loader: DataLoader,
    optimizer: SGD,
    device: torch.device,
    epoch: int,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    batches = 0

    progress = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = float(losses.detach().cpu().item())
        total_loss += loss_value
        batches += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return total_loss / max(1, batches)


def save_checkpoint(path: Path, model, optimizer: SGD, epoch: int, loss: float) -> None:
    """Save a training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def append_log_row(log_path: Path, row: dict[str, Any]) -> None:
    """Append one row to training_log.csv."""
    file_exists = log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "learning_rate", "epoch_time_seconds"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_summary(
    summary_path: Path,
    args: argparse.Namespace,
    data_yaml: Path,
    backbone: str,
    device: torch.device,
    num_classes: int,
    training_start_time: datetime,
    training_end_time: datetime,
    total_training_time_seconds: float,
    best_loss: float,
    best_weights_path: Path,
    last_weights_path: Path,
) -> None:
    """Save experiment_summary.json."""
    summary = {
        "model_name": "fasterrcnn_resnet50_fpn",
        "backbone": backbone,
        "data": str(data_yaml),
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "dataset_license": DATASET_LICENSE,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "optimizer": "SGD(momentum=0.9, weight_decay=0.0005)",
        "scheduler": "StepLR(step_size=10, gamma=0.1)",
        "device": str(device),
        "num_classes": num_classes,
        "training_start_time": training_start_time.isoformat(timespec="seconds"),
        "training_end_time": training_end_time.isoformat(timespec="seconds"),
        "total_training_time_seconds": round(total_training_time_seconds, 3),
        "best_loss": round(best_loss, 6),
        "path_to_best_weights": str(best_weights_path),
        "path_to_last_weights": str(last_weights_path),
        "model_size_mb": model_size_mb(best_weights_path),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": get_cuda_device_name(device),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "gpu_memory_total_mb": get_gpu_total_memory_mb(device),
        "gpu_memory_peak_mb": get_peak_memory_mb(device),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run Faster R-CNN training."""
    args = parse_args()
    config = load_config(args.config)

    args.data = value_from_cli_or_config(args.data, config, "data", "yaml_path", "data/traffic-signs/data.yaml")
    args.epochs = int(value_from_cli_or_config(args.epochs, config, "training", "epochs", 30))
    args.batch = int(value_from_cli_or_config(args.batch, config, "training", "batch_size", 4))
    args.lr = float(value_from_cli_or_config(args.lr, config, "training", "learning_rate", 0.005))
    args.device = value_from_cli_or_config(args.device, config, "training", "device", "cuda")
    args.output = value_from_cli_or_config(
        args.output,
        config,
        "project",
        "output_dir",
        "outputs/runs/fasterrcnn_resnet50_fpn",
    )
    args.workers = int(value_from_cli_or_config(args.workers, config, "training", "workers", 0))
    args.backbone = value_from_cli_or_config(args.backbone, config, "training", "backbone", "resnet50_fpn")

    data_yaml = Path(args.data)
    data_config = load_data_config(data_yaml)
    num_classes = int(data_config.get("nc", 55)) + 1
    output_dir = ensure_dir(args.output)
    log_path = output_dir / "training_log.csv"
    best_weights_path = output_dir / "fasterrcnn_best.pth"
    last_weights_path = output_dir / "fasterrcnn_last.pth"
    summary_path = output_dir / "experiment_summary.json"

    device = get_device(args.device)
    reset_peak_memory(device)

    train_dataset = YoloDetectionDataset(data_yaml, split="train")
    valid_dataset = YoloDetectionDataset(data_yaml, split="valid")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    print("Faster R-CNN training")
    print(f"Backbone: {args.backbone}")
    print(f"Dataset config: {data_yaml}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Valid images: {len(valid_dataset)}")
    print(f"Num classes: {num_classes}")
    print(f"Output directory: {output_dir}")

    model = create_model(num_classes).to(device)
    optimizer = SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float("inf")
    training_start_time = datetime.now()
    training_timer_start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_timer_start = perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()
        epoch_time_seconds = perf_counter() - epoch_timer_start
        learning_rate = optimizer.param_groups[0]["lr"]
        gpu_memory = get_peak_memory_mb(device)

        save_checkpoint(last_weights_path, model, optimizer, epoch, train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(best_weights_path, model, optimizer, epoch, train_loss)

        append_log_row(
            log_path,
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "learning_rate": learning_rate,
                "epoch_time_seconds": round(epoch_time_seconds, 3),
            },
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"learning_rate={learning_rate:.8f} | "
            f"epoch_time_seconds={epoch_time_seconds:.3f} | "
            f"gpu_memory_mb={gpu_memory if gpu_memory is not None else 'n/a'}"
        )

    training_end_time = datetime.now()
    total_training_time_seconds = perf_counter() - training_timer_start
    save_summary(
        summary_path=summary_path,
        args=args,
        data_yaml=data_yaml,
        backbone=args.backbone,
        device=device,
        num_classes=num_classes,
        training_start_time=training_start_time,
        training_end_time=training_end_time,
        total_training_time_seconds=total_training_time_seconds,
        best_loss=best_loss,
        best_weights_path=best_weights_path,
        last_weights_path=last_weights_path,
    )

    print("\nTraining completed.")
    print(f"Best weights: {best_weights_path}")
    print(f"Last weights: {last_weights_path}")
    print(f"Training log: {log_path}")
    print(f"Experiment summary: {summary_path}")


if __name__ == "__main__":
    main()
