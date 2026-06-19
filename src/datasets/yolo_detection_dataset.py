"""PyTorch dataset for YOLO-format object detection data."""

from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YoloDetectionDataset(Dataset):
    """Load YOLO-format annotations for torchvision detection models."""

    def __init__(self, data_yaml: str | Path, split: str = "train") -> None:
        """Create dataset from data.yaml and split name."""
        self.data_yaml = Path(data_yaml)
        self.split = split
        self.config = self._load_config(self.data_yaml)
        self.dataset_root = self.data_yaml.parent
        self.images_dir = self._resolve_images_dir(split)
        self.labels_dir = self._resolve_labels_dir(self.images_dir)
        self.image_paths = self._find_images(self.images_dir)

    @staticmethod
    def _load_config(data_yaml: Path) -> dict[str, Any]:
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

        with data_yaml.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(f"data.yaml must contain a mapping: {data_yaml}")
        return config

    def _resolve_images_dir(self, split: str) -> Path:
        split_key = "val" if split in {"valid", "val"} else split
        split_path = self.config.get(split_key)

        if split_path is None and split == "valid":
            split_path = self.config.get("valid")
        if split_path is None:
            split_path = f"{split}/images"

        path = Path(split_path)
        if not path.is_absolute():
            path = self.dataset_root / path
        return path

    @staticmethod
    def _resolve_labels_dir(images_dir: Path) -> Path:
        parts = list(images_dir.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return Path(*parts)
        return images_dir.parent / "labels"

    @staticmethod
    def _find_images(images_dir: Path) -> list[Path]:
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        return sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return image tensor and Faster R-CNN target dict."""
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes, labels = self._read_labels(label_path, image_width, image_height)

        image_tensor = F.to_tensor(image).to(dtype=torch.float32)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
        }
        return image_tensor, target

    @staticmethod
    def _read_labels(
        label_path: Path,
        image_width: int,
        image_height: int,
    ) -> tuple[list[list[float]], list[int]]:
        """Read one YOLO label file and convert boxes to pixel xyxy."""
        if not label_path.exists():
            return [], []

        boxes: list[list[float]] = []
        labels: list[int] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:]]

            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height

            x_min = max(0.0, min(float(image_width), x_min))
            y_min = max(0.0, min(float(image_height), y_min))
            x_max = max(0.0, min(float(image_width), x_max))
            y_max = max(0.0, min(float(image_height), y_max))

            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id + 1)

        return boxes, labels
