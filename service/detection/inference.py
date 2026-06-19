"""Inference helpers for image detection."""

from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import cv2
import torch
import yaml
from django.conf import settings
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from ultralytics import YOLO

from .models import ModelWeight


MODEL_CACHE: dict[tuple[str, str], Any] = {}
NUM_FASTERRCNN_CLASSES = 56


def resolve_project_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return settings.PROJECT_ROOT / relative_path


def load_class_names() -> dict[int, str]:
    """Load class names from project data.yaml."""
    with settings.DATA_YAML_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {}


def create_fasterrcnn_model(weights_path: Path):
    """Create and load torchvision Faster R-CNN."""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_FASTERRCNN_CLASSES)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device


def get_model(model_weight: ModelWeight, weights_path_value: str | None = None):
    """Return cached detector model."""
    selected_weights_path = weights_path_value or model_weight.weights_path
    cache_key = (model_weight.model_type, selected_weights_path)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    weights_path = resolve_project_path(selected_weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл весов не найден: {selected_weights_path}")

    if model_weight.model_type == ModelWeight.ULTRALYTICS:
        model = YOLO(str(weights_path))
        MODEL_CACHE[cache_key] = model
        return model

    if model_weight.model_type == ModelWeight.TORCHVISION_FASTERRCNN:
        model = create_fasterrcnn_model(weights_path)
        MODEL_CACHE[cache_key] = model
        return model

    raise ValueError(f"Неизвестный тип модели: {model_weight.model_type}")


def draw_box(image, box: list[float], color: tuple[int, int, int], text: str) -> None:
    """Draw bounding box and label."""
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def update_class_counts(class_counts: dict[str, int], class_name: str) -> None:
    """Increment class counter."""
    class_counts[class_name] = class_counts.get(class_name, 0) + 1


def predict_ultralytics_on_frame(
    frame,
    model_weight: ModelWeight,
    confidence: float,
    weights_path_value: str | None = None,
):
    """Run Ultralytics inference on one frame and draw detections."""
    model = get_model(model_weight, weights_path_value)
    results = model.predict(
        source=frame,
        imgsz=model_weight.image_size,
        conf=confidence,
        verbose=False,
    )
    names = getattr(model, "names", {})
    class_counts: dict[str, int] = {}
    detections_count = 0
    annotated = frame.copy()

    if results:
        boxes = getattr(results[0], "boxes", None)
        if boxes is not None:
            for box in boxes:
                score = float(box.conf[0].detach().cpu().item())
                class_id = int(box.cls[0].detach().cpu().item())
                class_name = str(names.get(class_id, class_id))
                xyxy = box.xyxy[0].detach().cpu().tolist()
                draw_box(annotated, xyxy, (126, 34, 206), f"{class_name} {score:.2f}")
                update_class_counts(class_counts, class_name)
                detections_count += 1

    return annotated, detections_count, class_counts


def predict_fasterrcnn_on_frame(
    frame,
    model_weight: ModelWeight,
    confidence: float,
    weights_path_value: str | None = None,
):
    """Run torchvision Faster R-CNN inference on one frame and draw detections."""
    model, device = get_model(model_weight, weights_path_value)
    class_names = load_class_names()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).to(device)
    class_counts: dict[str, int] = {}
    detections_count = 0
    annotated = frame.copy()

    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output["boxes"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < confidence:
            continue
        class_id = int(label) - 1
        class_name = class_names.get(class_id, str(class_id))
        draw_box(annotated, box, (126, 34, 206), f"{class_name} {float(score):.2f}")
        update_class_counts(class_counts, class_name)
        detections_count += 1

    return annotated, detections_count, class_counts


def predict_on_frame(
    frame,
    model_weight: ModelWeight,
    confidence: float,
    weights_path_value: str | None = None,
):
    """Run selected model type on one OpenCV BGR frame."""
    if model_weight.model_type == ModelWeight.ULTRALYTICS:
        return predict_ultralytics_on_frame(frame, model_weight, confidence, weights_path_value)
    return predict_fasterrcnn_on_frame(frame, model_weight, confidence, weights_path_value)


def merge_counts(total: dict[str, int], current: dict[str, int]) -> None:
    """Merge class counters."""
    for class_name, count in current.items():
        total[class_name] = total.get(class_name, 0) + count


def process_image(
    input_path: Path,
    model_weight: ModelWeight,
    confidence: float,
    weights_path_value: str | None = None,
) -> dict[str, Any]:
    """Run detection on an image and save annotated output."""
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError("Не удалось прочитать изображение.")

    start = perf_counter()
    annotated, detections_count, detected_classes = predict_on_frame(
        image,
        model_weight,
        confidence,
        weights_path_value,
    )
    latency_ms = (perf_counter() - start) * 1000

    output_dir = settings.MEDIA_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_{uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(output_path), annotated)

    return {
        "output_path": output_path,
        "detections_count": detections_count,
        "detected_classes": detected_classes,
        "latency_ms": round(latency_ms, 3),
    }
