"""Analyze YOLO successes and errors on validation images."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 0, 255)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze YOLO detection errors.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--data", default="data/traffic-signs/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--images", required=True, help="Path to validation images.")
    parser.add_argument("--labels", required=True, help="Path to validation labels.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument("--output-dir", default="outputs/error_analysis/yolo", help="Output directory.")
    parser.add_argument("--max-images", type=int, default=300, help="Maximum number of images.")
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> dict[int, str]:
    """Load class names from data.yaml."""
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {}


def list_images(images_dir: Path, max_images: int) -> list[Path]:
    """Return image files from a directory."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return images[:max_images]


def yolo_to_xyxy(values: list[float], image_width: int, image_height: int) -> list[float]:
    """Convert normalized YOLO bbox to pixel xyxy."""
    x_center, y_center, width, height = values
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return [x1, y1, x2, y2]


def read_ground_truth(label_path: Path, image_width: int, image_height: int) -> list[dict[str, Any]]:
    """Read YOLO labels for one image."""
    if not label_path.exists():
        return []

    objects = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        bbox = yolo_to_xyxy([float(value) for value in parts[1:]], image_width, image_height)
        objects.append({"class_id": class_id, "bbox": bbox})
    return objects


def run_predictions(model: YOLO, image_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run YOLO inference and return prediction dictionaries."""
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        verbose=False,
    )
    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []

    predictions = []
    for box in boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        class_id = int(box.cls[0].detach().cpu().item())
        confidence = float(box.conf[0].detach().cpu().item())
        predictions.append({"class_id": class_id, "confidence": confidence, "bbox": xyxy})
    return predictions


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU for two xyxy boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int], list[int], list[int]]:
    """Match predictions to ground truth and return TP/FP/FN details."""
    matches = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    class_confusions: set[int] = set()
    localization_errors: set[int] = set()

    candidates = []
    for pred_index, pred in enumerate(predictions):
        for gt_index, gt in enumerate(ground_truth):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            candidates.append((iou, pred_index, gt_index))

    for iou, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in used_pred or gt_index in used_gt:
            continue
        if iou >= iou_threshold and predictions[pred_index]["class_id"] == ground_truth[gt_index]["class_id"]:
            matches.append((pred_index, gt_index))
            used_pred.add(pred_index)
            used_gt.add(gt_index)
        elif iou >= iou_threshold:
            class_confusions.add(pred_index)
            used_pred.add(pred_index)
            used_gt.add(gt_index)
        elif iou >= max(0.1, iou_threshold * 0.5):
            localization_errors.add(pred_index)

    false_positives = [
        index
        for index in range(len(predictions))
        if index not in used_pred and index not in localization_errors
    ]
    false_negatives = [index for index in range(len(ground_truth)) if index not in used_gt]
    return matches, false_positives, false_negatives, list(localization_errors), list(class_confusions)


def error_reason(error_type: str) -> str:
    """Return a short human-readable explanation for an error type."""
    reasons = {
        "false_positive": "сложный фон или похожий объект привел к ложному срабатыванию",
        "false_negative": "маленький объект, перекрытие или низкая видимость могли привести к пропуску",
        "localization_error": "объект найден, но bounding box локализован недостаточно точно",
        "class_confusion": "объект похож на другой класс дорожного знака",
    }
    return reasons.get(error_type, "требуется ручная проверка причины ошибки")


def draw_box(
    image,
    bbox: list[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw one bbox and label on an image."""
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def save_visualization(
    image_path: Path,
    output_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: dict[int, str],
) -> None:
    """Save image visualization with GT and predicted boxes."""
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for gt in ground_truth:
        class_id = gt["class_id"]
        label = f"GT {class_id}:{class_names.get(class_id, class_id)}"
        draw_box(image, gt["bbox"], GT_COLOR, label)

    for pred in predictions:
        class_id = pred["class_id"]
        label = f"P {class_id}:{class_names.get(class_id, class_id)} {pred['confidence']:.2f}"
        draw_box(image, pred["bbox"], PRED_COLOR, label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    """Run error analysis and save JSON, CSV, and visual examples."""
    args = parse_args()
    weights = Path(args.weights)
    data_yaml = Path(args.data)
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    class_names = load_class_names(data_yaml)
    image_paths = list_images(images_dir, args.max_images)
    model = YOLO(str(weights))

    successful_examples = []
    error_examples = []
    rows = []
    counts = {"tp": 0, "fp": 0, "fn": 0}
    common_error_reasons: dict[str, int] = {}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_height, image_width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        ground_truth = read_ground_truth(label_path, image_width, image_height)
        predictions = run_predictions(model, image_path, args)
        matches, false_positives, false_negatives, localization_errors, class_confusions = match_predictions(
            ground_truth, predictions, args.iou
        )

        counts["tp"] += len(matches)
        counts["fp"] += len(false_positives) + len(localization_errors) + len(class_confusions)
        counts["fn"] += len(false_negatives)

        visual_path = vis_dir / f"{image_path.stem}.jpg"
        save_visualization(image_path, visual_path, ground_truth, predictions, class_names)

        confidences = [prediction["confidence"] for prediction in predictions]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        if len(successful_examples) < 5 and matches and not false_positives and not false_negatives:
            successful_examples.append(
                {
                    "image": str(image_path),
                    "visualization": str(visual_path),
                    "detected_objects": len(matches),
                    "average_confidence": round(avg_conf, 3),
                    "comment": "знак хорошо виден, объект крупный, фон не мешает",
                }
            )

        error_sources = []
        if false_positives:
            error_sources.append("false_positive")
        if false_negatives:
            error_sources.append("false_negative")
        if localization_errors:
            error_sources.append("localization_error")
        if class_confusions:
            error_sources.append("class_confusion")

        for error_type in error_sources:
            common_error_reasons[error_type] = common_error_reasons.get(error_type, 0) + 1
            if len(error_examples) < 5:
                error_examples.append(
                    {
                        "image": str(image_path),
                        "visualization": str(visual_path),
                        "error_type": error_type,
                        "reason": error_reason(error_type),
                    }
                )

        rows.append(
            {
                "image": str(image_path),
                "tp": len(matches),
                "fp": len(false_positives) + len(localization_errors) + len(class_confusions),
                "fn": len(false_negatives),
                "predictions": len(predictions),
                "ground_truth": len(ground_truth),
                "visualization": str(visual_path),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "weights": str(weights),
        "conf": args.conf,
        "iou": args.iou,
        "processed_images": len(image_paths),
        "true_positive_count": counts["tp"],
        "false_positive_count": counts["fp"],
        "false_negative_count": counts["fn"],
        "successful_examples": successful_examples,
        "error_examples": error_examples,
        "common_error_reasons": common_error_reasons,
    }

    with (output_dir / "error_analysis.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    with (output_dir / "error_analysis.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "tp", "fp", "fn", "predictions", "ground_truth", "visualization"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Error analysis completed.")
    print(f"Processed images: {len(image_paths)}")
    print(f"TP: {counts['tp']}, FP: {counts['fp']}, FN: {counts['fn']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
