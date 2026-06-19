"""Evaluate and benchmark a torchvision Faster R-CNN checkpoint."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

try:
    from src.datasets.yolo_detection_dataset import YoloDetectionDataset
except ModuleNotFoundError:
    from datasets.yolo_detection_dataset import YoloDetectionDataset


GT_COLOR = (0, 200, 0)
PRED_COLOR = (0, 0, 255)
NUM_CLASSES = 56


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on YOLO-format validation data.")
    parser.add_argument("--weights", required=True, help="Path to Faster R-CNN checkpoint.")
    parser.add_argument("--data", default="data/traffic-signs/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--batch", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--device", default="cuda", help="Device, for example cuda or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--max-images", type=int, default=300, help="Maximum validation images.")
    parser.add_argument("--output-json", default="outputs/reports/fasterrcnn_eval.json", help="Output JSON path.")
    parser.add_argument("--error-dir", default="outputs/error_analysis/fasterrcnn", help="Error analysis dir.")
    return parser.parse_args()


def collate_fn(batch):
    """Collate detection batch."""
    return tuple(zip(*batch))


def load_class_names(data_yaml: Path) -> dict[int, str]:
    """Load class names from data.yaml."""
    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    names = config.get("names", {})
    if isinstance(names, list):
        return {index + 1: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index) + 1: str(name) for index, name in names.items()}
    return {}


def get_device(device_arg: str) -> torch.device:
    """Resolve evaluation device safely."""
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def create_model(num_classes: int):
    """Create Faster R-CNN with a replaced classifier head."""
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(model, weights_path: Path, device: torch.device) -> None:
    """Load model weights from a training checkpoint."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


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
) -> tuple[list[tuple[int, int, float]], list[int], list[int], list[dict[str, Any]]]:
    """Match predictions to ground truth by class and IoU."""
    candidates = []
    for pred_index, pred in enumerate(predictions):
        for gt_index, gt in enumerate(ground_truth):
            iou = compute_iou(pred["box"], gt["box"])
            if pred["label"] == gt["label"]:
                candidates.append((iou, pred_index, gt_index))

    matches = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for iou, pred_index, gt_index in sorted(candidates, reverse=True):
        if iou < iou_threshold or pred_index in used_pred or gt_index in used_gt:
            continue
        matches.append((pred_index, gt_index, iou))
        used_pred.add(pred_index)
        used_gt.add(gt_index)

    false_positives = [index for index in range(len(predictions)) if index not in used_pred]
    false_negatives = [index for index in range(len(ground_truth)) if index not in used_gt]
    errors = build_errors(ground_truth, predictions, false_positives, false_negatives)
    return matches, false_positives, false_negatives, errors


def best_iou_for_prediction(prediction: dict[str, Any], ground_truth: list[dict[str, Any]]) -> float:
    """Find best IoU between one prediction and any GT box."""
    if not ground_truth:
        return 0.0
    return max(compute_iou(prediction["box"], gt["box"]) for gt in ground_truth)


def build_errors(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    false_positives: list[int],
    false_negatives: list[int],
) -> list[dict[str, Any]]:
    """Build error rows for FP and FN cases."""
    errors = []
    for pred_index in false_positives:
        pred = predictions[pred_index]
        errors.append(
            {
                "error_type": "false_positive",
                "class_id": int(pred["label"]) - 1,
                "score": round(float(pred["score"]), 6),
                "iou": round(best_iou_for_prediction(pred, ground_truth), 6),
            }
        )

    for gt_index in false_negatives:
        gt = ground_truth[gt_index]
        errors.append(
            {
                "error_type": "false_negative",
                "class_id": int(gt["label"]) - 1,
                "score": None,
                "iou": 0.0,
            }
        )
    return errors


def tensor_target_to_list(target: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Convert target tensors to plain dictionaries."""
    boxes = target["boxes"].detach().cpu().tolist()
    labels = target["labels"].detach().cpu().tolist()
    return [{"box": box, "label": int(label)} for box, label in zip(boxes, labels)]


def output_to_predictions(output: dict[str, torch.Tensor], conf: float) -> list[dict[str, Any]]:
    """Convert model output tensors to filtered prediction dictionaries."""
    boxes = output["boxes"].detach().cpu().tolist()
    labels = output["labels"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()
    predictions = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= conf:
            predictions.append({"box": box, "label": int(label), "score": float(score)})
    return predictions


def draw_box(image, box: list[float], color: tuple[int, int, int], label: str) -> None:
    """Draw one box on an image."""
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def save_visualization(
    image_path: Path,
    output_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: dict[int, str],
) -> None:
    """Save GT/prediction visualization."""
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for gt in ground_truth:
        label = gt["label"]
        draw_box(image, gt["box"], GT_COLOR, f"GT {label - 1}:{class_names.get(label, label)}")

    for pred in predictions:
        label = pred["label"]
        draw_box(
            image,
            pred["box"],
            PRED_COLOR,
            f"P {label - 1}:{class_names.get(label, label)} {pred['score']:.2f}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def model_size_mb(path: Path) -> float | None:
    """Return checkpoint size in MB."""
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 3)


def synchronize_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA before/after timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    """Evaluate Faster R-CNN and save benchmark/error-analysis outputs."""
    args = parse_args()
    weights_path = Path(args.weights)
    data_yaml = Path(args.data)
    output_json = Path(args.output_json)
    error_dir = Path(args.error_dir)
    visuals_dir = error_dir / "visualizations"
    class_names = load_class_names(data_yaml)
    device = get_device(args.device)

    dataset = YoloDetectionDataset(data_yaml, split="valid")
    max_images = min(args.max_images, len(dataset))
    dataset = Subset(dataset, range(max_images))
    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = create_model(NUM_CLASSES)
    load_checkpoint(model, weights_path, device)
    model.to(device)
    model.eval()

    tp = fp = fn = 0
    latencies_ms: list[float] = []
    detection_counts: list[int] = []
    successful_examples = []
    error_examples = []
    error_rows = []
    processed_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images_on_device = [image.to(device) for image in images]
            synchronize_if_cuda(device)
            start = perf_counter()
            outputs = model(images_on_device)
            synchronize_if_cuda(device)
            elapsed_ms_per_image = ((perf_counter() - start) * 1000) / len(images)

            for image, target, output in zip(images, targets, outputs):
                image_id = int(target["image_id"].item())
                image_path = dataset.dataset.image_paths[image_id]
                ground_truth = tensor_target_to_list(target)
                predictions = output_to_predictions(output, args.conf)
                matches, false_positives, false_negatives, errors = match_predictions(
                    ground_truth,
                    predictions,
                    args.iou,
                )

                tp += len(matches)
                fp += len(false_positives)
                fn += len(false_negatives)
                processed_images += 1
                latencies_ms.append(elapsed_ms_per_image)
                detection_counts.append(len(predictions))

                visual_path = visuals_dir / f"{image_path.stem}.jpg"
                if len(successful_examples) < 5 and matches and not false_positives and not false_negatives:
                    save_visualization(image_path, visual_path, ground_truth, predictions, class_names)
                    successful_examples.append(str(visual_path))

                if errors:
                    save_visualization(image_path, visual_path, ground_truth, predictions, class_names)
                    if len(error_examples) < 5:
                        error_examples.append(str(visual_path))
                    for error in errors:
                        class_name = class_names.get(error["class_id"] + 1, str(error["class_id"]))
                        row = {
                            "image_path": str(image_path),
                            "error_type": error["error_type"],
                            "class_id": error["class_id"],
                            "class_name": class_name,
                            "score": error["score"],
                            "iou": error["iou"],
                        }
                        error_rows.append(row)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    mean_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    fps = 1000 / mean_latency_ms if mean_latency_ms > 0 else 0.0
    avg_detections = sum(detection_counts) / len(detection_counts) if detection_counts else 0.0

    report = {
        "weights": str(weights_path),
        "data": str(data_yaml),
        "conf": args.conf,
        "iou": args.iou,
        "processed_images": processed_images,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "fps": round(fps, 3),
        "avg_detections_per_image": round(avg_detections, 3),
        "model_size_mb": model_size_mb(weights_path),
        "successful_examples": successful_examples,
        "error_examples": error_examples,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    error_dir.mkdir(parents=True, exist_ok=True)
    with (error_dir / "error_analysis.json").open("w", encoding="utf-8") as file:
        json.dump({"errors": error_rows}, file, ensure_ascii=False, indent=2)

    with (error_dir / "error_analysis.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "error_type", "class_id", "class_name", "score", "iou"],
        )
        writer.writeheader()
        writer.writerows(error_rows)

    print("Faster R-CNN evaluation completed.")
    print(f"Processed images: {processed_images}")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(f"Precision={precision:.6f}, Recall={recall:.6f}")
    print(f"Mean latency ms={mean_latency_ms:.3f}, FPS={fps:.3f}")
    print(f"Output JSON: {output_json}")
    print(f"Error analysis dir: {error_dir}")


if __name__ == "__main__":
    main()
