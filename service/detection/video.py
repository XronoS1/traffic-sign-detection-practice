"""Video processing helpers."""

from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any
from uuid import uuid4

import cv2
from django.conf import settings

from .inference import merge_counts, predict_on_frame
from .models import ModelWeight


MAX_VIDEO_SECONDS = 60
FRAME_STRIDE = 2


def transcode_to_browser_mp4(raw_output_path: Path, final_output_path: Path) -> None:
    """Transcode OpenCV output to browser-compatible H.264 MP4."""
    cmd = [
        settings.FFMPEG_BINARY,
        "-y",
        "-i",
        str(raw_output_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(final_output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg не найден. Не удалось создать обработанное видео.") from exc

    if result.returncode != 0:
        message = result.stderr.strip() or "ffmpeg завершился с ошибкой."
        raise RuntimeError(f"Не удалось создать обработанное видео через ffmpeg: {message}")

    if not final_output_path.exists():
        raise RuntimeError("Не удалось создать обработанное видео. Итоговый MP4-файл не найден.")


def process_video(
    input_path: Path,
    model_weight: ModelWeight,
    confidence: float,
    weights_path_value: str | None = None,
) -> dict[str, Any]:
    """Run detection on a video and save an annotated MP4."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Не удалось открыть видео.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps > 0 and frame_count > 0 and frame_count / fps > MAX_VIDEO_SECONDS:
        capture.release()
        raise ValueError("Видео длиннее 60 секунд. Загрузите более короткий файл.")
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Не удалось прочитать размер видео.")

    output_dir = settings.MEDIA_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_token = uuid4().hex[:8]
    raw_output_path = output_dir / f"{input_path.stem}_{file_token}_raw.avi"
    output_path = output_dir / f"{input_path.stem}_{file_token}.mp4"
    output_fps = fps if fps > 0 else 25
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("Не удалось создать временный видеофайл для результата.")

    total_detections = 0
    detected_classes: dict[str, int] = {}
    last_annotated = None
    frame_index = 0
    start = perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % FRAME_STRIDE == 0 or last_annotated is None:
            annotated, detections_count, class_counts = predict_on_frame(
                frame,
                model_weight,
                confidence,
                weights_path_value,
            )
            total_detections += detections_count
            merge_counts(detected_classes, class_counts)
            last_annotated = annotated
        else:
            annotated = frame

        writer.write(annotated)
        frame_index += 1

    capture.release()
    writer.release()
    transcode_to_browser_mp4(raw_output_path, output_path)
    raw_output_path.unlink(missing_ok=True)
    latency_ms = (perf_counter() - start) * 1000

    return {
        "output_path": output_path,
        "detections_count": total_detections,
        "detected_classes": detected_classes,
        "latency_ms": round(latency_ms, 3),
    }
