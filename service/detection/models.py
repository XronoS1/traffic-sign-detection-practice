"""Database models for detection runs."""

from django.conf import settings
from django.db import models


class ModelWeight(models.Model):
    """A selectable detector checkpoint."""

    ULTRALYTICS = "ultralytics"
    TORCHVISION_FASTERRCNN = "torchvision_fasterrcnn"

    MODEL_TYPE_CHOICES = [
        (ULTRALYTICS, "Ultralytics YOLO/RT-DETR"),
        (TORCHVISION_FASTERRCNN, "Torchvision Faster R-CNN"),
    ]
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    QUALITY_CHOICES = [
        (HIGH, "Высокое качество"),
        (MEDIUM, "Среднее качество"),
        (LOW, "Низкое качество"),
    ]

    title = models.CharField(max_length=120, unique=True)
    model_type = models.CharField(max_length=32, choices=MODEL_TYPE_CHOICES)
    weights_path = models.CharField(max_length=255)
    best_weights_path = models.CharField(max_length=255, blank=True)
    last_weights_path = models.CharField(max_length=255, blank=True)
    image_size = models.PositiveIntegerField(default=640)
    default_conf = models.FloatField(default=0.5)
    quality_label = models.CharField(max_length=16, choices=QUALITY_CHOICES, default=MEDIUM)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["title"]

    def __str__(self) -> str:
        return f"{self.title} — {self.get_quality_label_display().lower()}"

    def get_weights_path(self, variant: str) -> str:
        """Return selected best/last weights path with backward compatibility."""
        if variant == "last":
            return self.last_weights_path or self.weights_path
        return self.best_weights_path or self.weights_path


class DetectionRun(models.Model):
    """One user-submitted image or video detection run."""

    IMAGE = "image"
    VIDEO = "video"
    FILE_TYPE_CHOICES = [(IMAGE, "Изображение"), (VIDEO, "Видео")]

    CREATED = "created"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    STATUS_CHOICES = [
        (CREATED, "Создано"),
        (PROCESSING, "Обрабатывается"),
        (DONE, "Готово"),
        (FAILED, "Ошибка"),
    ]

    BEST = "best"
    LAST = "last"
    WEIGHT_VARIANT_CHOICES = [
        (BEST, "Лучшие веса"),
        (LAST, "Последние веса"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    model_weight = models.ForeignKey(ModelWeight, on_delete=models.PROTECT)
    input_file = models.FileField(upload_to="uploads/%Y/%m/%d/")
    output_file = models.FileField(upload_to="results/%Y/%m/%d/", blank=True)
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES)
    confidence = models.FloatField()
    image_size = models.PositiveIntegerField()
    weight_variant = models.CharField(max_length=8, choices=WEIGHT_VARIANT_CHOICES, default=BEST)
    used_weights_path = models.CharField(max_length=255, blank=True)
    detections_count = models.PositiveIntegerField(default=0)
    detected_classes = models.JSONField(default=dict, blank=True)
    latency_ms = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=CREATED)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user} - {self.model_weight} - {self.created_at:%Y-%m-%d %H:%M}"
