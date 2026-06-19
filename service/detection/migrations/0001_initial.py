"""Initial detection app migration."""

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="ModelWeight",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=120, unique=True)),
                (
                    "model_type",
                    models.CharField(
                        choices=[
                            ("ultralytics", "Ultralytics YOLO/RT-DETR"),
                            ("torchvision_fasterrcnn", "Torchvision Faster R-CNN"),
                        ],
                        max_length=32,
                    ),
                ),
                ("weights_path", models.CharField(max_length=255)),
                ("image_size", models.PositiveIntegerField(default=640)),
                ("default_conf", models.FloatField(default=0.5)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["title"]},
        ),
        migrations.CreateModel(
            name="DetectionRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("input_file", models.FileField(upload_to="uploads/%Y/%m/%d/")),
                ("output_file", models.FileField(blank=True, upload_to="results/%Y/%m/%d/")),
                ("file_type", models.CharField(choices=[("image", "Изображение"), ("video", "Видео")], max_length=10)),
                ("confidence", models.FloatField()),
                ("image_size", models.PositiveIntegerField()),
                ("detections_count", models.PositiveIntegerField(default=0)),
                ("detected_classes", models.JSONField(blank=True, default=dict)),
                ("latency_ms", models.FloatField(blank=True, null=True)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("created", "Создано"),
                            ("processing", "Обрабатывается"),
                            ("done", "Готово"),
                            ("failed", "Ошибка"),
                        ],
                        default="created",
                        max_length=16,
                    ),
                ),
                ("error_message", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "model_weight",
                    models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to="detection.modelweight"),
                ),
                (
                    "user",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
                ),
            ],
            options={"ordering": ["-created_at"]},
        ),
    ]
