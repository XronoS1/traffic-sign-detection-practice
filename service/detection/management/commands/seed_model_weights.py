"""Seed selectable model weights."""

from django.core.management.base import BaseCommand

from detection.models import ModelWeight


MODEL_WEIGHTS = [
    {
        "title": "YOLOv8n",
        "model_type": ModelWeight.ULTRALYTICS,
        "weights_path": "outputs/runs/yolov8n_640/weights/best.pt",
        "best_weights_path": "outputs/runs/yolov8n_640/weights/best.pt",
        "last_weights_path": "outputs/runs/yolov8n_640/weights/last.pt",
        "image_size": 640,
        "default_conf": 0.50,
        "quality_label": ModelWeight.HIGH,
        "is_active": True,
    },
    {
        "title": "YOLOv11n",
        "model_type": ModelWeight.ULTRALYTICS,
        "weights_path": "outputs/runs/yolo11n_640/weights/best.pt",
        "best_weights_path": "outputs/runs/yolo11n_640/weights/best.pt",
        "last_weights_path": "outputs/runs/yolo11n_640/weights/last.pt",
        "image_size": 640,
        "default_conf": 0.50,
        "quality_label": ModelWeight.MEDIUM,
        "is_active": True,
    },
    {
        "title": "YOLO26n",
        "model_type": ModelWeight.ULTRALYTICS,
        "weights_path": "outputs/runs/yolo26n_640/weights/best.pt",
        "best_weights_path": "outputs/runs/yolo26n_640/weights/best.pt",
        "last_weights_path": "outputs/runs/yolo26n_640/weights/last.pt",
        "image_size": 640,
        "default_conf": 0.50,
        "quality_label": ModelWeight.LOW,
        "is_active": True,
    },
    {
        "title": "RT-DETR-l",
        "model_type": ModelWeight.ULTRALYTICS,
        "weights_path": "outputs/runs/rtdetr_l_512_30ep/weights/best.pt",
        "best_weights_path": "outputs/runs/rtdetr_l_512_30ep/weights/best.pt",
        "last_weights_path": "outputs/runs/rtdetr_l_512_30ep/weights/last.pt",
        "image_size": 512,
        "default_conf": 0.50,
        "quality_label": ModelWeight.HIGH,
        "is_active": True,
    },
    {
        "title": "Faster R-CNN ResNet50-FPN",
        "model_type": ModelWeight.TORCHVISION_FASTERRCNN,
        "weights_path": "outputs/runs/fasterrcnn_resnet50_fpn_640_30ep/fasterrcnn_best.pth",
        "best_weights_path": "outputs/runs/fasterrcnn_resnet50_fpn_640_30ep/fasterrcnn_best.pth",
        "last_weights_path": "outputs/runs/fasterrcnn_resnet50_fpn_640_30ep/fasterrcnn_last.pth",
        "image_size": 640,
        "default_conf": 0.50,
        "quality_label": ModelWeight.MEDIUM,
        "is_active": True,
    },
]


class Command(BaseCommand):
    help = "Create or update default model weights."

    def handle(self, *args, **options):
        active_titles = {config["title"] for config in MODEL_WEIGHTS}
        for config in MODEL_WEIGHTS:
            config = config.copy()
            title = config["title"]
            ModelWeight.objects.update_or_create(title=title, defaults=config)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Seeded: {title} ({dict(ModelWeight.QUALITY_CHOICES)[config['quality_label']]})"
                )
            )
        ModelWeight.objects.exclude(title__in=active_titles).update(is_active=False)
        self.stdout.write(self.style.WARNING("Inactive old model weight records were hidden from the UI."))
