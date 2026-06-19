
from django.contrib import admin

from .models import DetectionRun, ModelWeight


@admin.register(ModelWeight)
class ModelWeightAdmin(admin.ModelAdmin):
    list_display = ("title", "model_type", "image_size", "default_conf", "is_active")
    list_filter = ("model_type", "is_active")
    search_fields = ("title", "weights_path", "best_weights_path", "last_weights_path")


@admin.register(DetectionRun)
class DetectionRunAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "model_weight", "file_type", "status", "detections_count", "created_at")
    list_filter = ("status", "file_type", "model_weight")
    search_fields = ("user__username", "user__email", "model_weight__title")
