"""Views for the road sign detection service."""

from collections import Counter
from pathlib import Path

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render

from .forms import DetectionRunForm, EmailRegistrationForm
from .class_names import get_display_class_name
from .inference import process_image
from .models import DetectionRun
from .video import process_video


def relative_media_path(path: Path) -> str:
    """Convert an absolute media path to a FileField-compatible relative path."""
    from django.conf import settings

    return str(path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")


def display_detected_classes(detected_classes: dict) -> list[tuple[str, int]]:
    """Return detected classes sorted by count with Russian display names."""
    return sorted(
        ((get_display_class_name(class_name), count) for class_name, count in (detected_classes or {}).items()),
        key=lambda item: item[1],
        reverse=True,
    )


def home(request):
    """Home page."""
    return render(request, "detection/home.html")


def register(request):
    """Register a user with email and password."""
    if request.method == "POST":
        form = EmailRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Регистрация выполнена.")
            return redirect("profile")
    else:
        form = EmailRegistrationForm()
    return render(request, "accounts/register.html", {"form": form})


@login_required
def profile(request):
    """User profile page."""
    runs_count = DetectionRun.objects.filter(user=request.user).count()
    return render(request, "detection/profile.html", {"runs_count": runs_count})


@login_required
def detect(request):
    """Upload file and run detection synchronously."""
    if request.method == "POST":
        form = DetectionRunForm(request.POST, request.FILES)
        if form.is_valid():
            model_weight = form.cleaned_data["model_weight"]
            run = form.save(commit=False)
            run.user = request.user
            run.model_weight = model_weight
            run.file_type = form.detect_file_type()
            run.confidence = min(0.95, max(0.05, model_weight.default_conf))
            run.image_size = model_weight.image_size
            run.weight_variant = form.cleaned_data["weight_variant"]
            run.used_weights_path = model_weight.get_weights_path(run.weight_variant)
            run.status = DetectionRun.CREATED
            run.save()

            try:
                run.status = DetectionRun.PROCESSING
                run.save(update_fields=["status", "updated_at"])
                input_path = Path(run.input_file.path)
                if run.file_type == DetectionRun.IMAGE:
                    result = process_image(input_path, model_weight, run.confidence, run.used_weights_path)
                else:
                    result = process_video(input_path, model_weight, run.confidence, run.used_weights_path)

                run.output_file.name = relative_media_path(result["output_path"])
                run.detections_count = result["detections_count"]
                run.detected_classes = result["detected_classes"]
                run.latency_ms = result["latency_ms"]
                run.status = DetectionRun.DONE
                run.error_message = ""
                run.save()
                return redirect("run_detail", run_id=run.id)
            except Exception as exc:
                run.status = DetectionRun.FAILED
                run.error_message = str(exc)
                run.save(update_fields=["status", "error_message", "updated_at"])
                messages.error(request, f"Не удалось выполнить распознавание: {exc}")
                return redirect("run_detail", run_id=run.id)
    else:
        form = DetectionRunForm()
    return render(request, "detection/detect.html", {"form": form})


@login_required
def run_detail(request, run_id: int):
    """Show one run owned by current user."""
    run = get_object_or_404(DetectionRun, id=run_id)
    if run.user_id != request.user.id:
        raise Http404("Запуск не найден.")
    return render(
        request,
        "detection/run_detail.html",
        {"run": run, "detected_classes_display": display_detected_classes(run.detected_classes)},
    )


@login_required
def history(request):
    """Show user's detection history."""
    runs = DetectionRun.objects.filter(user=request.user).select_related("model_weight")
    return render(request, "detection/history.html", {"runs": runs})


@login_required
def stats(request):
    """Show user's aggregate statistics."""
    runs = DetectionRun.objects.filter(user=request.user)
    done_runs = runs.filter(status=DetectionRun.DONE)
    model_distribution = (
        runs.values("model_weight__title")
        .annotate(total=Count("id"))
        .order_by("-total")
    )
    most_used_model = model_distribution[0] if model_distribution else None
    weight_distribution = (
        runs.values("weight_variant")
        .annotate(total=Count("id"))
        .order_by("-total")
    )
    top_classes_counter: Counter[str] = Counter()
    for run in done_runs:
        top_classes_counter.update(run.detected_classes or {})

    context = {
        "total_runs": runs.count(),
        "image_runs": runs.filter(file_type=DetectionRun.IMAGE).count(),
        "video_runs": runs.filter(file_type=DetectionRun.VIDEO).count(),
        "average_latency": done_runs.aggregate(avg=Avg("latency_ms"))["avg"],
        "model_distribution": model_distribution,
        "most_used_model": most_used_model,
        "weight_distribution": [
            {
                "label": dict(DetectionRun.WEIGHT_VARIANT_CHOICES).get(
                    item["weight_variant"],
                    item["weight_variant"],
                ),
                "total": item["total"],
            }
            for item in weight_distribution
        ],
        "top_classes": [
            (get_display_class_name(class_name), count)
            for class_name, count in top_classes_counter.most_common(10)
        ],
    }
    return render(request, "detection/stats.html", context)
