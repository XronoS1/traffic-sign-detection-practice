
from pathlib import Path

from django import forms
from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from .models import DetectionRun, ModelWeight


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class EmailRegistrationForm(UserCreationForm):

    email = forms.EmailField(label="Email")

    class Meta:
        model = User
        fields = ("email", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(username=email).exists():
            raise forms.ValidationError("Пользователь с таким email уже существует.")
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"].strip().lower()
        user.username = user.email
        if commit:
            user.save()
        return user


class EmailAuthenticationForm(AuthenticationForm):

    username = forms.EmailField(label="Email")


class DetectionRunForm(forms.ModelForm):

    weight_variant = forms.ChoiceField(
        label="Вариант весов",
        choices=DetectionRun.WEIGHT_VARIANT_CHOICES,
        initial=DetectionRun.BEST,
    )

    class Meta:
        model = DetectionRun
        fields = ("model_weight", "weight_variant", "input_file")
        labels = {
            "model_weight": "Модель",
            "input_file": "Файл",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model_weight"].queryset = ModelWeight.objects.filter(is_active=True)

    def clean_input_file(self):
        uploaded_file = self.cleaned_data["input_file"]
        if uploaded_file.size > settings.MAX_UPLOAD_SIZE:
            raise forms.ValidationError("Максимальный размер файла — 100 MB.")

        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
            raise forms.ValidationError("Поддерживаются изображения JPG/PNG и видео MP4/AVI/MOV/MKV.")
        return uploaded_file

    def detect_file_type(self) -> str:
        suffix = Path(self.cleaned_data["input_file"].name).suffix.lower()
        return DetectionRun.IMAGE if suffix in IMAGE_EXTENSIONS else DetectionRun.VIDEO
