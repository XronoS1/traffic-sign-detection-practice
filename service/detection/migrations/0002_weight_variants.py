
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("detection", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="modelweight",
            name="best_weights_path",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="modelweight",
            name="last_weights_path",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="detectionrun",
            name="used_weights_path",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="detectionrun",
            name="weight_variant",
            field=models.CharField(
                choices=[("best", "Лучшие веса"), ("last", "Последние веса")],
                default="best",
                max_length=8,
            ),
        ),
    ]
