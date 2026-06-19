"""Add model quality labels for UI selection."""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("detection", "0002_weight_variants"),
    ]

    operations = [
        migrations.AddField(
            model_name="modelweight",
            name="quality_label",
            field=models.CharField(
                choices=[
                    ("high", "Высокое качество"),
                    ("medium", "Среднее качество"),
                    ("low", "Низкое качество"),
                ],
                default="medium",
                max_length=16,
            ),
        ),
    ]
