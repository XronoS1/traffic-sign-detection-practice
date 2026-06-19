from pathlib import Path

from src.config import ensure_dir, load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("name: test\n", encoding="utf-8")

    assert load_config(config_path) == {"name": "test"}


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    directory = tmp_path / "outputs"

    result = ensure_dir(directory)

    assert result.exists()
    assert result.is_dir()
