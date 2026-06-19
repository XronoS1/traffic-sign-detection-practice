from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config or {}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
