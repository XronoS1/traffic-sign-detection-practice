
import argparse
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_DIRS = (
    Path("train/images"),
    Path("train/labels"),
    Path("valid/images"),
    Path("valid/labels"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect YOLO-format road sign dataset.")
    parser.add_argument(
        "--data",
        "--data-root",
        dest="data_root",
        default="data/traffic-signs",
        help="Path to dataset root directory.",
    )
    return parser.parse_args()


def load_data_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"Missing data.yaml: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        errors.append(f"Cannot parse data.yaml: {exc}")
        return {}

    if not isinstance(data, dict):
        errors.append("data.yaml must contain a YAML mapping.")
        return {}

    return data


def list_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def print_yaml_summary(config: dict[str, Any]) -> None:
    print("\ndata.yaml:")
    print(f"  train: {config.get('train', 'not specified')}")
    print(f"  val: {config.get('val', config.get('valid', 'not specified'))}")

    if "test" in config:
        print(f"  test: {config['test']}")

    print(f"  nc: {config.get('nc', 'not specified')}")
    print(f"  names: {config.get('names', 'not specified')}")


def resolve_class_count(config: dict[str, Any], errors: list[str]) -> int | None:
    nc = config.get("nc")

    if isinstance(nc, int):
        return nc

    if nc is not None:
        errors.append(f"nc must be an integer, got: {nc!r}")
        return None

    names = config.get("names")
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names)

    errors.append("Cannot determine class count: data.yaml has no valid nc or names.")
    return None


def check_required_paths(data_root: Path, errors: list[str]) -> None:
    for relative_path in REQUIRED_DIRS:
        path = data_root / relative_path
        if not path.exists():
            errors.append(f"Missing directory: {path}")
        elif not path.is_dir():
            errors.append(f"Expected directory, got file: {path}")


def check_image_label_pairs(
    split: str,
    image_files: list[Path],
    label_files: list[Path],
    errors: list[str],
) -> None:
    label_stems = {path.stem for path in label_files}
    image_stems = {path.stem for path in image_files}

    for image_file in image_files:
        if image_file.stem not in label_stems:
            errors.append(f"[{split}] Missing label for image: {image_file.name}")

    for label_file in label_files:
        if label_file.stem not in image_stems:
            errors.append(f"[{split}] Label without matching image: {label_file.name}")


def parse_label_line(
    line: str,
    label_file: Path,
    line_number: int,
    class_count: int | None,
    errors: list[str],
) -> None:
    parts = line.split()
    if len(parts) != 5:
        errors.append(
            f"{label_file}:{line_number}: expected 5 values, got {len(parts)}."
        )
        return

    class_text, *box_texts = parts

    try:
        class_id = int(class_text)
    except ValueError:
        errors.append(f"{label_file}:{line_number}: class_id must be an integer.")
        class_id = None

    if class_id is not None and class_count is not None:
        if not 0 <= class_id <= class_count - 1:
            errors.append(
                f"{label_file}:{line_number}: class_id {class_id} is outside "
                f"range 0..{class_count - 1}."
            )

    for field_name, value_text in zip(("x_center", "y_center", "width", "height"), box_texts):
        try:
            value = float(value_text)
        except ValueError:
            errors.append(f"{label_file}:{line_number}: {field_name} must be a number.")
            continue

        if not 0.0 <= value <= 1.0:
            errors.append(
                f"{label_file}:{line_number}: {field_name}={value} is outside range 0..1."
            )


def check_label_file(
    label_file: Path,
    class_count: int | None,
    errors: list[str],
) -> int:
    before = len(errors)

    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        errors.append(f"{label_file}: cannot read as UTF-8 text.")
        return len(errors) - before

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parse_label_line(line, label_file, line_number, class_count, errors)

    return len(errors) - before


def inspect_split(
    data_root: Path,
    split: str,
    class_count: int | None,
    errors: list[str],
) -> tuple[int, int, int, int]:
    image_dir = data_root / split / "images"
    label_dir = data_root / split / "labels"
    image_files = list_files(image_dir, IMAGE_SUFFIXES)
    label_files = list_files(label_dir, {".txt"})

    check_image_label_pairs(split, image_files, label_files, errors)

    checked_label_files = 0
    annotation_errors = 0
    for label_file in label_files:
        checked_label_files += 1
        annotation_errors += check_label_file(label_file, class_count, errors)

    return len(image_files), len(label_files), checked_label_files, annotation_errors


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_yaml = data_root / "data.yaml"
    errors: list[str] = []

    print("YOLO dataset inspection")
    print(f"Dataset root: {data_root}")

    if not data_root.exists():
        errors.append(f"Missing dataset root: {data_root}")
    elif not data_root.is_dir():
        errors.append(f"Dataset root is not a directory: {data_root}")

    check_required_paths(data_root, errors)

    config = load_data_yaml(data_yaml, errors)
    if config:
        print_yaml_summary(config)
    else:
        print("\ndata.yaml: not available")

    class_count = resolve_class_count(config, errors) if config else None

    print("\nFile counts:")
    total_checked_label_files = 0
    total_annotation_errors = 0

    for split in ("train", "valid"):
        image_count, label_count, checked_label_files, annotation_errors = inspect_split(
            data_root,
            split,
            class_count,
            errors,
        )
        total_checked_label_files += checked_label_files
        total_annotation_errors += annotation_errors

        print(f"  {split}/images: {image_count}")
        print(f"  {split}/labels: {label_count}")

    print("\nErrors:")
    if errors:
        for index, error in enumerate(errors, start=1):
            print(f"  {index}. {error}")
    else:
        print("  No errors found.")

    print("\nSummary:")
    print(f"  dataset status: {'HAS ERRORS' if errors else 'OK'}")
    print(f"  checked label files: {total_checked_label_files}")
    print(f"  detected annotation errors: {total_annotation_errors}")


if __name__ == "__main__":
    main()
