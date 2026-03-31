"""BDD100K dataset preparation entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_config
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
YOLO_EVAL_SPLIT_NAMES = ("val", "eval")
IMAGE_EXPORT_MODES = {"copy", "symlink", "none"}
DATA_ISSUE_POLICIES = {"warn", "error"}
SUBSET_STRATEGIES = {"random", "balanced_by_class_presence"}


@dataclass(frozen=True)
class ImageRecord:
    file_name: str
    width: int
    height: int
    labels: list[dict[str, Any]]


def validate_policy(name: str, value: str, allowed_values: set[str]) -> str:
    normalized_value = value.lower()
    if normalized_value not in allowed_values:
        raise ValueError(f"{name} must be one of {sorted(allowed_values)}.")
    return normalized_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the BDD100K dataset.")
    parser.add_argument("--config", required=True, help="Path to the dataset config YAML.")
    return parser.parse_args()


def load_bdd_annotations(path: str | Path) -> list[dict[str, Any]]:
    annotations_path = Path(path)
    with annotations_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of annotations in {annotations_path}.")

    return data


def select_subset(
    annotations: list[dict[str, Any]],
    subset_enabled: bool,
    max_images: int | None,
    seed: int,
    classes: list[str] | None = None,
    strategy: str = "random",
) -> list[dict[str, Any]]:
    if not subset_enabled or max_images is None or max_images <= 0 or len(annotations) <= max_images:
        return annotations

    normalized_strategy = validate_policy("dataset.subset.strategy", strategy, SUBSET_STRATEGIES)
    sampler = random.Random(seed)
    if normalized_strategy == "balanced_by_class_presence":
        sampled_indices = select_balanced_subset_indices(
            annotations=annotations,
            max_images=max_images,
            seed=seed,
            classes=classes or [],
        )
    else:
        sampled_indices = sorted(sampler.sample(range(len(annotations)), max_images))
    return [annotations[index] for index in sampled_indices]


def annotation_categories(annotation: dict[str, Any], classes: set[str]) -> set[str]:
    labels = annotation.get("labels", [])
    if not isinstance(labels, list):
        return set()

    categories: set[str] = set()
    for label in labels:
        if not isinstance(label, dict):
            continue
        category = label.get("category")
        if isinstance(category, str) and category in classes:
            categories.add(category)
    return categories


def select_balanced_subset_indices(
    annotations: list[dict[str, Any]],
    max_images: int,
    seed: int,
    classes: list[str],
) -> list[int]:
    if not classes:
        sampler = random.Random(seed)
        return sorted(sampler.sample(range(len(annotations)), max_images))

    sampler = random.Random(seed)
    class_names = list(classes)
    class_set = set(class_names)
    target_per_class = max(1, max_images // len(class_names))

    indices_by_class: dict[str, list[int]] = {class_name: [] for class_name in class_names}
    for index, annotation in enumerate(annotations):
        for class_name in annotation_categories(annotation, class_set):
            indices_by_class[class_name].append(index)

    for indices in indices_by_class.values():
        sampler.shuffle(indices)

    selected_indices: set[int] = set()
    class_image_counts: dict[str, int] = {class_name: 0 for class_name in class_names}

    for class_name in class_names:
        for index in indices_by_class[class_name]:
            if len(selected_indices) >= max_images:
                break
            if class_image_counts[class_name] >= target_per_class:
                break
            if index in selected_indices:
                continue
            selected_indices.add(index)
            for present_class in annotation_categories(annotations[index], class_set):
                class_image_counts[present_class] += 1

    remaining_indices = [index for index in range(len(annotations)) if index not in selected_indices]
    sampler.shuffle(remaining_indices)
    selected_indices.update(remaining_indices[: max_images - len(selected_indices)])
    return sorted(selected_indices)


def build_class_map(classes: list[str]) -> dict[str, int]:
    return {class_name: index for index, class_name in enumerate(classes)}


def resolve_eval_split_name(split_names: list[str]) -> str:
    for split_name in YOLO_EVAL_SPLIT_NAMES:
        if split_name in split_names:
            return split_name

    raise ValueError("dataset.splits must include either a 'val' or 'eval' split for YOLO export.")


def parse_image_record(annotation: dict[str, Any]) -> ImageRecord:
    return parse_image_record_with_fallback(annotation)


def read_image_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as handle:
        signature = handle.read(24)
        if signature.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", signature[16:24])
            return width, height

        if signature[:2] == b"\xff\xd8":
            handle.seek(2)
            while True:
                marker_prefix = handle.read(1)
                if marker_prefix != b"\xff":
                    raise ValueError(f"Unsupported JPEG structure in {image_path}.")

                marker_type = handle.read(1)
                while marker_type == b"\xff":
                    marker_type = handle.read(1)

                if marker_type in {b"\xd8", b"\xd9"}:
                    continue

                segment_length_bytes = handle.read(2)
                if len(segment_length_bytes) != 2:
                    break
                segment_length = struct.unpack(">H", segment_length_bytes)[0]

                if marker_type in {
                    b"\xc0",
                    b"\xc1",
                    b"\xc2",
                    b"\xc3",
                    b"\xc5",
                    b"\xc6",
                    b"\xc7",
                    b"\xc9",
                    b"\xca",
                    b"\xcb",
                    b"\xcd",
                    b"\xce",
                    b"\xcf",
                }:
                    handle.read(1)
                    height, width = struct.unpack(">HH", handle.read(4))
                    return width, height

                handle.seek(segment_length - 2, 1)

    raise ValueError(f"Unable to determine image size for {image_path}.")


def locate_image_path(split_images_dir: Path, file_name: str) -> Path | None:
    direct_path = split_images_dir / file_name
    if direct_path.exists():
        return direct_path

    matches = list(split_images_dir.rglob(file_name))
    if not matches:
        return None

    return matches[0]


def parse_image_record_with_fallback(annotation: dict[str, Any], image_path: Path | None = None) -> ImageRecord:
    if not isinstance(annotation, dict):
        raise ValueError("BDD100K annotation entry must be a mapping.")

    file_name = annotation.get("name")
    attributes = annotation.get("attributes", {})
    if attributes is None:
        attributes = {}
    if not isinstance(attributes, dict):
        raise ValueError("BDD100K annotation attributes must be a mapping when present.")

    width = attributes.get("width", annotation.get("width"))
    height = attributes.get("height", annotation.get("height"))

    if (not width or not height) and image_path is not None and image_path.exists():
        width, height = read_image_size(image_path)

    if not file_name or not width or not height:
        raise ValueError("BDD100K annotation is missing image name or dimensions.")
    if not isinstance(annotation.get("labels", []), list):
        raise ValueError("BDD100K annotation labels must be a list.")

    return ImageRecord(
        file_name=str(file_name),
        width=int(width),
        height=int(height),
        labels=list(annotation.get("labels", [])),
    )


def convert_box_to_yolo(box2d: dict[str, Any], image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    try:
        x1 = float(box2d["x1"])
        y1 = float(box2d["y1"])
        x2 = float(box2d["x2"])
        y2 = float(box2d["y2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BDD100K box2d must contain numeric x1, y1, x2, y2 values.") from exc

    x1 = min(max(x1, 0.0), float(image_width))
    y1 = min(max(y1, 0.0), float(image_height))
    x2 = min(max(x2, 0.0), float(image_width))
    y2 = min(max(y2, 0.0), float(image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height


def format_yolo_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_annotation_to_yolo_lines(annotation: dict[str, Any], class_map: dict[str, int]) -> list[str]:
    return convert_annotation_to_yolo_lines_with_image(annotation, class_map)


def convert_annotation_to_yolo_lines_with_image(
    annotation: dict[str, Any],
    class_map: dict[str, int],
    image_path: Path | None = None,
) -> list[str]:
    image = parse_image_record_with_fallback(annotation, image_path=image_path)
    yolo_lines: list[str] = []

    for label in image.labels:
        category = label.get("category")
        box2d = label.get("box2d")
        if category not in class_map or not isinstance(box2d, dict):
            continue

        try:
            normalized_box = convert_box_to_yolo(box2d, image.width, image.height)
        except ValueError:
            continue
        if normalized_box is None:
            continue

        yolo_lines.append(format_yolo_line(class_map[category], normalized_box))

    return yolo_lines


def ensure_split_dirs(output_dir: Path, split_name: str) -> tuple[Path, Path]:
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def export_image_file(source_path: Path, destination_path: Path, export_mode: str) -> bool:
    if export_mode == "none":
        return False

    if not source_path.exists():
        return False

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()

    if export_mode == "copy":
        shutil.copy2(source_path, destination_path)
        return True

    if export_mode == "symlink":
        destination_path.symlink_to(source_path.resolve())
        return True

    raise ValueError(f"Unsupported image export mode: {export_mode}")


def handle_data_issue(policy: str, message: str) -> None:
    if policy == "error":
        raise ValueError(message)

    LOGGER.warning(message)


def write_dataset_yaml(output_dir: Path, class_names: list[str], eval_split_name: str) -> None:
    dataset_yaml = output_dir / "dataset.yaml"
    names_block = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images/train",
                f"val: images/{eval_split_name}",
                f"nc: {len(class_names)}",
                "names:",
                names_block,
                "",
            ]
        ),
        encoding="utf-8",
    )


def convert_split(
    split_name: str,
    split_config: dict[str, Any],
    output_dir: Path,
    class_map: dict[str, int],
    image_export_mode: str,
    missing_image_policy: str,
    malformed_annotation_policy: str,
    subset_enabled: bool,
    subset_seed: int,
    subset_max_images: int | None,
    class_names: list[str],
    subset_strategy: str,
) -> dict[str, int]:
    annotations = load_bdd_annotations(split_config["annotations"])
    annotations = select_subset(
        annotations,
        subset_enabled,
        subset_max_images,
        subset_seed,
        classes=class_names,
        strategy=subset_strategy,
    )
    images_dir, labels_dir = ensure_split_dirs(output_dir, split_name)
    split_images_dir = Path(split_config["images"])

    images_seen = 0
    labels_written = 0
    images_exported = 0
    missing_images = 0
    malformed_annotations = 0

    for annotation in annotations:
        image_name = str(annotation.get("name", ""))
        source_image_path = locate_image_path(split_images_dir, image_name) if image_name else None
        try:
            image = parse_image_record_with_fallback(annotation, image_path=source_image_path)
            yolo_lines = convert_annotation_to_yolo_lines_with_image(annotation, class_map, image_path=source_image_path)
        except ValueError as exc:
            malformed_annotations += 1
            handle_data_issue(
                malformed_annotation_policy,
                f"Skipping malformed annotation in split '{split_name}': {exc}",
            )
            continue

        label_path = labels_dir / f"{Path(image.file_name).stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        destination_image_path = images_dir / image.file_name
        exported = False
        if source_image_path is not None:
            exported = export_image_file(source_image_path, destination_image_path, image_export_mode)
        if exported:
            images_exported += 1
        elif image_export_mode != "none" and source_image_path is None:
            missing_images += 1
            handle_data_issue(
                missing_image_policy,
                f"Missing source image for split '{split_name}': {split_images_dir / image.file_name}",
            )

        images_seen += 1
        labels_written += len(yolo_lines)

    return {
        "images": images_seen,
        "labels": labels_written,
        "images_exported": images_exported,
        "missing_images": missing_images,
        "malformed_annotations": malformed_annotations,
        "annotations_loaded": len(annotations),
    }


def prepare_dataset(config: dict[str, Any]) -> dict[str, dict[str, int]]:
    dataset_config = config.get("dataset", {})
    class_names = list(dataset_config.get("classes", []))
    if not class_names:
        raise ValueError("dataset.classes must contain at least one class.")

    splits = dataset_config.get("splits", {})
    if not isinstance(splits, dict) or not splits:
        raise ValueError("dataset.splits must define at least one split.")
    if "train" not in splits:
        raise ValueError("dataset.splits must include a 'train' split.")

    output_dir = Path(dataset_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    image_export_mode = validate_policy(
        "dataset.image_export_mode",
        str(dataset_config.get("image_export_mode", "copy")),
        IMAGE_EXPORT_MODES,
    )
    missing_image_policy = validate_policy(
        "dataset.missing_image_policy",
        str(dataset_config.get("missing_image_policy", "warn")),
        DATA_ISSUE_POLICIES,
    )
    malformed_annotation_policy = validate_policy(
        "dataset.malformed_annotation_policy",
        str(dataset_config.get("malformed_annotation_policy", "warn")),
        DATA_ISSUE_POLICIES,
    )
    subset_config = dataset_config.get("subset", {})
    if subset_config is None:
        subset_config = {}
    if not isinstance(subset_config, dict):
        raise ValueError("dataset.subset must be a mapping when present.")
    subset_enabled = bool(subset_config.get("enabled", False))
    subset_seed = int(subset_config.get("seed", 42))
    subset_strategy = str(subset_config.get("strategy", "random"))
    raw_subset_max_images = subset_config.get("max_images_per_split")
    subset_max_images = None if raw_subset_max_images in (None, "") else int(raw_subset_max_images)

    class_map = build_class_map(class_names)
    split_stats: dict[str, dict[str, int]] = {}
    eval_split_name = resolve_eval_split_name(list(splits.keys()))

    for split_name, split_config in splits.items():
        split_stats[split_name] = convert_split(
            split_name,
            split_config,
            output_dir,
            class_map,
            image_export_mode,
            missing_image_policy,
            malformed_annotation_policy,
            subset_enabled,
            subset_seed,
            subset_max_images,
            class_names,
            subset_strategy,
        )

    write_dataset_yaml(output_dir, class_names, eval_split_name)
    return split_stats


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))
    split_stats = prepare_dataset(config)
    LOGGER.info("Prepared BDD100K YOLO export for %s.", config.get("project_name", "RoadSight"))
    for split_name, stats in split_stats.items():
        LOGGER.info(
            "Split %s: loaded %s annotations, wrote %s images, %s labels, exported %s image files, missing %s image files, malformed annotations %s.",
            split_name,
            stats["annotations_loaded"],
            stats["images"],
            stats["labels"],
            stats["images_exported"],
            stats["missing_images"],
            stats["malformed_annotations"],
        )


if __name__ == "__main__":
    main()
