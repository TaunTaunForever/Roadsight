import json
from pathlib import Path

import pytest

from src.data.prepare_bdd import (
    build_class_map,
    convert_annotation_to_yolo_lines,
    locate_image_path,
    parse_image_record,
    parse_image_record_with_fallback,
    prepare_dataset,
    read_image_size,
    select_subset,
)


def test_convert_annotation_filters_classes_and_normalizes_boxes() -> None:
    class_map = build_class_map(["car", "person"])
    annotation = {
        "name": "frame-001.jpg",
        "attributes": {"width": 100, "height": 50},
        "labels": [
            {"category": "car", "box2d": {"x1": 10, "y1": 5, "x2": 30, "y2": 25}},
            {"category": "traffic sign", "box2d": {"x1": 1, "y1": 1, "x2": 5, "y2": 5}},
            {"category": "person", "box2d": {"x1": -10, "y1": 10, "x2": 10, "y2": 40}},
        ],
    }

    lines = convert_annotation_to_yolo_lines(annotation, class_map)

    assert lines == [
        "0 0.200000 0.300000 0.200000 0.400000",
        "1 0.050000 0.500000 0.100000 0.600000",
    ]


def test_select_subset_is_deterministic() -> None:
    annotations = [{"name": f"frame_{index}.jpg"} for index in range(10)]

    subset_a = select_subset(annotations, subset_enabled=True, max_images=4, seed=7)
    subset_b = select_subset(annotations, subset_enabled=True, max_images=4, seed=7)

    assert subset_a == subset_b
    assert len(subset_a) == 4


def test_select_subset_can_balance_rare_classes() -> None:
    annotations = [
        {
            "name": "car_only_1.jpg",
            "labels": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}],
        },
        {
            "name": "car_only_2.jpg",
            "labels": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}],
        },
        {
            "name": "bike_only.jpg",
            "labels": [{"category": "bike", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}],
        },
        {
            "name": "person_only.jpg",
            "labels": [{"category": "person", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}],
        },
    ]

    subset = select_subset(
        annotations,
        subset_enabled=True,
        max_images=3,
        seed=7,
        classes=["car", "person", "bike"],
        strategy="balanced_by_class_presence",
    )

    selected_names = {annotation["name"] for annotation in subset}
    assert len(subset) == 3
    assert "bike_only.jpg" in selected_names
    assert "person_only.jpg" in selected_names


def test_read_image_size_supports_png(tmp_path: Path) -> None:
    png_path = tmp_path / "tiny.png"
    png_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D49484452"
            "0000000200000003"
            "0802000000"
            "12345678"
        )
    )

    assert read_image_size(png_path) == (2, 3)


def test_parse_image_record_can_infer_dimensions_from_image(tmp_path: Path) -> None:
    png_path = tmp_path / "frame.png"
    png_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D49484452"
            "0000000400000005"
            "0802000000"
            "12345678"
        )
    )

    image = parse_image_record_with_fallback(
        {"name": "frame.png", "attributes": {"weather": "clear"}, "labels": []},
        image_path=png_path,
    )

    assert image.width == 4
    assert image.height == 5


def test_locate_image_path_finds_nested_file(tmp_path: Path) -> None:
    nested_dir = tmp_path / "train/trainA"
    nested_dir.mkdir(parents=True)
    image_path = nested_dir / "frame.jpg"
    image_path.write_bytes(b"test")

    assert locate_image_path(tmp_path / "train", "frame.jpg") == image_path


def test_prepare_dataset_writes_yolo_labels_and_dataset_yaml(tmp_path: Path) -> None:
    train_annotations = tmp_path / "train.json"
    train_images_dir = tmp_path / "images/train"
    train_images_dir.mkdir(parents=True)
    (train_images_dir / "train_a.jpg").write_bytes(b"train-image")
    train_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "train_a.jpg",
                    "attributes": {"width": 200, "height": 100},
                    "labels": [
                        {"category": "car", "box2d": {"x1": 20, "y1": 10, "x2": 60, "y2": 30}},
                        {"category": "lane", "poly2d": []},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    val_annotations = tmp_path / "val.json"
    val_images_dir = tmp_path / "images/val"
    val_images_dir.mkdir(parents=True)
    (val_images_dir / "val_a.jpg").write_bytes(b"val-image")
    val_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "val_a.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [
                        {"category": "person", "box2d": {"x1": 25, "y1": 20, "x2": 75, "y2": 80}}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "yolo"
    config = {
        "dataset": {
            "output_dir": str(output_dir),
            "classes": ["car", "person"],
            "splits": {
                "train": {"annotations": str(train_annotations), "images": str(tmp_path / "images/train")},
                "val": {"annotations": str(val_annotations), "images": str(tmp_path / "images/val")},
            },
        }
    }

    stats = prepare_dataset(config)

    assert stats["train"]["images"] == 1
    assert stats["train"]["annotations_loaded"] == 1
    assert stats["train"]["labels"] == 1
    assert stats["train"]["images_exported"] == 1
    assert stats["train"]["missing_images"] == 0
    assert stats["train"]["malformed_annotations"] == 0
    assert stats["val"]["images"] == 1
    assert stats["val"]["annotations_loaded"] == 1
    assert stats["val"]["labels"] == 1
    assert stats["val"]["images_exported"] == 1
    assert stats["val"]["missing_images"] == 0
    assert stats["val"]["malformed_annotations"] == 0
    assert stats["train"]["images_exported"] == 1
    assert stats["val"]["images_exported"] == 1
    assert (output_dir / "labels/train/train_a.txt").read_text(encoding="utf-8") == (
        "0 0.200000 0.200000 0.200000 0.200000\n"
    )
    assert (output_dir / "labels/val/val_a.txt").read_text(encoding="utf-8") == (
        "1 0.500000 0.500000 0.500000 0.600000\n"
    )
    dataset_yaml = (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert "nc: 2" in dataset_yaml
    assert "0: car" in dataset_yaml
    assert "1: person" in dataset_yaml
    assert (output_dir / "images/train/train_a.jpg").read_bytes() == b"train-image"
    assert (output_dir / "images/val/val_a.jpg").read_bytes() == b"val-image"


def test_parse_image_record_raises_when_dimensions_are_missing() -> None:
    with pytest.raises(ValueError, match="missing image name or dimensions"):
        parse_image_record({"name": "frame.jpg", "labels": []})


def test_prepare_dataset_supports_eval_split_name(tmp_path: Path) -> None:
    train_annotations = tmp_path / "train.json"
    eval_annotations = tmp_path / "eval.json"
    train_images_dir = tmp_path / "images/train"
    eval_images_dir = tmp_path / "images/eval"
    train_images_dir.mkdir(parents=True)
    eval_images_dir.mkdir(parents=True)
    (train_images_dir / "train_a.jpg").write_bytes(b"train-image")
    (eval_images_dir / "eval_a.jpg").write_bytes(b"eval-image")
    train_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "train_a.jpg",
                    "attributes": {"width": 100, "height": 50},
                    "labels": [
                        {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 20, "y2": 10}}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    eval_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "eval_a.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [
                        {"category": "person", "box2d": {"x1": 10, "y1": 10, "x2": 40, "y2": 60}}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "yolo_eval"
    stats = prepare_dataset(
        {
            "dataset": {
                "output_dir": str(output_dir),
                "classes": ["car", "person"],
                "splits": {
                    "train": {"annotations": str(train_annotations), "images": str(tmp_path / "images/train")},
                    "eval": {"annotations": str(eval_annotations), "images": str(tmp_path / "images/eval")},
                },
            }
        }
    )

    assert stats["eval"]["images"] == 1
    assert stats["eval"]["annotations_loaded"] == 1
    assert stats["eval"]["labels"] == 1
    assert stats["eval"]["images_exported"] == 1
    assert stats["eval"]["missing_images"] == 0
    assert stats["eval"]["malformed_annotations"] == 0
    assert (output_dir / "labels/eval/eval_a.txt").read_text(encoding="utf-8") == (
        "1 0.250000 0.350000 0.300000 0.500000\n"
    )
    dataset_yaml = (output_dir / "dataset.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in dataset_yaml
    assert "val: images/eval" in dataset_yaml


def test_prepare_dataset_can_symlink_images(tmp_path: Path) -> None:
    train_annotations = tmp_path / "train.json"
    eval_annotations = tmp_path / "eval.json"
    train_images_dir = tmp_path / "images/train"
    eval_images_dir = tmp_path / "images/eval"
    train_images_dir.mkdir(parents=True)
    eval_images_dir.mkdir(parents=True)
    (train_images_dir / "train_a.jpg").write_bytes(b"train-image")
    (eval_images_dir / "eval_a.jpg").write_bytes(b"eval-image")

    train_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "train_a.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    eval_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "eval_a.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [{"category": "person", "box2d": {"x1": 20, "y1": 20, "x2": 60, "y2": 80}}],
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "yolo_symlink"
    stats = prepare_dataset(
        {
            "dataset": {
                "output_dir": str(output_dir),
                "image_export_mode": "symlink",
                "classes": ["car", "person"],
                "splits": {
                    "train": {"annotations": str(train_annotations), "images": str(train_images_dir)},
                    "eval": {"annotations": str(eval_annotations), "images": str(eval_images_dir)},
                },
            }
        }
    )

    train_export = output_dir / "images/train/train_a.jpg"
    eval_export = output_dir / "images/eval/eval_a.jpg"
    assert stats["train"]["images_exported"] == 1
    assert stats["eval"]["images_exported"] == 1
    assert stats["train"]["annotations_loaded"] == 1
    assert stats["eval"]["annotations_loaded"] == 1
    assert stats["train"]["malformed_annotations"] == 0
    assert stats["eval"]["malformed_annotations"] == 0
    assert train_export.is_symlink()
    assert eval_export.is_symlink()
    assert train_export.resolve() == (train_images_dir / "train_a.jpg").resolve()
    assert eval_export.resolve() == (eval_images_dir / "eval_a.jpg").resolve()


def test_prepare_dataset_warns_and_skips_malformed_annotations(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    train_images_dir = tmp_path / "images/train"
    val_images_dir = tmp_path / "images/val"
    train_images_dir.mkdir(parents=True)
    val_images_dir.mkdir(parents=True)
    (train_images_dir / "good.jpg").write_bytes(b"good")
    (val_images_dir / "val.jpg").write_bytes(b"val")

    train_annotations = tmp_path / "train.json"
    val_annotations = tmp_path / "val.json"
    train_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "good.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 20}}],
                },
                {
                    "name": "bad.jpg",
                    "attributes": {"width": 100},
                    "labels": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    val_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "val.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        stats = prepare_dataset(
            {
                "dataset": {
                    "output_dir": str(tmp_path / "yolo_warn"),
                    "classes": ["car", "person"],
                    "malformed_annotation_policy": "warn",
                    "splits": {
                        "train": {"annotations": str(train_annotations), "images": str(train_images_dir)},
                        "val": {"annotations": str(val_annotations), "images": str(val_images_dir)},
                    },
                }
            }
        )

    assert stats["train"]["images"] == 1
    assert stats["train"]["annotations_loaded"] == 2
    assert stats["train"]["malformed_annotations"] == 1
    assert "Skipping malformed annotation" in caplog.text


def test_prepare_dataset_errors_on_missing_images_when_configured(tmp_path: Path) -> None:
    train_images_dir = tmp_path / "images/train"
    val_images_dir = tmp_path / "images/val"
    train_images_dir.mkdir(parents=True)
    val_images_dir.mkdir(parents=True)
    (val_images_dir / "val.jpg").write_bytes(b"val")

    train_annotations = tmp_path / "train.json"
    val_annotations = tmp_path / "val.json"
    train_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "missing.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}}],
                }
            ]
        ),
        encoding="utf-8",
    )
    val_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "val.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing source image"):
        prepare_dataset(
            {
                "dataset": {
                    "output_dir": str(tmp_path / "yolo_error"),
                    "classes": ["car", "person"],
                    "missing_image_policy": "error",
                    "splits": {
                        "train": {"annotations": str(train_annotations), "images": str(train_images_dir)},
                        "val": {"annotations": str(val_annotations), "images": str(val_images_dir)},
                    },
                }
            }
        )


def test_prepare_dataset_errors_on_malformed_annotations_when_configured(tmp_path: Path) -> None:
    train_images_dir = tmp_path / "images/train"
    val_images_dir = tmp_path / "images/val"
    train_images_dir.mkdir(parents=True)
    val_images_dir.mkdir(parents=True)
    (val_images_dir / "val.jpg").write_bytes(b"val")

    train_annotations = tmp_path / "train.json"
    val_annotations = tmp_path / "val.json"
    train_annotations.write_text(
        json.dumps([{"name": "bad.jpg", "attributes": {"width": 100}, "labels": []}]),
        encoding="utf-8",
    )
    val_annotations.write_text(
        json.dumps(
            [
                {
                    "name": "val.jpg",
                    "attributes": {"width": 100, "height": 100},
                    "labels": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Skipping malformed annotation"):
        prepare_dataset(
            {
                "dataset": {
                    "output_dir": str(tmp_path / "yolo_error"),
                    "classes": ["car", "person"],
                    "malformed_annotation_policy": "error",
                    "splits": {
                        "train": {"annotations": str(train_annotations), "images": str(train_images_dir)},
                        "val": {"annotations": str(val_annotations), "images": str(val_images_dir)},
                    },
                }
            }
        )


def test_prepare_dataset_can_limit_subset_size(tmp_path: Path) -> None:
    train_images_dir = tmp_path / "images/train"
    val_images_dir = tmp_path / "images/val"
    train_images_dir.mkdir(parents=True)
    val_images_dir.mkdir(parents=True)

    train_annotations_data = []
    for index in range(5):
        file_name = f"train_{index}.jpg"
        (train_images_dir / file_name).write_bytes(b"train")
        train_annotations_data.append(
            {
                "name": file_name,
                "attributes": {"width": 100, "height": 100},
                "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}}],
            }
        )

    val_file_name = "val_0.jpg"
    (val_images_dir / val_file_name).write_bytes(b"val")
    val_annotations_data = [
        {
            "name": val_file_name,
            "attributes": {"width": 100, "height": 100},
            "labels": [{"category": "person", "box2d": {"x1": 20, "y1": 20, "x2": 40, "y2": 60}}],
        }
    ]

    train_annotations = tmp_path / "train.json"
    val_annotations = tmp_path / "val.json"
    train_annotations.write_text(json.dumps(train_annotations_data), encoding="utf-8")
    val_annotations.write_text(json.dumps(val_annotations_data), encoding="utf-8")

    stats = prepare_dataset(
        {
            "dataset": {
                "output_dir": str(tmp_path / "yolo_subset"),
                "classes": ["car", "person"],
                "subset": {"enabled": True, "seed": 7, "max_images_per_split": 2},
                "splits": {
                    "train": {"annotations": str(train_annotations), "images": str(train_images_dir)},
                    "val": {"annotations": str(val_annotations), "images": str(val_images_dir)},
                },
            }
        }
    )

    assert stats["train"]["annotations_loaded"] == 2
    assert stats["train"]["images"] == 2
    assert stats["train"]["images_exported"] == 2
    assert len(list((tmp_path / "yolo_subset/labels/train").glob("*.txt"))) == 2
