from pathlib import Path

import pytest

from src.models import train as train_module


def test_validate_training_config_points_at_processed_dataset() -> None:
    settings = train_module.validate_training_config(
        {
            "model_weights": "yolov8n.pt",
            "data_config": "data/processed/bdd100k_yolo/dataset.yaml",
            "output_dir": "runs/train",
            "run_name": "roadsight_bdd100k_subset",
            "epochs": 5,
            "image_size": 640,
            "batch_size": 8,
            "device": "auto",
            "use_all_available_gpus": True,
            "fallback_to_cpu_on_incompatible_gpu": True,
        }
    )

    assert settings["data_config"].endswith("data/processed/bdd100k_yolo/dataset.yaml")
    assert settings["model_weights"] == "yolov8n.pt"
    assert settings["run_name"] == "roadsight_bdd100k_subset"
    assert settings["device"] == "auto"
    assert settings["use_all_available_gpus"] is True
    assert settings["fallback_to_cpu_on_incompatible_gpu"] is True
    assert settings["resume_checkpoint"] is None
    assert settings["resume"] is False


def test_validate_training_config_requires_existing_dataset_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Training data config does not exist"):
        train_module.validate_training_config(
            {
                "model_weights": "yolov8n.pt",
                "data_config": str(tmp_path / "missing.yaml"),
                "output_dir": "runs/train",
            }
        )


def test_train_model_calls_ultralytics_with_expected_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            captured["weights"] = weights

        def train(self, **kwargs: object) -> str:
            captured["kwargs"] = kwargs
            return "trained"

    monkeypatch.setattr(train_module, "load_yolo_class", lambda: FakeYOLO)
    monkeypatch.setattr(
        train_module,
        "resolve_training_device",
        lambda device, use_all_available_gpus, fallback_to_cpu_on_incompatible_gpu: "0,1,2",
    )

    result = train_module.train_model(
        {
            "model_weights": "yolov8n.pt",
            "data_config": "data/processed/bdd100k_yolo/dataset.yaml",
            "output_dir": "runs/train",
            "run_name": "roadsight_bdd100k_subset",
            "epochs": 3,
            "image_size": 320,
            "batch_size": 4,
            "workers": 2,
            "device": "auto",
            "use_all_available_gpus": True,
            "fallback_to_cpu_on_incompatible_gpu": True,
            "pretrained": True,
        }
    )

    assert captured["weights"] == "yolov8n.pt"
    assert captured["kwargs"] == {
        "data": "data/processed/bdd100k_yolo/dataset.yaml",
        "epochs": 3,
        "imgsz": 320,
        "batch": 4,
        "project": "runs/train",
        "name": "roadsight_bdd100k_subset",
        "device": "0,1,2",
        "workers": 2,
        "pretrained": True,
    }
    assert result["result"] == "trained"
    assert result["settings"]["resolved_device"] == "0,1,2"


def test_validate_training_config_accepts_resume_checkpoint(tmp_path: Path) -> None:
    resume_checkpoint = tmp_path / "last.pt"
    resume_checkpoint.write_bytes(b"weights")

    settings = train_module.validate_training_config(
        {
            "model_weights": "yolov8s.pt",
            "resume_checkpoint": str(resume_checkpoint),
            "resume": True,
            "data_config": "data/processed/bdd100k_yolo/dataset.yaml",
            "output_dir": "runs/train",
        }
    )

    assert settings["resume_checkpoint"] == str(resume_checkpoint)
    assert settings["resume"] is True


def test_train_model_can_resume_from_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    resume_checkpoint = tmp_path / "last.pt"
    resume_checkpoint.write_bytes(b"weights")

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            captured["weights"] = weights

        def train(self, **kwargs: object) -> str:
            captured["kwargs"] = kwargs
            return "trained"

    monkeypatch.setattr(train_module, "load_yolo_class", lambda: FakeYOLO)
    monkeypatch.setattr(
        train_module,
        "resolve_training_device",
        lambda device, use_all_available_gpus, fallback_to_cpu_on_incompatible_gpu: "0,1,2",
    )

    result = train_module.train_model(
        {
            "model_weights": "yolov8s.pt",
            "resume_checkpoint": str(resume_checkpoint),
            "resume": True,
            "data_config": "data/processed/bdd100k_yolo/dataset.yaml",
            "output_dir": "runs/train",
            "run_name": "roadsight_resume",
            "epochs": 20,
            "image_size": 640,
            "batch_size": 80,
            "workers": 4,
            "device": "auto",
            "use_all_available_gpus": True,
            "fallback_to_cpu_on_incompatible_gpu": True,
            "pretrained": True,
        }
    )

    assert captured["weights"] == str(resume_checkpoint)
    assert captured["kwargs"] == {
        "data": "data/processed/bdd100k_yolo/dataset.yaml",
        "epochs": 20,
        "imgsz": 640,
        "batch": 80,
        "project": "runs/train",
        "name": "roadsight_resume",
        "device": "0,1,2",
        "workers": 4,
        "pretrained": True,
        "resume": True,
    }
    assert result["result"] == "trained"


def test_resolve_training_device_uses_all_detected_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_module,
        "detect_available_gpus",
        lambda: [
            {"index": "0", "name": "GeForce GTX 1080"},
            {"index": "1", "name": "GeForce GTX 1080"},
            {"index": "2", "name": "GeForce GTX 1080"},
        ],
    )
    monkeypatch.setattr(train_module, "detect_incompatible_gpus", lambda: [])

    assert (
        train_module.resolve_training_device(
            "auto",
            use_all_available_gpus=True,
            fallback_to_cpu_on_incompatible_gpu=True,
        )
        == "0,1,2"
    )


def test_resolve_training_device_falls_back_to_cpu_when_no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train_module, "detect_available_gpus", lambda: [])

    assert (
        train_module.resolve_training_device(
            "auto",
            use_all_available_gpus=True,
            fallback_to_cpu_on_incompatible_gpu=True,
        )
        == "cpu"
    )


def test_resolve_training_device_falls_back_on_incompatible_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_module,
        "detect_available_gpus",
        lambda: [{"index": "0", "name": "GeForce GTX 1080"}],
    )
    monkeypatch.setattr(
        train_module,
        "detect_incompatible_gpus",
        lambda: [{"index": "0", "name": "GeForce GTX 1080", "arch": "sm_61"}],
    )

    assert (
        train_module.resolve_training_device(
            "auto",
            use_all_available_gpus=True,
            fallback_to_cpu_on_incompatible_gpu=True,
        )
        == "cpu"
    )


def test_resolve_training_device_raises_on_incompatible_gpu_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_module,
        "detect_available_gpus",
        lambda: [{"index": "0", "name": "GeForce GTX 1080"}],
    )
    monkeypatch.setattr(
        train_module,
        "detect_incompatible_gpus",
        lambda: [{"index": "0", "name": "GeForce GTX 1080", "arch": "sm_61"}],
    )

    with pytest.raises(ValueError, match="unsupported by the installed PyTorch build"):
        train_module.resolve_training_device(
            "auto",
            use_all_available_gpus=True,
            fallback_to_cpu_on_incompatible_gpu=False,
        )
