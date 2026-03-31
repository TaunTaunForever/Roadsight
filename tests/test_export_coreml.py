from pathlib import Path

import pytest

from src.models import export_coreml as export_module


def test_validate_export_config_uses_trained_checkpoint() -> None:
    settings = export_module.validate_export_config(
        {
            "model_weights": "runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.pt",
            "output_dir": "runs/export",
            "run_name": "roadsight_coreml",
            "image_size": 640,
            "compute_precision": "float16",
            "include_nms": True,
        }
    )

    assert settings["model_weights"].endswith(
        "runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.pt"
    )
    assert settings["output_dir"].endswith("runs/export")
    assert settings["run_name"] == "roadsight_coreml"
    assert settings["compute_precision"] == "float16"
    assert settings["include_nms"] is True


def test_validate_export_config_requires_existing_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Export weights do not exist"):
        export_module.validate_export_config(
            {
                "model_weights": str(tmp_path / "missing.pt"),
                "output_dir": str(tmp_path / "export"),
            }
        )


def test_export_coreml_model_calls_ultralytics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"weights")
    artifact_path = tmp_path / "runs/export/roadsight_coreml/best.mlpackage"

    class FakeYOLO:
        def export(self, **kwargs: object) -> str:
            captured["kwargs"] = kwargs
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("coreml", encoding="utf-8")
            return str(artifact_path)

    monkeypatch.setattr(export_module, "load_yolo_model", lambda weights: FakeYOLO())

    outcome = export_module.export_coreml_model(
        {
            "model_weights": str(weights_path),
            "output_dir": str(tmp_path / "runs/export"),
            "run_name": "roadsight_coreml",
            "image_size": 640,
            "compute_precision": "float16",
            "include_nms": True,
        }
    )

    assert captured["kwargs"] == {
        "format": "coreml",
        "imgsz": 640,
        "project": str(tmp_path / "runs/export"),
        "name": "roadsight_coreml",
        "nms": True,
        "half": True,
        "int8": False,
    }
    assert outcome["artifact_path"] == str(artifact_path)
