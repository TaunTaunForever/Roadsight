from pathlib import Path

import pytest

from src.inference import common as inference_common


def test_validate_inference_config_uses_trained_checkpoint() -> None:
    settings = inference_common.validate_inference_config(
        {
            "model_weights": "runs/detect/runs/train/roadsight_bdd100k_subset_1000/weights/best.pt",
            "output_dir": "runs/inference",
            "confidence_threshold": 0.3,
        }
    )

    assert settings["model_weights"].endswith(
        "runs/detect/runs/train/roadsight_bdd100k_subset_1000/weights/best.pt"
    )
    assert settings["output_dir"] == "runs/inference"
    assert settings["confidence_threshold"] == 0.3


def test_validate_inference_config_requires_existing_weights(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Inference weights do not exist"):
        inference_common.validate_inference_config(
            {
                "model_weights": str(tmp_path / "missing.pt"),
                "output_dir": "runs/inference",
            }
        )


def test_run_prediction_calls_ultralytics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    source_path = tmp_path / "frame.jpg"
    source_path.write_bytes(b"frame")

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            captured["weights"] = weights

        def predict(self, **kwargs: object) -> list[str]:
            captured["kwargs"] = kwargs
            return ["predicted"]

    monkeypatch.setattr(inference_common, "load_yolo_class", lambda: FakeYOLO)
    monkeypatch.setattr(inference_common, "resolve_training_device", lambda *args, **kwargs: "cpu")

    outcome = inference_common.run_prediction(
        inference={
            "model_weights": "runs/detect/runs/train/roadsight_bdd100k_subset_1000/weights/best.pt",
            "output_dir": "runs/inference",
            "confidence_threshold": 0.4,
            "save_visualizations": True,
        },
        source=str(source_path),
        run_name="predict_image",
    )

    assert captured["weights"].__str__().endswith(
        "runs/detect/runs/train/roadsight_bdd100k_subset_1000/weights/best.pt"
    )
    assert captured["kwargs"] == {
        "source": str(source_path),
        "conf": 0.4,
        "project": "runs/inference",
        "name": "predict_image",
        "device": "cpu",
        "save": True,
    }
    assert outcome["results"] == ["predicted"]
    assert outcome["save_dir"] == "runs/inference/predict_image"
