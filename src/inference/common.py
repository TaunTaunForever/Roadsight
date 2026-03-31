"""Shared helpers for RoadSight inference."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.models.train import resolve_training_device

LOGGER = logging.getLogger(__name__)


def load_yolo_class() -> type:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Install dependencies with "
            "`./Roadsight_venv/bin/pip install -r requirements.txt`."
        ) from exc

    return YOLO


def validate_inference_config(inference: dict[str, Any]) -> dict[str, Any]:
    model_weights = inference.get("model_weights")
    output_dir = inference.get("output_dir")
    data_config = inference.get("data_config")
    if not model_weights:
        raise ValueError("inference.model_weights is required.")
    if not output_dir:
        raise ValueError("inference.output_dir is required.")

    weights_path = Path(str(model_weights))
    if not weights_path.exists():
        raise ValueError(f"Inference weights do not exist: {weights_path}")

    data_config_path = None
    if data_config is not None:
        candidate = Path(str(data_config))
        if not candidate.exists():
            raise ValueError(f"Inference data config does not exist: {candidate}")
        data_config_path = str(candidate)

    return {
        "model_weights": str(weights_path),
        "data_config": data_config_path,
        "confidence_threshold": float(inference.get("confidence_threshold", 0.25)),
        "output_dir": str(output_dir),
        "eval_run_name": str(inference.get("eval_run_name", "evaluate")),
        "image_run_name": str(inference.get("image_run_name", "predict_image")),
        "video_run_name": str(inference.get("video_run_name", "predict_video")),
        "device": inference.get("device", "auto"),
        "save_visualizations": bool(inference.get("save_visualizations", True)),
        "save_json": bool(inference.get("save_json", False)),
    }


@lru_cache(maxsize=4)
def load_yolo_model(weights_path: str) -> Any:
    yolo_class = load_yolo_class()
    return yolo_class(weights_path)


def serialize_result(result: Any) -> dict[str, Any]:
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    detections: list[dict[str, Any]] = []
    if boxes is None:
        return {"detections": detections}

    xyxy_values = getattr(boxes, "xyxy", [])
    conf_values = getattr(boxes, "conf", [])
    cls_values = getattr(boxes, "cls", [])
    total = min(len(xyxy_values), len(conf_values), len(cls_values))
    for index in range(total):
        class_id = int(cls_values[index])
        detections.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "confidence": float(conf_values[index]),
                "box_xyxy": [float(value) for value in xyxy_values[index]],
            }
        )

    return {"detections": detections}


def predict_with_model(
    inference: dict[str, Any],
    source: str,
    run_name: str,
    save: bool | None = None,
) -> dict[str, Any]:
    settings = validate_inference_config(inference)
    source_path = Path(source)
    if not source_path.exists():
        raise ValueError(f"Inference source does not exist: {source_path}")

    resolved_device = resolve_training_device(
        settings["device"],
        use_all_available_gpus=False,
        fallback_to_cpu_on_incompatible_gpu=True,
    )
    model = load_yolo_model(settings["model_weights"])
    predict_kwargs = {
        "source": str(source_path),
        "conf": settings["confidence_threshold"],
        "project": settings["output_dir"],
        "name": run_name,
        "device": resolved_device,
        "save": settings["save_visualizations"] if save is None else save,
    }
    results = model.predict(**predict_kwargs)
    save_dir = Path(settings["output_dir"]) / run_name

    LOGGER.info(
        "Inference completed with weights=%s source=%s device=%s output=%s.",
        settings["model_weights"],
        source_path,
        resolved_device,
        save_dir,
    )
    return {
        "settings": settings | {"resolved_device": resolved_device},
        "source": str(source_path),
        "save_dir": str(save_dir),
        "results": results,
    }


def run_prediction(
    inference: dict[str, Any],
    source: str,
    run_name: str,
) -> dict[str, Any]:
    return predict_with_model(inference=inference, source=source, run_name=run_name)
