"""Core ML export entrypoint for Apple deployment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.inference.common import load_yolo_model
from src.utils.config import load_yaml_config
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained YOLO model to Core ML.")
    parser.add_argument("--config", required=True, help="Path to the export config YAML.")
    return parser.parse_args()


def validate_export_config(export: dict[str, Any]) -> dict[str, Any]:
    model_weights = export.get("model_weights")
    output_dir = export.get("output_dir")

    if not model_weights:
        raise ValueError("export.model_weights is required.")
    if not output_dir:
        raise ValueError("export.output_dir is required.")

    weights_path = Path(str(model_weights))
    if not weights_path.exists():
        raise ValueError(f"Export weights do not exist: {weights_path}")

    output_path = Path(str(output_dir))
    output_path.mkdir(parents=True, exist_ok=True)

    return {
        "model_weights": str(weights_path),
        "output_dir": str(output_path),
        "run_name": str(export.get("run_name", "roadsight_coreml")),
        "image_size": int(export.get("image_size", 640)),
        "compute_precision": str(export.get("compute_precision", "float16")),
        "include_nms": bool(export.get("include_nms", True)),
    }


def export_coreml_model(export: dict[str, Any]) -> dict[str, Any]:
    settings = validate_export_config(export)
    model = load_yolo_model(settings["model_weights"])

    export_kwargs = {
        "format": "coreml",
        "imgsz": settings["image_size"],
        "project": settings["output_dir"],
        "name": settings["run_name"],
        "nms": settings["include_nms"],
        "half": settings["compute_precision"].lower() == "float16",
        "int8": settings["compute_precision"].lower() == "int8",
    }
    artifact_path = model.export(**export_kwargs)
    save_path = Path(str(artifact_path))

    LOGGER.info(
        "Core ML export completed with weights=%s output=%s.",
        settings["model_weights"],
        save_path,
    )
    return {
        "settings": settings,
        "artifact_path": str(save_path),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))
    outcome = export_coreml_model(config.get("export", {}))
    LOGGER.info("Core ML artifact saved to %s.", outcome["artifact_path"])


if __name__ == "__main__":
    main()
