"""Model evaluation entrypoint."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.inference.common import load_yolo_model, validate_inference_config
from src.models.train import resolve_training_device
from src.utils.config import load_yaml_config
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", required=True, help="Path to the inference config YAML.")
    return parser.parse_args()


def run_evaluation(inference: dict[str, Any]) -> dict[str, Any]:
    settings = validate_inference_config(inference)
    if not settings["data_config"]:
        raise ValueError("inference.data_config is required for evaluation.")

    resolved_device = resolve_training_device(
        settings["device"],
        use_all_available_gpus=False,
        fallback_to_cpu_on_incompatible_gpu=True,
    )
    model = load_yolo_model(settings["model_weights"])
    val_kwargs = {
        "data": settings["data_config"],
        "conf": settings["confidence_threshold"],
        "project": settings["output_dir"],
        "name": settings["eval_run_name"],
        "device": resolved_device,
        "save_json": settings["save_json"],
        "split": "val",
    }
    results = model.val(**val_kwargs)
    save_dir = Path(settings["output_dir"]) / settings["eval_run_name"]

    LOGGER.info(
        "Evaluation completed with weights=%s data=%s device=%s output=%s.",
        settings["model_weights"],
        settings["data_config"],
        resolved_device,
        save_dir,
    )
    return {
        "settings": settings | {"resolved_device": resolved_device},
        "save_dir": str(save_dir),
        "results": results,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))
    outcome = run_evaluation(config.get("inference", {}))
    LOGGER.info("Evaluation artifacts saved to %s.", outcome["save_dir"])


if __name__ == "__main__":
    main()
