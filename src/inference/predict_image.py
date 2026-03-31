"""Image inference entrypoint."""

from __future__ import annotations

import argparse
import logging

from src.inference.common import run_prediction
from src.utils.config import load_yaml_config
from src.utils.logging_utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("--config", required=True, help="Path to the inference config YAML.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))
    inference = config.get("inference", {})
    outcome = run_prediction(
        inference=inference,
        source=args.image,
        run_name=str(inference.get("image_run_name", "predict_image")),
    )
    logging.getLogger(__name__).info(
        "Image inference saved outputs to %s.",
        outcome["save_dir"],
    )


if __name__ == "__main__":
    main()
