"""YOLO training entrypoint."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_config
from src.utils.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    parser.add_argument("--config", required=True, help="Path to the training config YAML.")
    return parser.parse_args()


def validate_training_config(training: dict[str, Any]) -> dict[str, Any]:
    data_config = training.get("data_config")
    model_weights = training.get("model_weights") or training.get("model_name")
    resume_checkpoint = training.get("resume_checkpoint")
    output_dir = training.get("output_dir")

    if not data_config:
        raise ValueError("training.data_config is required.")
    if not model_weights:
        raise ValueError("training.model_weights or training.model_name is required.")
    if not output_dir:
        raise ValueError("training.output_dir is required.")

    data_config_path = Path(str(data_config))
    if not data_config_path.exists():
        raise ValueError(f"Training data config does not exist: {data_config_path}")

    resume_checkpoint_path = None
    if resume_checkpoint:
        resume_checkpoint_path = Path(str(resume_checkpoint))
        if not resume_checkpoint_path.exists():
            raise ValueError(f"Training resume checkpoint does not exist: {resume_checkpoint_path}")

    return {
        "data_config": str(data_config_path),
        "model_weights": str(model_weights),
        "resume_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        "output_dir": str(output_dir),
        "run_name": str(training.get("run_name", "roadsight_train")),
        "epochs": int(training.get("epochs", 10)),
        "image_size": int(training.get("image_size", 640)),
        "batch_size": int(training.get("batch_size", 16)),
        "workers": int(training.get("workers", 4)),
        "device": training.get("device", "auto"),
        "use_all_available_gpus": bool(training.get("use_all_available_gpus", False)),
        "fallback_to_cpu_on_incompatible_gpu": bool(
            training.get("fallback_to_cpu_on_incompatible_gpu", True)
        ),
        "pretrained": bool(training.get("pretrained", True)),
        "resume": bool(training.get("resume", False)),
    }


def load_yolo_class() -> type:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Install dependencies with "
            "`./Roadsight_venv/bin/pip install -r requirements.txt`."
        ) from exc

    return YOLO


def detect_available_gpus() -> list[dict[str, str]]:
    try:
        import torch

        if torch.cuda.is_available():
            return [
                {"index": str(index), "name": torch.cuda.get_device_name(index)}
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        LOGGER.warning("PyTorch GPU detection failed: %s", exc)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    gpus: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        index, name = [part.strip() for part in line.split(",", 1)]
        gpus.append({"index": index, "name": name})
    return gpus


def detect_incompatible_gpus() -> list[dict[str, str]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        supported_arches = set(getattr(torch.cuda, "get_arch_list", lambda: [])())
        if not supported_arches:
            return []

        incompatible: list[dict[str, str]] = []
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            arch = f"sm_{major}{minor}"
            if arch not in supported_arches and not can_run_cuda_ops(index):
                incompatible.append(
                    {
                        "index": str(index),
                        "name": torch.cuda.get_device_name(index),
                        "arch": arch,
                    }
                )
        return incompatible
    except Exception as exc:
        LOGGER.warning("PyTorch GPU compatibility detection failed: %s", exc)
        return []


def can_run_cuda_ops(device_index: int) -> bool:
    try:
        import torch

        with torch.cuda.device(device_index):
            # Some PyTorch builds omit older SMs from get_arch_list even though
            # basic CUDA kernels still execute correctly on those devices.
            probe = torch.zeros(1, device=f"cuda:{device_index}")
            probe.add_(1)
            torch.cuda.synchronize(device_index)
        return True
    except Exception as exc:
        LOGGER.warning("CUDA runtime probe failed on GPU %s: %s", device_index, exc)
        return False


def normalize_device_value(device: Any) -> str:
    if isinstance(device, int):
        return str(device)
    if isinstance(device, (list, tuple)):
        return ",".join(str(item) for item in device)
    return str(device).strip()


def resolve_training_device(
    device: Any,
    use_all_available_gpus: bool,
    fallback_to_cpu_on_incompatible_gpu: bool,
) -> str:
    normalized = normalize_device_value(device).lower()
    if normalized in {"cpu", "mps"}:
        return normalized

    if normalized == "auto":
        gpus = detect_available_gpus()
        if not gpus:
            LOGGER.warning("No GPUs detected. Falling back to CPU.")
            return "cpu"
        incompatible_gpus = detect_incompatible_gpus()
        if incompatible_gpus:
            gpu_summary = ", ".join(
                f"{gpu['index']}:{gpu['name']} ({gpu['arch']})" for gpu in incompatible_gpus
            )
            message = (
                "Detected GPU(s) that are unsupported by the installed PyTorch build: "
                f"{gpu_summary}. "
                "Install a PyTorch build that supports these compute capabilities or use CPU."
            )
            if fallback_to_cpu_on_incompatible_gpu:
                LOGGER.warning("%s Falling back to CPU.", message)
                return "cpu"
            raise ValueError(message)
        if use_all_available_gpus and len(gpus) > 1:
            return ",".join(gpu["index"] for gpu in gpus)
        return gpus[0]["index"]

    return normalize_device_value(device)


def train_model(training: dict[str, Any]) -> dict[str, Any]:
    settings = validate_training_config(training)
    resolved_device = resolve_training_device(
        settings["device"],
        settings["use_all_available_gpus"],
        settings["fallback_to_cpu_on_incompatible_gpu"],
    )
    yolo_class = load_yolo_class()
    model_source = settings["resume_checkpoint"] or settings["model_weights"]
    model = yolo_class(model_source)
    train_kwargs = {
        "data": settings["data_config"],
        "epochs": settings["epochs"],
        "imgsz": settings["image_size"],
        "batch": settings["batch_size"],
        "project": settings["output_dir"],
        "name": settings["run_name"],
        "device": resolved_device,
        "workers": settings["workers"],
        "pretrained": settings["pretrained"],
    }
    if settings["resume"]:
        train_kwargs["resume"] = True
    result = model.train(**train_kwargs)

    LOGGER.info(
        "Training started with model=%s data=%s epochs=%s imgsz=%s batch=%s device=%s output=%s/%s resume=%s.",
        model_source,
        settings["data_config"],
        settings["epochs"],
        settings["image_size"],
        settings["batch_size"],
        resolved_device,
        settings["output_dir"],
        settings["run_name"],
        settings["resume"],
    )
    settings["resolved_device"] = resolved_device
    return {"settings": settings, "result": result}


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    configure_logging(config.get("runtime", {}).get("log_level", "INFO"))
    train_model(config.get("training", {}))


if __name__ == "__main__":
    main()
