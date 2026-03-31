"""FastAPI application for RoadSight inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from src.inference.common import (
    load_yolo_model,
    predict_with_model,
    serialize_result,
    validate_inference_config,
)
from src.utils.config import load_yaml_config


def get_inference_settings(config_path: str | Path | None = None) -> dict[str, str]:
    resolved_path = Path(config_path or os.environ.get("ROADSIGHT_INFERENCE_CONFIG", "configs/inference.yaml"))
    config = load_yaml_config(resolved_path)
    return validate_inference_config(config.get("inference", {}))


def create_app(config_path: str | Path | None = None) -> FastAPI:
    inference_settings = get_inference_settings(config_path)
    app = FastAPI(title="RoadSight API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "roadsight",
            "model_weights": inference_settings["model_weights"],
        }

    @app.post("/predict/image")
    async def predict_image(request: Request) -> dict[str, object]:
        filename = request.headers.get("x-filename", "upload.jpg")
        suffix = Path(filename).suffix or ".jpg"
        temp_path = None
        try:
            payload = await request.body()
            if not payload:
                raise HTTPException(status_code=400, detail="Request body is empty.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)

            outcome = predict_with_model(
                inference=inference_settings,
                source=str(temp_path),
                run_name=inference_settings["image_run_name"],
                save=False,
            )
            first_result = outcome["results"][0] if outcome["results"] else None
            return {
                "model_weights": inference_settings["model_weights"],
                "source_filename": filename,
                "detections": serialize_result(first_result)["detections"] if first_result is not None else [],
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    return app


app = create_app()
