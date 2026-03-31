
# AGENTS.md

This file provides instructions for AI coding agents working on this repository.

## Project
RoadSight — an autonomous driving perception demo built with:

- BDD100K dataset
- YOLOv8 object detection
- FastAPI inference service
- OpenCV visualization
- pytest tests

The purpose of this project is to demonstrate **ML engineering capability**, including:

- dataset pipelines
- model training
- evaluation
- inference scripts
- API serving
- reproducible configuration

## Build commands

Install dependencies:

pip install -r requirements.txt

Run dataset preparation:

python -m src.data.prepare_bdd --config configs/data.yaml

Train model:

python -m src.models.train --config configs/train.yaml

Evaluate model:

python -m src.models.evaluate --config configs/inference.yaml

Run image inference:

python -m src.inference.predict_image --config configs/inference.yaml --image data/samples/example.jpg

Run video inference:

python -m src.inference.predict_video --config configs/inference.yaml --video data/samples/example.mp4

Start API:

uvicorn src.serving.app:app --reload

Run tests:

pytest -q

## Development priorities

1. Build repository scaffold
2. Implement BDD100K JSON → YOLO conversion
3. Run YOLOv8 training on subset
4. Implement evaluation export
5. Add inference scripts
6. Implement FastAPI prediction endpoint
7. Add tests and polish

## Coding style

- Prefer modular Python scripts
- Avoid large monolithic scripts
- Use configuration files rather than hardcoded paths
- Use logging instead of print where possible
