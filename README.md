# RoadSight

RoadSight is an autonomous driving perception demo built to showcase end-to-end ML engineering work around the BDD100K dataset, YOLOv8 object detection, offline inference, and API serving.

## Project Status

The core milestone plan is complete:

- repository scaffold and configuration
- BDD100K JSON to YOLO dataset conversion
- YOLOv8 training on a subset workflow
- evaluation export
- image and video inference entrypoints
- FastAPI serving with health and prediction endpoints
- pytest coverage across the pipeline

The project is now in documentation, validation, and portfolio-polish territory rather than initial feature buildout.

## Implemented Components

### Dataset Preparation

The dataset preparation pipeline in [src/data/prepare_bdd.py](/home/daniel/Development/Roadsight/src/data/prepare_bdd.py) supports:

- target class filtering
- BDD100K `box2d` to YOLO label conversion
- image export via copy, symlink, or label-only mode
- deterministic subset sampling for smaller first-pass experiments
- configurable warning or error policies for missing images and malformed annotations

The processed YOLO dataset is written to [data/processed/bdd100k_yolo](/home/daniel/Development/Roadsight/data/processed/bdd100k_yolo).

### Training

The training entrypoint in [src/models/train.py](/home/daniel/Development/Roadsight/src/models/train.py) loads configuration from [configs/train.yaml](/home/daniel/Development/Roadsight/configs/train.yaml) and runs Ultralytics YOLO training against the processed dataset.

The current recommended checkpoint is:

- `runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.pt`

This aligns with the active inference configuration in [configs/inference.yaml](/home/daniel/Development/Roadsight/configs/inference.yaml).

### Evaluation

The evaluation entrypoint in [src/models/evaluate.py](/home/daniel/Development/Roadsight/src/models/evaluate.py) runs validation on the configured YOLO dataset and writes artifacts under `runs/inference`.

Evaluation assets currently exist under:

- [runs/detect/runs/inference/evaluate](/home/daniel/Development/Roadsight/runs/detect/runs/inference/evaluate)
- [runs/detect/runs/inference/evaluate2](/home/daniel/Development/Roadsight/runs/detect/runs/inference/evaluate2)
- [runs/detect/runs/inference/evaluate_subset_10003_best](/home/daniel/Development/Roadsight/runs/detect/runs/inference/evaluate_subset_10003_best)
- [runs/detect/runs/inference/evaluate_full_yolov8s6_best2](/home/daniel/Development/Roadsight/runs/detect/runs/inference/evaluate_full_yolov8s6_best2)
- [runs/detect/runs/inference/evaluate_full_yolov8s6_continue10_best](/home/daniel/Development/Roadsight/runs/detect/runs/inference/evaluate_full_yolov8s6_continue10_best)

The latest formal evaluation for the recommended checkpoint produced:

- all classes: `precision 0.679`, `recall 0.563`, `mAP50 0.656`, `mAP50-95 0.377`
- car: `mAP50 0.797`, `mAP50-95 0.548`
- person: `mAP50 0.649`, `mAP50-95 0.371`
- traffic light: `mAP50 0.650`, `mAP50-95 0.277`
- traffic sign: `mAP50 0.675`, `mAP50-95 0.399`
- bike: `mAP50 0.511`, `mAP50-95 0.290`

### Inference

Inference entrypoints are implemented for both still images and video:

- [src/inference/predict_image.py](/home/daniel/Development/Roadsight/src/inference/predict_image.py)
- [src/inference/predict_video.py](/home/daniel/Development/Roadsight/src/inference/predict_video.py)

Shared model loading, device selection, and prediction helpers live in [src/inference/common.py](/home/daniel/Development/Roadsight/src/inference/common.py).

### API Serving

The FastAPI app in [src/serving/app.py](/home/daniel/Development/Roadsight/src/serving/app.py) provides:

- `GET /health`
- `POST /predict/image`

The API reads its default model settings from [configs/inference.yaml](/home/daniel/Development/Roadsight/configs/inference.yaml).

### Tests

Pytest coverage exists for:

- config loading
- dataset preparation
- training configuration and device resolution
- evaluation wiring
- inference wiring
- FastAPI health and prediction behavior

The current suite passes with `27 passed`.

## Repository Layout

```text
RoadSight/
├── configs/
├── data/
├── docs/
├── runs/
├── src/
│   ├── data/
│   ├── inference/
│   ├── models/
│   ├── serving/
│   └── utils/
└── tests/
```

See [docs/data_flow.md](/home/daniel/Development/Roadsight/docs/data_flow.md) for the filesystem and handoff flow between preparation, training, evaluation, inference, and serving.

For Apple deployment, see [docs/ios_deployment.md](/home/daniel/Development/Roadsight/docs/ios_deployment.md) and the Swift scaffold in [ios/RoadSightMobile](/home/daniel/Development/Roadsight/ios/RoadSightMobile).

## Setup

Install dependencies:

```bash
./Roadsight_venv/bin/python -m pip install -r requirements.txt
```

If you are not using the provided virtual environment, replace `./Roadsight_venv/bin/python` with your project Python interpreter.

## Common Commands

Prepare the YOLO dataset:

```bash
./Roadsight_venv/bin/python -m src.data.prepare_bdd --config configs/data.yaml
```

Train the model:

```bash
./Roadsight_venv/bin/python -m src.models.train --config configs/train.yaml
```

Evaluate the configured model:

```bash
./Roadsight_venv/bin/python -m src.models.evaluate --config configs/inference.yaml
```

Export the trained model for Apple deployment:

```bash
./Roadsight_venv/bin/python -m src.models.export_coreml --config configs/export.yaml
```

The current exported Core ML package is:

- [best.mlpackage](/home/daniel/Development/Roadsight/runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.mlpackage)

Run image inference:

```bash
./Roadsight_venv/bin/python -m src.inference.predict_image --config configs/inference.yaml --image path/to/image.jpg
```

Run video inference:

```bash
./Roadsight_venv/bin/python -m src.inference.predict_video --config configs/inference.yaml --video path/to/video.mp4
```

Start the API:

```bash
./Roadsight_venv/bin/python -m uvicorn src.serving.app:app --reload
```

Run tests:

```bash
./Roadsight_venv/bin/python -m pytest -q
```

## Current Defaults

The repo is currently configured around the full-dataset `yolov8s` workflow:

- classes: `car`, `person`, `traffic light`, `traffic sign`, `bike`
- subset sampling disabled in [configs/data.yaml](/home/daniel/Development/Roadsight/configs/data.yaml)
- training run name: `roadsight_bdd100k_full_yolov8s`
- training settings in [configs/train.yaml](/home/daniel/Development/Roadsight/configs/train.yaml): `model=yolov8s`, `epochs=10`, `batch_size=80`, `image_size=640`
- inference weights: `runs/detect/runs/train/roadsight_bdd100k_full_yolov8s6_continue10/weights/best.pt`

## Next Polishing Steps

What remains is not core feature implementation but final polish:

- expand project narrative and benchmark summary
- validate end-to-end demo flows with chosen sample assets
- tighten reproducibility and onboarding
- package the repo as a stronger portfolio-ready ML engineering showcase
