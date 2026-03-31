# RoadSight Data Flow

This repository is organized so raw BDD100K annotations and images flow into a processed YOLO dataset, then into training, evaluation, offline inference, and API serving.

## Filesystem map

```text
RoadSight/
├── configs/
│   ├── data.yaml          # Dataset prep configuration, classes, split paths, subset settings
│   ├── train.yaml         # Training configuration
│   └── inference.yaml     # Evaluation / inference / API configuration
├── data/
│   ├── raw/               # Real BDD100K files go here
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── bdd100k_labels_images_train.json
│   │       └── bdd100k_labels_images_val.json
│   └── processed/
│       └── bdd100k_yolo/  # Output from src.data.prepare_bdd
│           ├── dataset.yaml
│           ├── images/
│           │   ├── train/
│           │   └── val/ or eval/
│           └── labels/
│               ├── train/
│               └── val/ or eval/
├── demo/                  # Demo assets and walkthrough materials
├── scripts/               # Optional helper scripts
├── src/
│   ├── data/              # Dataset preparation logic
│   ├── models/            # Training and evaluation entrypoints
│   ├── inference/         # Image and video inference entrypoints
│   ├── serving/           # FastAPI application
│   └── utils/             # Shared helpers
└── tests/                 # Pytest coverage
```

## How the pieces connect

1. `configs/data.yaml` tells `src.data.prepare_bdd` where the raw BDD100K JSON and images live.
2. `src.data.prepare_bdd` filters classes, converts `box2d` annotations to YOLO labels, and copies or symlinks image files into `data/processed/bdd100k_yolo/`.
3. The generated `data/processed/bdd100k_yolo/dataset.yaml` becomes the handoff point for YOLO training.
4. `configs/train.yaml` points the training stage at the processed dataset.
5. `configs/inference.yaml` points evaluation, inference, and API serving at trained weights.

## Current configured defaults

The repo is currently aligned around these defaults:

- classes: `car`, `person`, `traffic light`, `traffic sign`, `bike`
- processed dataset path: `data/processed/bdd100k_yolo/dataset.yaml`
- training run name: `roadsight_bdd100k_subset_1000`
- recommended weights: `runs/detect/runs/train/roadsight_bdd100k_subset_1000/weights/best.pt`

This means evaluation, image inference, video inference, and the FastAPI app all share the same default model checkpoint.

## Subset workflow

When you have BDD100K locally:

1. Place the JSON label files under `data/raw/labels/`.
2. Place split image folders where `configs/data.yaml` expects them. In the current config, the labels come from `data/raw/labels/` while the image roots point at `bdd100k/bdd100k/images/100k/train` and `bdd100k/bdd100k/images/100k/val`.
3. Adjust `configs/data.yaml` if your local paths differ.
4. Set `dataset.subset.max_images_per_split` to a small number like `100`, `1000`, or `3000`.
5. Run:

```bash
./Roadsight_venv/bin/python -m src.data.prepare_bdd --config configs/data.yaml
```

That produces a smaller processed YOLO dataset under `data/processed/bdd100k_yolo/` that is suitable for initial training experiments.

## End-to-end flow

Once the processed dataset exists, the typical sequence is:

1. Train with `configs/train.yaml`.
2. Evaluate and infer with `configs/inference.yaml`.
3. Serve predictions through `src.serving.app`.
4. Verify behavior with `./Roadsight_venv/bin/python -m pytest -q`.
