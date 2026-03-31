
# RoadSight Implementation Plan

## Objective

Create a GitHub-ready ML project demonstrating an end‑to‑end perception pipeline.

## Core components

1. Dataset preparation
2. Model training
3. Evaluation
4. Image inference
5. Video inference
6. API service
7. Tests

## Repo structure

roadsight/
├── configs/
├── data/
├── demo/
├── scripts/
├── src/
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── serving/
│   └── utils/
└── tests/

## Milestones

Milestone 1 — Scaffold
Milestone 2 — Dataset conversion
Milestone 3 — Training
Milestone 4 — Evaluation
Milestone 5 — Inference
Milestone 6 — API
Milestone 7 — Tests

## Dataset subset

Start with classes:

- car
- person
- traffic light
- traffic sign
- bike

Train on 3k–10k images initially.
