# GAPS

Federated gas sensor learning system for cloud-edge deployment experiments.

This repository contains the core algorithm code only. Large datasets, checkpoints, result tables, and local notes are intentionally excluded from Git.

## Current Deployment Target

- Cloud side: Alibaba Cloud ECS running the Flower server.
- Edge side: local PC first, later Raspberry Pi, running the Flower client and holding local data.

## Core Files

- `model.py`: model definitions.
- `client.py`: local client training logic.
- `server.py`: aggregation and server-side learning logic.
- `config.py`: experiment and runtime configuration.
- `federated_dataset.py`: dataset and dataloader helpers.
- `exp_improved.py`: main experiment reproduction entry.
- `route_aware_response_anchoring.py`: route-aware QC/anchoring post-processing.

## Notes

Do not commit `dataset/`, `results/`, checkpoints, or generated figures. Keep those local or transfer them separately when needed.
