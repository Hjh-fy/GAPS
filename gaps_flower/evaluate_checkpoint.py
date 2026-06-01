"""Evaluate a Flower server checkpoint on local GAPS client datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from federated_dataset import create_client_full_test_loader
from gaps_flower.task import create_model, make_config
from utils import evaluate_model


def parse_client_ids(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_checkpoint_model(checkpoint_path: str, device: str, batch_size: int):
    config = make_config(device=device, local_epochs=1, batch_size=batch_size)
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state")
    if state is None:
        raise ValueError(f"Checkpoint has no model_state: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GAPS Flower checkpoint")
    parser.add_argument("--checkpoint", default="server_latest.pth")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--client-ids", default="1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    model, config, checkpoint = load_checkpoint_model(args.checkpoint, args.device, args.batch_size)
    client_ids = parse_client_ids(args.client_ids)

    rows = []
    total_examples = 0
    weighted_acc = 0.0
    for client_id in client_ids:
        loader = create_client_full_test_loader(
            Path(args.data_root) / f"client_{client_id}",
            batch_size=config.BATCH_SIZE,
        )
        accuracy = float(evaluate_model(model, loader, torch.device(config.DEVICE)))
        num_examples = len(loader.dataset)
        total_examples += num_examples
        weighted_acc += accuracy * num_examples
        rows.append({
            "client_id": client_id,
            "num_examples": num_examples,
            "accuracy": accuracy,
        })

    summary = {
        "checkpoint": str(args.checkpoint),
        "round": int(checkpoint.get("round", -1)),
        "num_tensors": len(checkpoint.get("model_state", {})),
        "clients": rows,
        "weighted_accuracy": weighted_acc / total_examples if total_examples else 0.0,
        "total_examples": total_examples,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
