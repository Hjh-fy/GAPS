"""Evaluate a three-gas checkpoint at window and exposure levels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gaps_flower.evaluate_checkpoint import (  # noqa: E402
    load_checkpoint_model,
    make_loader,
    parse_client_ids,
    resolve_device,
)
from federated_dataset import GasSensorWindowDataset  # noqa: E402
from scripts.lab_three_gas_3class.train_centralized_baseline import (  # noqa: E402
    classification_metrics,
    exposure_metrics,
    load_manifest,
    predict,
)


def make_named_loader(
    client_dir: Path,
    prefix: str,
    batch_size: int,
) -> DataLoader:
    """Load a pre-normalized named evaluation split such as early or full."""
    features = np.load(client_dir / f"{prefix}_features.npy")
    labels = np.load(client_dir / f"{prefix}_classification_labels.npy")
    regression_path = client_dir / f"{prefix}_regression_labels.npy"
    regression = np.load(regression_path) if regression_path.is_file() else None
    phase_path = client_dir / f"{prefix}_phase_labels.npy"
    phase = (
        np.load(phase_path, allow_pickle=True)
        if phase_path.is_file()
        else np.full(len(features), -1, dtype=np.int64)
    )
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=regression,
        classification_labels=labels,
        phase_labels=phase,
        normalize=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--client-ids", default="1,2,3")
    parser.add_argument(
        "--split",
        choices=("test", "calibration", "early", "full"),
        default="test",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, config, checkpoint = load_checkpoint_model(
        str(args.checkpoint),
        device,
        args.batch_size,
    )
    client_results = {}
    all_labels = []
    all_probabilities = []
    all_manifest = []

    for client_id in parse_client_ids(args.client_ids):
        if args.split in {"early", "full"}:
            loader = make_named_loader(
                args.data_root / f"client_{client_id}",
                args.split,
                args.batch_size,
            )
        else:
            loader = make_loader(
                args.data_root,
                client_id,
                args.split,
                args.batch_size,
            )
        prefix = args.split
        manifest = load_manifest(
            args.data_root
            / f"client_{client_id}"
            / f"{prefix}_window_manifest.csv"
        )
        labels, probabilities, loss = predict(model, loader, device)
        client_results[str(client_id)] = {
            "loss": loss,
            "window": classification_metrics(labels, probabilities),
            "exposure": exposure_metrics(labels, probabilities, manifest),
        }
        all_labels.append(labels)
        all_probabilities.append(probabilities)
        all_manifest.extend(manifest)

    import numpy as np

    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    result = {
        "checkpoint": str(args.checkpoint),
        "round": int(checkpoint.get("round", -1)),
        "split": args.split,
        "device": str(device),
        "model_config": {
            "num_classes": config.NUM_CLASSES,
            "input_dim": config.INPUT_DIM,
            "num_clients": config.NUM_CLIENTS,
            "num_phases": config.NUM_PHASES,
            "seq_len": config.SEQ_LEN,
        },
        "clients": client_results,
        "global": {
            "window": classification_metrics(labels, probabilities),
            "exposure": exposure_metrics(labels, probabilities, all_manifest),
        },
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
