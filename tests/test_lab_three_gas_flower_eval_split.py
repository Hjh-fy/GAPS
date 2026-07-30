from pathlib import Path

import numpy as np
import torch

from federated_dataset import create_merged_calibration_loader
from gaps_flower.task import load_client_loaders, make_config


def _write_classification_split(
    client_dir: Path,
    prefix: str,
    rows: int,
) -> None:
    features = np.zeros((rows, 100, 6), dtype=np.float32)
    labels = np.arange(rows, dtype=np.int64) % 3
    phases = np.zeros(rows, dtype=np.int64)
    np.save(client_dir / f"{prefix}_features.npy", features)
    np.save(client_dir / f"{prefix}_classification_labels.npy", labels)
    np.save(client_dir / f"{prefix}_phase_labels.npy", phases)


def test_calibration_loader_accepts_classification_only_data(tmp_path: Path) -> None:
    client_dir = tmp_path / "client_1"
    client_dir.mkdir()
    _write_classification_split(client_dir, "calibration", rows=6)

    loader = create_merged_calibration_loader([client_dir], batch_size=4)

    assert len(loader.dataset) == 6
    batch = next(iter(loader))
    assert tuple(batch[0].shape[1:]) == (100, 6)
    assert batch[1].dtype == torch.int64


def test_flower_formal_eval_can_use_calibration_without_touching_test(
    tmp_path: Path,
) -> None:
    client_dir = tmp_path / "client_1"
    client_dir.mkdir()
    _write_classification_split(client_dir, "train", rows=9)
    _write_classification_split(client_dir, "calibration", rows=6)
    # Deliberately omit test_* to prove the formal round path is validation-only.

    config = make_config(
        device="cpu",
        local_epochs=3,
        batch_size=4,
        profile="smoke",
        seed=42,
        num_classes=3,
        input_dim=6,
        num_clients=3,
        num_phases=1,
    )
    train_loader, eval_loader = load_client_loaders(
        tmp_path,
        1,
        config,
        eval_split="calibration",
    )

    assert len(train_loader.dataset) == 9
    assert len(eval_loader.dataset) == 6
