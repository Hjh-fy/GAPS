from pathlib import Path

import numpy as np

from scripts.lab_three_gas_3class.build_fivefold_dataset import (
    BuildConfig,
    build_folds,
    parse_normalization_clients,
)


def _records() -> list[dict]:
    records = []
    for platform in (1, 2, 3):
        for fold_group in (1, 2, 3, 4, 5):
            exposure_id = f"P{platform}_G{fold_group}"
            features = np.full(
                (2, 100, 6),
                platform * 10 + fold_group,
                dtype=np.float32,
            )
            records.append(
                {
                    "platform": platform,
                    "fold_group": fold_group,
                    "gas_label": (fold_group - 1) % 3,
                    "features": features,
                    "window_rows": [
                        {
                            "exposure_id": exposure_id,
                            "gas_label": (fold_group - 1) % 3,
                        }
                        for _ in range(len(features))
                    ],
                }
            )
    return records


def test_parse_normalization_clients() -> None:
    assert parse_normalization_clients("2") == (2,)
    assert parse_normalization_clients("1,2") == (1, 2)


def test_fold_normalization_uses_only_selected_source_client(
    tmp_path: Path,
) -> None:
    config = BuildConfig(
        raw_root="unused",
        output_root=str(tmp_path),
        normalization_clients=(2,),
    )
    build_folds(_records(), config, tmp_path)

    stats = np.load(tmp_path / "fold_1" / "norm_stats.npz")
    assert np.allclose(stats["mean"], 24.0)

    p2_train = np.load(
        tmp_path / "fold_1" / "client_2" / "train_features.npy"
    )
    p1_train = np.load(
        tmp_path / "fold_1" / "client_1" / "train_features.npy"
    )
    assert np.allclose(p2_train.mean(axis=(0, 1)), 0.0, atol=1e-6)
    assert np.allclose(p2_train.std(axis=(0, 1)), 1.0, atol=1e-6)
    assert not np.allclose(p1_train.mean(axis=(0, 1)), 0.0, atol=1e-3)
