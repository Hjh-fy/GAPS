from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.lab_three_gas_3class.evaluate_crossboard_scopes import (
    evaluate_scopes,
)
from scripts.lab_three_gas_3class.evaluate_exposure_checkpoint import (
    make_named_loader,
)


def test_one_checkpoint_is_evaluated_on_three_target_scopes(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[int], str]] = []

    def fake_evaluate(
        checkpoint: Path,
        data_root: Path,
        client_ids: list[int],
        split: str,
        output: Path,
        device: str,
    ) -> dict:
        calls.append((client_ids, split))
        return {"global": {"window": {"accuracy": 1.0}}}

    summary = evaluate_scopes(
        checkpoint=tmp_path / "server_round_025_adapted.pth",
        data_root=tmp_path / "fold_1",
        target_client=3,
        output_dir=tmp_path / "evaluation",
        device="cpu",
        evaluator=fake_evaluate,
    )

    assert calls == [([3], "test"), ([3], "early"), ([3], "full")]
    assert tuple(summary["scopes"]) == ("stable360", "early60", "full420")
    assert summary["target_client"] == 3


def test_named_loader_reads_early_prefix_without_renaming(tmp_path: Path) -> None:
    np.save(tmp_path / "early_features.npy", np.zeros((2, 100, 6), np.float32))
    np.save(
        tmp_path / "early_classification_labels.npy",
        np.asarray([0, 2], dtype=np.int64),
    )
    np.save(tmp_path / "early_phase_labels.npy", np.zeros(2, np.int64))

    batch = next(iter(make_named_loader(tmp_path, "early", batch_size=2)))

    assert tuple(batch[0].shape) == (2, 100, 6)
    assert batch[1].tolist() == [0, 2]
