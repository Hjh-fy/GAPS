from pathlib import Path

import pytest

from scripts.evaluate_iotj_b5_server_da_round_curve import (
    VARIANTS,
    _checkpoints,
    _summary,
)


def test_checkpoints_requires_exact_rounds(tmp_path: Path) -> None:
    for round_id in range(1, 26):
        (tmp_path / f"server_round_{round_id:03d}_adapted.pth").touch()
    assert sorted(_checkpoints(tmp_path)) == list(range(1, 26))
    (tmp_path / "server_round_013_adapted.pth").unlink()
    with pytest.raises(RuntimeError, match="checkpoint rounds mismatch"):
        _checkpoints(tmp_path)


def test_summary_keeps_variant_evidence_status() -> None:
    rows = []
    for variant in VARIANTS:
        for round_id in range(1, 26):
            rows.append(
                {
                    "variant": variant,
                    "round": round_id,
                    "accuracy": 0.9 + round_id / 1000.0,
                }
            )
    summary = _summary(rows)
    assert summary["DA100"]["best_accuracy_round_descriptive_only"] == 25
    assert summary["DA30"]["evidence_status"].startswith(
        "blocked_observability_contract"
    )
