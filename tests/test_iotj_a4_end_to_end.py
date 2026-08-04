from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.finalize_iotj_a4_end_to_end import (
    build_classifier_manifest,
    checkpoint_identity,
    prepare_output_root,
)


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_ROOT = ROOT / "results/iotj_final_classification_le1_20260804"


def test_cli_can_run_directly_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/finalize_iotj_a4_end_to_end.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--freeze-only" in completed.stdout


def test_checkpoint_identity_uses_ordered_state_content_not_container_bytes(
    tmp_path: Path,
) -> None:
    state = {
        "layer.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "layer.bias": torch.tensor([3.0], dtype=torch.float32),
    }
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    torch.save({"round": 25, "model_state": state, "note": "first"}, first)
    torch.save({"round": 25, "model_state": state, "note": "second"}, second)

    first_id = checkpoint_identity(first)
    second_id = checkpoint_identity(second)

    assert first_id["ordered_state_content_fingerprint"] == second_id[
        "ordered_state_content_fingerprint"
    ]
    assert first_id["whole_file_sha256"] != second_id["whole_file_sha256"]
    assert first_id["equality_basis"] == "ordered_state_content_fingerprint"
    assert first_id["whole_file_sha256_role"] == "provenance_only"


def test_classifier_manifest_freezes_c5_and_blocks_unavailable_c3_c4() -> None:
    manifest = build_classifier_manifest(CLASSIFICATION_ROOT)

    assert manifest["protocol"] == {
        "method": "server-centric A4",
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "seed": 42,
        "source_clients": ["C1", "C2"],
        "target_ce_weight": 0.0,
        "selective_aggregation": False,
        "fixed_endpoint_only": True,
    }
    assert manifest["targets"]["C5"]["status"] == "complete"
    assert manifest["targets"]["C5"]["accuracy"] == pytest.approx(
        0.9933823529411765
    )
    assert manifest["targets"]["C3"]["status"] == "blocked"
    assert manifest["targets"]["C4"]["status"] == "blocked"
    assert manifest["targets"]["C3"]["checkpoint"] is None
    assert manifest["targets"]["C4"]["checkpoint"] is None
    assert manifest["classification_retrained"] is False


def test_prepare_output_root_refuses_nonempty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "final"
    destination.mkdir()
    (destination / "existing.txt").write_text("frozen", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_output_root(destination)
