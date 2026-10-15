import json
from pathlib import Path

import pytest

from scripts.evaluate_iotj_canonical_v1_classification import completion_gate


def _write_run(root: Path, target: str) -> None:
    run = root / f"CANONICAL-V1-A4-{target}"
    run.mkdir(parents=True)
    checkpoint = run / "server_round_025_adapted.pth"
    checkpoint.write_bytes(b"checkpoint")
    (run / "fixed_endpoint_complete.json").write_text(
        json.dumps({"experiment_id": f"CANONICAL-V1-A4-{target}", "fixed_endpoint": {"round": 25}, "target_test_opened": False}),
        encoding="utf-8",
    )
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": f"CANONICAL-V1-A4-{target}",
                "checkpoint": str(checkpoint),
                "target_test_opened": False,
                "protocol": {
                    "classifier_router": "A4",
                    "local_epochs": 1,
                    "rounds": 25,
                    "checkpoint_reuse": False,
                    "target_test_selection": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_completion_gate_requires_all_three_a4_fixed_endpoints(tmp_path: Path) -> None:
    for target in ("C3", "C4", "C5"):
        _write_run(tmp_path, target)
    gate = completion_gate(tmp_path)
    assert gate["status"] == "PASS"
    assert gate["targets"] == ["C3", "C4", "C5"]


def test_completion_gate_fails_before_any_sealed_test_access(tmp_path: Path) -> None:
    _write_run(tmp_path, "C3")
    _write_run(tmp_path, "C4")
    with pytest.raises(RuntimeError, match="C5|completion"):
        completion_gate(tmp_path)
