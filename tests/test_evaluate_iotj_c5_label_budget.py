import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_iotj_c5_label_budget import (
    completion_gate,
    per_class_rows,
    source_retention_row,
)
from scripts.run_iotj_c5_label_budget import BUDGETS, METHODS, experiment_id


def make_completed_matrix(root: Path, protocol_hash: str = "frozen-protocol") -> None:
    for method in METHODS:
        for budget in BUDGETS:
            run_id = experiment_id(method, budget)
            run = root / run_id
            remote = run / "remote_server"
            remote.mkdir(parents=True)
            checkpoint = remote / "server_latest_adapted.pth"
            checkpoint.write_bytes(f"checkpoint-{run_id}".encode())
            manifest = {
                "experiment_id": run_id,
                "protocol_hash": protocol_hash,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "target_test_opened": False,
                "protocol": {
                    "method": method,
                    "budget_pct": budget,
                    "target": "C5",
                    "rounds": 25,
                    "local_epochs": 1,
                    "seed": 42,
                    "checkpoint_reuse": False,
                    "checkpoint_selection": "fixed_round_25",
                    "target_test_selection": False,
                },
            }
            marker = {
                "experiment_id": run_id,
                "protocol_hash": protocol_hash,
                "fixed_endpoint": {"round": 25, "checkpoint": checkpoint.name},
                "target_test_opened": False,
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run / "fixed_endpoint_complete.json").write_text(json.dumps(marker), encoding="utf-8")


def test_completion_gate_requires_all_six_unopened_round25_endpoints(tmp_path: Path) -> None:
    make_completed_matrix(tmp_path)
    gate = completion_gate(tmp_path, "frozen-protocol")
    assert gate["status"] == "PASS"
    assert len(gate["runs"]) == 6
    assert {item["method"] for item in gate["runs"].values()} == {"A0T", "A4"}


@pytest.mark.parametrize("mutation", ["missing", "round", "opened", "sha"])
def test_completion_gate_fails_closed_on_invalid_endpoint(tmp_path: Path, mutation: str) -> None:
    make_completed_matrix(tmp_path)
    run = tmp_path / experiment_id("A0T", 15)
    if mutation == "missing":
        (run / "fixed_endpoint_complete.json").unlink()
    elif mutation == "round":
        marker = json.loads((run / "fixed_endpoint_complete.json").read_text())
        marker["fixed_endpoint"]["round"] = 24
        (run / "fixed_endpoint_complete.json").write_text(json.dumps(marker))
    elif mutation == "opened":
        marker = json.loads((run / "fixed_endpoint_complete.json").read_text())
        marker["target_test_opened"] = True
        (run / "fixed_endpoint_complete.json").write_text(json.dumps(marker))
    else:
        manifest = json.loads((run / "run_manifest.json").read_text())
        manifest["checkpoint_sha256"] = "0" * 64
        (run / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="FAIL_CLOSED"):
        completion_gate(tmp_path, "frozen-protocol")


def test_per_class_rows_reports_precision_recall_and_f1() -> None:
    rows = per_class_rows([[8, 2], [1, 9]], class_names=("A", "B"))
    assert [(row["class_id"], row["class_name"], row["support"]) for row in rows] == [
        (0, "A", 10), (1, "B", 10)
    ]
    assert rows[0]["precision"] == pytest.approx(8 / 9)
    assert rows[0]["recall"] == pytest.approx(0.8)
    assert rows[0]["f1"] == pytest.approx(16 / 19)
    assert rows[1]["precision"] == pytest.approx(9 / 11)
    assert rows[1]["recall"] == pytest.approx(0.9)
    assert rows[1]["f1"] == pytest.approx(18 / 21)


def test_source_retention_row_is_relative_to_frozen_fedavg() -> None:
    row = source_retention_row(
        method="A4",
        budget=20,
        source_metrics={"N": 1360, "accuracy": 0.98, "macro_f1": 0.97},
        baseline_metrics={"accuracy": 0.99, "macro_f1": 0.985},
        checkpoint_sha256="abc",
    )
    assert row["budget_pct"] == 20
    assert row["source_accuracy_retention_delta"] == pytest.approx(-0.01)
    assert row["source_macro_f1_retention_delta"] == pytest.approx(-0.015)
