from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_registered_matrix_has_frozen_21_config_budget() -> None:
    from scripts.run_iotj_final_classification_le1 import (
        MATRIX_PATH,
        execution_counts,
        load_registered_matrix,
    )

    rows = load_registered_matrix(MATRIX_PATH)
    assert len(rows) == 21
    assert len({row["experiment_id"] for row in rows}) == 21
    assert execution_counts(rows) == {
        "registered_configs": 21,
        "new_full_fl_runs": 10,
        "e2_adaptation_branches": 9,
    }


@pytest.mark.parametrize(
    ("experiment_id", "server_tokens", "client_tokens", "optimizer"),
    [
        (
            "FCL-E1-FEDPROX",
            ["--strategy", "fedavg", "--rounds", "25"],
            ["--local-epochs", "1", "--proximal-mu", "0.01"],
            "Adam",
        ),
        (
            "FCL-E1-SCAFFOLD",
            ["--strategy", "scaffold", "--scaffold-lr", "0.0005"],
            ["--optimizer", "scaffold_sgd", "--local-epochs", "1"],
            "SGD",
        ),
        (
            "FCL-E3-GAPS-C5",
            [
                "--strategy",
                "gaps",
                "--selective-warmup",
                "5",
                "--require-selective-after-warmup",
                "true",
            ],
            ["--profile", "proto_replay", "--local-epochs", "1"],
            "Adam",
        ),
    ],
)
def test_three_host_commands_lock_optimizer_and_protocol(
    experiment_id: str,
    server_tokens: list[str],
    client_tokens: list[str],
    optimizer: str,
) -> None:
    from scripts.run_iotj_final_classification_le1 import build_flower_commands

    commands = build_flower_commands(experiment_id)
    server = commands["server"]
    client = commands["client_c1"]
    for token in server_tokens:
        assert token in server
    for token in client_tokens:
        assert token in client
    assert commands["protocol"]["optimizer"] == optimizer
    assert commands["protocol"]["seed"] == 42
    assert commands["protocol"]["batch_size"] == 32
    assert "test" not in " ".join(server + client + commands["client_c2"]).lower()


def test_e2_branches_are_exact_x_only_fixed_endpoint_definitions() -> None:
    from scripts.run_iotj_final_classification_le1 import build_e2_spec

    for method in ("CORAL", "MMD", "DANN"):
        for target in ("C3", "C4", "C5"):
            spec = build_e2_spec(f"FCL-E2-{method}-{target}")
            assert spec["source_checkpoint_role"] == "P0A_round25"
            assert spec["target_fields"] == ["x"]
            assert spec["steps"] == 100
            assert spec["optimizer"] == "Adam"
            assert spec["optimizer_lr"] == 5e-4
            assert spec["coefficient"] == 0.5
            assert spec["target_ce"] is False
            assert spec["conditional"] is False
            assert spec["checkpoint_selection"] == "fixed_step_100"


def test_protocol_freeze_hash_and_formal_lock_are_immutable(tmp_path: Path) -> None:
    from scripts.run_iotj_final_classification_le1 import (
        assert_formal_lock_matches,
        protocol_freeze_hash,
        start_formal_lock,
    )

    matrix = tmp_path / "matrix.csv"
    protocol = tmp_path / "PROTOCOL.md"
    matrix.write_text("id,value\na,1\n", encoding="utf-8")
    protocol.write_text("frozen\n", encoding="utf-8")
    digest = protocol_freeze_hash([matrix, protocol])
    lock = tmp_path / "formal_training_started.lock"
    start_formal_lock(lock, digest=digest, freeze_commit="abc123")
    assert_formal_lock_matches(lock, digest=digest)

    matrix.write_text("id,value\na,2\n", encoding="utf-8")
    changed = protocol_freeze_hash([matrix, protocol])
    with pytest.raises(RuntimeError, match="protocol hash mismatch"):
        assert_formal_lock_matches(lock, digest=changed)
    with pytest.raises(FileExistsError, match="already exists"):
        start_formal_lock(lock, digest=changed, freeze_commit="def456")


def test_completion_marker_is_immutable_and_drives_resume(tmp_path: Path) -> None:
    from scripts.run_iotj_final_classification_le1 import (
        experiment_resume_status,
        write_completion_marker,
    )

    run_dir = tmp_path / "FCL-E2-CORAL-C5"
    run_dir.mkdir()
    assert experiment_resume_status(run_dir, expected_protocol_hash="hash1") == "pending"
    marker = write_completion_marker(
        run_dir,
        experiment_id="FCL-E2-CORAL-C5",
        protocol_hash="hash1",
        endpoint={"steps": 100},
    )
    assert marker.is_file()
    assert experiment_resume_status(run_dir, expected_protocol_hash="hash1") == "complete"
    with pytest.raises(FileExistsError, match="immutable completion marker"):
        write_completion_marker(
            run_dir,
            experiment_id="FCL-E2-CORAL-C5",
            protocol_hash="hash1",
            endpoint={"steps": 100},
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["fixed_endpoint"] == {"steps": 100}


def test_static_command_audit_hard_fails_target_test_paths() -> None:
    from scripts.audit_iotj_final_classification_le1 import audit_training_commands

    clean = audit_training_commands(
        {"server": ["python", "adapt.py", "--target-calibration", "client_5"]}
    )
    assert clean["passed"] is True
    leaked = audit_training_commands(
        {"server": ["python", "adapt.py", "--target-test", "client_5/test_features.npy"]}
    )
    assert leaked["passed"] is False
    assert leaked["severity"] == "blocking"


def test_strict_audit_summary_fails_closed_on_any_blocking_finding() -> None:
    from scripts.audit_iotj_final_classification_le1 import summarize_findings

    summary = summarize_findings(
        [
            {"check_id": "A", "passed": True, "severity": "blocking"},
            {"check_id": "B", "passed": False, "severity": "blocking"},
            {"check_id": "C", "passed": False, "severity": "informational"},
        ]
    )
    assert summary["status"] == "FAIL"
    assert summary["blocking_failures"] == ["B"]


def test_remote_training_ssh_closes_stdin() -> None:
    from scripts.run_iotj_final_classification_le1 import _process_command

    command = _process_command("example-host", "/runtime", ["python", "-m", "client"])
    assert command[:3] == ["ssh", "-n", "-o"]
