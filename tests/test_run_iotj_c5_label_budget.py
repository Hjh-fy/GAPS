import json
from pathlib import Path

import pytest

from scripts.run_iotj_c5_label_budget import (
    BUDGETS,
    METHODS,
    audit_commands,
    build_budget_commands,
    experiment_id,
    load_or_create_freeze,
)


def value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_matrix_contains_only_six_fresh_c5_fixed_endpoint_runs() -> None:
    identities = [experiment_id(method, budget) for method in METHODS for budget in BUDGETS]
    assert len(identities) == len(set(identities)) == 6
    assert identities == [
        "CAN-V1-C5-LB-A0T-B15-S42",
        "CAN-V1-C5-LB-A0T-B10-S42",
        "CAN-V1-C5-LB-A0T-B05-S42",
        "CAN-V1-C5-LB-A4-B15-S42",
        "CAN-V1-C5-LB-A4-B10-S42",
        "CAN-V1-C5-LB-A4-B05-S42",
    ]
    for method in METHODS:
        for budget in BUDGETS:
            commands = build_budget_commands(method, budget)
            protocol = commands["protocol"]
            flat = " ".join(
                token
                for role in ("server", "client_c1", "client_c2")
                for token in commands[role]
            )
            assert protocol["target"] == "C5"
            assert protocol["budget_pct"] == budget
            assert protocol["rounds"] == 25
            assert protocol["local_epochs"] == 1
            assert protocol["seed"] == 42
            assert protocol["checkpoint_reuse"] is False
            assert protocol["checkpoint_selection"] == "fixed_round_25"
            assert "--checkpoint" not in flat
            assert "--resume" not in flat
            assert "test_features" not in flat
            assert "test_classification" not in flat


@pytest.mark.parametrize("budget,expected_suffix", [(15, "15"), (10, "10"), (5, "05")])
def test_budget_commands_change_only_c5_calibration_identity_for_each_method(
    budget: int, expected_suffix: str
) -> None:
    for method in METHODS:
        commands = build_budget_commands(method, budget)
        server = commands["server"]
        assert value(server, "--server-calib-data").endswith(f"client_5_budget_{expected_suffix}")
        assert value(server, "--rounds") == "25"
        assert value(server, "--domain-adapt-steps") == "100"
        assert value(server, "--da-server-opt-lr") == "0.0005"
        assert value(commands["client_c1"], "--local-epochs") == "1"
        assert value(commands["client_c2"], "--local-epochs") == "1"


def test_a0t_and_a4_keep_their_frozen_loss_surfaces() -> None:
    a0t = build_budget_commands("A0T", 5)["server"]
    a4 = build_budget_commands("A4", 5)["server"]
    assert value(a0t, "--profile") == "ce_only"
    assert value(a0t, "--ablation-variant") == "A0T"
    assert value(a0t, "--da-lambda-target-ce") == "1.0"
    assert value(a0t, "--use-proto-mmd") == "false"
    for option in (
        "--da-lambda-coral", "--da-lambda-global-mmd", "--da-lambda-class-mmd",
        "--da-lambda-proto-anchor", "--da-lambda-adv", "--da-lambda-proto",
        "--da-lambda-consistency", "--da-lambda-residual", "--da-lambda-proto-mmd",
        "--da-lambda-stage-mmd",
    ):
        assert float(value(a0t, option)) == 0.0
    assert value(a4, "--profile") == "ce_stats"
    assert value(a4, "--ablation-variant") == "A4"
    assert value(a4, "--da-lambda-target-ce") == "0.0"
    assert value(a4, "--da-lambda-coral") == "0.5"
    assert value(a4, "--da-lambda-global-mmd") == "0.5"
    assert value(a4, "--da-lambda-stage-mmd") == "0.2"


def test_command_audit_passes_only_the_frozen_six_run_surface() -> None:
    audit = audit_commands()
    assert audit["status"] == "PASS"
    assert audit["run_count"] == 6
    assert audit["target_test_referenced"] is False
    assert audit["checkpoint_reuse_referenced"] is False


def test_pre_run_freeze_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "PRE_RUN_FREEZE.json"
    payload = {"status": "FROZEN", "protocol_hash": "abc", "freeze_commit": "head"}
    assert load_or_create_freeze(path, payload) == payload
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert load_or_create_freeze(path, payload) == payload
    with pytest.raises(RuntimeError, match="differs"):
        load_or_create_freeze(path, {**payload, "protocol_hash": "changed"})
