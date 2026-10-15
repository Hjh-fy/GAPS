from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from client import Client
from gaps_flower.task import make_config


def test_g2_profile_is_only_ce_plus_direct_prototype_alignment() -> None:
    config = make_config(
        device="cpu", local_epochs=1, batch_size=32, profile="dg_proto", seed=42
    )
    assert config.USE_ALIGN is True
    assert config.USE_CONTRASTIVE_ALIGN is False
    assert config.USE_REPLAY_DISTILL is False
    assert config.USE_PROTO_DECOUPLING is False
    assert config.UPLOAD_PROTO_STATS is True
    assert config.LAMBDA_ALIGN == 0.05


def test_g2_alignment_normalizes_feature_and_global_prototype() -> None:
    config = make_config(
        device="cpu", local_epochs=1, batch_size=32, profile="dg_proto", seed=42
    )
    client = Client(client_id=1, config=config)
    features = torch.tensor([[2.0, 0.0]])
    classes = torch.tensor([0])
    phases = torch.tensor([1])
    prototypes = {(0, 1): torch.tensor([0.0, 3.0])}
    loss = client._compute_align_loss(features, classes, phases, prototypes)
    assert torch.isclose(loss, torch.tensor(0.1), atol=1e-7)


def test_g2_commands_have_no_target_access_and_no_extra_mechanisms() -> None:
    from scripts.run_iotj_source_dg_g2 import build_g2_commands

    commands = build_g2_commands()
    joined = " ".join(commands["server"] + commands["client_c1"] + commands["client_c2"])
    assert "client_5" not in joined
    assert "server-calib-data" not in joined
    assert "server-val-data" not in joined
    assert "--strategy gaps" in " ".join(commands["server"])
    assert "--profile dg_proto" in " ".join(commands["server"])
    assert "--use-selective-agg false" in " ".join(commands["server"])
    assert "--use-domain-adapt false" in " ".join(commands["server"])
    assert "--use-proto-mmd false" in " ".join(commands["server"])
    assert commands["protocol"]["target_x"] is False
    assert commands["protocol"]["target_y"] is False
    assert commands["protocol"]["target_phase"] is False
    assert commands["protocol"]["lambda_proto"] == 0.05


def test_g2_round_one_has_no_prototype_broadcast() -> None:
    from scripts.run_iotj_source_dg_g2 import g2_round_contract

    assert g2_round_contract(1) == "CE_ONLY_UPLOAD_PROTOTYPES"
    assert g2_round_contract(2) == "CE_PLUS_GLOBAL_PROTOTYPE_ALIGNMENT"
    assert g2_round_contract(25) == "CE_PLUS_GLOBAL_PROTOTYPE_ALIGNMENT"


def test_g2_preserves_and_retries_exact_lock_only_preflight_failure(tmp_path: Path) -> None:
    from scripts.run_iotj_source_dg_g2 import prepare_lock_only_retry

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    commands = {"server": ["fixed"], "protocol": {"seed": 42}}
    (run_dir / "locked_run_spec.json").write_text(json.dumps(commands), encoding="utf-8")
    archived = prepare_lock_only_retry(run_dir, commands)
    assert not run_dir.exists()
    assert archived.name.startswith("preflight_failure_lock_only")
    assert json.loads((archived / "locked_run_spec.json").read_text(encoding="utf-8")) == commands


def test_g2_refuses_retry_if_partial_run_has_any_execution_artifact(tmp_path: Path) -> None:
    from scripts.run_iotj_source_dg_g2 import prepare_lock_only_retry

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    commands = {"server": ["fixed"], "protocol": {"seed": 42}}
    (run_dir / "locked_run_spec.json").write_text(json.dumps(commands), encoding="utf-8")
    (run_dir / "server.stderr.log").write_text("started", encoding="utf-8")
    with pytest.raises(FileExistsError, match="execution artifacts"):
        prepare_lock_only_retry(run_dir, commands)


def test_g2_ssh_transport_retries_only_return_code_255() -> None:
    from scripts.run_iotj_source_dg_g2 import ssh_with_transport_retry

    calls = []

    def transient(_host: str, _command: str, *, timeout: float = 120.0) -> str:
        calls.append(timeout)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(255, ["ssh"])
        return "OK"

    assert ssh_with_transport_retry(transient, "host", "command", attempts=3) == "OK"
    assert len(calls) == 3

    def remote_failure(_host: str, _command: str, *, timeout: float = 120.0) -> str:
        raise subprocess.CalledProcessError(1, ["ssh"])

    with pytest.raises(subprocess.CalledProcessError) as exc:
        ssh_with_transport_retry(remote_failure, "host", "command", attempts=3)
    assert exc.value.returncode == 1
