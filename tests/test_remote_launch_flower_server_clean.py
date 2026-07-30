from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path


def _launch_command(
    tmp_path: Path,
    monkeypatch,
    *,
    da_mode: str,
    target_ce_weight: float,
) -> list[str]:
    captured: dict[str, list[str]] = {}

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **_kwargs):
        captured["command"] = list(command)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remote_launch_flower_server_clean.py",
            "--project",
            str(tmp_path),
            "--results-root",
            str(tmp_path / "results"),
            "--run-id",
            f"test_{da_mode}",
            "--data-root",
            str(tmp_path / "data"),
            "--source-clients",
            "2",
            "--target-clients",
            "3",
            "--profile",
            "proto_replay",
            "--num-classes",
            "3",
            "--input-dim",
            "6",
            "--num-clients",
            "3",
            "--num-phases",
            "1",
            "--da-mode",
            da_mode,
            "--target-ce-weight",
            str(target_ce_weight),
        ],
    )
    runpy.run_path(
        "scripts/remote_launch_flower_server_clean.py",
        run_name="__main__",
    )
    return captured["command"]


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_corrected_b2_launch_contract(tmp_path: Path, monkeypatch) -> None:
    command = _launch_command(
        tmp_path,
        monkeypatch,
        da_mode="corrected_b2",
        target_ce_weight=0.0,
    )

    assert _value(command, "--profile") == "proto_replay"
    assert _value(command, "--da-preset") == "none"
    assert _value(command, "--da-use-coral") == "false"
    assert _value(command, "--da-use-adversarial") == "false"
    assert _value(command, "--use-proto-mmd") == "false"
    assert _value(command, "--da-mmd-objective") == "mmd2"
    assert (
        _value(command, "--da-stage-alignment")
        == "cross_domain_same_class_phase"
    )
    assert _value(command, "--da-lambda-target-ce") == "0.0"


def test_target_ce_launch_contract(tmp_path: Path, monkeypatch) -> None:
    command = _launch_command(
        tmp_path,
        monkeypatch,
        da_mode="corrected_b2",
        target_ce_weight=1.0,
    )

    assert _value(command, "--da-lambda-target-ce") == "1.0"
