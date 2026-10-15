from pathlib import Path
import subprocess

import scripts.run_iotj_classification_cloud_edge as controller


def test_ssh_allows_20_second_hotspot_handshake(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(controller, "_run", fake_run)

    controller._ssh("gaps@pi", "echo PI_READY", timeout=15)

    assert calls[0][0] == [
        "ssh",
        "-n",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
        "gaps@pi",
        "echo PI_READY",
    ]


def test_wait_for_pi_allows_full_hotspot_connect_budget(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_ssh(_host: str, _command: str, **kwargs):
        calls.append(dict(kwargs))
        return subprocess.CompletedProcess([], 0, "PI_READY\n", "")

    monkeypatch.setattr(controller, "_ssh", fake_ssh)

    assert controller._wait_for_pi(["gaps@pi"], 0, 60) == "gaps@pi"
    assert calls == [{"timeout": 30, "check": False}]


def test_wait_for_pi_retries_after_single_ssh_timeout(monkeypatch) -> None:
    calls = 0

    def fake_ssh(_host: str, _command: str, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(["ssh"], 30)
        return subprocess.CompletedProcess([], 0, "PI_READY\n", "")

    monkeypatch.setattr(controller, "_ssh", fake_ssh)
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: None)

    assert controller._wait_for_pi(["gaps@pi"], 1, 1) == "gaps@pi"
    assert calls == 2


def test_v2_default_queue_contains_only_core_screening_groups() -> None:
    assert controller.DEFAULT_GROUPS == (
        "A0", "A0T", "A2", "A3", "A4", "A4S", "A5", "A6", "A7"
    )


def test_ecs_sync_uploads_runtime_and_frozen_commands(monkeypatch, tmp_path: Path) -> None:
    command_root = tmp_path / "iotj_classification_ablation_20260711_v2_commands"
    command_root.mkdir()
    ssh_calls: list[tuple[str, str]] = []
    scp_calls: list[tuple[list[Path], str]] = []
    run_calls: list[list[str]] = []

    monkeypatch.setattr(
        controller,
        "_ssh",
        lambda host, command, **_kwargs: ssh_calls.append((host, command)),
    )
    monkeypatch.setattr(
        controller,
        "_scp_to_remote",
        lambda paths, destination, **_kwargs: scp_calls.append((list(paths), destination)),
    )
    monkeypatch.setattr(
        controller,
        "_run",
        lambda command, **_kwargs: run_calls.append(list(command)),
    )

    controller._sync_ecs("root@example", command_root, "/root/GAPS")

    assert ssh_calls == [
        ("root@example", "mkdir -p '/root/GAPS/gaps_flower' '/root/GAPS/results'")
    ]
    uploaded_names = {path.name for paths, _destination in scp_calls for path in paths}
    assert {"client.py", "task.py", "server_app.py", "strategy.py", "domain_adaptation.py"} <= uploaded_names
    assert ["scp", "-pr", str(command_root), "root@example:/root/GAPS/results/"] in run_calls


def test_remote_launcher_normalizes_shell_line_endings(monkeypatch) -> None:
    captured: list[str] = []

    def fake_remote_python(_host: str, _python_bin: str, source: str) -> str:
        captured.append(source)
        return "1234"

    monkeypatch.setattr(controller, "_remote_python", fake_remote_python)

    pid = controller._remote_launch_script(
        "host", "python", "/project", "/project/run.sh", "/project/run.log"
    )

    assert pid == 1234
    assert "replace(b'\\r\\n', b'\\n').replace(b'\\r', b'\\n')" in captured[0]
