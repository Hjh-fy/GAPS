from pathlib import Path

import scripts.run_iotj_classification_cloud_edge as controller


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
