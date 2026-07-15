from __future__ import annotations

import inspect
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

import scripts.run_iotj_confirmation_observability as controller
from scripts.freeze_iotj_confirmation_protocol import (
    CONFIRMATION_SCHEDULE,
    canonical_sha256,
    sha256_file,
)
from scripts.run_iotj_confirmation_observability import (
    ArchiveMismatch,
    LifecycleHooks,
    Provenance,
    ValidationOutcome,
    allocate_attempt,
    copy_evidence_without_overwrite,
    deploy_source_archive,
    mark_attempt,
    run_confirmation_attempt,
    validate_host_preflight,
    validate_requested_run,
    verify_and_extract_archive,
)


PROVENANCE = Provenance(
    confirmation_commit="a" * 40,
    source_archive_sha256="b" * 64,
    dataset_manifest_sha256="c" * 64,
    algorithm_config_sha256="d" * 64,
)


def _read_status(path: Path) -> dict[str, object]:
    return json.loads((path / "attempt_status.json").read_text(encoding="utf-8"))


def _source_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    payload = b"print('frozen source')\n"
    archive_path = tmp_path / "source.tar"
    info = tarfile.TarInfo("app.py")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:") as archive:
        archive.addfile(info, io.BytesIO(payload))
    member = {
        "relative_path": "app.py",
        "byte_size": len(payload),
        "sha256": controller.sha256_bytes(payload),
    }
    members_sha256 = canonical_sha256({"regular_members": [member]})
    return archive_path, {
        "confirmation_commit": "a" * 40,
        "source_archive_sha256": sha256_file(archive_path),
        "regular_members_sha256": members_sha256,
        "tracked_files_manifest_sha256": members_sha256,
        "regular_members": [member],
        "dependency_versions": {
            "flwr": "1.23.0",
            "protobuf": "4.25.8",
            "psutil": "7.0.0",
        },
    }


def _hooks(events: list[object], *, validation_success: bool = True) -> LifecycleHooks:
    def launch(label: str, token: object):
        def inner(_attempt):
            events.append(label)
            return token

        return inner

    return LifecycleHooks(
        prepare=lambda _attempt: events.append("prepare"),
        launch_server=launch("server", "server-process"),
        start_tunnels=lambda _attempt: events.append("tunnels") or ["tunnel"],
        launch_pi_client=launch("pi-client", "pi-process"),
        launch_pi_sampler=lambda _attempt, process: events.append(
            ("pi-sampler", process)
        )
        or "pi-sampler-process",
        launch_pc_client=launch("pc-client", "pc-process"),
        launch_pc_sampler=lambda _attempt, process: events.append(
            ("pc-sampler", process)
        )
        or "pc-sampler-process",
        monitor_server=lambda _attempt, process: events.append(("monitor", process)),
        stop_sampler=lambda _attempt, process: events.append(("stop", process)),
        wait_sampler=lambda _attempt, process: events.append(("wait", process)),
        recover_evidence=lambda _attempt: events.append("recover"),
        validate_attempt=lambda _attempt: events.append("validate")
        or ValidationOutcome(
            success=validation_success,
            audit_sha256="e" * 64 if validation_success else None,
            reason=None if validation_success else "audit rejected attempt",
        ),
        cleanup_owned=lambda processes: events.append(("cleanup-owned", tuple(processes))),
        cleanup_tunnels=lambda processes: events.append(
            ("cleanup-tunnels", tuple(processes))
        ),
    )


def test_allocate_attempt_never_overwrites_and_stops_after_canonical(
    tmp_path: Path,
) -> None:
    first = allocate_attempt(tmp_path, "c12_to_c5__b2__s42")
    second = allocate_attempt(tmp_path, "c12_to_c5__b2__s42")
    assert first.attempt_id.endswith("__a001")
    assert second.attempt_id.endswith("__a002")
    mark_attempt(first.path, "canonical", audit_sha256="a" * 64)
    with pytest.raises(RuntimeError, match="canonical"):
        allocate_attempt(tmp_path, "c12_to_c5__b2__s42")


@pytest.mark.parametrize(
    ("group", "seed"),
    [("A6", 42), ("B1", 42), ("B2", 41), ("B5", 47)],
)
def test_controller_rejects_out_of_scope_runs(group: str, seed: int) -> None:
    with pytest.raises(ValueError, match="allowlist"):
        validate_requested_run(group, seed)


def test_default_queue_and_cli_are_exact() -> None:
    assert controller.DEFAULT_QUEUE == CONFIRMATION_SCHEDULE
    option_strings = {
        option
        for action in controller.build_parser()._actions
        for option in action.option_strings
    }
    assert "--skip-ecs-sync" not in option_strings
    assert "--skip-pi-sync" not in option_strings


def test_status_events_append_and_current_status_is_atomically_replaced(
    tmp_path: Path,
) -> None:
    attempt = allocate_attempt(tmp_path, "c12_to_c5__b5__s43")
    first = mark_attempt(
        attempt.path,
        "running",
        event_type="attempt_start",
        reason="controller allocated attempt",
        reason_category="controller",
        provenance=PROVENANCE,
    )
    first_bytes = (attempt.path / "status_events" / "status_001.json").read_bytes()
    second = mark_attempt(
        attempt.path,
        "failed",
        event_type="attempt_failure",
        reason="transport disconnected",
        reason_category="transport",
        provenance=PROVENANCE,
    )

    assert first["state"] == "running"
    assert second["state"] == "failed"
    assert (attempt.path / "status_events" / "status_001.json").read_bytes() == first_bytes
    assert sorted(path.name for path in (attempt.path / "status_events").iterdir()) == [
        "status_001.json",
        "status_002.json",
    ]
    assert _read_status(attempt.path) == second
    assert not list(attempt.path.glob(".attempt_status.*.tmp"))
    assert {
        "confirmation_commit",
        "source_archive_sha256",
        "dataset_manifest_sha256",
        "algorithm_config_sha256",
        "reason",
        "reason_category",
        "wall_time_utc",
    } <= set(second)
    serialized = json.dumps(second).lower()
    assert "accuracy" not in serialized
    assert "loss" not in serialized


def test_new_attempt_requires_objective_failure_category(tmp_path: Path) -> None:
    attempt = allocate_attempt(tmp_path, "c12_to_c5__b2__s44")
    mark_attempt(
        attempt.path,
        "failed",
        reason="operator disliked the outcome",
        reason_category="subjective_outcome",
        provenance=PROVENANCE,
    )
    with pytest.raises(RuntimeError, match="objective"):
        allocate_attempt(tmp_path, attempt.run_id)


def test_rerun_policy_accepts_only_state_and_reason_category() -> None:
    assert tuple(inspect.signature(controller.is_objective_rerun_allowed).parameters) == (
        "state",
        "reason_category",
    )
    assert controller.is_objective_rerun_allowed("failed", "transport")
    assert not controller.is_objective_rerun_allowed("failed", "subjective_outcome")
    assert not controller.is_objective_rerun_allowed("canonical", "transport")


def test_archive_hash_mismatch_leaves_fresh_destination_absent(tmp_path: Path) -> None:
    archive_path, manifest = _source_fixture(tmp_path)
    manifest["source_archive_sha256"] = "0" * 64
    destination = tmp_path / "src"

    with pytest.raises(ArchiveMismatch, match="SHA-256"):
        verify_and_extract_archive(archive_path, destination, manifest)

    assert not destination.exists()


def test_source_deployment_uses_one_tar_for_all_hosts_and_fresh_pc_src(
    tmp_path: Path,
) -> None:
    archive_path, manifest = _source_fixture(tmp_path)
    run_calls: list[list[str]] = []
    ssh_calls: list[tuple[str, str]] = []

    def fake_run(command, **_kwargs):
        run_calls.append([str(item) for item in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_ssh(host: str, command: str, **_kwargs):
        ssh_calls.append((host, command))
        return subprocess.CompletedProcess(command, 0, "", "")

    remote_report = json.dumps(
        {
            "source_archive_sha256": manifest["source_archive_sha256"],
            "regular_members_sha256": manifest["regular_members_sha256"],
        }
    )
    deployments = deploy_source_archive(
        archive_path,
        manifest,
        ecs_host="root@ecs",
        pi_host="gaps@pi",
        pc_runtime_root=tmp_path / "runtime",
        run=fake_run,
        ssh=fake_ssh,
        remote_python=lambda *_args, **_kwargs: remote_report,
    )

    scp_calls = [call for call in run_calls if call and call[0] == "scp"]
    assert len(scp_calls) == 2
    assert all(call[2] == str(archive_path) for call in scp_calls)
    assert {call[3] for call in scp_calls} == {
        f"root@ecs:/root/GAPS/confirmation_runtime/{manifest['source_archive_sha256']}/source.tar",
        f"gaps@pi:/home/gaps/GAPS/confirmation_runtime/{manifest['source_archive_sha256']}/source.tar",
    }
    assert all("sync" not in " ".join(call).lower() for call in run_calls)
    assert len(ssh_calls) == 2
    pc_archive = Path(deployments["pc"].archive_path)
    assert pc_archive.read_bytes() == archive_path.read_bytes()
    assert (Path(deployments["pc"].src_path) / "app.py").is_file()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dependency_versions", {"flwr": "1.22.0", "protobuf": "4.25.8", "psutil": "7.0.0"}, "dependency"),
        ("dataset_manifest_sha256", "0" * 64, "dataset"),
        ("algorithm_config_sha256", "0" * 64, "algorithm"),
        ("existing_attempt_processes", [123], "process"),
    ],
)
def test_host_preflight_fails_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    report: dict[str, object] = {
        "host_id": "ecs",
        "dependency_versions": {
            "flwr": "1.23.0",
            "protobuf": "4.25.8",
            "psutil": "7.0.0",
        },
        "source_archive_sha256": PROVENANCE.source_archive_sha256,
        "regular_members_sha256": "e" * 64,
        "dataset_manifest_sha256": PROVENANCE.dataset_manifest_sha256,
        "algorithm_config_sha256": PROVENANCE.algorithm_config_sha256,
        "existing_attempt_processes": [],
    }
    report[field] = value
    with pytest.raises(RuntimeError, match=match):
        validate_host_preflight(
            report,
            host_id="ecs",
            provenance=PROVENANCE,
            regular_members_sha256="e" * 64,
        )


def test_archive_mismatch_aborts_before_server_and_retains_attempt(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    hooks = _hooks(events)

    def fail_prepare(_attempt) -> None:
        events.append("prepare")
        raise ArchiveMismatch("archive SHA-256 mismatch")

    hooks = controller.replace(hooks, prepare=fail_prepare)
    with pytest.raises(ArchiveMismatch, match="mismatch"):
        run_confirmation_attempt(
            tmp_path,
            "c12_to_c5__b2__s42",
            provenance=PROVENANCE,
            hooks=hooks,
        )

    attempt_path = tmp_path / "c12_to_c5__b2__s42" / "c12_to_c5__b2__s42__a001"
    assert attempt_path.is_dir()
    assert _read_status(attempt_path)["state"] == "failed"
    assert "server" not in events


def test_lifecycle_stops_samplers_cleans_owned_processes_then_validates(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    attempt = run_confirmation_attempt(
        tmp_path,
        "c12_to_c5__b5__s45",
        provenance=PROVENANCE,
        hooks=_hooks(events),
    )

    assert _read_status(attempt.path)["state"] == "canonical"
    assert events.index(("stop", "pi-sampler-process")) < events.index("validate")
    assert events.index(("wait", "pc-sampler-process")) < events.index("validate")
    assert events.index("recover") < events.index("validate")
    assert events.index(("cleanup-tunnels", ("tunnel",))) < events.index("validate")
    owned = next(item[1] for item in events if isinstance(item, tuple) and item[0] == "cleanup-owned")
    assert owned == (
        "server-process",
        "pi-process",
        "pi-sampler-process",
        "pc-process",
        "pc-sampler-process",
    )


def test_validator_is_the_only_path_to_canonical(tmp_path: Path) -> None:
    events: list[object] = []
    attempt = run_confirmation_attempt(
        tmp_path,
        "c12_to_c5__b2__s46",
        provenance=PROVENANCE,
        hooks=_hooks(events, validation_success=False),
    )

    assert _read_status(attempt.path)["state"] == "invalid"
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["state"] != "canonical"
        for path in (attempt.path / "status_events").glob("status_*.json")
    )


def test_monitor_failure_stops_samplers_and_retains_logs_and_status(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    hooks = _hooks(events)

    def fail_monitor(attempt, _process) -> None:
        (attempt.path / "controller.log").write_text("partial log\n", encoding="utf-8")
        raise RuntimeError("server exited")

    hooks = controller.replace(hooks, monitor_server=fail_monitor)
    with pytest.raises(RuntimeError, match="server exited"):
        run_confirmation_attempt(
            tmp_path,
            "c12_to_c5__b5__s46",
            provenance=PROVENANCE,
            hooks=hooks,
        )

    attempt_path = tmp_path / "c12_to_c5__b5__s46" / "c12_to_c5__b5__s46__a001"
    assert (attempt_path / "controller.log").read_text(encoding="utf-8") == "partial log\n"
    assert _read_status(attempt_path)["state"] == "failed"
    assert ("stop", "pi-sampler-process") in events
    assert any(item[0] == "cleanup-owned" for item in events if isinstance(item, tuple))


def test_raw_evidence_copy_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "remote-evidence"
    source.mkdir()
    (source / "events.jsonl").write_text("new\n", encoding="utf-8")
    destination = tmp_path / "attempt" / "raw" / "ecs"
    destination.mkdir(parents=True)
    existing = destination / "events.jsonl"
    existing.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        copy_evidence_without_overwrite(source, destination)

    assert existing.read_text(encoding="utf-8") == "existing\n"
