from __future__ import annotations

import ast
import inspect
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import copy
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pytest

import scripts.run_iotj_confirmation_observability as controller
from scripts.freeze_iotj_confirmation_protocol import (
    CONFIRMATION_SCHEDULE,
    canonical_sha256,
    confirmation_run_id,
    sha256_file,
)
from scripts.run_iotj_confirmation_observability import (
    ArchiveMismatch,
    LifecycleHooks,
    Provenance,
    ValidationOutcome,
    allocate_attempt,
    bind_attempt_provenance,
    copy_evidence_without_overwrite,
    deploy_source_archive,
    mark_attempt,
    run_confirmation_attempt,
    validate_archive_member_path,
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


def _allocate_bound(tmp_path: Path, run_id: str):
    attempt = allocate_attempt(tmp_path, run_id)
    bind_attempt_provenance(attempt, PROVENANCE)
    return attempt


def _source_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _frozen_input_fixture(tmp_path: Path) -> dict[str, object]:
    archive_path, source_manifest = _source_fixture(tmp_path)
    source_manifest["confirmation_commit"] = PROVENANCE.confirmation_commit
    dataset_manifest: dict[str, object] = {
        "schema_version": 1,
        "direction": "C1/C2 -> C5",
        "files": [],
    }
    dataset_manifest["dataset_manifest_sha256"] = canonical_sha256(dataset_manifest)
    schedule: list[dict[str, object]] = []
    command_payloads: dict[str, dict[str, object]] = {}
    for group, seed in CONFIRMATION_SCHEDULE:
        run_id = confirmation_run_id(group, seed)
        algorithm = {
            "protocol": {
                "source_clients": [1, 2],
                "target_clients": [5],
                "training_seed": seed,
                "data_root": "frozen_dataset",
            },
            "training": {"rounds": 25, "profile": group.lower()},
            "causal_factors": {"full": group == "B5"},
            "server_adaptation": {"enabled": True},
        }
        algorithm_hash = canonical_sha256(algorithm)
        schedule.append(
            {
                "run_id": run_id,
                "group_id": group,
                "seed": seed,
                "algorithm_config_sha256": algorithm_hash,
            }
        )
        command_payloads[run_id] = {
            "run_id": run_id,
            "group_id": group,
            **algorithm,
            "topology": copy.deepcopy(controller.EXPECTED_TOPOLOGY),
            "commands": {
                "server_ecs": [
                    "python", "-m", "gaps_flower.server_app",
                    "--data-root", "dataset/frozen_dataset",
                ],
                "client_c1_pi": [
                    "python", "-m", "gaps_flower.client_app", "--client-id", "1",
                    "--data-root", "/home/gaps/GAPS/flower_runtime/dataset/frozen_dataset",
                ],
                "client_c2_pc": [
                    "python", "-m", "gaps_flower.client_app", "--client-id", "2",
                    "--data-root", str(tmp_path / "dataset" / "frozen_dataset"),
                ],
            },
            "provenance": {"code_revision": PROVENANCE.confirmation_commit},
            "source_archive_sha256": source_manifest["source_archive_sha256"],
            "regular_members_sha256": source_manifest["regular_members_sha256"],
            "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
            "algorithm_config_sha256": algorithm_hash,
        }
    protocol_manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": "iotj_main_direction_confirmation",
        "direction": "C1/C2 -> C5",
        "confirmation_commit": PROVENANCE.confirmation_commit,
        "source_archive_sha256": source_manifest["source_archive_sha256"],
        "regular_members_sha256": source_manifest["regular_members_sha256"],
        "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
        "schedule": schedule,
    }
    protocol_manifest["protocol_manifest_sha256"] = canonical_sha256(protocol_manifest)
    for payload in command_payloads.values():
        payload["protocol_manifest_sha256"] = protocol_manifest[
            "protocol_manifest_sha256"
        ]
    protocol_path = tmp_path / "manifests" / "confirmation_protocol_manifest.json"
    source_path = tmp_path / "manifests" / "source_archive_manifest.json"
    dataset_path = tmp_path / "manifests" / "dataset_manifest.json"
    command_root = tmp_path / "commands"
    _write_json(protocol_path, protocol_manifest)
    _write_json(source_path, source_manifest)
    _write_json(dataset_path, dataset_manifest)
    for run_id, payload in command_payloads.items():
        _write_json(command_root / run_id / "command_manifest.json", payload)
    return {
        "archive": archive_path,
        "protocol_path": protocol_path,
        "source_path": source_path,
        "dataset_path": dataset_path,
        "command_root": command_root,
        "protocol": protocol_manifest,
        "source": source_manifest,
        "dataset": dataset_manifest,
        "commands": command_payloads,
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
    bind_attempt_provenance(first, PROVENANCE)
    mark_attempt(
        first.path,
        "canonical",
        audit_sha256="a" * 64,
        reason="validator_accepted",
    )
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
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b5__s43")
    first = mark_attempt(
        attempt.path,
        "running",
        event_type="attempt_start",
        reason="attempt_allocated",
    )
    first_bytes = (attempt.path / "status_events" / "status_001.json").read_bytes()
    second = mark_attempt(
        attempt.path,
        "failed",
        event_type="attempt_failure",
        reason="transport_failure",
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
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b2__s44")
    mark_attempt(
        attempt.path,
        "failed",
        reason="operator_abort",
    )
    with pytest.raises(RuntimeError, match="objective"):
        allocate_attempt(tmp_path, attempt.run_id)


def test_rerun_policy_accepts_only_state_and_reason_category() -> None:
    assert tuple(inspect.signature(controller.is_objective_rerun_allowed).parameters) == (
        "state",
        "reason_category",
    )
    assert controller.is_objective_rerun_allowed("failed", "transport")
    assert not controller.is_objective_rerun_allowed("failed", "operator")
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
    def fake_remote_python(_host, _python, source, **_kwargs):
        if "REMOTE_DEPLOY_STATE_V1" in source:
            return json.dumps({"state": "absent"})
        if "REMOTE_RESERVE_ARCHIVE_V1" in source:
            return json.dumps({"state": "reserved"})
        if "REMOTE_INSTALL_ARCHIVE_V1" in source:
            return json.dumps(
                {"source_archive_sha256": manifest["source_archive_sha256"]}
            )
        assert "REMOTE_EXTRACT_SOURCE_V1" in source
        return remote_report

    deployments = deploy_source_archive(
        archive_path,
        manifest,
        ecs_host="root@ecs",
        pi_host="gaps@pi",
        pc_runtime_root=tmp_path / "runtime",
        run=fake_run,
        ssh=fake_ssh,
        remote_python=fake_remote_python,
    )

    scp_calls = [call for call in run_calls if call and call[0] == "scp"]
    assert len(scp_calls) == 2
    assert all(call[2] == str(archive_path) for call in scp_calls)
    destinations = {call[3].split(":", 1)[0]: call[3].split(":", 1)[1] for call in scp_calls}
    assert set(destinations) == {"root@ecs", "gaps@pi"}
    assert destinations["root@ecs"].startswith(
        f"/root/GAPS/confirmation_runtime/{manifest['source_archive_sha256']}/.source.tar."
    )
    assert destinations["gaps@pi"].startswith(
        f"/home/gaps/GAPS/confirmation_runtime/{manifest['source_archive_sha256']}/.source.tar."
    )
    assert all(destination.endswith(".tmp") for destination in destinations.values())
    assert all("sync" not in " ".join(call).lower() for call in run_calls)
    assert ssh_calls == []
    pc_archive = Path(deployments["pc"].archive_path)
    assert pc_archive.read_bytes() == archive_path.read_bytes()
    assert (Path(deployments["pc"].src_path) / "app.py").is_file()


def test_remote_extract_source_verifies_complete_reuse_and_rejects_partial(
    tmp_path: Path,
) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    remote_archive = tmp_path / "remote" / "source.tar"
    remote_archive.parent.mkdir(parents=True)
    remote_archive.write_bytes(archive.read_bytes())
    remote_src = remote_archive.parent / "src"

    with contextlib.redirect_stdout(io.StringIO()):
        exec(
            controller._remote_extract_source(
                str(remote_archive),
                str(remote_src),
                manifest,
                allow_fresh_extract=True,
            ),
            {},
        )
        exec(
            controller._remote_extract_source(
                str(remote_archive), str(remote_src), manifest
            ),
            {},
        )

    (remote_src / "app.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked member"):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(
                controller._remote_extract_source(
                    str(remote_archive), str(remote_src), manifest
                ),
                {},
            )

    partial_archive = tmp_path / "partial" / "source.tar"
    partial_archive.parent.mkdir()
    partial_archive.write_bytes(archive.read_bytes())
    with pytest.raises(RuntimeError, match="partial"):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(
                controller._remote_extract_source(
                    str(partial_archive), str(partial_archive.parent / "src"), manifest
                ),
                {},
            )


def test_content_addressed_deploy_reuses_only_verified_complete_runtime(
    tmp_path: Path,
) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    remote_states = {"root@ecs": "absent", "gaps@pi": "absent"}
    scp_calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = [str(item) for item in command]
        scp_calls.append(command)
        host = command[-1].split(":", 1)[0]
        assert remote_states[host] == "absent"
        assert "/.source.tar." in command[-1]
        assert command[-1].endswith(".tmp")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_remote_python(host, _python, source, **_kwargs):
        if "REMOTE_DEPLOY_STATE_V1" in source:
            return json.dumps({"state": remote_states[host]})
        if "REMOTE_RESERVE_ARCHIVE_V1" in source:
            assert remote_states[host] == "absent"
            return json.dumps({"state": "reserved"})
        if "REMOTE_INSTALL_ARCHIVE_V1" in source:
            assert remote_states[host] == "absent"
            remote_states[host] = "archive_only"
            return json.dumps(
                {"source_archive_sha256": manifest["source_archive_sha256"]}
            )
        assert "REMOTE_EXTRACT_SOURCE_V1" in source
        if remote_states[host] == "archive_only":
            remote_states[host] = "complete"
        assert remote_states[host] == "complete"
        return json.dumps(
            {
                "source_archive_sha256": manifest["source_archive_sha256"],
                "regular_members_sha256": manifest["regular_members_sha256"],
            }
        )

    kwargs = {
        "ecs_host": "root@ecs",
        "pi_host": "gaps@pi",
        "pc_runtime_root": tmp_path / "runtime",
        "run": fake_run,
        "ssh": lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
        "remote_python": fake_remote_python,
    }
    first = deploy_source_archive(archive, manifest, **kwargs)
    second = deploy_source_archive(archive, manifest, **kwargs)

    assert first == second
    assert len(scp_calls) == 2

    (Path(first["pc"].src_path) / "app.py").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ArchiveMismatch, match="tracked member"):
        deploy_source_archive(archive, manifest, **kwargs)


def test_content_addressed_deploy_rejects_partial_local_runtime(tmp_path: Path) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    runtime = tmp_path / "runtime"
    partial = runtime / str(manifest["source_archive_sha256"])
    partial.mkdir(parents=True)
    (partial / "source.tar").write_bytes(archive.read_bytes())

    with pytest.raises(RuntimeError, match="partial"):
        deploy_source_archive(
            archive,
            manifest,
            ecs_host="root@ecs",
            pi_host="gaps@pi",
            pc_runtime_root=runtime,
            run=lambda *_args, **_kwargs: pytest.fail("partial runtime transferred"),
            ssh=lambda *_args, **_kwargs: pytest.fail("partial runtime contacted host"),
            remote_python=lambda *_args, **_kwargs: pytest.fail("partial runtime contacted host"),
        )


def test_content_addressed_deploy_rejects_partial_remote_before_transfer(
    tmp_path: Path,
) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")

    with pytest.raises(ArchiveMismatch, match="partial ecs"):
        deploy_source_archive(
            archive,
            manifest,
            ecs_host="root@ecs",
            pi_host="gaps@pi",
            pc_runtime_root=tmp_path / "runtime",
            run=lambda *_args, **_kwargs: pytest.fail("partial remote transferred"),
            ssh=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
            remote_python=lambda host, _python, source, **_kwargs: (
                json.dumps({"state": "partial"})
                if host == "root@ecs" and "REMOTE_DEPLOY_STATE_V1" in source
                else pytest.fail("partial remote continued")
            ),
        )


@pytest.mark.parametrize("component", ["archive", "dangling_archive", "root"])
def test_remote_deploy_state_rejects_synthetic_symlink_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    root = tmp_path / "runtime" / "hash"
    root.mkdir(parents=True)
    archive = root / "source.tar"
    src = root / "src"
    if component != "dangling_archive":
        archive.write_bytes(b"archive")
    src.mkdir()
    target = root if component == "root" else archive
    original_lstat = os.lstat

    class _SyntheticSymlinkStat:
        st_mode = stat.S_IFLNK | 0o777
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    def fake_lstat(path, *args, **kwargs):
        if Path(path) == target:
            return _SyntheticSymlinkStat()
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", fake_lstat)
    source = controller._remote_deploy_state_source(str(archive), str(src))

    with pytest.raises(RuntimeError, match="symlink|reparse"):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, {})


def test_remote_extract_rejects_synthetic_archive_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    root = tmp_path / "runtime" / "hash"
    root.mkdir(parents=True)
    remote_archive = root / "source.tar"
    remote_archive.write_bytes(archive.read_bytes())
    original_lstat = os.lstat

    class _SyntheticSymlinkStat:
        st_mode = stat.S_IFLNK | 0o777
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    monkeypatch.setattr(
        os,
        "lstat",
        lambda path, *args, **kwargs: (
            _SyntheticSymlinkStat()
            if Path(path) == remote_archive
            else original_lstat(path, *args, **kwargs)
        ),
    )

    with pytest.raises(RuntimeError, match="symlink|reparse"):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(
                controller._remote_extract_source(
                    str(remote_archive),
                    str(root / "src"),
                    manifest,
                    allow_fresh_extract=True,
                ),
                {},
            )


def test_remote_runtime_sources_use_lstat_not_exists_for_pinned_components(
    tmp_path: Path,
) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    state_source = controller._remote_deploy_state_source(
        "/runtime/hash/source.tar", "/runtime/hash/src"
    )
    extract_source = controller._remote_extract_source(
        "/runtime/hash/source.tar", "/runtime/hash/src", manifest
    )

    assert "os.lstat" in state_source
    assert ".exists()" not in state_source
    assert "os.lstat" in extract_source
    assert "src_path.exists()" not in extract_source


def test_pc_runtime_parent_reparse_is_rejected_before_content_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    original_lstat = os.lstat
    original = original_lstat(runtime)

    class _SyntheticReparseStat:
        st_mode = original.st_mode
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    monkeypatch.setattr(
        os,
        "lstat",
        lambda path, *args, **kwargs: (
            _SyntheticReparseStat()
            if Path(path) == runtime
            else original_lstat(path, *args, **kwargs)
        ),
    )

    with pytest.raises(RuntimeError, match="reparse"):
        deploy_source_archive(
            archive,
            manifest,
            ecs_host="root@ecs",
            pi_host="gaps@pi",
            pc_runtime_root=runtime,
            run=lambda *_args, **_kwargs: pytest.fail("reparse runtime transferred"),
            ssh=lambda *_args, **_kwargs: pytest.fail("reparse runtime contacted host"),
            remote_python=lambda *_args, **_kwargs: pytest.fail("reparse runtime contacted host"),
        )


def test_fresh_remote_archive_transfer_uses_verified_temp_then_atomic_install(
    tmp_path: Path,
) -> None:
    archive, manifest = _source_fixture(tmp_path / "source")
    scp_destinations: list[str] = []
    install_sources: list[str] = []

    def fake_run(command, **_kwargs):
        scp_destinations.append(str(command[-1]))
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_remote_python(_host, _python, source, **_kwargs):
        if "REMOTE_DEPLOY_STATE_V1" in source:
            return json.dumps({"state": "absent"})
        if "REMOTE_RESERVE_ARCHIVE_V1" in source:
            return json.dumps({"state": "reserved"})
        if "REMOTE_INSTALL_ARCHIVE_V1" in source:
            install_sources.append(source)
            return json.dumps(
                {"source_archive_sha256": manifest["source_archive_sha256"]}
            )
        assert "REMOTE_EXTRACT_SOURCE_V1" in source
        return json.dumps(
            {
                "source_archive_sha256": manifest["source_archive_sha256"],
                "regular_members_sha256": manifest["regular_members_sha256"],
            }
        )

    deploy_source_archive(
        archive,
        manifest,
        ecs_host="root@ecs",
        pi_host="gaps@pi",
        pc_runtime_root=tmp_path / "runtime",
        run=fake_run,
        ssh=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
        remote_python=fake_remote_python,
    )

    assert len(scp_destinations) == 2
    assert all("/.source.tar." in destination and destination.endswith(".tmp") for destination in scp_destinations)
    assert len(install_sources) == 2
    assert all("os.lstat" in source and "os.link" in source for source in install_sources)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("dependency_versions", {"flwr": "1.22.0", "protobuf": "4.25.8", "psutil": "7.0.0"}, "dependency"),
        ("confirmation_commit", "f" * 40, "commit"),
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
        "confirmation_commit": PROVENANCE.confirmation_commit,
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
    controller_log = (attempt_path / "controller.log").read_text(encoding="utf-8")
    assert controller_log.startswith("partial log\n")
    assert "RuntimeError: server exited" in controller_log
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


def test_status_updates_are_serialized_and_current_matches_latest_event(
    tmp_path: Path,
) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b2__s43")

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(
            pool.map(
                lambda _index: mark_attempt(
                    attempt.path,
                    "running",
                    event_type="controller_progress",
                    reason="preflight_passed",
                ),
                range(24),
            )
        )

    assert sorted(int(status["sequence"]) for status in statuses) == list(range(1, 25))
    events = sorted((attempt.path / "status_events").glob("status_*.json"))
    assert len(events) == 24
    latest = json.loads(events[-1].read_text(encoding="utf-8"))
    assert _read_status(attempt.path) == latest


def test_terminal_status_cannot_be_replaced(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b5__s44")
    mark_attempt(attempt.path, "failed", reason="process_failure")

    with pytest.raises(RuntimeError, match="terminal"):
        mark_attempt(attempt.path, "running", reason="preflight_passed")


def test_allocator_reads_canonical_from_immutable_events(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b2__s45")
    mark_attempt(
        attempt.path,
        "canonical",
        audit_sha256="e" * 64,
        reason="validator_accepted",
    )
    (attempt.path / "attempt_status.json").write_text(
        json.dumps({"state": "failed"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="canonical"):
        allocate_attempt(tmp_path, attempt.run_id)


def test_malformed_or_gapped_status_chain_fails_closed(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b5__s42")
    mark_attempt(attempt.path, "running", reason="attempt_allocated")
    mark_attempt(attempt.path, "running", reason="preflight_passed")
    first = attempt.path / "status_events" / "status_001.json"
    first.unlink()

    with pytest.raises(RuntimeError, match="status.*gap"):
        allocate_attempt(tmp_path, attempt.run_id)
    with pytest.raises(RuntimeError, match="status.*gap"):
        mark_attempt(attempt.path, "failed", reason="process_failure")


@pytest.mark.parametrize("metric_key", ["accuracy", "loss", "nll", "ece", "recall", "f1", "metric", "metrics"])
def test_status_chain_rejects_metric_keys_even_when_event_and_current_match(
    tmp_path: Path, metric_key: str
) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b5__s43")
    mark_attempt(attempt.path, "running", reason="attempt_allocated")
    event_path = attempt.path / "status_events" / "status_001.json"
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["nested"] = {metric_key: 0.25}
    _write_json(event_path, payload)
    _write_json(attempt.path / "attempt_status.json", payload)

    with pytest.raises(RuntimeError, match="classification metric"):
        allocate_attempt(tmp_path, attempt.run_id)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda row: row.update(sequence=True), "type"),
        (lambda row: row.update(wall_time_utc="2026-07-15 12:00:00"), "UTC"),
        (lambda row: row.update(unexpected="field"), "schema"),
        (lambda row: row.update(audit_sha256="e" * 64), "combination"),
        (
            lambda row: row.update(
                state="canonical",
                event_type="attempt_end",
                reason="preflight_passed",
                reason_category="controller",
                audit_sha256="e" * 64,
            ),
            "combination",
        ),
    ],
)
def test_status_chain_rejects_malformed_schema_types_time_and_combinations(
    tmp_path: Path, mutation, match: str
) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b2__s44")
    mark_attempt(attempt.path, "running", reason="attempt_allocated")
    event_path = attempt.path / "status_events" / "status_001.json"
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(event_path, payload)
    _write_json(attempt.path / "attempt_status.json", payload)

    with pytest.raises(RuntimeError, match=match):
        allocate_attempt(tmp_path, attempt.run_id)


def test_status_chain_rejects_non_exact_bound_provenance_schema(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b5__s44")
    provenance_path = attempt.path / "attempt_provenance.json"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["controller_owner"]["accuracy"] = 0.5
    _write_json(provenance_path, payload)

    with pytest.raises(RuntimeError, match="classification metric"):
        mark_attempt(attempt.path, "running", reason="attempt_allocated")


def test_status_rejects_unbound_or_incomplete_provenance(tmp_path: Path) -> None:
    attempt = allocate_attempt(tmp_path, "c12_to_c5__b2__s42")
    with pytest.raises(RuntimeError, match="provenance"):
        mark_attempt(attempt.path, "running", reason="attempt_allocated")

    incomplete = dataclass_replace(PROVENANCE, algorithm_config_sha256=None)
    with pytest.raises(ValueError, match="algorithm_config_sha256"):
        bind_attempt_provenance(attempt, incomplete)


@pytest.mark.parametrize(
    "reason",
    ["classification_accuracy_low", "training_loss_high", "target_metric_ranking"],
)
def test_status_rejects_result_driven_reason_codes(tmp_path: Path, reason: str) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b5__s45")
    with pytest.raises(ValueError, match="reason"):
        mark_attempt(attempt.path, "failed", reason=reason)


def test_failure_status_uses_stable_code_and_details_remain_in_log(tmp_path: Path) -> None:
    events: list[object] = []
    hooks = _hooks(events)
    hooks = controller.replace(
        hooks,
        prepare=lambda _attempt: (_ for _ in ()).throw(
            RuntimeError("SECRET detailed exception with classification_accuracy=0.1")
        ),
    )

    with pytest.raises(RuntimeError, match="SECRET"):
        run_confirmation_attempt(
            tmp_path,
            "c12_to_c5__b2__s44",
            provenance=PROVENANCE,
            hooks=hooks,
        )

    attempt_path = tmp_path / "c12_to_c5__b2__s44" / "c12_to_c5__b2__s44__a001"
    status_text = (attempt_path / "attempt_status.json").read_text(encoding="utf-8")
    assert "SECRET" not in status_text
    assert "classification_accuracy" not in status_text
    assert _read_status(attempt_path)["reason"] == "process_failure"
    assert "SECRET detailed exception" in (attempt_path / "controller.log").read_text(
        encoding="utf-8"
    )


def test_attempt_sequence_over_999_fails_closed(tmp_path: Path) -> None:
    run_id = "c12_to_c5__b2__s46"
    run_root = tmp_path / run_id
    run_root.mkdir()
    (run_root / f"{run_id}__a999").mkdir()
    with pytest.raises(RuntimeError, match="exhausted"):
        allocate_attempt(tmp_path, run_id)


@pytest.mark.parametrize(
    "member_name",
    [
        "../escape.py",
        "dir/../escape.py",
        "dir/./file.py",
        ".",
        "/absolute.py",
        "C:/drive.py",
        "C:\\drive.py",
        "\\\\server\\share\\file.py",
        "//server/share/file.py",
        "colon:name.py",
        "back\\slash.py",
        "nul\x00name.py",
    ],
)
def test_archive_member_path_rejects_cross_platform_escape(member_name: str) -> None:
    with pytest.raises(ArchiveMismatch, match="member path"):
        validate_archive_member_path(member_name)


@pytest.mark.parametrize(
    "member_name",
    ["../escape.py", "C:/drive.py", "//server/share.py", "back\\slash.py"],
)
def test_malicious_tar_member_is_rejected_before_extract(
    tmp_path: Path, member_name: str
) -> None:
    payload = b"malicious\n"
    archive_path = tmp_path / "malicious.tar"
    info = tarfile.TarInfo(member_name)
    info.size = len(payload)
    with tarfile.open(archive_path, "w:") as archive:
        archive.addfile(info, io.BytesIO(payload))
    row = {
        "relative_path": member_name,
        "byte_size": len(payload),
        "sha256": controller.sha256_bytes(payload),
    }
    members_hash = canonical_sha256({"regular_members": [row]})
    manifest = {
        "source_archive_sha256": sha256_file(archive_path),
        "regular_members_sha256": members_hash,
        "tracked_files_manifest_sha256": members_hash,
        "regular_members": [row],
    }

    with pytest.raises(ArchiveMismatch, match="member path"):
        verify_and_extract_archive(archive_path, tmp_path / "src", manifest)
    assert not (tmp_path / "src").exists()


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def test_attempt_and_evidence_paths_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    run_id = "c12_to_c5__b5__s46"
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _symlink_or_skip(raw_root / run_id, outside)
    with pytest.raises(RuntimeError, match="symlink"):
        allocate_attempt(raw_root, run_id)

    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.jsonl").write_text("evidence\n", encoding="utf-8")
    destination_parent = tmp_path / "destination-parent"
    _symlink_or_skip(destination_parent, outside)
    with pytest.raises(RuntimeError, match="symlink"):
        copy_evidence_without_overwrite(source, destination_parent / "raw")


def test_path_guard_rejects_reported_symlink_without_platform_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    real_lstat = controller.os.lstat

    def synthetic_lstat(path):
        entry = real_lstat(path)
        if Path(path) != raw_root:
            return entry
        return os.stat_result((stat.S_IFLNK | 0o777, *tuple(entry)[1:]))

    monkeypatch.setattr(controller.os, "lstat", synthetic_lstat)
    with pytest.raises(RuntimeError, match="symlink"):
        allocate_attempt(raw_root, "c12_to_c5__b2__s42")


def _load_frozen(fixture: dict[str, object]):
    return controller.load_frozen_inputs(
        fixture["protocol_path"],
        fixture["source_path"],
        fixture["dataset_path"],
        fixture["command_root"],
        fixture["archive"],
    )


def test_frozen_loader_binds_all_ten_runs_and_self_hashes(tmp_path: Path) -> None:
    fixture = _frozen_input_fixture(tmp_path)

    frozen = _load_frozen(fixture)

    assert [(run.group_id, run.seed) for run in frozen.runs] == list(
        CONFIRMATION_SCHEDULE
    )
    assert all(
        run.provenance.confirmation_commit == PROVENANCE.confirmation_commit
        for run in frozen.runs
    )


def test_frozen_loader_rejects_protocol_self_hash_mismatch(tmp_path: Path) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    protocol = copy.deepcopy(fixture["protocol"])
    protocol["direction"] = "tampered"
    _write_json(fixture["protocol_path"], protocol)

    with pytest.raises(ValueError, match="protocol.*self.*SHA-256"):
        _load_frozen(fixture)


def test_frozen_loader_rejects_source_and_command_commit_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    source = copy.deepcopy(fixture["source"])
    source["confirmation_commit"] = "f" * 40
    _write_json(fixture["source_path"], source)
    with pytest.raises(ValueError, match="confirmation_commit"):
        _load_frozen(fixture)

    fixture = _frozen_input_fixture(tmp_path / "second")
    first_run = confirmation_run_id(*CONFIRMATION_SCHEDULE[0])
    command = copy.deepcopy(fixture["commands"][first_run])
    command["provenance"]["code_revision"] = "f" * 40
    _write_json(
        fixture["command_root"] / first_run / "command_manifest.json", command
    )
    with pytest.raises(ValueError, match="confirmation_commit"):
        _load_frozen(fixture)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("protocol_manifest_sha256", "protocol"),
        ("source_archive_sha256", "source"),
        ("dataset_manifest_sha256", "dataset"),
    ],
)
def test_frozen_loader_rejects_command_binding_mismatch(
    tmp_path: Path, field: str, match: str
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    first_run = confirmation_run_id(*CONFIRMATION_SCHEDULE[0])
    command = copy.deepcopy(fixture["commands"][first_run])
    command[field] = "0" * 64
    _write_json(
        fixture["command_root"] / first_run / "command_manifest.json", command
    )

    with pytest.raises(ValueError, match=match):
        _load_frozen(fixture)


def test_frozen_loader_rejects_algorithm_hash_mismatch_in_command_or_schedule(
    tmp_path: Path,
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    first_group, first_seed = CONFIRMATION_SCHEDULE[0]
    first_run = confirmation_run_id(first_group, first_seed)
    command = copy.deepcopy(fixture["commands"][first_run])
    command["training"]["rounds"] = 24
    _write_json(
        fixture["command_root"] / first_run / "command_manifest.json", command
    )
    with pytest.raises(ValueError, match="algorithm"):
        _load_frozen(fixture)


def _controller_argv(
    fixture: dict[str, object], tmp_path: Path, *extra: str
) -> list[str]:
    return [
        "--protocol-manifest",
        str(fixture["protocol_path"]),
        "--source-archive-manifest",
        str(fixture["source_path"]),
        "--dataset-manifest",
        str(fixture["dataset_path"]),
        "--command-root",
        str(fixture["command_root"]),
        "--source-archive",
        str(fixture["archive"]),
        "--raw-root",
        str(tmp_path / "raw"),
        "--pc-runtime-root",
        str(tmp_path / "runtime"),
        "--runs",
        "B2:42",
        "--poll-seconds",
        "0.01",
        "--run-timeout-seconds",
        "1",
        *extra,
    ]


def test_validator_cli_argv_and_audit_sha_are_fail_closed(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path / "raw", "c12_to_c5__b2__s42")
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, {"protocol": "frozen"})
    validator = tmp_path / "validator.py"
    validator.write_text("# fake path only\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append([str(item) for item in command])
        assert kwargs["check"] is False
        output = Path(command[command.index("--output") + 1])
        _write_json(output, {"status": "valid", "reasons": []})
        digest = sha256_file(output)
        return subprocess.CompletedProcess(command, 0, json.dumps({"audit_sha256": digest}), "")

    outcome = controller.invoke_validator(
        attempt,
        validator=validator,
        protocol_manifest=protocol_path,
        run=fake_run,
    )

    audit_path = attempt.path / "attempt_audit.json"
    assert calls == [
        [
            sys.executable,
            str(validator),
            "--attempt-dir",
            str(attempt.path),
            "--protocol-manifest",
            str(protocol_path),
            "--output",
            str(audit_path),
        ]
    ]
    assert outcome == ValidationOutcome(True, sha256_file(audit_path), None)


@pytest.mark.parametrize(
    ("returncode", "stdout", "match"),
    [(2, "{}", "return code"), (0, "{}", "audit_sha256"), (0, '{"audit_sha256":"bad"}', "audit_sha256")],
)
def test_validator_rejects_nonzero_or_malformed_sha(
    tmp_path: Path, returncode: int, stdout: str, match: str
) -> None:
    attempt = _allocate_bound(tmp_path / "raw", "c12_to_c5__b5__s42")
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, {})

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        _write_json(output, {"status": "valid", "reasons": []})
        return subprocess.CompletedProcess(command, returncode, stdout, "validator detail")

    with pytest.raises(RuntimeError, match=match):
        controller.invoke_validator(
            attempt,
            validator=tmp_path / "validator.py",
            protocol_manifest=protocol_path,
            run=fake_run,
        )


class _FakePopen:
    next_pid = 7000

    def __init__(self, command, **_kwargs):
        type(self).next_pid += 1
        self.command = [str(item) for item in command]
        self.pid = type(self).next_pid
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


def _literal_assignment(source: str, name: str) -> object:
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Path"
                and len(node.value.args) == 1
            ):
                return ast.literal_eval(node.value.args[0])
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name!r} not found")


def _json_assignment(source: str, name: str) -> dict[str, object]:
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Call)
            and node.value.args
        ):
            return json.loads(ast.literal_eval(node.value.args[0]))
    raise AssertionError(f"JSON assignment {name!r} not found")


def _supervisor_source(outer_source: str) -> str:
    for node in ast.walk(ast.parse(outer_source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            command = ast.literal_eval(node.args[0])
            if len(command) == 3 and command[1] == "-c":
                return str(command[2])
    raise AssertionError("remote supervisor source not found")


def test_remote_launcher_publishes_child_pid_separately_from_owned_group() -> None:
    captured: list[str] = []
    identity: dict[str, object] = {}

    def fake_remote_python(_host, _python, source, **_kwargs):
        captured.append(source)
        if "REMOTE_LAUNCH_V1" in source:
            identity.update(
                {
                    "schema_version": 1,
                    "label": "pi-client",
                    "launch_token": _literal_assignment(source, "launch_token"),
                    "registration_path": _literal_assignment(
                        source, "registration_path"
                    ),
                    "child_pid": 4202,
                    "owner_pid": 4201,
                    "owner_pgid": 4201,
                    "owner_start_ticks": 7654321,
                }
            )
            return json.dumps(identity)
        assert "REMOTE_READ_REGISTRATION_V1" in source
        return json.dumps(identity)

    process = controller._remote_launch_process(
        host_id="pi",
        label="pi-client",
        host="gaps@pi",
        python_bin="/venv/bin/python",
        command=["/venv/bin/python", "-m", "gaps_flower.client_app"],
        cwd="/runtime/hash/src",
        log_path="/runtime/attempt/client.log",
        exit_path="/runtime/attempt/client.exit",
        python_path="/runtime/hash/src",
        registration_path="/runtime/attempt/client.registration.json",
        remote_python=fake_remote_python,
    )

    assert process.pid == 4202
    assert process.owner_pid == 4201
    assert process.owner_pgid == 4201
    assert process.owner_start_ticks == 7654321
    assert process.registration_path == "/runtime/attempt/client.registration.json"
    assert process.launch_token == identity["launch_token"]
    assert "'owner_pgid': process.pid" in captured[0]
    assert "owner_start_ticks = process_start_ticks(process.pid)" in captured[0]
    assert "registration_path.open('x'" in captured[0]
    assert captured[0].index("registration_path.open('x'") < captured[0].index(
        "print(json.dumps(identity"
    )
    supervisor = _supervisor_source(captured[0])
    assert _literal_assignment(supervisor, "cwd") == "/runtime/hash/src"
    assert "child = subprocess.Popen(" in supervisor
    assert "pid_path.open('x'" in supervisor
    assert "handle.write(str(child.pid))" in supervisor
    assert "returncode = child.wait()" in supervisor


def test_lost_launch_ack_reads_registration_then_identity_gated_terminates() -> None:
    calls: list[str] = []
    identity: dict[str, object] = {}

    def fake_remote_python(_host, _python, source, **_kwargs):
        calls.append(source)
        if "REMOTE_LAUNCH_V1" in source:
            identity.update(
                {
                    "schema_version": 1,
                    "label": "server",
                    "launch_token": _literal_assignment(source, "launch_token"),
                    "registration_path": _literal_assignment(
                        source, "registration_path"
                    ),
                    "child_pid": 5202,
                    "owner_pid": 5201,
                    "owner_pgid": 5201,
                    "owner_start_ticks": 8765432,
                }
            )
            raise ConnectionError("SSH response lost after registration")
        if "REMOTE_READ_REGISTRATION_V1" in source:
            return json.dumps(identity)
        assert "REMOTE_TERMINATE_REGISTRATION_V1" in source
        return json.dumps({"recovered": True})

    with pytest.raises(RuntimeError, match="acknowledgement.*recovered"):
        controller._remote_launch_process(
            host_id="ecs",
            label="server",
            host="root@ecs",
            python_bin="/venv/bin/python",
            command=["/venv/bin/python", "-m", "gaps_flower.server_app"],
            cwd="/runtime/hash/src",
            log_path="/runtime/attempt/server.log",
            exit_path="/runtime/attempt/server.exit",
            python_path="/runtime/hash/src",
            registration_path="/runtime/attempt/server.registration.json",
            remote_python=fake_remote_python,
        )

    assert any("REMOTE_READ_REGISTRATION_V1" in source for source in calls)
    terminate = next(
        source for source in calls if "REMOTE_TERMINATE_REGISTRATION_V1" in source
    )
    assert "registration != expected" in terminate
    assert terminate.index("registration != expected") < terminate.index("os.killpg")
    assert "process_start_ticks(owner_pid) != owner_start_ticks" in terminate


@pytest.mark.parametrize("registration_kind", ["malformed", "mismatch"])
def test_bad_launch_registration_never_blind_kills_and_attempt_is_audited(
    tmp_path: Path, registration_kind: str
) -> None:
    remote_calls: list[str] = []
    identity: dict[str, object] = {}

    def fake_remote_python(_host, _python, source, **_kwargs):
        remote_calls.append(source)
        if "REMOTE_LAUNCH_V1" in source:
            identity.update(
                {
                    "schema_version": 1,
                    "label": "server",
                    "launch_token": _literal_assignment(source, "launch_token"),
                    "registration_path": _literal_assignment(
                        source, "registration_path"
                    ),
                    "child_pid": 6202,
                    "owner_pid": 6201,
                    "owner_pgid": 6201,
                    "owner_start_ticks": 9876543,
                }
            )
            raise ConnectionError("launch response truncated")
        assert "REMOTE_READ_REGISTRATION_V1" in source
        if registration_kind == "malformed":
            return json.dumps({"schema_version": 1})
        return json.dumps({**identity, "launch_token": "0" * 32})

    hooks = _hooks([])

    def launch_server(attempt):
        return controller._remote_launch_process(
            host_id="ecs",
            label="server",
            host="root@ecs",
            python_bin="/venv/bin/python",
            command=["/venv/bin/python", "-m", "gaps_flower.server_app"],
            cwd="/runtime/hash/src",
            log_path=f"/runtime/{attempt.attempt_id}/server.log",
            exit_path=f"/runtime/{attempt.attempt_id}/server.exit",
            python_path="/runtime/hash/src",
            registration_path=f"/runtime/{attempt.attempt_id}/server.registration.json",
            remote_python=fake_remote_python,
        )

    hooks = controller.replace(hooks, launch_server=launch_server)
    with pytest.raises(RuntimeError, match="registration"):
        run_confirmation_attempt(
            tmp_path,
            "c12_to_c5__b2__s45",
            provenance=PROVENANCE,
            hooks=hooks,
        )

    assert not any("REMOTE_TERMINATE_REGISTRATION_V1" in source for source in remote_calls)
    run_id = "c12_to_c5__b2__s45"
    attempt_path = tmp_path / run_id / f"{run_id}__a001"
    assert _read_status(attempt_path)["state"] == "failed"
    controller_log = (attempt_path / "controller.log").read_text(encoding="utf-8")
    assert "registration" in controller_log


def test_ecs_rewrite_preserves_algorithm_argv_and_absolutizes_all_dataset_paths() -> None:
    frozen = [
        "/root/gaps_env/bin/python",
        "-m",
        "gaps_flower.server_app",
        "--rounds",
        "25",
        "--server-val-data",
        "dataset/frozen/client_1,dataset/frozen/client_2",
        "--server-calib-data",
        "dataset/frozen/client_5",
        "--da-lambda-coral",
        "0.5",
    ]

    assert controller._rewrite_ecs_dataset_paths(frozen) == [
        "/root/gaps_env/bin/python",
        "-m",
        "gaps_flower.server_app",
        "--rounds",
        "25",
        "--server-val-data",
        "/root/GAPS/dataset/frozen/client_1,/root/GAPS/dataset/frozen/client_2",
        "--server-calib-data",
        "/root/GAPS/dataset/frozen/client_5",
        "--da-lambda-coral",
        "0.5",
    ]


def _install_production_fakes(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, object],
    events: list[str],
    launches: list[str] | None = None,
    launch_identities: dict[str, dict[str, object]] | None = None,
    cleanups: list[str] | None = None,
) -> None:
    source = fixture["source"]
    dataset = fixture["dataset"]
    remote_states = {"root@121.40.139.213": "absent", "gaps@pi": "absent"}
    registrations: dict[str, dict[str, object]] = {}

    def report(host_id: str, source_code: str) -> str:
        expected = _json_assignment(source_code, "expected")
        return json.dumps(
            {
                "host_id": host_id,
                "dependency_versions": copy.deepcopy(controller.EXPECTED_DEPENDENCY_VERSIONS),
                "confirmation_commit": PROVENANCE.confirmation_commit,
                "source_archive_sha256": source["source_archive_sha256"],
                "regular_members_sha256": source["regular_members_sha256"],
                "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
                "algorithm_config_sha256": expected["algorithm_config_sha256"],
                "existing_attempt_processes": [],
            },
            sort_keys=True,
        )

    def fake_remote_python(host, _python, source_code, **_kwargs):
        if "REMOTE_DEPLOY_STATE_V1" in source_code:
            events.append(f"deploy-state:{host}:{remote_states[host]}")
            return json.dumps({"state": remote_states[host]})
        if "REMOTE_RESERVE_ARCHIVE_V1" in source_code:
            assert remote_states[host] == "absent"
            return json.dumps({"state": "reserved"})
        if "REMOTE_INSTALL_ARCHIVE_V1" in source_code:
            assert remote_states[host] == "absent"
            remote_states[host] = "archive_only"
            return json.dumps(
                {"source_archive_sha256": source["source_archive_sha256"]}
            )
        if "REMOTE_EXTRACT_SOURCE_V1" in source_code:
            mode = "fresh" if remote_states[host] == "archive_only" else "reuse"
            events.append(f"deploy:{host}:{mode}")
            if mode == "fresh":
                remote_states[host] = "complete"
            return json.dumps(
                {
                    "source_archive_sha256": source["source_archive_sha256"],
                    "regular_members_sha256": source["regular_members_sha256"],
                }
            )
        if "HOST_PREFLIGHT_V1:" in source_code:
            host_id = source_code.split("HOST_PREFLIGHT_V1:", 1)[1].splitlines()[0].strip()
            events.append(f"preflight:{host_id}")
            return report(host_id, source_code)
        if "REMOTE_LAUNCH_V1:" in source_code:
            label = source_code.split("REMOTE_LAUNCH_V1:", 1)[1].splitlines()[0].strip()
            events.append(f"launch:{label}")
            if launches is not None:
                launches.append(source_code)
            owner_pid = 5000 + len(events) * 2
            identity = {
                "schema_version": 1,
                "label": label,
                "launch_token": _literal_assignment(source_code, "launch_token"),
                "registration_path": _literal_assignment(
                    source_code, "registration_path"
                ),
                "child_pid": owner_pid + 1,
                "owner_pid": owner_pid,
                "owner_pgid": owner_pid,
                "owner_start_ticks": owner_pid * 100,
            }
            registrations[str(identity["registration_path"])] = identity
            if launch_identities is not None:
                launch_identities[label] = identity
            return json.dumps(identity)
        if "REMOTE_READ_REGISTRATION_V1" in source_code:
            registration_path = str(_literal_assignment(source_code, "registration_path"))
            return json.dumps(registrations[registration_path])
        if "REMOTE_PROCESS_STATE_V1" in source_code:
            events.append("monitor:server")
            return json.dumps({"running": False, "returncode": 0})
        if "REMOTE_WAIT_V1:" in source_code:
            label = source_code.split("REMOTE_WAIT_V1:", 1)[1].splitlines()[0].strip()
            events.append(f"wait:{label}")
            return json.dumps({"returncode": 0})
        if "REMOTE_TERMINATE_REGISTRATION_V1" in source_code:
            events.append("cleanup:remote")
            if cleanups is not None:
                cleanups.append(source_code)
            expected = _json_assignment(source_code, "expected")
            registration_path = str(expected["registration_path"])
            assert registrations.pop(registration_path) == expected
            return json.dumps({"recovered": True})
        raise AssertionError(f"unexpected remote source: {source_code[:120]}")

    def fake_run(command, **_kwargs):
        command = [str(item) for item in command]
        if command[:2] == [sys.executable, "-c"] and "HOST_PREFLIGHT_V1:pc" in command[2]:
            events.append("preflight:pc")
            return subprocess.CompletedProcess(command, 0, report("pc", command[2]), "")
        if "--attempt-dir" in command:
            events.append("validator")
            output = Path(command[command.index("--output") + 1])
            _write_json(output, {"status": "valid", "reasons": []})
            digest = sha256_file(output)
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"audit_sha256": digest}), ""
            )
        if command and command[0] == "scp":
            if "-pr" in command:
                events.append("recover")
                destination = Path(command[-1])
                destination.mkdir(parents=True)
                (destination / "events.jsonl").write_text("raw\n", encoding="utf-8")
            else:
                if "/.source.tar." in command[-1] and command[-1].endswith(".tmp"):
                    host = command[-1].split(":", 1)[0]
                    assert remote_states[host] == "absent"
                    events.append(f"source-scp:{host}")
                events.append("scp")
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected run command: {command}")

    def fake_popen(command, **kwargs):
        process = _FakePopen(command, **kwargs)
        label = "pc-sampler" if "scripts.sample_iotj_process_resources" in process.command else "pc-client"
        events.append(f"launch:{label}")
        return process

    monkeypatch.setattr(controller, "_run", fake_run)
    monkeypatch.setattr(controller, "_ssh", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(controller, "_remote_python", fake_remote_python)
    monkeypatch.setattr(controller, "_wait_for_pi", lambda *_args, **_kwargs: "gaps@pi")
    monkeypatch.setattr(
        controller,
        "_start_tunnels",
        lambda *_args, **_kwargs: events.append("tunnels") or [_FakePopen(["tunnel"])],
    )
    monkeypatch.setattr(
        controller,
        "_terminate_processes",
        lambda processes: events.append("cleanup:local"),
    )
    monkeypatch.setattr(controller.subprocess, "Popen", fake_popen)


def test_formal_deploy_failure_is_audited_inside_allocated_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    validator = tmp_path / "validator.py"
    validator.write_text("# fake validator boundary\n", encoding="utf-8")
    monkeypatch.setattr(controller, "_wait_for_pi", lambda *_args, **_kwargs: "gaps@pi")
    monkeypatch.setattr(
        controller,
        "_ssh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    def fake_remote_python(_host, _python, source_code, **_kwargs):
        if "REMOTE_DEPLOY_STATE_V1" in source_code:
            return json.dumps({"state": "absent"})
        if "REMOTE_RESERVE_ARCHIVE_V1" in source_code:
            return json.dumps({"state": "reserved"})
        raise AssertionError(f"unexpected remote source: {source_code[:120]}")

    monkeypatch.setattr(controller, "_remote_python", fake_remote_python)

    def fail_transfer(command, **_kwargs):
        if str(command[0]) == "scp":
            raise ArchiveMismatch("forced deployment transfer failure")
        raise AssertionError(command)

    monkeypatch.setattr(controller, "_run", fail_transfer)

    with pytest.raises(ArchiveMismatch, match="forced deployment"):
        controller.main(
            _controller_argv(fixture, tmp_path, "--validator", str(validator))
        )

    run_id = confirmation_run_id("B2", 42)
    attempt = tmp_path / "raw" / run_id / f"{run_id}__a001"
    assert _read_status(attempt)["state"] == "failed"
    assert "forced deployment transfer failure" in (attempt / "controller.log").read_text(
        encoding="utf-8"
    )


def test_two_formal_runs_verify_and_reuse_same_content_addressed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    validator = tmp_path / "validator.py"
    validator.write_text("# fake validator boundary\n", encoding="utf-8")
    events: list[str] = []
    _install_production_fakes(monkeypatch, fixture, events)

    assert controller.main(
        _controller_argv(
            fixture,
            tmp_path,
            "--runs",
            "B2:42,B2:43",
            "--validator",
            str(validator),
        )
    ) == 0

    for host in ("root@121.40.139.213", "gaps@pi"):
        assert events.count(f"source-scp:{host}") == 1
        assert events.count(f"deploy:{host}:fresh") == 1
        assert events.count(f"deploy:{host}:reuse") == 1
    for seed in (42, 43):
        run_id = confirmation_run_id("B2", seed)
        assert _read_status(tmp_path / "raw" / run_id / f"{run_id}__a001")["state"] == "canonical"


def test_formal_server_launch_uses_only_frozen_source_and_absolute_ecs_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    validator = tmp_path / "validator.py"
    validator.write_text("# fake validator boundary\n", encoding="utf-8")
    events: list[str] = []
    launches: list[str] = []
    identities: dict[str, dict[str, object]] = {}
    cleanups: list[str] = []
    _install_production_fakes(
        monkeypatch, fixture, events, launches, identities, cleanups
    )

    assert controller.main(
        _controller_argv(fixture, tmp_path, "--validator", str(validator))
    ) == 0

    server_outer = next(source for source in launches if "REMOTE_LAUNCH_V1:server" in source)
    supervisor = _supervisor_source(server_outer)
    source_hash = fixture["source"]["source_archive_sha256"]
    extracted_src = f"/root/GAPS/confirmation_runtime/{source_hash}/src"
    assert _literal_assignment(supervisor, "cwd") == extracted_src
    assert f"environment['PYTHONPATH'] = {extracted_src!r}" in supervisor
    command = _literal_assignment(supervisor, "command")
    data_index = command.index("--data-root")
    assert command[data_index + 1] == "/root/GAPS/dataset/frozen_dataset"
    assert "/root/GAPS" != _literal_assignment(supervisor, "cwd")
    run_id = confirmation_run_id("B2", 42)
    remote_attempt = f"/root/GAPS/confirmation_runtime/{source_hash}/attempts/{run_id}__a001"
    assert command == [
        "python",
        "-m",
        "gaps_flower.server_app",
        "--data-root",
        "/root/GAPS/dataset/frozen_dataset",
        "--output-dir",
        f"{remote_attempt}/raw/server/training",
        "--observer-context",
        f"{remote_attempt}/contexts/server.json",
        "--observer-events",
        f"{remote_attempt}/raw/server/events.jsonl",
    ]

    pi_sampler_outer = next(
        source for source in launches if "REMOTE_LAUNCH_V1:pi-sampler" in source
    )
    sampler_command = _literal_assignment(_supervisor_source(pi_sampler_outer), "command")
    pid_index = sampler_command.index("--pid")
    assert int(sampler_command[pid_index + 1]) == identities["pi-client"]["child_pid"]
    assert int(sampler_command[pid_index + 1]) != identities["pi-client"]["owner_pid"]

    owned_registrations: list[dict[str, object]] = []
    for cleanup in cleanups:
        expected = _json_assignment(cleanup, "expected")
        owned_registrations.append(expected)
        assert cleanup.index("registration != expected") < cleanup.index("os.killpg")
        assert "actual_pgid != owner_pgid" in cleanup
        assert "process_start_ticks(owner_pid) != owner_start_ticks" in cleanup
        assert str(expected["registration_path"]).endswith(".registration.json")
    owned_pgids = {row["owner_pgid"] for row in owned_registrations}
    expected_remote_pgids = {
        identity["owner_pgid"]
        for label, identity in identities.items()
        if label in {"server", "pi-client", "pi-sampler"}
    }
    assert owned_pgids == expected_remote_pgids
    assert not ({identity["child_pid"] for identity in identities.values()} & owned_pgids)


def test_main_validate_only_and_preflight_only_never_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_input_fixture(tmp_path / "validate")
    monkeypatch.setattr(
        controller,
        "_wait_for_pi",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validate-only performed an external action")
        ),
    )
    assert controller.main(_controller_argv(fixture, tmp_path / "validate", "--validate-inputs-only")) == 0
    monkeypatch.undo()

    fixture = _frozen_input_fixture(tmp_path / "preflight")
    events: list[str] = []
    _install_production_fakes(monkeypatch, fixture, events)
    assert controller.main(_controller_argv(fixture, tmp_path / "preflight", "--preflight-only")) == 0
    assert {"preflight:ecs", "preflight:pi", "preflight:pc"} <= set(events)
    assert not any(event.startswith("launch:") for event in events)
    assert "validator" not in events
    assert not (tmp_path / "preflight" / "raw").exists()


def test_main_formal_binds_real_helpers_in_fail_closed_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    validator = tmp_path / "validator.py"
    validator.write_text("# path is executed by fake _run\n", encoding="utf-8")
    events: list[str] = []
    _install_production_fakes(monkeypatch, fixture, events)

    assert controller.main(
        _controller_argv(fixture, tmp_path, "--validator", str(validator))
    ) == 0

    preflight_end = max(events.index(name) for name in ("preflight:ecs", "preflight:pi", "preflight:pc"))
    assert preflight_end < events.index("launch:server")
    assert events.index("launch:server") < events.index("tunnels")
    assert events.index("tunnels") < events.index("launch:pi-client")
    assert events.index("launch:pi-client") < events.index("launch:pi-sampler")
    assert events.index("launch:pi-sampler") < events.index("launch:pc-client")
    assert events.index("launch:pc-client") < events.index("launch:pc-sampler")
    assert events.index("monitor:server") < events.index("recover")
    assert events.index("recover") < events.index("validator")
    first_run = confirmation_run_id("B2", 42)
    attempt_path = (
        tmp_path
        / "raw"
        / first_run
        / f"{first_run}__a001"
    )
    assert _read_status(attempt_path)["state"] == "canonical"

    fixture = _frozen_input_fixture(tmp_path / "second")
    protocol = copy.deepcopy(fixture["protocol"])
    protocol["schedule"][0]["algorithm_config_sha256"] = "0" * 64
    protocol_without_hash = dict(protocol)
    protocol_without_hash.pop("protocol_manifest_sha256")
    protocol["protocol_manifest_sha256"] = canonical_sha256(protocol_without_hash)
    _write_json(fixture["protocol_path"], protocol)
    for run_id, original in fixture["commands"].items():
        command = copy.deepcopy(original)
        command["protocol_manifest_sha256"] = protocol["protocol_manifest_sha256"]
        _write_json(
            fixture["command_root"] / run_id / "command_manifest.json", command
        )
    with pytest.raises(ValueError, match="algorithm"):
        _load_frozen(fixture)
