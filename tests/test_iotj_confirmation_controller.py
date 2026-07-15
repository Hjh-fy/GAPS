from __future__ import annotations

import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
import copy
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
                "server_ecs": ["python", "-m", "gaps_flower.server_app"],
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
    real_islink = controller.os.path.islink
    monkeypatch.setattr(
        controller.os.path,
        "islink",
        lambda path: Path(path) == raw_root or real_islink(path),
    )
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


def _install_production_fakes(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, object],
    events: list[str],
) -> None:
    source = fixture["source"]
    dataset = fixture["dataset"]
    first_run = confirmation_run_id("B2", 42)
    algorithm_hash = fixture["commands"][first_run]["algorithm_config_sha256"]

    def report(host_id: str) -> str:
        return json.dumps(
            {
                "host_id": host_id,
                "dependency_versions": copy.deepcopy(controller.EXPECTED_DEPENDENCY_VERSIONS),
                "confirmation_commit": PROVENANCE.confirmation_commit,
                "source_archive_sha256": source["source_archive_sha256"],
                "regular_members_sha256": source["regular_members_sha256"],
                "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
                "algorithm_config_sha256": algorithm_hash,
                "existing_attempt_processes": [],
            },
            sort_keys=True,
        )

    def fake_remote_python(host, _python, source_code, **_kwargs):
        if "REMOTE_EXTRACT_SOURCE_V1" in source_code:
            events.append(f"deploy:{host}")
            return json.dumps(
                {
                    "source_archive_sha256": source["source_archive_sha256"],
                    "regular_members_sha256": source["regular_members_sha256"],
                }
            )
        if "HOST_PREFLIGHT_V1:" in source_code:
            host_id = source_code.split("HOST_PREFLIGHT_V1:", 1)[1].splitlines()[0].strip()
            events.append(f"preflight:{host_id}")
            return report(host_id)
        if "REMOTE_LAUNCH_V1:" in source_code:
            label = source_code.split("REMOTE_LAUNCH_V1:", 1)[1].splitlines()[0].strip()
            events.append(f"launch:{label}")
            return str(5000 + len(events))
        if "REMOTE_PROCESS_STATE_V1" in source_code:
            events.append("monitor:server")
            return json.dumps({"running": False, "returncode": 0})
        if "REMOTE_WAIT_V1:" in source_code:
            label = source_code.split("REMOTE_WAIT_V1:", 1)[1].splitlines()[0].strip()
            events.append(f"wait:{label}")
            return json.dumps({"returncode": 0})
        if "REMOTE_CLEANUP_V1" in source_code:
            events.append("cleanup:remote")
            return "CLEANED"
        raise AssertionError(f"unexpected remote source: {source_code[:120]}")

    def fake_run(command, **_kwargs):
        command = [str(item) for item in command]
        if command[:2] == [sys.executable, "-c"] and "HOST_PREFLIGHT_V1:pc" in command[2]:
            events.append("preflight:pc")
            return subprocess.CompletedProcess(command, 0, report("pc"), "")
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


def test_main_validate_only_and_preflight_only_never_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_input_fixture(tmp_path / "validate")
    monkeypatch.setattr(
        controller,
        "deploy_source_archive",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validate-only deployed")
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
