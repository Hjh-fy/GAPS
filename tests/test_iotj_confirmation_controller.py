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

import scripts.build_iotj_c2_dataset_subset_manifest as c2_subset_generator
import scripts.generate_iotj_ecs_c2_topology_manifest as topology_generator
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
    bind_remote_attempt_paths,
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


def _write_ecs_c2_topology_manifest(
    path: Path, *, archive_sha: str, config_by_run: dict[str, str]
) -> Path:
    payload = {
        "topology_id": "ecs_c2_pi_c1",
        "source_archive_sha256": archive_sha,
        "algorithm_config_sha256_by_run": config_by_run,
        "hosts": {
            "C2": {
                "host_id": "ecs-c2",
                "ssh_host": "root@114.55.171.63",
            }
        },
    }
    payload["execution_topology_manifest_sha256"] = canonical_sha256(payload)
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_ecs_c2_topology_manifest_binds_every_frozen_run_hash(
    tmp_path: Path,
) -> None:
    configs = {
        "c12_to_c5__b2__s42": "a" * 64,
        "c12_to_c5__b5__s42": "b" * 64,
    }
    path = _write_ecs_c2_topology_manifest(
        tmp_path / "topology.json", archive_sha="c" * 64, config_by_run=configs
    )

    actual = controller.load_execution_topology_manifest(
        path, expected_archive_sha="c" * 64, expected_config_by_run=configs
    )

    assert actual["algorithm_config_sha256_by_run"] == configs


def test_ecs_c2_topology_manifest_rejects_single_run_config_mismatch(
    tmp_path: Path,
) -> None:
    path = _write_ecs_c2_topology_manifest(
        tmp_path / "topology.json",
        archive_sha="c" * 64,
        config_by_run={"c12_to_c5__b2__s42": "a" * 64},
    )

    with pytest.raises(RuntimeError, match="algorithm config"):
        controller.load_execution_topology_manifest(
            path,
            expected_archive_sha="c" * 64,
            expected_config_by_run={"c12_to_c5__b2__s42": "b" * 64},
        )


def test_ecs_c2_topology_manifest_rejects_self_hash_mismatch(tmp_path: Path) -> None:
    configs = {"c12_to_c5__b2__s42": "a" * 64}
    path = _write_ecs_c2_topology_manifest(
        tmp_path / "topology.json", archive_sha="c" * 64, config_by_run=configs
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["hosts"]["C2"]["ssh_host"] = "root@wrong-host"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="self SHA-256"):
        controller.load_execution_topology_manifest(
            path, expected_archive_sha="c" * 64, expected_config_by_run=configs
        )


def test_ecs_c2_topology_generator_binds_exact_protocol_schedule(tmp_path: Path) -> None:
    protocol = {
        "source_archive_sha256": "a" * 64,
        "schedule": [
            {"run_id": "c12_to_c5__b2__s42", "algorithm_config_sha256": "b" * 64},
            {"run_id": "c12_to_c5__b5__s42", "algorithm_config_sha256": "c" * 64},
        ],
    }

    manifest = topology_generator.build_execution_topology_manifest(protocol)

    assert manifest["source_archive_sha256"] == "a" * 64
    assert manifest["algorithm_config_sha256_by_run"] == {
        "c12_to_c5__b2__s42": "b" * 64,
        "c12_to_c5__b5__s42": "c" * 64,
    }
    assert manifest["hosts"]["C2"]["host_id"] == "ecs-c2"


def test_c2_subset_manifest_contains_only_client_2_files() -> None:
    full = {
        "dataset_manifest_sha256": "a" * 64,
        "files": [
            {"relative_path": "client_1/train_features.npy", "sha256": "b" * 64, "byte_size": 1},
            {"relative_path": "client_2/train_features.npy", "sha256": "c" * 64, "byte_size": 2},
            {"relative_path": "client_2/train_classification_labels.npy", "sha256": "d" * 64, "byte_size": 3},
        ],
    }

    subset = c2_subset_generator.build_client_subset_manifest(full, client_id=2)

    assert subset["parent_dataset_manifest_sha256"] == "a" * 64
    assert [row["relative_path"] for row in subset["files"]] == [
        "client_2/train_classification_labels.npy",
        "client_2/train_features.npy",
    ]
    assert all(row["relative_path"].startswith("client_2/") for row in subset["files"])


def test_remote_c2_contexts_use_ecs_identity_and_distinct_names(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b2__s42")

    contexts = controller.write_host_contexts(
        attempt,
        group_id="B2",
        seed=42,
        provenance=PROVENANCE,
        c2_host_id="ecs-c2",
    )

    assert {"ecs_c2_client", "ecs_c2_sampler"}.issubset(contexts)
    assert "pc_client" not in contexts
    client = json.loads(contexts["ecs_c2_client"].read_text(encoding="utf-8"))
    sampler = json.loads(contexts["ecs_c2_sampler"].read_text(encoding="utf-8"))
    assert client["host_id"] == sampler["host_id"] == "ecs-c2"
    assert client["client_id"] == sampler["client_id"] == "C2"


def _fake_remote_deployments(root: str) -> dict[str, controller.HostDeployment]:
    return {
        host_id: controller.HostDeployment(
            host_id=host_id,
            archive_path=f"{root}/{host_id}/source.tar",
            src_path=f"{root}/{host_id}/src",
            source_archive_sha256=PROVENANCE.source_archive_sha256,
            regular_members_sha256="e" * 64,
        )
        for host_id in ("ecs", "pi", "ecs_c2")
    }


def test_ecs_c2_remote_binding_is_unique(
    tmp_path: Path,
) -> None:
    topology_path = _write_ecs_c2_topology_manifest(
        tmp_path / "topology.json",
        archive_sha=str(PROVENANCE.source_archive_sha256),
        config_by_run={"c12_to_c5__b2__s42": str(PROVENANCE.algorithm_config_sha256)},
    )
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    first = _allocate_bound(tmp_path / "first", "c12_to_c5__b2__s42")
    second = _allocate_bound(tmp_path / "second", "c12_to_c5__b2__s42")

    first_binding = bind_remote_attempt_paths(
        first,
        topology=topology,
        deployments=_fake_remote_deployments("/runtime"),
    )
    second_binding = bind_remote_attempt_paths(
        second,
        topology=topology,
        deployments=_fake_remote_deployments("/runtime"),
    )

    assert first.attempt_id == second.attempt_id
    assert first_binding["remote_directory_name"] != second_binding["remote_directory_name"]
    assert set(first_binding["remote_roots"]) == {"ecs", "pi", "ecs_c2"}
    for root in first_binding["remote_roots"].values():
        assert f"/attempts/ecs_c2_pi_c1/{first.attempt_id}__" in root
        assert f"/attempts/{first.attempt_id}" not in root
    stored = json.loads(
        (first.path / "remote_attempt_binding.json").read_text(encoding="utf-8")
    )
    assert stored == first_binding
    with pytest.raises(FileExistsError):
        bind_remote_attempt_paths(
            first,
            topology=topology,
            deployments=_fake_remote_deployments("/runtime"),
        )


def _read_status(path: Path) -> dict[str, object]:
    return json.loads((path / "attempt_status.json").read_text(encoding="utf-8"))


def _allocate_bound(tmp_path: Path, run_id: str):
    attempt = allocate_attempt(tmp_path, run_id)
    bind_attempt_provenance(attempt, PROVENANCE)
    return attempt


def _execute_generated_remote_source(source: str) -> dict[str, object]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, "<generated-remote-source>", "exec"), {})
    lines = [line for line in output.getvalue().splitlines() if line.strip()]
    assert lines, "generated remote source did not emit its acknowledgement"
    return json.loads(lines[-1])


def _write_tampered_status_chain(attempt, events: list[dict[str, object]]) -> None:
    events_root = attempt.path / "status_events"
    events_root.mkdir(exist_ok=True)
    for existing in events_root.iterdir():
        existing.unlink()
    for sequence, event in enumerate(events, start=1):
        payload = {
            "sequence": sequence,
            "run_id": attempt.run_id,
            "attempt_id": attempt.attempt_id,
            "state": event["state"],
            "event_type": event["event_type"],
            "reason": event["reason"],
            "reason_category": controller.STATUS_REASON_CATEGORIES[event["reason"]],
            "wall_time_utc": "2026-07-15T12:00:00.000000Z",
            "confirmation_commit": PROVENANCE.confirmation_commit,
            "source_archive_sha256": PROVENANCE.source_archive_sha256,
            "dataset_manifest_sha256": PROVENANCE.dataset_manifest_sha256,
            "algorithm_config_sha256": PROVENANCE.algorithm_config_sha256,
            "audit_sha256": event.get("audit_sha256"),
        }
        _write_json(events_root / f"status_{sequence:03d}.json", payload)
        if sequence == len(events):
            _write_json(attempt.path / "attempt_status.json", payload)


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
                "transport_status": "not_collected",
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
            "transport_status": "not_collected",
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
    bind_attempt_provenance(second, PROVENANCE)
    mark_attempt(second.path, "running", reason="attempt_allocated")
    mark_attempt(second.path, "running", reason="preflight_passed")
    mark_attempt(
        second.path,
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
    mark_attempt(attempt.path, "running", reason="attempt_allocated")
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
    ecs_runtime_base = getattr(controller, "ECS_REMOTE_RUNTIME_BASE", None)
    pi_runtime_base = getattr(controller, "PI_REMOTE_RUNTIME_BASE", None)
    assert ecs_runtime_base == "/root/GAPS/confirmation_runtime"
    assert pi_runtime_base == "/home/gaps/GAPS/confirmation_runtime"
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
        f"{ecs_runtime_base}/{manifest['source_archive_sha256']}/.source.tar."
    )
    assert destinations["gaps@pi"].startswith(
        f"{pi_runtime_base}/{manifest['source_archive_sha256']}/.source.tar."
    )
    assert all(destination.endswith(".tmp") for destination in destinations.values())
    assert all("sync" not in " ".join(call).lower() for call in run_calls)
    assert ssh_calls == []
    pc_archive = Path(deployments["pc"].archive_path)
    assert pc_archive.read_bytes() == archive_path.read_bytes()
    assert (Path(deployments["pc"].src_path) / "app.py").is_file()


def test_remote_c2_deployment_skips_local_pc_runtime(tmp_path: Path) -> None:
    archive_path, manifest = _source_fixture(tmp_path / "source")
    run_calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        run_calls.append([str(item) for item in command])
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
        return remote_report

    runtime = tmp_path / "must-not-exist"
    deployments = deploy_source_archive(
        archive_path,
        manifest,
        ecs_host="root@ecs",
        pi_host="gaps@pi",
        c2_host="root@c2",
        c2_python="/root/gaps_c2_cpu_env/bin/python",
        c2_runtime_base="/root/GAPS/confirmation_runtime_c2",
        pc_runtime_root=runtime,
        run=fake_run,
        ssh=lambda *_args, **_kwargs: pytest.fail("deployment used ssh helper"),
        remote_python=fake_remote_python,
    )

    assert set(deployments) == {"ecs", "pi", "ecs_c2"}
    assert deployments["ecs_c2"].host_id == "ecs_c2"
    assert str(deployments["ecs_c2"].src_path).startswith(
        "/root/GAPS/confirmation_runtime_c2/"
    )
    assert not runtime.exists()
    assert {call[3].split(":", 1)[0] for call in run_calls if call[0] == "scp"} == {
        "root@ecs",
        "gaps@pi",
        "root@c2",
    }


def test_ecs_c2_tunnel_commands_keep_all_flower_endpoints_loopback() -> None:
    commands = controller.build_ecs_c2_tunnel_commands(
        "root@server", "gaps@pi", "root@c2"
    )

    assert len(commands) == 3
    assert commands[0][-2:] == ["127.0.0.1:18080:127.0.0.1:8080", "root@server"]
    assert commands[1][-2:] == ["127.0.0.1:18080:127.0.0.1:18080", "gaps@pi"]
    assert commands[2][-2:] == ["127.0.0.1:18080:127.0.0.1:18080", "root@c2"]
    assert "-L" in commands[0]
    assert "-R" in commands[1] and "-R" in commands[2]
    assert all("ServerAliveCountMax=20" in command for command in commands)
    assert all("0.0.0.0" not in " ".join(command) for command in commands)


def test_ecs_c2_tunnel_start_failure_reclaims_prior_tunnels() -> None:
    events: list[str] = []

    class Tunnel:
        def __init__(self, label: str, failed: bool) -> None:
            self.label = label
            self.failed = failed
            self.terminated = False

        def poll(self):
            return 1 if self.failed else None

        def terminate(self) -> None:
            self.terminated = True
            events.append(f"terminate:{self.label}")

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:
            events.append(f"kill:{self.label}")

    created: list[Tunnel] = []

    def fake_popen(command, **_kwargs):
        tunnel = Tunnel(str(command[-1]), failed=len(created) == 2)
        created.append(tunnel)
        return tunnel

    with pytest.raises(RuntimeError, match="ecs-c2"):
        controller._start_ecs_c2_tunnels(
            "root@server", "gaps@pi", "root@c2", popen=fake_popen, sleeper=lambda _x: None
        )

    assert events == ["terminate:root@server", "terminate:gaps@pi"]
    assert not created[2].terminated


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH regression")
def test_pc_deploy_temp_name_fits_long_content_root(tmp_path: Path) -> None:
    archive_path, manifest = _source_fixture(tmp_path / "source")
    source_hash = str(manifest["source_archive_sha256"])
    target_root_length = 219
    padding = target_root_length - len(str(tmp_path.absolute())) - len(source_hash) - 2
    if padding < 1:
        pytest.skip("pytest temporary path is too long for the MAX_PATH boundary fixture")
    runtime = tmp_path / ("r" * padding)
    content_root = runtime / source_hash
    legacy_temp = content_root / f".source.tar.{'0' * 32}.tmp"
    assert len(str(content_root.absolute())) == target_root_length
    assert len(str(legacy_temp.absolute())) >= 260

    remote_report = json.dumps(
        {
            "source_archive_sha256": manifest["source_archive_sha256"],
            "regular_members_sha256": manifest["regular_members_sha256"],
        }
    )

    def fake_remote_python(_host, _python, source, **_kwargs):
        if "REMOTE_DEPLOY_STATE_V1" in source:
            return json.dumps({"state": "complete"})
        assert "REMOTE_EXTRACT_SOURCE_V1" in source
        return remote_report

    deployments = deploy_source_archive(
        archive_path,
        manifest,
        ecs_host="root@ecs",
        pi_host="gaps@pi",
        pc_runtime_root=runtime,
        run=lambda *_args, **_kwargs: pytest.fail("complete remote runtime transferred"),
        ssh=lambda *_args, **_kwargs: pytest.fail("complete remote runtime used ssh"),
        remote_python=fake_remote_python,
    )

    assert Path(deployments["pc"].archive_path).read_bytes() == archive_path.read_bytes()
    assert (Path(deployments["pc"].src_path) / "app.py").is_file()

    (Path(deployments["pc"].src_path) / "app.py").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ArchiveMismatch, match="tracked member"):
        deploy_source_archive(
            archive_path,
            manifest,
            ecs_host="root@ecs",
            pi_host="gaps@pi",
            pc_runtime_root=runtime,
            run=lambda *_args, **_kwargs: pytest.fail("tampered runtime transferred"),
            ssh=lambda *_args, **_kwargs: pytest.fail("tampered runtime used ssh"),
            remote_python=lambda *_args, **_kwargs: pytest.fail(
                "tampered runtime contacted host"
            ),
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH regression")
def test_local_launch_imports_module_beyond_max_path(tmp_path: Path) -> None:
    target_source_length = 223
    padding = target_source_length - len(str(tmp_path.absolute())) - 1
    if padding < 1:
        pytest.skip("pytest temporary path is too long for the MAX_PATH fixture")
    source = tmp_path / ("r" * padding)
    source_io = controller._windows_extended_local_path(source.absolute())
    module = source_io / "scripts" / "sample_iotj_process_resources.py"
    module.parent.mkdir(parents=True)
    (module.parent / "__init__.py").write_text("", encoding="utf-8")
    module.write_text("print('long-path-module-ok')\n", encoding="utf-8")
    assert len(str(source.absolute())) == target_source_length
    assert len(str(source.absolute() / "scripts" / module.name)) >= 260

    process = controller._local_launch_process(
        label="pc-sampler",
        command=[sys.executable, "-m", "scripts.sample_iotj_process_resources"],
        cwd=source,
        log_root=tmp_path / "logs",
    )
    try:
        assert process.handle.wait(timeout=30) == 0
    finally:
        for handle in process.log_handles:
            handle.close()
    assert (tmp_path / "logs" / "stdout.log").read_text(
        encoding="utf-8"
    ).strip() == "long-path-module-ok"
    assert (tmp_path / "logs" / "stderr.log").read_text(encoding="utf-8") == ""


def test_controller_remote_python_streams_long_probe_over_utf8_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "# 传感器 probe\n" + ("value = 1\n" * 7000)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append(([str(item) for item in command], dict(kwargs)))
        return subprocess.CompletedProcess(command, 0, " acknowledged\n", "")

    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    output = controller._remote_python(
        "root@ecs", "/root/gaps_env/bin/python", source, timeout=123
    )

    assert output == "acknowledged"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "root@ecs",
        "/root/gaps_env/bin/python",
        "-",
    ]
    assert "-n" not in command
    assert len(subprocess.list2cmdline(command)) < 1024
    assert kwargs == {
        "cwd": controller.REPO_ROOT,
        "input": source,
        "text": True,
        "encoding": "utf-8",
        "capture_output": True,
        "timeout": 123,
    }


def test_controller_remote_python_default_timeout_allows_slow_content_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(_command, **kwargs):
        calls.append(dict(kwargs))
        return subprocess.CompletedProcess([], 0, "ok\n", "")

    monkeypatch.setattr(controller.subprocess, "run", fake_run)

    assert controller._remote_python("root@ecs", "/root/gaps_env/bin/python", "print(1)") == "ok"
    assert calls[0]["timeout"] == 120


def test_controller_remote_python_rejects_unapproved_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("unapproved remote Python executed"),
    )
    with pytest.raises(ValueError, match="approved absolute path"):
        controller._remote_python("root@ecs", "/tmp/python", "print('unsafe')")


def test_controller_remote_python_preserves_nonzero_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 7, "", "remote probe failed"
        ),
    )
    with pytest.raises(RuntimeError, match=r"command failed \(7\)") as exc_info:
        controller._remote_python(
            "gaps@pi", "/home/gaps/GAPS/gaps_rpi_env/bin/python", "print('probe')"
        )
    assert "remote probe failed" in str(exc_info.value)


def test_remote_python_classifies_ssh_255_as_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["ssh"], 255, "", "ssh: transient network failure"
        ),
    )

    with pytest.raises(controller.RemoteTransportError, match="transient network"):
        controller._remote_python(
            "root@ecs", "/root/gaps_env/bin/python", "print('probe')"
        )


def test_remote_monitor_retries_transient_transport_and_recovers() -> None:
    process = controller.OwnedProcess(host_id="ecs", label="server", pid=123)
    states: list[object] = [
        controller.RemoteTransportError("temporary outage"),
        (True, None),
        (False, 0),
    ]
    sleeps: list[float] = []
    events: list[str] = []

    def read_state(_process):
        value = states.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    ticks = iter((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    controller._monitor_remote_server(
        process,
        timeout_seconds=60,
        poll_seconds=1,
        transport_grace_seconds=10,
        state_reader=read_state,
        sleeper=sleeps.append,
        monotonic=lambda: next(ticks),
        transport_event=events.append,
    )

    assert states == []
    assert sleeps == [1, 1]
    assert events[0].startswith("transport_lost:")
    assert events[1] == "transport_recovered"


def test_remote_monitor_fails_after_transport_grace() -> None:
    process = controller.OwnedProcess(host_id="ecs", label="server", pid=123)

    def fail_state(_process):
        raise controller.RemoteTransportError("still offline")

    ticks = iter((0.0, 1.0, 5.0, 12.0))
    with pytest.raises(controller.RemoteTransportError, match="grace exceeded"):
        controller._monitor_remote_server(
            process,
            timeout_seconds=60,
            poll_seconds=1,
            transport_grace_seconds=10,
            state_reader=fail_state,
            sleeper=lambda _seconds: None,
            monotonic=lambda: next(ticks),
        )


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


def test_generated_remote_reserve_hash_and_install_source_succeeds_atomically(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime" / "hash"
    runtime_root.mkdir(parents=True)
    temporary = runtime_root / ".source.tar.behavior.tmp"
    final = runtime_root / "source.tar"
    payload = b"exact frozen source archive bytes\x00\xff"
    expected_sha256 = controller.sha256_bytes(payload)

    reserve = _execute_generated_remote_source(
        controller._remote_reserve_archive_source(
            str(temporary), str(runtime_root)
        )
    )
    assert reserve == {"state": "reserved"}
    assert temporary.is_file() and temporary.read_bytes() == b""
    reserved_inode = temporary.stat().st_ino

    temporary.write_bytes(payload)
    installed = _execute_generated_remote_source(
        controller._remote_install_archive_source(
            str(temporary), str(final), expected_sha256, str(runtime_root)
        )
    )

    assert installed == {"source_archive_sha256": expected_sha256}
    assert final.is_file() and not final.is_symlink()
    assert final.read_bytes() == payload
    assert final.stat().st_ino == reserved_inode
    assert not temporary.exists() and not temporary.is_symlink()


def test_generated_remote_install_wrong_hash_leaves_final_absent_and_temp_intact(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime" / "hash"
    runtime_root.mkdir(parents=True)
    temporary = runtime_root / ".source.tar.wrong.tmp"
    final = runtime_root / "source.tar"
    payload = b"wrong archive bytes"
    _execute_generated_remote_source(
        controller._remote_reserve_archive_source(
            str(temporary), str(runtime_root)
        )
    )
    temporary.write_bytes(payload)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _execute_generated_remote_source(
            controller._remote_install_archive_source(
                str(temporary), str(final), "0" * 64, str(runtime_root)
            )
        )

    assert not final.exists() and not final.is_symlink()
    assert temporary.is_file() and temporary.read_bytes() == payload


def test_generated_remote_reserve_rejects_existing_and_unsafe_leaves(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime" / "hash"
    runtime_root.mkdir(parents=True)
    existing = runtime_root / ".source.tar.existing.tmp"
    existing.write_bytes(b"do not truncate")

    with pytest.raises(RuntimeError, match="already exists"):
        _execute_generated_remote_source(
            controller._remote_reserve_archive_source(
                str(existing), str(runtime_root)
            )
        )
    assert existing.read_bytes() == b"do not truncate"

    escaped = runtime_root.parent / ".source.tar.escaped.tmp"
    with pytest.raises(RuntimeError, match="escapes runtime root"):
        _execute_generated_remote_source(
            controller._remote_reserve_archive_source(
                str(escaped), str(runtime_root)
            )
        )
    assert not escaped.exists()


@pytest.mark.parametrize("dangling", [False, True], ids=["symlink", "dangling"])
def test_generated_remote_reserve_rejects_symlink_and_dangling_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dangling: bool
) -> None:
    runtime_root = tmp_path / "runtime" / "hash"
    runtime_root.mkdir(parents=True)
    temporary = runtime_root / ".source.tar.unsafe.tmp"
    if not dangling:
        temporary.write_bytes(b"symlink target bytes")
    original_lstat = os.lstat

    class _SyntheticSymlinkStat:
        st_mode = stat.S_IFLNK | 0o777
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

    monkeypatch.setattr(
        os,
        "lstat",
        lambda path, *args, **kwargs: (
            _SyntheticSymlinkStat()
            if Path(path) == temporary
            else original_lstat(path, *args, **kwargs)
        ),
    )

    with pytest.raises(RuntimeError, match="symlink|reparse"):
        _execute_generated_remote_source(
            controller._remote_reserve_archive_source(
                str(temporary), str(runtime_root)
            )
        )


def test_generated_remote_install_refuses_existing_final_without_overwrite(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime" / "hash"
    runtime_root.mkdir(parents=True)
    temporary = runtime_root / ".source.tar.install.tmp"
    final = runtime_root / "source.tar"
    incoming = b"incoming frozen archive"
    final.write_bytes(b"existing final bytes")
    temporary.write_bytes(incoming)

    with pytest.raises(RuntimeError, match="already exists"):
        _execute_generated_remote_source(
            controller._remote_install_archive_source(
                str(temporary),
                str(final),
                controller.sha256_bytes(incoming),
                str(runtime_root),
            )
        )

    assert final.read_bytes() == b"existing final bytes"
    assert temporary.read_bytes() == incoming


def test_generated_remote_install_rejects_nonregular_temp(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime" / "hash"
    runtime_root.mkdir(parents=True)
    temporary = runtime_root / ".source.tar.directory.tmp"
    temporary.mkdir()
    final = runtime_root / "source.tar"

    with pytest.raises(RuntimeError, match="not a real file"):
        _execute_generated_remote_source(
            controller._remote_install_archive_source(
                str(temporary), str(final), "0" * 64, str(runtime_root)
            )
        )

    assert temporary.is_dir()
    assert not final.exists()


def test_generated_remote_install_rejects_leaves_outside_reserved_runtime(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime" / "hash"
    runtime_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    temporary = outside / ".source.tar.unsafe.tmp"
    final = outside / "source.tar"
    payload = b"must not install outside the pinned runtime"
    temporary.write_bytes(payload)

    with pytest.raises(RuntimeError, match="escapes runtime root"):
        _execute_generated_remote_source(
            controller._remote_install_archive_source(
                str(temporary),
                str(final),
                controller.sha256_bytes(payload),
                str(runtime_root),
            )
        )

    assert temporary.read_bytes() == payload
    assert not final.exists()


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
    first = mark_attempt(attempt.path, "running", reason="attempt_allocated")

    def append_preflight(_index: int):
        try:
            return mark_attempt(
                attempt.path,
                "running",
                event_type="preflight_passed",
                reason="preflight_passed",
            )
        except RuntimeError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(append_preflight, range(24)))

    statuses = [result for result in results if isinstance(result, dict)]
    failures = [result for result in results if isinstance(result, RuntimeError)]
    assert first["sequence"] == 1
    assert len(statuses) == 1 and statuses[0]["sequence"] == 2
    assert len(failures) == 23
    assert all("preflight_passed" in str(failure) for failure in failures)
    events = sorted((attempt.path / "status_events").glob("status_*.json"))
    assert len(events) == 2
    latest = json.loads(events[-1].read_text(encoding="utf-8"))
    assert _read_status(attempt.path) == latest


def test_terminal_status_cannot_be_replaced(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b5__s44")
    mark_attempt(attempt.path, "running", reason="attempt_allocated")
    mark_attempt(attempt.path, "failed", reason="process_failure")

    with pytest.raises(RuntimeError, match="terminal"):
        mark_attempt(attempt.path, "running", reason="preflight_passed")


def test_allocator_reads_canonical_from_immutable_events(tmp_path: Path) -> None:
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b2__s45")
    mark_attempt(attempt.path, "running", reason="attempt_allocated")
    mark_attempt(attempt.path, "running", reason="preflight_passed")
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


@pytest.mark.parametrize(
    ("case", "events"),
    [
        (
            "canonical_first",
            [
                {
                    "state": "canonical",
                    "event_type": "attempt_end",
                    "reason": "validator_accepted",
                    "audit_sha256": "e" * 64,
                }
            ],
        ),
        (
            "preflight_first",
            [
                {
                    "state": "running",
                    "event_type": "preflight_passed",
                    "reason": "preflight_passed",
                }
            ],
        ),
        (
            "duplicated_start",
            [
                {
                    "state": "running",
                    "event_type": "attempt_start",
                    "reason": "attempt_allocated",
                },
                {
                    "state": "running",
                    "event_type": "attempt_start",
                    "reason": "attempt_allocated",
                },
            ],
        ),
        (
            "reordered_start",
            [
                {
                    "state": "running",
                    "event_type": "preflight_passed",
                    "reason": "preflight_passed",
                },
                {
                    "state": "running",
                    "event_type": "attempt_start",
                    "reason": "attempt_allocated",
                },
            ],
        ),
        (
            "duplicated_preflight",
            [
                {
                    "state": "running",
                    "event_type": "attempt_start",
                    "reason": "attempt_allocated",
                },
                {
                    "state": "running",
                    "event_type": "preflight_passed",
                    "reason": "preflight_passed",
                },
                {
                    "state": "running",
                    "event_type": "preflight_passed",
                    "reason": "preflight_passed",
                },
            ],
        ),
        (
            "canonical_before_preflight",
            [
                {
                    "state": "running",
                    "event_type": "attempt_start",
                    "reason": "attempt_allocated",
                },
                {
                    "state": "canonical",
                    "event_type": "attempt_end",
                    "reason": "validator_accepted",
                    "audit_sha256": "e" * 64,
                },
            ],
        ),
        (
            "invalid_before_preflight",
            [
                {
                    "state": "running",
                    "event_type": "attempt_start",
                    "reason": "attempt_allocated",
                },
                {
                    "state": "invalid",
                    "event_type": "attempt_failure",
                    "reason": "validator_rejected",
                },
            ],
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_status_chain_rejects_exact_lifecycle_tampering_even_when_current_matches(
    tmp_path: Path, case: str, events: list[dict[str, object]]
) -> None:
    del case
    attempt = _allocate_bound(tmp_path, "c12_to_c5__b2__s46")
    _write_tampered_status_chain(attempt, events)

    with pytest.raises(RuntimeError, match="lifecycle"):
        controller._read_status_chain(attempt.path, verify_current=True)


def test_prepare_failure_chain_and_objective_rerun_remain_legal(tmp_path: Path) -> None:
    events: list[object] = []
    hooks = _hooks(events)

    def fail_prepare(_attempt) -> None:
        raise ArchiveMismatch("archive mismatch before preflight")

    with pytest.raises(ArchiveMismatch, match="archive mismatch"):
        run_confirmation_attempt(
            tmp_path,
            "c12_to_c5__b5__s46",
            provenance=PROVENANCE,
            hooks=controller.replace(hooks, prepare=fail_prepare),
        )

    run_id = "c12_to_c5__b5__s46"
    first_path = tmp_path / run_id / f"{run_id}__a001"
    chain = controller._read_status_chain(first_path, verify_current=True)
    assert [
        (row["state"], row["event_type"], row["reason"]) for row in chain
    ] == [
        ("running", "attempt_start", "attempt_allocated"),
        ("failed", "attempt_failure", "archive_integrity_failure"),
    ]
    assert allocate_attempt(tmp_path, run_id).attempt_id.endswith("__a002")


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


@pytest.mark.parametrize("bad_status", [None, "collected", "NOT_COLLECTED"])
def test_frozen_loader_rejects_protocol_transport_status_before_external_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_status: str | None,
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    protocol = copy.deepcopy(fixture["protocol"])
    if bad_status is None:
        protocol["schedule"][0].pop("transport_status")
    else:
        protocol["schedule"][0]["transport_status"] = bad_status
    protocol.pop("protocol_manifest_sha256")
    protocol["protocol_manifest_sha256"] = canonical_sha256(protocol)
    _write_json(fixture["protocol_path"], protocol)
    for run_id, original in fixture["commands"].items():
        command = copy.deepcopy(original)
        command["protocol_manifest_sha256"] = protocol["protocol_manifest_sha256"]
        _write_json(
            fixture["command_root"] / run_id / "command_manifest.json", command
        )
    monkeypatch.setattr(
        controller,
        "_wait_for_pi",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid protocol transport status reached an external action"
        ),
    )

    with pytest.raises(ValueError, match="transport"):
        controller.main(_controller_argv(fixture, tmp_path, "--validate-inputs-only"))


def test_frozen_loader_rejects_conflicting_protocol_transport_declaration(
    tmp_path: Path,
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    protocol = copy.deepcopy(fixture["protocol"])
    protocol["transport_status"] = "collected"
    protocol.pop("protocol_manifest_sha256")
    protocol["protocol_manifest_sha256"] = canonical_sha256(protocol)
    _write_json(fixture["protocol_path"], protocol)
    for run_id, original in fixture["commands"].items():
        command = copy.deepcopy(original)
        command["protocol_manifest_sha256"] = protocol["protocol_manifest_sha256"]
        _write_json(
            fixture["command_root"] / run_id / "command_manifest.json", command
        )

    with pytest.raises(ValueError, match="transport"):
        controller.main(_controller_argv(fixture, tmp_path, "--validate-inputs-only"))


@pytest.mark.parametrize("bad_status", [None, "collected", "NOT_COLLECTED"])
def test_frozen_loader_rejects_command_transport_status_before_external_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_status: str | None,
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    first_run = confirmation_run_id(*CONFIRMATION_SCHEDULE[0])
    command = copy.deepcopy(fixture["commands"][first_run])
    if bad_status is None:
        command.pop("transport_status")
    else:
        command["transport_status"] = bad_status
    _write_json(
        fixture["command_root"] / first_run / "command_manifest.json", command
    )
    monkeypatch.setattr(
        controller,
        "_wait_for_pi",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid command transport status reached an external action"
        ),
    )

    with pytest.raises(ValueError, match="transport"):
        controller.main(_controller_argv(fixture, tmp_path, "--validate-inputs-only"))


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


def test_validator_exit_two_with_invalid_report_returns_formal_rejection(
    tmp_path: Path,
) -> None:
    attempt = _allocate_bound(tmp_path / "raw", "c12_to_c5__b5__s42")
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, {})

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        _write_json(output, {"status": "invalid", "reasons": ["missing event"]})
        return subprocess.CompletedProcess(
            command,
            2,
            json.dumps({"audit_sha256": sha256_file(output)}),
            "validator rejected evidence",
        )

    outcome = controller.invoke_validator(
        attempt,
        validator=tmp_path / "validator.py",
        protocol_manifest=protocol_path,
        run=fake_run,
    )

    assert outcome.success is False
    assert outcome.audit_sha256 is None


@pytest.mark.parametrize(
    ("returncode", "status", "match"),
    [
        (1, "valid", "return code"),
        (3, "invalid", "return code"),
        (0, "invalid", "status"),
        (2, "valid", "status"),
    ],
)
def test_validator_rejects_unsupported_or_exit_status_mismatch(
    tmp_path: Path, returncode: int, status: str, match: str
) -> None:
    attempt = _allocate_bound(tmp_path / "raw", "c12_to_c5__b5__s42")
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, {})

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        _write_json(output, {"status": status, "reasons": []})
        return subprocess.CompletedProcess(
            command,
            returncode,
            json.dumps({"audit_sha256": sha256_file(output)}),
            "validator detail",
        )

    with pytest.raises(RuntimeError, match=match):
        controller.invoke_validator(
            attempt,
            validator=tmp_path / "validator.py",
            protocol_manifest=protocol_path,
            run=fake_run,
        )


@pytest.mark.parametrize(
    ("report", "write_report", "match"),
    [
        ({"reasons": []}, True, "status"),
        ({"status": "unknown", "reasons": []}, True, "status"),
        ("not a mapping", True, "report"),
        (b"{not-json", True, "report"),
        (None, False, "regular audit output"),
    ],
)
def test_validator_rejects_malformed_or_missing_report(
    tmp_path: Path, report: object, write_report: bool, match: str
) -> None:
    attempt = _allocate_bound(tmp_path / "raw", "c12_to_c5__b5__s42")
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, {})

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        if write_report:
            if isinstance(report, bytes):
                output.write_bytes(report)
            else:
                _write_json(output, report)
        return subprocess.CompletedProcess(command, 2, "{}", "validator detail")

    with pytest.raises(RuntimeError, match=match):
        controller.invoke_validator(
            attempt,
            validator=tmp_path / "validator.py",
            protocol_manifest=protocol_path,
            run=fake_run,
        )


@pytest.mark.parametrize("stdout", ["{}", '{"audit_sha256":"bad"}', "not json"])
def test_validator_valid_report_requires_matching_stdout_sha(
    tmp_path: Path, stdout: str
) -> None:
    attempt = _allocate_bound(tmp_path / "raw", "c12_to_c5__b5__s42")
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, {})

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        _write_json(output, {"status": "valid", "reasons": []})
        return subprocess.CompletedProcess(command, 0, stdout, "validator detail")

    with pytest.raises(RuntimeError, match="JSON|audit_sha256"):
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


def test_local_content_addressed_launch_disables_bytecode_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    popen_kwargs: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        popen_kwargs.update(kwargs)
        return _FakePopen(command, **kwargs)

    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.setattr(controller.subprocess, "Popen", fake_popen)
    process = controller._local_launch_process(
        label="pc-client",
        command=[sys.executable, "-m", "gaps_flower.client_app"],
        cwd=tmp_path / "runtime" / ("a" * 64) / "src",
        log_root=tmp_path / "attempt" / "client_logs",
    )
    try:
        environment = popen_kwargs.get("env", {})
        assert isinstance(environment, dict)
        assert environment.get("PYTHONDONTWRITEBYTECODE") == "1"
    finally:
        for handle in process.log_handles:
            handle.close()


def test_remote_content_addressed_launch_disables_child_bytecode_writes() -> None:
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

    controller._remote_launch_process(
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

    supervisor = _supervisor_source(captured[0])
    assert "environment['PYTHONDONTWRITEBYTECODE'] = '1'" in supervisor


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
    *,
    validator_status: str = "valid",
    validator_returncode: int = 0,
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
            _write_json(
                output,
                {
                    "status": validator_status,
                    "reasons": (
                        [] if validator_status == "valid" else ["forced rejection"]
                    ),
                },
            )
            digest = sha256_file(output)
            return subprocess.CompletedProcess(
                command,
                validator_returncode,
                json.dumps({"audit_sha256": digest}),
                "",
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

    assert (
        controller.main(
            _controller_argv(fixture, tmp_path, "--validator", str(validator))
        )
        == 0
    )

    server_outer = next(source for source in launches if "REMOTE_LAUNCH_V1:server" in source)
    supervisor = _supervisor_source(server_outer)
    source_hash = fixture["source"]["source_archive_sha256"]
    extracted_src = f"{controller.ECS_REMOTE_RUNTIME_BASE}/{source_hash}/src"
    assert _literal_assignment(supervisor, "cwd") == extracted_src
    assert f"environment['PYTHONPATH'] = {extracted_src!r}" in supervisor
    command = _literal_assignment(supervisor, "command")
    data_index = command.index("--data-root")
    assert command[data_index + 1] == "/root/GAPS/dataset/frozen_dataset"
    assert "/root/GAPS" != _literal_assignment(supervisor, "cwd")
    run_id = confirmation_run_id("B2", 42)
    remote_attempt = (
        f"{controller.ECS_REMOTE_RUNTIME_BASE}/{source_hash}/attempts/{run_id}__a001"
    )
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


def test_formal_validator_rejection_records_invalid_attempt_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    validator = tmp_path / "validator.py"
    validator.write_text("# path is executed by fake _run\n", encoding="utf-8")
    events: list[str] = []
    _install_production_fakes(
        monkeypatch,
        fixture,
        events,
        validator_status="invalid",
        validator_returncode=2,
    )

    assert controller.main(
        _controller_argv(fixture, tmp_path, "--validator", str(validator))
    ) == 0

    run_id = confirmation_run_id("B2", 42)
    attempt_path = tmp_path / "raw" / run_id / f"{run_id}__a001"
    assert _read_status(attempt_path)["state"] == "invalid"
    final_event = json.loads(
        sorted((attempt_path / "status_events").glob("status_*.json"))[-1].read_text(
            encoding="utf-8"
        )
    )
    assert final_event["event_type"] == "attempt_failure"
    assert final_event["reason"] == "validator_rejected"
    assert final_event["audit_sha256"] is None


class _CleanupToken:
    def __init__(self, label: str) -> None:
        self.label = label


class _TrackedCleanupLog:
    def __init__(
        self, label: str, events: list[str], *, close_error: BaseException | None = None
    ) -> None:
        self.label = label
        self.events = events
        self.close_error = close_error
        self.closed = False

    def close(self) -> None:
        self.events.append(f"close:{self.label}")
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def test_production_cleanup_is_best_effort_and_aggregates_every_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _load_frozen(_frozen_input_fixture(tmp_path / "inputs"))
    runtime = controller.ProductionRuntime(
        frozen=frozen,
        frozen_run=frozen.runs[0],
        deployments={},
        ecs_host="root@ecs",
        pi_host="gaps@pi",
        validator=tmp_path / "validator.py",
        poll_seconds=0.01,
        timeout_seconds=1.0,
        pc_runtime_root=tmp_path / "runtime",
    )
    events: list[str] = []
    logs = [
        _TrackedCleanupLog("server", events),
        _TrackedCleanupLog("pi-client", events),
        _TrackedCleanupLog("pi-sampler", events),
        _TrackedCleanupLog(
            "pc-client",
            events,
            close_error=OSError("pc client log close exploded"),
        ),
        _TrackedCleanupLog("pc-sampler", events),
    ]

    def remote_process(
        host_id: str, label: str, ordinal: int, *, valid_identity: bool = True
    ) -> controller.OwnedProcess:
        owner_pid = 5000 + ordinal if valid_identity else None
        return controller.OwnedProcess(
            host_id=host_id,
            label=label,
            pid=6000 + ordinal,
            owner_pid=owner_pid,
            owner_pgid=owner_pid,
            owner_start_ticks=7000 + ordinal if valid_identity else None,
            registration_path=f"/attempt/{label}.registration.json",
            launch_token=f"{ordinal:032x}",
            host="root@ecs" if host_id == "ecs" else "gaps@pi",
            python_bin="/venv/bin/python",
            log_handles=(logs[ordinal],),
        )

    server = remote_process("ecs", "server", 0)
    pi_client = remote_process("pi", "pi-client", 1)
    pi_sampler = remote_process("pi", "pi-sampler", 2, valid_identity=False)
    pc_client_handle = _CleanupToken("pc-client")
    pc_sampler_handle = _CleanupToken("pc-sampler")
    pc_client = controller.OwnedProcess(
        "pc", "pc-client", 8001, handle=pc_client_handle, log_handles=(logs[3],)
    )
    pc_sampler = controller.OwnedProcess(
        "pc", "pc-sampler", 8002, handle=pc_sampler_handle, log_handles=(logs[4],)
    )
    tunnel = _CleanupToken("tunnel")

    def fake_remote_terminate(*, registration, **_kwargs) -> None:
        label = str(registration["label"])
        events.append(f"terminate-remote:{label}")
        if label == "server":
            raise RuntimeError("ECS terminate exploded")

    def fake_terminate_processes(processes) -> None:
        labels = [process.label for process in processes]
        events.extend(f"terminate-local:{label}" for label in labels)
        if labels == ["pc-client"]:
            raise RuntimeError("PC client terminate exploded")
        if labels == ["tunnel"]:
            raise RuntimeError("tunnel terminate exploded")

    monkeypatch.setattr(
        controller, "_terminate_remote_launch_registration", fake_remote_terminate
    )
    monkeypatch.setattr(controller, "_terminate_processes", fake_terminate_processes)

    hooks = controller.build_production_hooks(runtime)
    hooks = controller.replace(
        hooks,
        prepare=lambda _attempt: None,
        launch_server=lambda _attempt: server,
        start_tunnels=lambda _attempt: [tunnel],
        launch_pi_client=lambda _attempt: pi_client,
        launch_pi_sampler=lambda _attempt, _client: pi_sampler,
        launch_pc_client=lambda _attempt: pc_client,
        launch_pc_sampler=lambda _attempt, _client: pc_sampler,
        monitor_server=lambda _attempt, _server: None,
        stop_sampler=lambda _attempt, _sampler: None,
        wait_sampler=lambda _attempt, _sampler: None,
        recover_evidence=lambda _attempt: pytest.fail("cleanup failure recovered evidence"),
        validate_attempt=lambda _attempt: pytest.fail("cleanup failure invoked validator"),
    )

    with pytest.raises(controller.AttemptFailure):
        run_confirmation_attempt(
            tmp_path / "raw",
            frozen.runs[0].run_id,
            provenance=frozen.runs[0].provenance,
            hooks=hooks,
        )

    assert "terminate-remote:server" in events
    assert "terminate-remote:pi-client" in events
    assert "terminate-local:pc-client" in events
    assert "terminate-local:pc-sampler" in events
    assert "terminate-local:tunnel" in events
    assert all(log.closed for log in logs)
    attempt_path = (
        tmp_path
        / "raw"
        / frozen.runs[0].run_id
        / f"{frozen.runs[0].run_id}__a001"
    )
    assert _read_status(attempt_path)["state"] == "failed"
    controller_log = (attempt_path / "controller.log").read_text(encoding="utf-8")
    for detail in (
        "server: RuntimeError: ECS terminate exploded",
        "pi-sampler: RuntimeError: remote process lacks owned identity",
        "pc-client: RuntimeError: PC client terminate exploded",
        "pc-client: OSError: pc client log close exploded",
        "RuntimeError: tunnel terminate exploded",
    ):
        assert detail in controller_log


def test_formal_smoke_config_is_explicit_noncanonical_and_fixed() -> None:
    config = controller.FormalSmokeConfig(
        observer_enabled=False,
        trace_output="/attempt/common_trace.jsonl",
        initial_checkpoint="/attempt/frozen_initial_checkpoint.pth",
    )
    assert config.namespace == "noncanonical_smoke"
    assert config.rounds == 2
    assert config.local_epochs == 1
    assert config.observer_enabled is False
    with pytest.raises(ValueError, match="rounds"):
        dataclass_replace(config, rounds=25)
    with pytest.raises(ValueError, match="namespace"):
        dataclass_replace(config, namespace="canonical")


def test_smoke_commands_are_ephemeral_and_do_not_mutate_frozen_manifest(
    tmp_path: Path,
) -> None:
    fixture = _frozen_input_fixture(tmp_path)
    manifest = next(iter(fixture["commands"].values()))
    original = copy.deepcopy(manifest["commands"])
    config = controller.FormalSmokeConfig(
        observer_enabled=False,
        trace_output="/attempt/common_trace.jsonl",
        initial_checkpoint="/attempt/frozen_initial_checkpoint.pth",
    )

    derived = controller.derive_noncanonical_smoke_commands(manifest, config)

    assert manifest["commands"] == original
    assert manifest["algorithm_config_sha256"] == fixture["protocol"]["schedule"][0][
        "algorithm_config_sha256"
    ]
    assert controller._command_option(derived["server_ecs"], "--rounds") == "2"
    assert controller._command_option(derived["client_c1_pi"], "--local-epochs") == "1"
    assert controller._command_option(derived["client_c2_pc"], "--local-epochs") == "1"
    assert "--observer-context" not in derived["server_ecs"]
    assert "--observer-events" not in derived["server_ecs"]
    assert derived["smoke_identity"]["noncanonical_smoke"] is True
    assert derived["smoke_identity"]["algorithm_config_sha256"] == manifest[
        "algorithm_config_sha256"
    ]


def test_default_production_hook_path_is_unchanged_without_explicit_smoke() -> None:
    signature = inspect.signature(controller.build_production_hooks)
    assert signature.parameters["smoke"].default is None


def test_noncanonical_smoke_lifecycle_never_allocates_or_marks_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    hooks = LifecycleHooks(
        prepare=lambda _attempt: events.append("prepare"),
        launch_server=lambda _attempt: events.append("server") or object(),
        start_tunnels=lambda _attempt: events.append("tunnels") or [],
        launch_pi_client=lambda _attempt: events.append("pi-client") or object(),
        launch_pi_sampler=lambda _attempt, _client: events.append("pi-sampler") or object(),
        launch_pc_client=lambda _attempt: events.append("pc-client") or object(),
        launch_pc_sampler=lambda _attempt, _client: events.append("pc-sampler") or object(),
        monitor_server=lambda _attempt, _server: events.append("monitor"),
        stop_sampler=lambda _attempt, _sampler: events.append("stop-sampler"),
        wait_sampler=lambda _attempt, _sampler: events.append("wait-sampler"),
        recover_evidence=lambda _attempt: events.append("recover"),
        validate_attempt=lambda _attempt: pytest.fail("smoke called 25-round validator"),
        cleanup_owned=lambda _owned: events.append("cleanup-owned") or [],
        cleanup_tunnels=lambda _tunnels: events.append("cleanup-tunnels"),
    )
    monkeypatch.setattr(
        controller,
        "allocate_attempt",
        lambda *_args, **_kwargs: pytest.fail("smoke allocated canonical attempt"),
    )
    monkeypatch.setattr(
        controller,
        "mark_attempt",
        lambda *_args, **_kwargs: pytest.fail("smoke wrote attempt registry"),
    )
    output = tmp_path / "formal-smoke" / "off"

    attempt = controller.run_noncanonical_smoke_attempt(
        output,
        run_id="c12_to_c5__b2__s42",
        mode="off",
        provenance=PROVENANCE,
        hooks=hooks,
    )

    assert attempt.path == output
    assert json.loads((output / "noncanonical_smoke.json").read_text(encoding="utf-8"))[
        "noncanonical_smoke"
    ] is True
    assert not (output / "attempt_status.json").exists()
    assert "recover" in events
    assert events.count("stop-sampler") == 2
    assert events.count("wait-sampler") == 2


def test_noncanonical_smoke_failure_retains_output_and_cleans_all_owned(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    hooks = LifecycleHooks(
        prepare=lambda _attempt: events.append("prepare"),
        launch_server=lambda _attempt: events.append("server") or object(),
        start_tunnels=lambda _attempt: [object()],
        launch_pi_client=lambda _attempt: (_ for _ in ()).throw(RuntimeError("pi failed")),
        launch_pi_sampler=lambda *_args: pytest.fail("unexpected sampler"),
        launch_pc_client=lambda *_args: pytest.fail("unexpected PC"),
        launch_pc_sampler=lambda *_args: pytest.fail("unexpected PC sampler"),
        monitor_server=lambda *_args: pytest.fail("unexpected monitor"),
        stop_sampler=lambda *_args: None,
        wait_sampler=lambda *_args: None,
        recover_evidence=lambda *_args: pytest.fail("failed smoke recovered"),
        validate_attempt=lambda *_args: pytest.fail("failed smoke validated"),
        cleanup_owned=lambda owned: events.append(f"cleanup-owned:{len(owned)}") or [],
        cleanup_tunnels=lambda tunnels: events.append(f"cleanup-tunnels:{len(tunnels)}"),
    )
    output = tmp_path / "failed-smoke"
    with pytest.raises(RuntimeError, match="pi failed"):
        controller.run_noncanonical_smoke_attempt(
            output,
            run_id="c12_to_c5__b2__s42",
            mode="on",
            provenance=PROVENANCE,
            hooks=hooks,
        )
    assert output.is_dir()
    assert "cleanup-owned:1" in events
    assert "cleanup-tunnels:1" in events
