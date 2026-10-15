"""Run DA80 -> DA50 -> DA30 sequentially with fail-closed gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results/iotj_b5_server_da_budget_ablation_20260731"
RUN_ID = "c12_to_c5__b5__s42"
ATTEMPT_ID = f"{RUN_ID}__a001"
LEVELS = (80, 50, 30)

DATA_ROOT = (
    REPO_ROOT
    / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
)
SOURCE_ARCHIVE = REPO_ROOT / "results/c2e/source/confirmation_source.tar"
SOURCE_ARCHIVE_MANIFEST = (
    REPO_ROOT / "results/c2e_summary/source_archive_manifest.json"
)
DATASET_MANIFEST = REPO_ROOT / "results/c2e_summary/dataset_manifest.json"
C2_SUBSET_MANIFEST = (
    REPO_ROOT
    / "results/c2e_ecs_c2_topology/c2_dataset_subset_manifest.json"
)
V4_ROOT = REPO_ROOT / "results/iotj_b5_c5_deployment_p1_20260722"
RUNTIME_CONTRACT = (
    V4_ROOT / "c5_h8_runtime_contract_b5_v4/runtime_contract.json"
)
ROW_MAP = V4_ROOT / "c5_h8_runtime_contract_b5_v4/row_map_1360.json"
BUNDLE_MANIFEST = V4_ROOT / "bundle_candidate/manifest.json"
HC95 = V4_ROOT / "high_coverage_qc/test_hc95_records.csv"
HC90 = V4_ROOT / "high_coverage_qc/test_hc90_records.csv"

FROZEN_SHA256 = {
    RUNTIME_CONTRACT: "54a42bb9f622c441a889a36fb1e585cb437e04c11128eb0578cfef6fd7711c3c",
    BUNDLE_MANIFEST: "a2514bd74ba0a98334d146af218922ee84884a53b93b0d4c44414723abee73b5",
    HC95: "33d04439376852bb976d9a4ed5f09235107b296c5f839c75ed667fdecc598860",
    HC90: "6051e7787915e0163ffd815dc089626e751906474c858072c5c0520c615dccb3",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_event(event: str, **details: Any) -> None:
    path = RESULT_ROOT / "sequence/sequence_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    status_path = RESULT_ROOT / "sequence/sequence_status.json"
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(status_path)


def _verify_frozen_assets() -> None:
    mismatches = {}
    for path, expected in FROZEN_SHA256.items():
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[str(path)] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(f"FAIL_CLOSED frozen asset mismatch: {mismatches}")


def _run(command: list[str], log_name: str) -> subprocess.CompletedProcess[str]:
    log_root = RESULT_ROOT / "sequence"
    log_root.mkdir(parents=True, exist_ok=True)
    _append_event("subprocess_start", log_name=log_name, command=command)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (log_root / f"{log_name}.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (log_root / f"{log_name}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(
            f"FAIL_CLOSED {log_name} exited {completed.returncode}"
        )
    _append_event("subprocess_pass", log_name=log_name)
    return completed


def _controller_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        payload = json.loads(line)
        if (
            isinstance(payload, dict)
            and payload.get("status") == "preflighted"
            and payload.get("queue") == [RUN_ID]
        ):
            return payload
    raise RuntimeError("FAIL_CLOSED controller preflight receipt is missing")


def _level_dir(steps: int) -> Path:
    return RESULT_ROOT / f"da{steps}"


def _raw_root(steps: int) -> Path:
    return _level_dir(steps) / "raw"


def _attempt_dir(steps: int) -> Path:
    return _raw_root(steps) / RUN_ID / ATTEMPT_ID


def _wait_for_canonical(steps: int, timeout_seconds: int) -> Path:
    attempt = _attempt_dir(steps)
    status_path = attempt / "attempt_status.json"
    deadline = time.monotonic() + timeout_seconds
    last_sequence: int | None = None
    while time.monotonic() < deadline:
        if status_path.is_file():
            status = _json(status_path)
            sequence = int(status["sequence"])
            if sequence != last_sequence:
                _append_event(
                    "attempt_status",
                    da_steps=steps,
                    state=status["state"],
                    reason=status["reason"],
                    sequence=sequence,
                )
                last_sequence = sequence
            if status["state"] == "canonical":
                if status["reason"] != "validator_accepted":
                    raise RuntimeError(
                        "FAIL_CLOSED canonical reason is not validator_accepted"
                    )
                return attempt
            if status["state"] == "failed":
                raise RuntimeError(
                    f"FAIL_CLOSED DA{steps}: {status['reason_category']}/"
                    f"{status['reason']}"
                )
        time.sleep(30)
    raise TimeoutError(f"FAIL_CLOSED timed out waiting for DA{steps}")


def _controller_command(steps: int, preflight_only: bool) -> list[str]:
    protocol_root = _level_dir(steps) / "protocol_inputs"
    command = [
        sys.executable,
        "-m",
        "scripts.run_iotj_confirmation_observability",
        "--protocol-manifest",
        str(protocol_root / "confirmation_protocol_manifest.json"),
        "--source-archive-manifest",
        str(SOURCE_ARCHIVE_MANIFEST),
        "--dataset-manifest",
        str(DATASET_MANIFEST),
        "--command-root",
        str(protocol_root / "commands"),
        "--source-archive",
        str(SOURCE_ARCHIVE),
        "--raw-root",
        str(_raw_root(steps)),
        "--runs",
        "B5:42",
        "--ecs-host",
        "root@121.40.139.213",
        "--pi-hosts",
        "gaps@192.168.137.172",
        "--wait-for-pi-minutes",
        "5",
        "--pi-retry-seconds",
        "10",
        "--c2-host",
        "root@114.55.171.63",
        "--c2-python",
        "/root/gaps_c2_cpu_env/bin/python",
        "--c2-data-root",
        (
            "/root/GAPS/confirmation_c2_data/"
            "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
        ),
        "--c2-dataset-subset-manifest",
        str(C2_SUBSET_MANIFEST),
        "--execution-topology-manifest",
        str(protocol_root / "execution_topology_manifest.json"),
        "--run-timeout-seconds",
        "172800",
        "--poll-seconds",
        "30",
    ]
    if preflight_only:
        command.append("--preflight-only")
    return command


def _write_preflight(
    steps: int, controller_result: dict[str, Any]
) -> None:
    level_dir = _level_dir(steps)
    protocol_root = level_dir / "protocol_inputs"
    command_path = (
        protocol_root / f"commands/{RUN_ID}/command_manifest.json"
    )
    payload = {
        "schema_version": "iotj.b5_server_da_budget_preflight.v1",
        "experiment_id": f"IOTJ-B5-LE1-DA{steps}-S42-20260731",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "local_epochs": 1,
        "server_da_steps_per_round": steps,
        "server_da_total_steps": 25 * steps,
        "checks": {
            "three_host_controller_preflight": "PASS",
            "residual_flower_processes": "none",
            "formal_ports": "free",
            "source_archive_sha256": _sha256(SOURCE_ARCHIVE),
            "runtime_v4_frozen_assets": "unchanged",
            "test_used_for_training_selection_or_stopping": False,
        },
        "command_manifest": {
            "path": str(command_path.relative_to(REPO_ROOT)),
            "sha256": _sha256(command_path),
            "algorithm_config_sha256": _json(command_path)[
                "algorithm_config_sha256"
            ],
        },
        "controller_result": controller_result,
        "decision": "PASS",
        "formal_training_authorized": True,
    }
    _write_json_exclusive(level_dir / f"da{steps}_preflight.json", payload)


def _postprocess(steps: int, next_authorized: bool) -> None:
    level_dir = _level_dir(steps)
    attempt = _attempt_dir(steps)
    checkpoint = (
        attempt / "raw/ecs/training/server_round_025_adapted.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FAIL_CLOSED missing checkpoint: {checkpoint}")
    evaluation_dir = level_dir / "classification_evaluation"
    _run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_iotj_b5_multiseed_seed",
            "--seed",
            "42",
            "--checkpoint",
            str(checkpoint),
            "--data-root",
            str(DATA_ROOT),
            "--row-map",
            str(ROW_MAP),
            "--runtime-contract",
            str(RUNTIME_CONTRACT),
            "--output-dir",
            str(evaluation_dir),
            "--run-id",
            RUN_ID,
            "--attempt-id",
            ATTEMPT_ID,
            "--output-prefix",
            f"da{steps}",
            "--device",
            "cpu",
            "--batch-size",
            "64",
        ],
        f"da{steps}_classification_evaluation",
    )
    audit_command = [
        sys.executable,
        "-m",
        "scripts.audit_iotj_b5_multiseed_seed",
        "--seed",
        "42",
        "--seed-dir",
        str(level_dir),
        "--raw-root",
        str(_raw_root(steps)),
        "--attempt-id",
        ATTEMPT_ID,
        "--local-epochs",
        "1",
        "--server-da-steps-per-round",
        str(steps),
        "--output-prefix",
        f"da{steps}",
        "--evidence-boundary",
        "post-freeze single-seed server-DA compute-budget sensitivity only",
        "--runtime-contract",
        str(RUNTIME_CONTRACT),
        "--bundle-manifest",
        str(BUNDLE_MANIFEST),
        "--hc95",
        str(HC95),
        "--hc90",
        str(HC90),
    ]
    audit_command.append(
        "--next-experiment-authorized"
        if next_authorized
        else "--no-next-experiment-authorized"
    )
    _run(audit_command, f"da{steps}_postflight_audit")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds-per-run", type=int, default=172800)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    events_path = RESULT_ROOT / "sequence/sequence_events.jsonl"
    if events_path.exists() and not args.resume:
        raise FileExistsError("REFUSE_TO_OVERWRITE existing sequence state")
    _append_event(
        "sequence_resumed" if args.resume else "sequence_started",
        levels=list(LEVELS),
    )
    try:
        for index, steps in enumerate(LEVELS):
            postflight_path = _level_dir(steps) / f"da{steps}_postflight.json"
            if postflight_path.is_file():
                postflight = _json(postflight_path)
                if postflight.get("verdict") != "PASS":
                    raise RuntimeError(
                        f"FAIL_CLOSED existing DA{steps} postflight is not PASS"
                    )
                _append_event(
                    "level_reused", da_steps=steps, verdict="PASS"
                )
                continue
            _verify_frozen_assets()
            preflight = _run(
                _controller_command(steps, preflight_only=True),
                f"da{steps}_controller_preflight",
            )
            _write_preflight(steps, _controller_result(preflight.stdout))
            _run(
                _controller_command(steps, preflight_only=False),
                f"da{steps}_controller_formal",
            )
            _wait_for_canonical(steps, args.timeout_seconds_per_run)
            _postprocess(
                steps,
                next_authorized=index < len(LEVELS) - 1,
            )
            _verify_frozen_assets()
            _append_event("level_complete", da_steps=steps, verdict="PASS")
        _append_event("sequence_complete", verdict="PASS")
    except BaseException as exc:
        _append_event(
            "sequence_failed",
            verdict="FAIL_CLOSED",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
