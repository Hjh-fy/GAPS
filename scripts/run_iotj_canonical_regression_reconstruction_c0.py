"""Run frozen C0 adaptation-timing validation for canonical-v1.

The source Flower stage accepts C1/C2 only.  C3/C4/C5 calibration is opened
only after the common round25 source checkpoint and complete A4 context have
been locked.  Target test is opened only after all three step100 endpoints are
complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_dataset import GasSensorWindowDataset  # noqa: E402
from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays  # noqa: E402
from gaps_flower.evaluate_checkpoint import load_checkpoint_model  # noqa: E402
from gaps_flower.final_adaptation_context import load_final_adaptation_context  # noqa: E402
from gaps_flower.posthoc_commissioning import (  # noqa: E402
    A4_C0_A_CONTEXT_AVAILABILITY,
    BATCH_SIZE,
    SEED,
    STEPS,
    a4_final_adapt,
    ordered_state_fingerprint,
    sha256_file,
)
from scripts import run_iotj_final_classification_le1 as frozen  # noqa: E402
from scripts.summarize_iotj_classification_ablation import (  # noqa: E402
    evaluate_checkpoint_stream,
)


STUDY_ID = "CAN-V1-CRRQ-20260811"
SOURCE_EXECUTION_ID = "source_fl"
SOURCE_FORMAL_ID = "CAN-V1-CRRQ-C0-B-SOURCE"
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
REMOTE_DATA_ROOT = "/root/GAPS/dataset/iotj_canonical_v1"
PI_DATA_ROOT = "/home/gaps/GAPS/flower_runtime/dataset/iotj_canonical_v1"
C2_DATA_ROOT = "/root/GAPS/confirmation_c2_data/iotj_canonical_v1"
OUTPUT_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811"
C0_ROOT = OUTPUT_ROOT / "C0"
SOURCE_RUN = C0_ROOT / SOURCE_EXECUTION_ID
DESIGN_MANIFEST = ROOT / "docs/experiments/iotj_canonical_v1_final/canonical_regression_reconstruction_qc_20260811/protocol_manifest.json"
INTERLEAVED_ROOT = ROOT / "results/iotj_canonical_v1_final_20260808/classification"
INTERLEAVED_METRICS = ROOT / "results/iotj_canonical_v1_final_20260808/classification_evaluation/classification_metrics.csv"
TARGETS = ("C3", "C4", "C5")
MARGIN = 0.005
_FROZEN_BUILDER = frozen.build_flower_commands


def _set_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def build_c0_source_commands() -> dict[str, Any]:
    """Derive the A4 source-only trajectory without any target path."""

    commands = _FROZEN_BUILDER("FCL-E4-A1")
    replacements = (
        ("FCL-E4-A1", SOURCE_EXECUTION_ID),
        (frozen.REMOTE_DATA_ROOT, REMOTE_DATA_ROOT),
        (frozen.PI_DATA_ROOT, PI_DATA_ROOT),
        (frozen.C2_DATA_ROOT, C2_DATA_ROOT),
    )
    for role in ("server", "client_c1", "client_c2"):
        values = list(commands[role])
        for old, new in replacements:
            values = [value.replace(old, new) for value in values]
        commands[role] = values
        _set_option(commands[role], "--profile", "ce_stats")
    server = commands["server"]
    _set_option(server, "--ablation-variant", "A4")
    _set_option(server, "--target-information-method", "a4")
    _set_option(server, "--use-selective-agg", "false")
    _set_option(server, "--require-selective-after-warmup", "false")
    _set_option(server, "--use-proto-mmd", "true")
    _set_option(server, "--use-domain-adapt", "false")
    server.extend(["--final-adaptation-context-round", "25"])
    commands["protocol"].update(
        {
            "experiment_id": SOURCE_FORMAL_ID,
            "dataset": "iotj_canonical_v1",
            "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
            "source_clients": ["C1", "C2"],
            "target": None,
            "target_information_in_source_api": False,
            "target_x": False,
            "target_class": False,
            "target_phase": False,
            "target_concentration": False,
            "target_test": False,
            "method": "A4_SOURCE_SIDE_WITH_FINAL_CONTEXT_CAPTURE",
            "optimizer": "Adam",
            "optimizer_lr": 5e-4,
            "checkpoint_selection": "fixed_round_25",
            "hyperparameter_search": False,
        }
    )
    return commands


def protocol_hash() -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(build_c0_source_commands(), sort_keys=True).encode("utf-8"))
    for path in (DESIGN_MANIFEST,):
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def decide_c0(
    final_macro_f1: dict[str, float], interleaved_macro_f1: dict[str, float]
) -> dict[str, Any]:
    if set(final_macro_f1) != set(TARGETS) or set(interleaved_macro_f1) != set(TARGETS):
        raise ValueError("C0 decision requires exactly C3/C4/C5")
    targets: dict[str, Any] = {}
    for target in TARGETS:
        delta = float(final_macro_f1[target]) - float(interleaved_macro_f1[target])
        targets[target] = {
            "final_macro_f1": float(final_macro_f1[target]),
            "interleaved_macro_f1": float(interleaved_macro_f1[target]),
            "delta": delta,
            "margin": -MARGIN,
            "pass": bool(delta >= -MARGIN),
        }
    all_pass = all(item["pass"] for item in targets.values())
    return {
        "decision": "V1_FINAL_ADAPT_SUPPORTED" if all_pass else "V1_INTERLEAVED_RETAINED",
        "all_targets_pass": all_pass,
        "targets": targets,
        "rescue_search_performed": False,
    }


def verify_final_adaptation_endpoints(c0_root: Path = C0_ROOT) -> dict[str, dict[str, Any]]:
    endpoints: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        marker = Path(c0_root) / f"final_adapt_{target}" / "fixed_endpoint_complete.json"
        if not marker.is_file():
            raise RuntimeError(f"FAIL_CLOSED missing fixed endpoint for {target}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if int(payload.get("step", -1)) != STEPS:
            raise RuntimeError(f"FAIL_CLOSED {target} is not fixed step100")
        if payload.get("target_test_opened") is not False:
            raise RuntimeError(f"FAIL_CLOSED target test opened before common gate: {target}")
        if payload.get("target") != target:
            raise RuntimeError(f"FAIL_CLOSED endpoint target identity mismatch: {target}")
        endpoints[target] = payload
    source_fingerprints = {item.get("source_state_fingerprint") for item in endpoints.values()}
    source_fingerprints.discard(None)
    if len(source_fingerprints) > 1:
        raise RuntimeError("FAIL_CLOSED target branches did not reload one source state")
    return endpoints


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def target_from_metric_row(row: dict[str, str]) -> str:
    """Resolve only a per-target row from the formal classification schema."""

    for field in ("scope", "target", "Target"):
        value = str(row.get(field, "")).upper()
        if value in TARGETS:
            return value
    return ""


def validate_remote_source_hashes(
    expected: dict[str, str], observed: dict[str, str], *, host: str
) -> dict[str, Any]:
    for path, expected_sha in expected.items():
        observed_sha = observed.get(path)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"FAIL_CLOSED {host} source hash mismatch for {path}: "
                f"expected={expected_sha} observed={observed_sha}"
            )
    extras = sorted(set(observed) - set(expected))
    return {
        "status": "PASS",
        "host": host,
        "hashes": {path: observed[path] for path in sorted(expected)},
        "unregistered_rows": extras,
    }


def _remote_sha256(host: str, root: str, paths: list[str]) -> dict[str, str]:
    remote = f"cd {root} && sha256sum " + " ".join(paths)
    completed = subprocess.run(
        [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            host,
            remote,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        digest, path = line.strip().split(maxsplit=1)
        observed[path.lstrip("*")] = digest.lower()
    return observed


def audit_remote_source_datasets(
    *, pi_host: str, c2_host: str, design: dict[str, Any]
) -> dict[str, Any]:
    common = {
        "dataset_sha256.json": sha256_file(DATA_ROOT / "dataset_sha256.json"),
        "canonical_preprocessing_manifest.json": sha256_file(
            DATA_ROOT / "canonical_preprocessing_manifest.json"
        ),
    }
    expected_pi = {
        **common,
        "client_1/train_experiment_info.json": design["manifest_sha256"]["C1_train"],
        "client_1/calibration_experiment_info.json": design["manifest_sha256"]["C1_calibration"],
        "client_1/test_experiment_info.json": design["manifest_sha256"]["C1_test"],
    }
    expected_c2 = {
        **common,
        "client_2/train_experiment_info.json": design["manifest_sha256"]["C2_train"],
        "client_2/calibration_experiment_info.json": design["manifest_sha256"]["C2_calibration"],
        "client_2/test_experiment_info.json": design["manifest_sha256"]["C2_test"],
    }
    return {
        "pi_c1": validate_remote_source_hashes(
            expected_pi,
            _remote_sha256(pi_host, PI_DATA_ROOT, list(expected_pi)),
            host=pi_host,
        ),
        "c2": validate_remote_source_hashes(
            expected_c2,
            _remote_sha256(c2_host, C2_DATA_ROOT, list(expected_c2)),
            host=c2_host,
        ),
    }


def _interleaved_reference() -> tuple[dict[str, float], list[dict[str, Any]]]:
    manifest = json.loads(DESIGN_MANIFEST.read_text(encoding="utf-8"))
    frozen_reference = manifest["C0"]["interleaved_reuse"]
    by_target: dict[str, dict[str, str]] = {}
    with INTERLEAVED_METRICS.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            target = target_from_metric_row(row)
            if target in TARGETS:
                by_target[target] = row
    metrics: dict[str, float] = {}
    audit_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        run = INTERLEAVED_ROOT / f"CANONICAL-V1-A4-{target}"
        run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        checkpoint = run / "remote_server/server_latest_adapted.pth"
        expected_sha = frozen_reference[target]["checkpoint_sha256"]
        if sha256_file(checkpoint) != expected_sha or run_manifest["checkpoint_sha256"] != expected_sha:
            raise RuntimeError(f"FAIL_CLOSED {target} interleaved checkpoint hash mismatch")
        observed_metric = float(by_target[target]["macro_f1"])
        expected_metric = float(frozen_reference[target]["macro_f1"])
        if abs(observed_metric - expected_metric) > 1e-15:
            raise RuntimeError(f"FAIL_CLOSED {target} interleaved metric mismatch")
        loss_path = run / "remote_server/domain_adapt_round_025.json"
        loss_payload = json.loads(loss_path.read_text(encoding="utf-8"))
        activity = {row["loss_name"]: row for row in loss_payload["loss_activity"]}
        observed_context = {
            "semantic_prototypes": bool(activity["proto_anchor"]["input_available"]),
            "client_prototypes": bool(activity["proto_loss"]["input_available"]),
            "two_client_prototypes": bool(activity["proto_mmd"]["input_available"]),
            "client_residuals": bool(activity["device_residual"]["input_available"]),
        }
        if observed_context != A4_C0_A_CONTEXT_AVAILABILITY:
            raise RuntimeError(f"FAIL_CLOSED {target} C0-A context availability changed")
        metrics[target] = observed_metric
        audit_rows.append(
            {
                "target": target,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": expected_sha,
                "macro_f1": observed_metric,
                "loss_activity_sha256": sha256_file(loss_path),
                **{f"availability_{key}": value for key, value in observed_context.items()},
            }
        )
    return metrics, audit_rows


def preflight(
    output_root: Path = OUTPUT_ROOT,
    *,
    remote_source_audit: dict[str, Any],
) -> dict[str, Any]:
    if not DATA_ROOT.is_dir():
        raise RuntimeError(f"FAIL_CLOSED canonical-v1 dataset missing: {DATA_ROOT}")
    design = json.loads(DESIGN_MANIFEST.read_text(encoding="utf-8"))
    if design["canonical_freeze"]["window_shape"] != [50, 8]:
        raise RuntimeError("FAIL_CLOSED canonical window shape is not 50x8")
    observed_manifest_hashes: dict[str, str] = {}
    for client in (1, 2, 3, 4, 5):
        for role in ("train", "calibration", "test"):
            key = f"C{client}_{role}"
            if key not in design["manifest_sha256"]:
                continue
            path = DATA_ROOT / f"client_{client}/{role}_experiment_info.json"
            observed = sha256_file(path)
            if observed != design["manifest_sha256"][key]:
                raise RuntimeError(f"FAIL_CLOSED canonical manifest hash mismatch: {key}")
            observed_manifest_hashes[key] = observed
    _metrics, reuse_rows = _interleaved_reference()
    commands = build_c0_source_commands()
    serialized = json.dumps(commands, sort_keys=True)
    forbidden = ("client_3", "client_4", "client_5", "--server-calib-data", "--server-val-data")
    if any(value in serialized for value in forbidden):
        raise RuntimeError("FAIL_CLOSED target path entered C0 source commands")
    if commands["client_c1"][commands["client_c1"].index("--data-root") + 1] != PI_DATA_ROOT:
        raise RuntimeError("FAIL_CLOSED C1 source command is not canonical-v1")
    if commands["client_c2"][commands["client_c2"].index("--data-root") + 1] != C2_DATA_ROOT:
        raise RuntimeError("FAIL_CLOSED C2 source command is not canonical-v1")
    payload = {
        "schema_version": "iotj.canonical_v1.crrq.c0.preflight.v1",
        "status": "PASS",
        "study_id": STUDY_ID,
        "git_head": _git_head(),
        "protocol_hash": protocol_hash(),
        "formal_execution_started": False,
        "dataset_aggregate_sha256": design["canonical_freeze"]["dataset_aggregate_sha256"],
        "manifest_sha256": observed_manifest_hashes,
        "source_commands": commands,
        "source_target_access": False,
        "remote_source_dataset_audit": remote_source_audit,
        "interleaved_reuse": reuse_rows,
        "c0a_context_availability": A4_C0_A_CONTEXT_AVAILABILITY,
        "target_test_opened": False,
        "hyperparameter_search": False,
    }
    return _write_json(Path(output_root) / "C0/C0_PRE_EXECUTION_AUDIT.json", payload) and payload


def execute_source_fl(
    *,
    ecs_host: str,
    pi_host: str,
    c2_host: str,
    timeout_hours: float,
) -> None:
    preflight_path = C0_ROOT / "C0_PRE_EXECUTION_AUDIT.json"
    if not preflight_path.is_file():
        raise RuntimeError("FAIL_CLOSED C0 preflight has not passed")
    if (SOURCE_RUN / "fixed_endpoint_complete.json").is_file():
        return
    if SOURCE_RUN.exists():
        raise FileExistsError(f"FAIL_CLOSED partial source endpoint exists: {SOURCE_RUN}")
    original_root = frozen.RESULT_ROOT
    original_builder = frozen.build_flower_commands
    try:
        frozen.RESULT_ROOT = C0_ROOT
        frozen.build_flower_commands = lambda _experiment_id: build_c0_source_commands()
        frozen.execute_full_fl(
            SOURCE_EXECUTION_ID,
            protocol_hash=protocol_hash(),
            lock_payload={"freeze_commit": _git_head(), "study_id": STUDY_ID, "gate": "C0"},
            ecs_host=ecs_host,
            pi_host=pi_host,
            c2_host=c2_host,
            timeout_hours=timeout_hours,
        )
    finally:
        frozen.RESULT_ROOT = original_root
        frozen.build_flower_commands = original_builder
    required = (
        SOURCE_RUN / "remote_server/server_round_025.pth",
        SOURCE_RUN / "remote_server/final_adaptation_context_round_025.json",
        SOURCE_RUN / "fixed_endpoint_complete.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("FAIL_CLOSED source endpoint lacks round25 context: " + ",".join(missing))


def _da_loader(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> DataLoader:
    features, classes, phases = arrays
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=np.zeros((len(features), 4), dtype=np.float32),
        classification_labels=classes,
        phase_labels=phases,
        normalize=False,
        mean_std=None,
    )
    count = min(len(dataset), 500)
    indices = np.random.RandomState(SEED).choice(len(dataset), size=count, replace=False)
    return DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


def _save_adapted_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    target: str,
    source_sha: str,
    source_state_fingerprint: str,
    context_sha: str,
    diagnostics: dict[str, Any],
) -> None:
    torch.save(
        {
            "step": STEPS,
            "model_state": model.state_dict(),
            "method": "V1_FINAL_ADAPT_A4",
            "target": target,
            "source_checkpoint_sha256": source_sha,
            "source_state_fingerprint": source_state_fingerprint,
            "final_adaptation_context_sha256": context_sha,
            "seed": SEED,
            "diagnostics": diagnostics,
        },
        path,
    )


def adapt_targets(device: torch.device) -> None:
    source_checkpoint = SOURCE_RUN / "remote_server/server_round_025.pth"
    context_path = SOURCE_RUN / "remote_server/final_adaptation_context_round_025.json"
    if not source_checkpoint.is_file() or not context_path.is_file():
        raise RuntimeError("FAIL_CLOSED common round25 source endpoint is unavailable")
    source_container = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_fingerprint = ordered_state_fingerprint(source_container["model_state"])
    source_sha = sha256_file(source_checkpoint)
    context_sha = sha256_file(context_path)
    source_arrays = load_domain_adaptation_arrays(
        [DATA_ROOT / "client_1", DATA_ROOT / "client_2"],
        strict=True,
        expected_window_shape=(50, 8),
    )
    for target in TARGETS:
        directory = C0_ROOT / f"final_adapt_{target}"
        marker = directory / "fixed_endpoint_complete.json"
        if marker.is_file():
            continue
        if directory.exists():
            raise FileExistsError(f"FAIL_CLOSED partial target endpoint exists: {directory}")
        directory.mkdir(parents=True)
        target_id = int(target[1:])
        # Each branch independently reloads both source model and context.
        source_model, _config, reloaded_container = load_checkpoint_model(
            str(source_checkpoint), device, BATCH_SIZE
        )
        reloaded_fingerprint = ordered_state_fingerprint(reloaded_container["model_state"])
        if reloaded_fingerprint != source_fingerprint:
            raise RuntimeError(f"FAIL_CLOSED independent source reload mismatch: {target}")
        context = load_final_adaptation_context(context_path, source_checkpoint)
        target_arrays = load_domain_adaptation_arrays(
            [DATA_ROOT / f"client_{target_id}"],
            strict=True,
            expected_window_shape=(50, 8),
        )
        source_loader = _da_loader(source_arrays)
        target_loader = _da_loader(target_arrays)
        adapted, step_rows, summary = a4_final_adapt(
            source_model,
            source_loader,
            target_loader,
            context=context,
            device=device,
        )
        checkpoint = directory / f"final_adapt_a4_{target}.pth"
        _save_adapted_checkpoint(
            checkpoint,
            adapted,
            target=target,
            source_sha=source_sha,
            source_state_fingerprint=source_fingerprint,
            context_sha=context_sha,
            diagnostics=summary,
        )
        _write_csv(directory / "adaptation_step_diagnostics.csv", step_rows)
        _write_json(directory / "system_metrics.json", summary)
        _write_json(
            directory / "run_manifest.json",
            {
                "schema_version": "iotj.canonical_v1.crrq.c0.final_adapt.v1",
                "experiment_id": f"CAN-V1-CRRQ-C0-B-{target}",
                "target": target,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_checkpoint": str(source_checkpoint.resolve()),
                "source_checkpoint_sha256": source_sha,
                "source_state_fingerprint": source_fingerprint,
                "final_adaptation_context": str(context_path.resolve()),
                "final_adaptation_context_sha256": context_sha,
                "calibration_manifest_sha256": sha256_file(
                    DATA_ROOT / f"client_{target_id}/calibration_experiment_info.json"
                ),
                "steps": STEPS,
                "optimizer": "Adam",
                "optimizer_lr": 5e-4,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
                "target_test_opened": False,
                "checkpoint_selection": "fixed_step_100",
                "hyperparameter_search": False,
            },
        )
        _write_json(
            marker,
            {
                "status": "COMPLETE",
                "target": target,
                "step": STEPS,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_state_fingerprint": source_fingerprint,
                "target_test_opened": False,
            },
        )


def evaluate_and_decide(device: torch.device) -> dict[str, Any]:
    endpoints = verify_final_adaptation_endpoints(C0_ROOT)
    opened = {
        target: sha256_file(DATA_ROOT / f"client_{int(target[1:])}/test_experiment_info.json")
        for target in TARGETS
    }
    _write_json(
        C0_ROOT / "SEALED_TEST_OPEN.json",
        {
            "status": "OPENED_AFTER_ALL_THREE_STEP100_ENDPOINTS_LOCKED",
            "opened_at_unix": time.time(),
            "test_manifest_sha256": opened,
            "selection_performed": False,
        },
    )
    interleaved, _reuse_rows = _interleaved_reference()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    final_f1: dict[str, float] = {}
    for target in TARGETS:
        checkpoint = Path(endpoints[target]["checkpoint"])
        rows, metrics = evaluate_checkpoint_stream(
            checkpoint,
            data_root=DATA_ROOT,
            target_client=int(target[1:]),
            split="test",
            device=device,
            batch_size=BATCH_SIZE,
            expected_endpoint=("step", STEPS),
        )
        final_f1[target] = float(metrics["macro_f1"])
        metric_rows.append(
            {
                "target": target,
                "lifecycle": "V1_FINAL_ADAPT",
                "N": metrics["N"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "ece": metrics["ece"],
                "checkpoint_sha256": sha256_file(checkpoint),
                "target_test_manifest_sha256": opened[target],
                "target_adaptation_steps": STEPS,
            }
        )
        prediction_rows.extend({"target": target, **row} for row in rows)
        system = json.loads(
            (C0_ROOT / f"final_adapt_{target}/system_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for row in system["loss_activity"]:
            loss_rows.append({"target": target, "lifecycle": "V1_FINAL_ADAPT", **row})
    for target in TARGETS:
        metric_rows.append(
            {
                "target": target,
                "lifecycle": "V1_INTERLEAVED_REUSE",
                "N": "",
                "accuracy": "",
                "macro_f1": interleaved[target],
                "nll": "",
                "ece": "",
                "checkpoint_sha256": json.loads(DESIGN_MANIFEST.read_text(encoding="utf-8"))["C0"]["interleaved_reuse"][target]["checkpoint_sha256"],
                "target_test_manifest_sha256": opened[target],
                "target_adaptation_steps": 2500,
            }
        )
    decision = decide_c0(final_f1, interleaved)
    _write_csv(C0_ROOT / "classification_v1_timing_comparison.csv", metric_rows)
    _write_csv(C0_ROOT / "classification_predictions.csv", prediction_rows)
    _write_csv(C0_ROOT / "classification_loss_activity.csv", loss_rows)
    _write_json(C0_ROOT / "C0_DECISION.json", decision)
    report = [
        "# Classification V1 Final Adaptation Report",
        "",
        f"- Decision: `{decision['decision']}`.",
        "- C0 isolates lifecycle timing: 25x100 interleaved target steps versus one final 100-step invocation.",
        "- Optimizer, A4 losses, coefficients, source batch convention, calibration identities, seed and fixed endpoint were not searched.",
        "- The C0-A device-residual input was baseline-unavailable and remained unavailable; prototype and semantic inputs retained parity.",
        "",
        "| Target | Final Macro-F1 | Interleaved Macro-F1 | Delta | Pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for target in TARGETS:
        item = decision["targets"][target]
        report.append(
            f"| {target} | {item['final_macro_f1']:.9f} | {item['interleaved_macro_f1']:.9f} | {item['delta']:+.9f} | {item['pass']} |"
        )
    (C0_ROOT / "CLASSIFICATION_V1_FINAL_ADAPT_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return decision


def write_hash_index() -> dict[str, str]:
    files = [
        path
        for path in C0_ROOT.rglob("*")
        if path.is_file() and path.name != "C0_SHA256_INDEX.json"
    ]
    index = {str(path.relative_to(C0_ROOT)).replace("\\", "/"): sha256_file(path) for path in sorted(files)}
    _write_json(C0_ROOT / "C0_SHA256_INDEX.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("preflight", "run-source", "adapt-targets", "evaluate", "audit"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ecs-host", default="root@121.40.139.213")
    parser.add_argument("--pi-host", default="gaps@192.168.137.172")
    parser.add_argument("--c2-host", default="root@114.55.171.63")
    parser.add_argument("--timeout-hours", type=float, default=10.0)
    args = parser.parse_args()
    if args.stage == "preflight":
        design = json.loads(DESIGN_MANIFEST.read_text(encoding="utf-8"))
        remote_audit = audit_remote_source_datasets(
            pi_host=args.pi_host, c2_host=args.c2_host, design=design
        )
        print(json.dumps(preflight(remote_source_audit=remote_audit), sort_keys=True))
    elif args.stage == "run-source":
        execute_source_fl(
            ecs_host=args.ecs_host,
            pi_host=args.pi_host,
            c2_host=args.c2_host,
            timeout_hours=args.timeout_hours,
        )
    elif args.stage == "adapt-targets":
        adapt_targets(torch.device(args.device))
    elif args.stage == "evaluate":
        print(json.dumps(evaluate_and_decide(torch.device(args.device)), sort_keys=True))
    else:
        print(json.dumps({"status": "PASS", "files": len(write_hash_index())}))


if __name__ == "__main__":
    main()
