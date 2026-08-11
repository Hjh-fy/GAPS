"""Run Gate B lightweight post-hoc C5 personalization at frozen step 100."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays  # noqa: E402
from gaps_flower.evaluate_checkpoint import load_checkpoint_model  # noqa: E402
from gaps_flower.posthoc_commissioning import (  # noqa: E402
    BATCH_SIZE,
    LR,
    SEED,
    STEPS,
    load_calibration_identity_manifest,
    low_rank_adapter_adapt,
    ordered_state_fingerprint,
    sha256_file,
    supervised_ce_adapt,
)
from scripts.run_iotj_posthoc_commissioning_g1 import (  # noqa: E402
    _RSSMonitor,
    _evaluate_one,
    _loader,
)


DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
SOURCE_RUN = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/source_fl/CAN-V1-CMP-FEDAVG"
SOURCE_CHECKPOINT = SOURCE_RUN / "remote_server/server_latest.pth"
G1_ROOT = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate1_posthoc"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/gate_b_lightweight_posthoc"
NEW_METHODS = ("classifier_only", "rank4_adapter")
METHODS = ("source_only", "a0t_full", "classifier_only", "projection_head", "rank4_adapter")
DISPLAY = {
    "source_only": "B0 Source-only",
    "a0t_full": "B1 Posthoc A0T-full",
    "classifier_only": "B2 Classifier-only",
    "projection_head": "B3 Projection+Head",
    "rank4_adapter": "B4 Rank-4 Adapter",
}


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def gate_b_protocol() -> dict[str, Any]:
    return {
        "gate": "B_LIGHTWEIGHT_POSTHOC_TARGET_PERSONALIZATION",
        "source_set": "S2",
        "target": "C5",
        "calibration_count": 320,
        "calibration_role": "canonical_v1_20_percent",
        "steps": STEPS,
        "optimizer": "Adam",
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "adapter_rank": 4,
        "adapter_initialization": "kaiming_down_zero_up",
        "adapter_deployment": "exact_classifier_fold",
        "hyperparameter_search": False,
        "checkpoint_selection": "fixed_step_100",
        "target_test_selection": False,
        "sufficient_macro_f1_gap_pp": 0.5,
        "substantial_parameter_fraction": 0.25,
    }


def decide_gate_b(
    *,
    full_f1: float,
    candidates: dict[str, dict[str, float]],
    full_trainable_parameters: int,
) -> dict[str, Any]:
    sufficient: dict[str, bool] = {}
    for method, row in candidates.items():
        sufficient[method] = bool(
            full_f1 - float(row["macro_f1"]) <= 0.005
            and int(row["trainable_parameters"]) <= 0.25 * int(full_trainable_parameters)
        )
    preference = ("classifier_only", "projection_head", "rank4_adapter")
    selected = next((method for method in preference if sufficient.get(method, False)), None)
    if selected is None:
        return {
            "decision": "FULL_ADAPTATION_REQUIRED",
            "selected_method": "a0t_full",
            "sufficient": sufficient,
        }
    return {
        "decision": "LIGHTWEIGHT_PERSONALIZATION_SUPPORTED",
        "selected_method": selected,
        "sufficient": sufficient,
    }


def _validate_locked_endpoint(directory: Path, method: str) -> dict[str, Any]:
    manifest_path = directory / "run_manifest.json"
    marker_path = directory / "fixed_endpoint_complete.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"missing endpoint for {method}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if int(marker.get("step", -1)) != STEPS or int(manifest.get("steps", -1)) != STEPS:
        raise RuntimeError(f"wrong fixed endpoint for {method}")
    if marker.get("target_test_opened") is not False or manifest.get("target_test_opened") is not False:
        raise RuntimeError(f"target test opened before endpoint lock: {method}")
    if not checkpoint.is_file() or sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
        raise RuntimeError(f"checkpoint hash mismatch for {method}")
    if marker.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise RuntimeError(f"marker checkpoint mismatch for {method}")
    if marker.get("source_state_fingerprint") != manifest.get("source_state_fingerprint"):
        raise RuntimeError(f"source fingerprint mismatch for {method}")
    return manifest


def verify_new_endpoint_locks(output: Path) -> dict[str, dict[str, Any]]:
    locked = {method: _validate_locked_endpoint(output / method, method) for method in NEW_METHODS}
    if len({row["source_state_fingerprint"] for row in locked.values()}) != 1:
        raise RuntimeError("new endpoints did not independently reload the same source")
    return locked


def _audit_reused_endpoints(source_fingerprint: str, calibration_sha: str) -> dict[str, dict[str, Any]]:
    mapping = {"a0t_full": "a0t_full", "projection_head": "target_head"}
    result: dict[str, dict[str, Any]] = {}
    for method, directory_name in mapping.items():
        manifest = _validate_locked_endpoint(G1_ROOT / directory_name, method)
        if manifest.get("source_state_fingerprint") != source_fingerprint:
            raise RuntimeError(f"reused source fingerprint mismatch: {method}")
        if manifest.get("calibration_manifest_sha256") != calibration_sha:
            raise RuntimeError(f"reused calibration mismatch: {method}")
        if float(manifest.get("lr", -1.0)) != LR or int(manifest.get("seed", -1)) != SEED:
            raise RuntimeError(f"reused protocol mismatch: {method}")
        result[method] = manifest
    return result


def _source_provenance() -> tuple[str, str, dict[str, Any]]:
    source_manifest = json.loads((SOURCE_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(SOURCE_CHECKPOINT)
    if source_manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("source checkpoint SHA mismatch")
    protocol = source_manifest.get("protocol", {})
    if (
        int(protocol.get("rounds", -1)) != 25
        or int(protocol.get("local_epochs", -1)) != 1
        or int(protocol.get("seed", -1)) != SEED
        or protocol.get("target_x") is not False
        or protocol.get("target_y") is not False
    ):
        raise RuntimeError("source endpoint protocol mismatch")
    _model, _config, container = load_checkpoint_model(str(SOURCE_CHECKPOINT), torch.device("cpu"), BATCH_SIZE)
    state_fingerprint = ordered_state_fingerprint(container["model_state"])
    return checkpoint_sha, state_fingerprint, source_manifest


def write_pre_run_freeze(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED Gate B output exists: {output}")
    output.mkdir(parents=True)
    source_sha, source_state, _source_manifest = _source_provenance()
    calibration_manifest = DATA_ROOT / "client_5/calibration_experiment_info.json"
    load_calibration_identity_manifest(calibration_manifest, expected_client=5, expected_count=320)
    calibration_sha = sha256_file(calibration_manifest)
    reused = _audit_reused_endpoints(source_state, calibration_sha)
    freeze = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.gate_b.freeze.v1",
        "status": "FROZEN_BEFORE_ADAPTATION",
        "producer_commit": _git_head(),
        "protocol": gate_b_protocol(),
        "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
        "source_checkpoint_sha256": source_sha,
        "source_state_fingerprint": source_state,
        "calibration_manifest": str(calibration_manifest.resolve()),
        "calibration_manifest_sha256": calibration_sha,
        "reused_endpoint_sha256": {
            method: row["checkpoint_sha256"] for method, row in reused.items()
        },
        "target_test_access_before_lock": "NONE",
    }
    _json(output / "PRE_RUN_FREEZE.json", freeze)
    return freeze


def _save_checkpoint(path: Path, model: torch.nn.Module, method: str, freeze: dict[str, Any]) -> None:
    torch.save(
        {
            "step": STEPS,
            "model_state": model.state_dict(),
            "method": method,
            "source_checkpoint_sha256": freeze["source_checkpoint_sha256"],
            "source_state_fingerprint": freeze["source_state_fingerprint"],
            "seed": SEED,
        },
        path,
    )


def _save_classifier_delta(path: Path, model: torch.nn.Module, *, method: str) -> int:
    torch.save(
        {
            "method": method,
            "deployment_scope": "classifier_only",
            "classifier_state": model.classifier.state_dict(),
        },
        path,
    )
    return path.stat().st_size


def _serialized_delta_bytes(checkpoint: Path, prefixes: tuple[str, ...]) -> int:
    container = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = container.get("model_state", container)
    selected = {
        name: value.detach().cpu()
        for name, value in state.items()
        if not prefixes or name.startswith(prefixes)
    }
    buffer = io.BytesIO()
    torch.save(selected, buffer)
    return len(buffer.getvalue())


def _adapt_new_endpoints(output: Path, freeze: dict[str, Any], device: torch.device) -> None:
    target_arrays = load_domain_adaptation_arrays(
        [DATA_ROOT / "client_5"], strict=True, expected_window_shape=(50, 8)
    )
    for method in NEW_METHODS:
        method_dir = output / method
        method_dir.mkdir()
        source_model, _config, container = load_checkpoint_model(
            str(SOURCE_CHECKPOINT), device, BATCH_SIZE
        )
        if ordered_state_fingerprint(container["model_state"]) != freeze["source_state_fingerprint"]:
            raise RuntimeError(f"source reload fingerprint mismatch: {method}")
        target_loader = _loader(target_arrays, limit=None)
        with _RSSMonitor() as monitor:
            if method == "classifier_only":
                adapted, diagnostics, system = supervised_ce_adapt(
                    source_model,
                    target_loader,
                    method="classifier_only",
                    device=device,
                    steps=STEPS,
                    lr=LR,
                    seed=SEED,
                )
            else:
                adapted, diagnostics, system = low_rank_adapter_adapt(
                    source_model,
                    target_loader,
                    device=device,
                    rank=4,
                    steps=STEPS,
                    lr=LR,
                    seed=SEED,
                )
        system["peak_rss_bytes"] = int(monitor.peak) if monitor.process is not None else None
        checkpoint = method_dir / f"posthoc_{method}_c5.pth"
        _save_checkpoint(checkpoint, adapted, method, freeze)
        delta = method_dir / "personalization_delta.pth"
        system["checkpoint_delta_bytes"] = _save_classifier_delta(delta, adapted, method=method)
        system["checkpoint_bytes"] = checkpoint.stat().st_size
        system["checkpoint_sha256"] = sha256_file(checkpoint)
        system["prediction_checkpoint_form"] = "ordinary_folded_model" if method == "rank4_adapter" else "ordinary_model"
        _write_csv(method_dir / "adaptation_diagnostics.csv", diagnostics)
        _json(method_dir / "system_metrics.json", system)
        manifest = {
            "schema_version": "iotj.canonical_v1.method_breakthrough.gate_b.run.v1",
            "experiment_id": f"CAN-V1-MB-B-{method.upper().replace('_', '-')}",
            "method": method,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": system["checkpoint_sha256"],
            "personalization_delta": str(delta.resolve()),
            "personalization_delta_sha256": sha256_file(delta),
            "source_checkpoint": freeze["source_checkpoint"],
            "source_checkpoint_sha256": freeze["source_checkpoint_sha256"],
            "source_state_fingerprint": freeze["source_state_fingerprint"],
            "calibration_manifest": freeze["calibration_manifest"],
            "calibration_manifest_sha256": freeze["calibration_manifest_sha256"],
            "calibration_count": 320,
            "steps": STEPS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "target_test_opened": False,
            "system_metrics": system,
        }
        _json(method_dir / "run_manifest.json", manifest)
        _json(
            method_dir / "fixed_endpoint_complete.json",
            {
                "status": "COMPLETE",
                "method": method,
                "step": STEPS,
                "checkpoint_sha256": system["checkpoint_sha256"],
                "source_state_fingerprint": freeze["source_state_fingerprint"],
                "target_test_opened": False,
            },
        )


def _evaluate_and_analyze(output: Path, freeze: dict[str, Any], device: torch.device) -> dict[str, Any]:
    new = verify_new_endpoint_locks(output)
    reused = _audit_reused_endpoints(
        freeze["source_state_fingerprint"], freeze["calibration_manifest_sha256"]
    )
    test_manifest = DATA_ROOT / "client_5/test_experiment_info.json"
    _json(
        output / "SEALED_TEST_OPEN.json",
        {
            "status": "OPENED_AFTER_B2_B4_LOCKED_AND_B0_B1_B3_AUDITED",
            "target_test_manifest_sha256": sha256_file(test_manifest),
            "target_test_selection": False,
        },
    )
    checkpoints = {
        "source_only": SOURCE_CHECKPOINT,
        "a0t_full": Path(reused["a0t_full"]["checkpoint"]),
        "classifier_only": Path(new["classifier_only"]["checkpoint"]),
        "projection_head": Path(reused["projection_head"]["checkpoint"]),
        "rank4_adapter": Path(new["rank4_adapter"]["checkpoint"]),
    }
    systems = {
        "a0t_full": dict(reused["a0t_full"]["system_metrics"]),
        "projection_head": dict(reused["projection_head"]["system_metrics"]),
        "classifier_only": dict(new["classifier_only"]["system_metrics"]),
        "rank4_adapter": dict(new["rank4_adapter"]["system_metrics"]),
    }
    systems["a0t_full"]["checkpoint_delta_bytes"] = _serialized_delta_bytes(
        checkpoints["a0t_full"], ()
    )
    systems["projection_head"]["checkpoint_delta_bytes"] = _serialized_delta_bytes(
        checkpoints["projection_head"], ("feat_proj.", "classifier.")
    )
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        endpoint = ("round", 25) if method == "source_only" else ("step", STEPS)
        evaluated = _evaluate_one(checkpoints[method], endpoint, device)
        metrics[method] = {scope: values[1] for scope, values in evaluated.items()}
        system = systems.get(method, {})
        source_f1 = metrics["source_only"]["C1+C2"]["macro_f1"] if method != "source_only" else metrics[method]["C1+C2"]["macro_f1"]
        rows.append(
            {
                "method": DISPLAY[method],
                "method_key": method,
                "C5_accuracy": metrics[method]["C5"]["accuracy"],
                "C5_macro_f1": metrics[method]["C5"]["macro_f1"],
                "C5_nll": metrics[method]["C5"]["nll"],
                "C5_ece": metrics[method]["C5"]["ece"],
                "C1_macro_f1": metrics[method]["C1"]["macro_f1"],
                "C2_macro_f1": metrics[method]["C2"]["macro_f1"],
                "C1_C2_macro_f1": metrics[method]["C1+C2"]["macro_f1"],
                "C1_C2_retention_delta": metrics[method]["C1+C2"]["macro_f1"] - source_f1,
                "trainable_parameters": system.get("trainable_parameters", 0),
                "trainable_parameter_ratio": system.get("trainable_parameter_ratio", 0.0),
                "adaptation_seconds": system.get("adaptation_seconds", 0.0),
                "checkpoint_delta_bytes": system.get("checkpoint_delta_bytes", 0),
                "peak_rss_bytes": system.get("peak_rss_bytes", ""),
                "checkpoint_sha256": sha256_file(checkpoints[method]),
            }
        )
    _write_csv(output / "GATE_B_LIGHTWEIGHT_COMPARISON.csv", rows)
    by_key = {row["method_key"]: row for row in rows}
    decision = decide_gate_b(
        full_f1=float(by_key["a0t_full"]["C5_macro_f1"]),
        candidates={
            method: {
                "macro_f1": float(by_key[method]["C5_macro_f1"]),
                "trainable_parameters": int(by_key[method]["trainable_parameters"]),
            }
            for method in ("classifier_only", "projection_head", "rank4_adapter")
        },
        full_trainable_parameters=int(by_key["a0t_full"]["trainable_parameters"]),
    )
    _json(output / "GATE_B_DECISION.json", decision)
    selected = decision["selected_method"]
    report = f"""# Gate B Lightweight Post-hoc Target Personalization

## [Scientific Question]

Can a new C5 node reach full A0T performance by personalizing only a small endpoint rather than the complete source model?

## [Protocol]

All methods use the immutable canonical S2 round25 source checkpoint and the same 320-window canonical-v1 C5 calibration set. B2 and B4 independently reload the source, use 100 Adam steps at 5e-4 with seed42, and lock step100 before C5 test evaluation. B1 and B3 are immutable audited G1 endpoints. The rank-4 adapter is exactly folded into the ordinary classifier for deployment.

## [Primary Result]

- Full A0T C5 Macro-F1: {float(by_key['a0t_full']['C5_macro_f1']):.6f}
- Classifier-only C5 Macro-F1: {float(by_key['classifier_only']['C5_macro_f1']):.6f}
- Projection+Head C5 Macro-F1: {float(by_key['projection_head']['C5_macro_f1']):.6f}
- Rank-4 adapter C5 Macro-F1: {float(by_key['rank4_adapter']['C5_macro_f1']):.6f}

## [Negative Result / Limitation]

This is seed42 on C5 with a fixed 100-step budget. Source-retention scores describe the hypothetical adapted checkpoint; the operational global source checkpoint remains immutable.

## [Leakage Audit]

Only the canonical C5 calibration loader entered B2/B4 adaptation. The C5 test manifest was opened after both new endpoints were locked. No target-test checkpoint selection or hyperparameter search occurred.

## [Decision]

`{decision['decision']}`; selected path: `{selected}`.

## [Paper Implication]

The method story may claim lightweight commissioning only if the registered 0.5-point and parameter-reduction gates are met; otherwise full target adaptation remains necessary.

## [Next Action]

Proceed to the already frozen read-only Gate C routing-cost audit. Do not start Gate D/E/F.
"""
    (output / "GATE_B_REPORT.md").write_text(report, encoding="utf-8")
    audit = """# Gate B Experiment Audit

## Verdict: PASS

- B2/B4 independently reloaded the same ordered source state.
- B1/B3 checkpoint, calibration, fixed-step, and source fingerprints were audited before reuse.
- C5 test opened only after all endpoint locks; no target-test selection occurred.
- No rank, learning-rate, step-count, or method search was performed.
"""
    (output / "EXPERIMENT_AUDIT.md").write_text(audit, encoding="utf-8")
    _json(
        output / "protocol_manifest.json",
        {
            "status": "PASS",
            "protocol": gate_b_protocol(),
            "producer_commit": freeze["producer_commit"],
            "source_checkpoint_sha256": freeze["source_checkpoint_sha256"],
            "source_state_fingerprint": freeze["source_state_fingerprint"],
            "calibration_manifest_sha256": freeze["calibration_manifest_sha256"],
            "target_test_manifest_sha256": sha256_file(test_manifest),
            "target_test_selection": False,
            "decision": decision,
        },
    )
    excluded = {"sha256_index.json", "runner.stdout.log", "runner.stderr.log", "runner.pid"}
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in excluded)
    _json(
        output / "sha256_index.json",
        {str(path.relative_to(output)).replace("\\", "/"): sha256_file(path) for path in files},
    )
    return decision


def run(output: Path, device: torch.device) -> dict[str, Any]:
    output = output.resolve()
    freeze = write_pre_run_freeze(output)
    _adapt_new_endpoints(output, freeze, device)
    decision = _evaluate_and_analyze(output, freeze, device)
    return {"status": "PASS", "output": str(output), **decision}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate B lightweight post-hoc personalization")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    print(json.dumps(run(args.output, device), indent=2))


if __name__ == "__main__":
    main()
