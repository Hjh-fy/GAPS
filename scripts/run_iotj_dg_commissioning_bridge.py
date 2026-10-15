"""Run frozen Phase-2 source-initialization to Full-A0T commissioning bridge."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays
from gaps_flower.evaluate_checkpoint import load_checkpoint_model
from gaps_flower.posthoc_commissioning import (
    BATCH_SIZE,
    LR,
    SEED,
    STEPS,
    load_calibration_identity_manifest,
    ordered_state_fingerprint,
    sha256_file,
    supervised_ce_adapt,
)
from scripts.run_iotj_posthoc_commissioning_g1 import _RSSMonitor, _loader
from scripts.summarize_iotj_classification_ablation import (
    classification_metrics,
    evaluate_checkpoint_stream,
)


OUTPUT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/phase2_dg_commissioning"
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
S4_DATA_ROOT = ROOT / "dataset/iotj_canonical_v1_s4_role_view"
BUDGET_STUDY = ROOT / "results/iotj_canonical_v1_c5_budget_20260810"
BUDGET_ROOT = BUDGET_STUDY / "budget_data"
G1_ROOT = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate1_posthoc"
PHASE1_ROOT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/phase1_s4_dg_multiseed"
S2_RUN = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/source_fl/CAN-V1-CMP-FEDAVG"
GATE_A_ROOT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/gate_a_source_diversity"
SOURCE_CONFIG = {
    "I0": {"run": S2_RUN, "method": "S2-FedAvg", "data_root": DATA_ROOT, "clients": (1, 2)},
    "I1": {"run": GATE_A_ROOT / "fedavg", "method": "S4-FedAvg", "data_root": S4_DATA_ROOT, "clients": (1, 2, 3, 4)},
    "I2": {"run": GATE_A_ROOT / "gaps_dg_p", "method": "S4-DG-P", "data_root": S4_DATA_ROOT, "clients": (1, 2, 3, 4)},
}


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phase2_specs() -> list[dict[str, Any]]:
    rows = []
    for identity in ("I0", "I1", "I2"):
        for budget in (20, 5):
            rows.append(
                {
                    "experiment_id": f"CAN-V1-MB-P2-{identity}-A0T-B{budget:02d}-S42",
                    "identity": identity,
                    "source_method": SOURCE_CONFIG[identity]["method"],
                    "budget": budget,
                    "calibration_n": 320 if budget == 20 else 80,
                    "execution": "reuse" if (identity, budget) == ("I0", 20) else "train",
                    "steps": STEPS,
                    "lr": LR,
                    "seed": SEED,
                }
            )
    return rows


def calibration_dir(budget: int) -> Path:
    if int(budget) == 20:
        return DATA_ROOT / "client_5"
    if int(budget) == 5:
        return BUDGET_ROOT / "client_5_budget_05"
    raise ValueError(f"unregistered Phase-2 budget: {budget}")


def audit_budget_inputs() -> dict[str, Any]:
    audit = json.loads((BUDGET_STUDY / "calibration_budget_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("nested") is not True:
        raise RuntimeError("FAIL_CLOSED frozen calibration-budget audit failed")
    identities: dict[int, set[str]] = {}
    for budget, count in ((20, 320), (5, 80)):
        directory = calibration_dir(budget)
        manifest = load_calibration_identity_manifest(
            directory / "calibration_experiment_info.json", expected_client=5, expected_count=count
        )
        identities[budget] = set(manifest)
        if budget == 5 and any(directory.glob("test_*")):
            raise RuntimeError(f"FAIL_CLOSED target test array entered budget directory B{budget:02d}")
    if not identities[5].issubset(identities[20]):
        raise RuntimeError("FAIL_CLOSED B05 is not nested in B20")
    if int(audit.get("calibration_test_exact_identity_overlap", -1)) != 0:
        raise RuntimeError("FAIL_CLOSED calibration/test identity overlap")
    return {
        "status": "PASS",
        "counts": {"20": 320, "5": 80},
        "per_stratum": {"20": 8, "5": 2},
        "strata": 40,
        "nested": True,
        "calibration_test_identity_overlap": 0,
        "test_arrays_available_to_adaptation": False,
        "adaptation_input_contract": "explicit calibration arrays only; no path enters supervised_ce_adapt",
        "manifest_sha256": {
            str(budget): sha256_file(calibration_dir(budget) / "calibration_experiment_info.json")
            for budget in (20, 5)
        },
    }


def _resolve_checkpoint(run: Path, manifest: dict[str, Any]) -> Path:
    candidates = [Path(str(manifest.get("checkpoint", ""))), run / "remote_server/server_latest.pth"]
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == manifest.get("checkpoint_sha256"):
            return candidate.resolve()
    raise RuntimeError(f"FAIL_CLOSED source checkpoint unavailable/hash mismatch: {run}")


def audit_source_identities() -> dict[str, dict[str, Any]]:
    result = {}
    for identity, config in SOURCE_CONFIG.items():
        run = Path(config["run"])
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        checkpoint = _resolve_checkpoint(run, manifest)
        _model, _config, payload = load_checkpoint_model(str(checkpoint), torch.device("cpu"), BATCH_SIZE)
        if int(payload.get("round", -1)) != 25:
            raise RuntimeError(f"FAIL_CLOSED {identity} is not round25")
        protocol = manifest.get("protocol", {})
        if protocol.get("seed") != 42 or protocol.get("checkpoint_selection") != "fixed_round_25":
            raise RuntimeError(f"FAIL_CLOSED {identity} source protocol mismatch")
        if any(protocol.get(key) is not False for key in ("target_x", "target_y")):
            raise RuntimeError(f"FAIL_CLOSED {identity} source has target access")
        state_fingerprint = ordered_state_fingerprint(payload["model_state"])
        expected_fingerprint = manifest.get("checkpoint_state_fingerprint")
        if expected_fingerprint and expected_fingerprint != state_fingerprint:
            raise RuntimeError(f"FAIL_CLOSED {identity} state fingerprint mismatch")
        result[identity] = {
            "identity": identity,
            "source_method": config["method"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_state_fingerprint": state_fingerprint,
            "round": 25,
            "seed": 42,
            "target_access": "NONE",
            "source_clients": list(config["clients"]),
        }
    return result


def audit_i0_b20_reuse() -> dict[str, Any]:
    identities = audit_source_identities()
    manifest_path = G1_ROOT / "a0t_full/run_manifest.json"
    marker_path = G1_ROOT / "a0t_full/fixed_endpoint_complete.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint = G1_ROOT / "a0t_full/posthoc_a0t_full_c5.pth"
    budget_manifest = calibration_dir(20) / "calibration_experiment_info.json"
    expected = identities["I0"]
    checks = (
        manifest.get("method") == "a0t_full",
        manifest.get("steps") == STEPS,
        manifest.get("lr") == LR,
        manifest.get("seed") == SEED,
        manifest.get("calibration_count") == 320,
        manifest.get("target_test_opened") is False,
        marker.get("step") == STEPS,
        manifest.get("source_checkpoint_sha256") == expected["checkpoint_sha256"],
        manifest.get("source_state_fingerprint") == expected["checkpoint_state_fingerprint"],
        manifest.get("calibration_manifest_sha256") == sha256_file(budget_manifest),
        checkpoint.is_file() and sha256_file(checkpoint) == manifest.get("checkpoint_sha256"),
    )
    if not all(checks):
        raise RuntimeError("FAIL_CLOSED I0+B20 G1 reuse is not exact")
    return {
        "status": "PASS",
        "identity": "I0",
        "budget": 20,
        "method": "a0t_full",
        "steps": STEPS,
        "lr": LR,
        "seed": SEED,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_state_fingerprint": manifest["source_state_fingerprint"],
        "calibration_manifest_sha256": manifest["calibration_manifest_sha256"],
        "adaptation_seconds": manifest["system_metrics"]["adaptation_seconds"],
    }


def decide_dg_commissioning(
    macro_f1: dict[tuple[str, int], float], *, seed42_zero_shot_dg_gain: float
) -> dict[str, Any]:
    expected = {(identity, budget) for identity in ("I0", "I1", "I2") for budget in (20, 5)}
    if set(macro_f1) != expected:
        raise RuntimeError("FAIL_CLOSED Phase-2 decision requires all six endpoints")
    dg = {budget: float(macro_f1[("I2", budget)] - macro_f1[("I1", budget)]) for budget in (20, 5)}
    diversity = {budget: float(macro_f1[("I1", budget)] - macro_f1[("I0", budget)]) for budget in (20, 5)}
    if all(dg[budget] >= 0.01 for budget in (20, 5)):
        decision = "DG_TO_COMMISSIONING_SUPPORTED"
    elif dg[5] >= 0.01 and dg[20] < 0.01:
        decision = "DG_LOW_BUDGET_VALUE_SUPPORTED"
    elif all(abs(dg[budget]) < 0.01 for budget in (20, 5)) and seed42_zero_shot_dg_gain >= 0.01:
        decision = "DG_ZERO_SHOT_ONLY"
    elif any(diversity[budget] >= 0.01 for budget in (20, 5)) and all(dg[budget] < 0.01 for budget in (20, 5)):
        decision = "SOURCE_DIVERSITY_ONLY"
    else:
        decision = "DG_TO_COMMISSIONING_NOT_SUPPORTED"
    return {
        "decision": decision,
        "dg_minus_s4_fedavg": {str(key): value for key, value in dg.items()},
        "s4_minus_s2_fedavg": {str(key): value for key, value in diversity.items()},
        "seed42_zero_shot_dg_gain": float(seed42_zero_shot_dg_gain),
        "meaningful_gain": 0.01,
        "next_action": "ENTER_PHASE3_REGISTERED_SELECTION",
    }


def _endpoint_dir(output: Path, identity: str, budget: int) -> Path:
    return Path(output) / identity / f"B{budget:02d}"


def verify_new_endpoint_locks(
    output: Path, expected_source_fingerprints: dict[str, str]
) -> dict[tuple[str, int], dict[str, Any]]:
    locked = {}
    for spec in phase2_specs():
        if spec["execution"] != "train":
            continue
        identity, budget = spec["identity"], spec["budget"]
        directory = _endpoint_dir(output, identity, budget)
        manifest_path = directory / "run_manifest.json"
        marker_path = directory / "fixed_endpoint_complete.json"
        if not manifest_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"FAIL_CLOSED missing Phase-2 endpoint: {identity}/B{budget:02d}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(manifest.get("checkpoint", "")))
        if not checkpoint.is_absolute():
            checkpoint = (manifest_path.parent / checkpoint).resolve()
        if marker.get("step") != STEPS or manifest.get("steps") != STEPS:
            raise RuntimeError(f"FAIL_CLOSED Phase-2 step mismatch: {identity}/B{budget:02d}")
        if manifest.get("target_test_opened") is not False or marker.get("target_test_opened") is not False:
            raise RuntimeError(f"FAIL_CLOSED target test opened before lock: {identity}/B{budget:02d}")
        if manifest.get("source_state_fingerprint") != expected_source_fingerprints[identity]:
            raise RuntimeError(f"FAIL_CLOSED source fingerprint mismatch: {identity}/B{budget:02d}")
        if manifest.get("budget") != budget or manifest.get("seed") != SEED or manifest.get("lr") != LR:
            raise RuntimeError(f"FAIL_CLOSED protocol mismatch: {identity}/B{budget:02d}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
            raise RuntimeError(f"FAIL_CLOSED checkpoint hash mismatch: {identity}/B{budget:02d}")
        locked[(identity, budget)] = {**manifest, "checkpoint": str(checkpoint)}
    if len(locked) != 5:
        raise RuntimeError("FAIL_CLOSED Phase-2 requires exactly five new endpoints")
    return locked


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def write_freeze(output: Path) -> dict[str, Any]:
    phase1 = json.loads((PHASE1_ROOT / "PHASE1_DECISION.json").read_text(encoding="utf-8"))
    if phase1.get("decision") not in {"SOURCE_DG_SUPPORTED", "SOURCE_DG_UNSTABLE", "SOURCE_DG_NOT_CONFIRMED"}:
        raise RuntimeError("FAIL_CLOSED Phase-1 predecessor decision missing")
    sources = audit_source_identities()
    payload = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.phase2.freeze.v1",
        "status": "FROZEN",
        "freeze_commit": _git_head(),
        "phase1_decision": phase1,
        "specs": phase2_specs(),
        "sources": sources,
        "budgets": audit_budget_inputs(),
        "i0_b20_reuse": audit_i0_b20_reuse(),
        "adaptation": {"method": "a0t_full", "steps": STEPS, "optimizer": "Adam", "lr": LR, "seed": SEED},
        "target_calibration_fields": {"x": True, "class": True, "phase": "schema_only_not_used_by_loss", "concentration": False},
        "target_test_available_to_adaptation": False,
        "target_test_selection": False,
        "hyperparameter_search": False,
    }
    path = Path(output) / "PRE_RUN_FREEZE.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("FAIL_CLOSED Phase-2 freeze differs")
    else:
        _json(path, payload)
    return payload


def _progress(output: Path, status: str, completed: list[str], active: str | None) -> None:
    _json(Path(output) / "RUN_PROGRESS.json", {"status": status, "completed": completed, "completed_count": len(completed), "required_new_count": 5, "active": active, "updated_at_utc": datetime.now(timezone.utc).isoformat()})


def train_new_endpoints(output: Path, freeze: dict[str, Any], device: torch.device) -> None:
    sources = freeze["sources"]
    completed: list[str] = []
    for spec in phase2_specs():
        if spec["execution"] != "train":
            continue
        identity, budget = spec["identity"], spec["budget"]
        name = f"{identity}/B{budget:02d}"
        directory = _endpoint_dir(output, identity, budget)
        if (directory / "fixed_endpoint_complete.json").is_file():
            completed.append(name)
            continue
        if directory.exists():
            raise RuntimeError(f"FAIL_CLOSED partial Phase-2 endpoint: {name}")
        directory.mkdir(parents=True)
        _progress(output, "RUNNING", completed, name)
        source = sources[identity]
        source_model, _config, source_payload = load_checkpoint_model(source["checkpoint"], device, BATCH_SIZE)
        if ordered_state_fingerprint(source_payload["model_state"]) != source["checkpoint_state_fingerprint"]:
            raise RuntimeError(f"FAIL_CLOSED independent source reload mismatch: {name}")
        budget_dir = calibration_dir(budget)
        arrays = load_domain_adaptation_arrays([budget_dir], strict=True, expected_window_shape=(50, 8))
        if len(arrays[0]) != spec["calibration_n"]:
            raise RuntimeError(f"FAIL_CLOSED calibration count mismatch: {name}")
        loader = _loader(arrays, limit=None)
        with _RSSMonitor() as monitor:
            adapted, diagnostics, system = supervised_ce_adapt(
                source_model, loader, method="a0t_full", device=device, steps=STEPS, lr=LR, seed=SEED
            )
        system["peak_rss_bytes"] = int(monitor.peak) if monitor.process is not None else None
        checkpoint = directory / "posthoc_a0t_full_c5.pth"
        torch.save(
            {"step": STEPS, "model_state": adapted.state_dict(), "method": "a0t_full", "identity": identity, "budget": budget, "source_checkpoint_sha256": source["checkpoint_sha256"], "source_state_fingerprint": source["checkpoint_state_fingerprint"], "seed": SEED},
            checkpoint,
        )
        system["checkpoint_bytes"] = checkpoint.stat().st_size
        system["checkpoint_sha256"] = sha256_file(checkpoint)
        _csv(directory / "adaptation_diagnostics.csv", diagnostics)
        _json(directory / "system_metrics.json", system)
        manifest = {
            "schema_version": "iotj.canonical_v1.method_breakthrough.phase2.run.v1",
            "experiment_id": spec["experiment_id"],
            "identity": identity,
            "source_method": spec["source_method"],
            "budget": budget,
            "calibration_count": spec["calibration_n"],
            "method": "a0t_full",
            "steps": STEPS,
            "optimizer": "Adam",
            "lr": LR,
            "seed": SEED,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": system["checkpoint_sha256"],
            "source_checkpoint": source["checkpoint"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_state_fingerprint": source["checkpoint_state_fingerprint"],
            "calibration_manifest": str((budget_dir / "calibration_experiment_info.json").resolve()),
            "calibration_manifest_sha256": freeze["budgets"]["manifest_sha256"][str(budget)],
            "target_test_opened": False,
            "target_test_selection": False,
            "system_metrics": system,
        }
        _json(directory / "run_manifest.json", manifest)
        _json(directory / "fixed_endpoint_complete.json", {"status": "COMPLETE", "step": STEPS, "checkpoint_sha256": system["checkpoint_sha256"], "source_state_fingerprint": source["checkpoint_state_fingerprint"], "target_test_opened": False})
        completed.append(name)
    verify_new_endpoint_locks(output, {key: value["checkpoint_state_fingerprint"] for key, value in sources.items()})
    _progress(output, "ALL_ENDPOINTS_LOCKED", completed, None)


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[f"prob_{class_id}"]) for class_id in range(4)] for row in rows], dtype=np.float64)


def _evaluate_pooled(checkpoint: Path, data_root: Path, clients: tuple[int, ...], endpoint: tuple[str, int], device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for client in clients:
        selected, _metrics = evaluate_checkpoint_stream(checkpoint, data_root=data_root, target_client=client, split="test", device=device, batch_size=BATCH_SIZE, expected_endpoint=endpoint)
        rows.extend(selected)
    return classification_metrics([int(row["true_class"]) for row in rows], _probabilities(rows))


def evaluate_and_analyze(output: Path, freeze: dict[str, Any], device: torch.device) -> dict[str, Any]:
    sources = freeze["sources"]
    new = verify_new_endpoint_locks(output, {key: value["checkpoint_state_fingerprint"] for key, value in sources.items()})
    reuse = freeze["i0_b20_reuse"]
    endpoints = {**new, ("I0", 20): {**reuse, "system_metrics": {"adaptation_seconds": reuse["adaptation_seconds"]}}}
    if len(endpoints) != 6:
        raise RuntimeError("FAIL_CLOSED Phase-2 endpoint set differs")
    _json(Path(output) / "SEALED_TEST_OPEN.json", {"status": "OPENED_AFTER_ALL_SIX_ENDPOINTS_LOCKED", "opened_at_utc": datetime.now(timezone.utc).isoformat(), "target_test_manifest_sha256": sha256_file(DATA_ROOT / "client_5/test_experiment_info.json"), "target_test_selection": False})
    source_metrics = {}
    rows: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    c5_f1: dict[tuple[str, int], float] = {}
    for spec in phase2_specs():
        identity, budget = spec["identity"], spec["budget"]
        endpoint = endpoints[(identity, budget)]
        checkpoint = Path(endpoint["checkpoint"])
        target_rows, target_metrics = evaluate_checkpoint_stream(checkpoint, data_root=DATA_ROOT, target_client=5, split="test", device=device, batch_size=BATCH_SIZE, expected_endpoint=("step", STEPS))
        if identity not in source_metrics:
            config = SOURCE_CONFIG[identity]
            source_metrics[identity] = _evaluate_pooled(Path(sources[identity]["checkpoint"]), Path(config["data_root"]), tuple(config["clients"]), ("round", 25), device)
        config = SOURCE_CONFIG[identity]
        adapted_source = _evaluate_pooled(checkpoint, Path(config["data_root"]), tuple(config["clients"]), ("step", STEPS), device)
        seconds = float(endpoint.get("system_metrics", {}).get("adaptation_seconds", endpoint.get("adaptation_seconds", 0.0)))
        rows.append({"identity": identity, "source_method": spec["source_method"], "budget": budget, "calibration_n": spec["calibration_n"], "C5_N": target_metrics["N"], "C5_accuracy": target_metrics["accuracy"], "C5_macro_f1": target_metrics["macro_f1"], "C5_nll": target_metrics["nll"], "C5_ece": target_metrics["ece"], "source_macro_f1_before": source_metrics[identity]["macro_f1"], "source_macro_f1_after": adapted_source["macro_f1"], "source_retention_delta": adapted_source["macro_f1"] - source_metrics[identity]["macro_f1"], "adaptation_seconds": seconds, "checkpoint_sha256": endpoint["checkpoint_sha256"], "source_checkpoint_sha256": sources[identity]["checkpoint_sha256"]})
        c5_f1[(identity, budget)] = float(target_metrics["macro_f1"])
        for class_id, gas in enumerate(("Ethanol", "CO", "Ethylene", "Methane")):
            per_class.append({"identity": identity, "budget": budget, "class_id": class_id, "gas": gas, "recall": target_metrics["per_class_recall"][str(class_id)], "f1": target_metrics["per_class_f1"][str(class_id)]})
        predictions.extend({"identity": identity, "budget": budget, **row} for row in target_rows)
    _csv(Path(output) / "DG_COMMISSIONING_BRIDGE.csv", rows)
    _csv(Path(output) / "DG_COMMISSIONING_BRIDGE_PER_CLASS.csv", per_class)
    _csv(Path(output) / "DG_COMMISSIONING_BRIDGE_PREDICTIONS.csv", predictions)
    phase1 = freeze["phase1_decision"]
    seed42_gain = float(phase1["paired_macro_f1_gains"]["42"])
    decision = decide_dg_commissioning(c5_f1, seed42_zero_shot_dg_gain=seed42_gain)
    _json(Path(output) / "PHASE2_DECISION.json", decision)
    table = "\n".join(f"| {row['identity']} | {row['budget']} | {row['C5_macro_f1']:.6f} | {row['source_retention_delta']:+.6f} | {row['adaptation_seconds']:.3f} |" for row in rows)
    (Path(output) / "DG_COMMISSIONING_BRIDGE_REPORT.md").write_text(f"# DG-to-commissioning bridge report\n\n| Identity | Budget | C5 Macro-F1 | Source retention delta | Seconds |\n|---|---:|---:|---:|---:|\n{table}\n\nDecision: `{decision['decision']}`.\n\nAll six endpoints use Full A0T at fixed step100. I0+B20 is exact G1 reuse; the other five endpoints independently reload their registered original round25 source state. C5 test was not used for stopping, tuning, or selection.\n", encoding="utf-8")
    _json(Path(output) / "protocol_manifest.json", {"status": "PASS", "freeze_commit": freeze["freeze_commit"], "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6", "target_test_manifest_sha256": sha256_file(DATA_ROOT / "client_5/test_experiment_info.json"), "target_test_selection": False, "decision": decision})
    _progress(output, "COMPLETE", [f"{spec['identity']}/B{spec['budget']:02d}" for spec in phase2_specs() if spec["execution"] == "train"], None)
    excluded = {"RUN_PROGRESS.json", "runner.pid", "runner.stdout.log", "runner.stderr.log", "sha256_index.json"}
    files = sorted(path for path in Path(output).rglob("*") if path.is_file() and path.name not in excluded)
    _json(Path(output) / "sha256_index.json", {str(path.relative_to(output)).replace("\\", "/"): sha256_file(path) for path in files})
    return decision


def run(output: Path, device: torch.device) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = write_freeze(output)
    train_new_endpoints(output, freeze, device)
    return evaluate_and_analyze(output, freeze, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    print(json.dumps(run(args.output, device), indent=2))


if __name__ == "__main__":
    main()
