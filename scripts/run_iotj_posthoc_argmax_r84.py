"""Run Phase-3 registered Posthoc Argmax R84 baseline on C5."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.posthoc_commissioning import ordered_state_fingerprint, sha256_file
from scripts import run_gaps_cross_target_r84_full as r84_common
from scripts.run_iotj_a0t_vs_a4_regression import (
    DATA_ROOT,
    EXPECTED_DATASET_SHA256,
    EXPECTED_H1_SHA256,
    FROZEN_ALPHAS,
    H1_MANIFEST,
    EndpointSpec,
    _evaluate_endpoint_test,
    _model_manifest,
    _without_features,
    build_four_scopes,
    fit_fixed_alpha_models,
    prepare_rows,
    route_rows,
)
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


OUTPUT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/phase3_posthoc_argmax"
PHASE2_ROOT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/phase2_dg_commissioning"
G1_ROOT = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate1_posthoc/a0t_full"
SEED = 42
STEPS = 100
EXPERIMENT_ID = "CAN-V1-MB-P3-POSTHOC-R84-S42"


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _phase2_b20_metrics() -> tuple[str, dict[str, float]]:
    decision = json.loads((PHASE2_ROOT / "PHASE2_DECISION.json").read_text(encoding="utf-8"))["decision"]
    import csv
    with (PHASE2_ROOT / "DG_COMMISSIONING_BRIDGE.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {row["identity"]: float(row["C5_macro_f1"]) for row in rows if int(row["budget"]) == 20}
    if set(values) != {"I0", "I1", "I2"}:
        raise RuntimeError("FAIL_CLOSED Phase-2 B20 metrics incomplete")
    return decision, values


def select_phase3_identity(phase2_decision: str, b20_macro_f1: dict[str, float]) -> dict[str, Any]:
    if set(b20_macro_f1) != {"I0", "I1", "I2"}:
        raise RuntimeError("FAIL_CLOSED selection requires I0/I1/I2 B20")
    if phase2_decision in {"DG_TO_COMMISSIONING_SUPPORTED", "DG_LOW_BUDGET_VALUE_SUPPORTED"}:
        identity = "I2"
        rule = "registered_dg_commissioning_decision"
    else:
        best = max(float(value) for value in b20_macro_f1.values())
        identity = next(name for name in ("I0", "I1", "I2") if best - float(b20_macro_f1[name]) <= 0.01)
        rule = "simplest_effective_within_0.01_of_best_B20"
    return {
        "identity": identity,
        "budget": 20,
        "phase2_decision": phase2_decision,
        "b20_macro_f1": {key: float(value) for key, value in b20_macro_f1.items()},
        "selection_rule": rule,
        "effectiveness_band": 0.01,
        "target_test_checkpoint_selection": False,
    }


def _classifier_paths(identity: str) -> tuple[Path, Path, Path]:
    if identity == "I0":
        return G1_ROOT / "posthoc_a0t_full_c5.pth", G1_ROOT / "run_manifest.json", G1_ROOT / "fixed_endpoint_complete.json"
    directory = PHASE2_ROOT / identity / "B20"
    return directory / "posthoc_a0t_full_c5.pth", directory / "run_manifest.json", directory / "fixed_endpoint_complete.json"


def audit_selected_classifier(identity: str) -> dict[str, Any]:
    checkpoint, manifest_path, marker_path = _classifier_paths(identity)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not checkpoint.is_file() or sha256_file(checkpoint) != manifest.get("checkpoint_sha256"):
        raise RuntimeError("FAIL_CLOSED selected classifier hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_fingerprint = ordered_state_fingerprint(payload["model_state"])
    if state_fingerprint == manifest.get("source_state_fingerprint"):
        raise RuntimeError("FAIL_CLOSED selected checkpoint was not adapted")
    if int(payload.get("step", -1)) != STEPS or int(marker.get("step", -1)) != STEPS:
        raise RuntimeError("FAIL_CLOSED selected classifier is not step100")
    if manifest.get("target_test_opened") is not False or marker.get("target_test_opened") is not False:
        raise RuntimeError("FAIL_CLOSED target test opened before selected endpoint lock")
    observed_identity = str(manifest.get("identity", "I0" if identity == "I0" else ""))
    observed_budget = int(manifest.get("budget", 20))
    if observed_identity != identity or observed_budget != 20:
        raise RuntimeError("FAIL_CLOSED selected classifier identity/budget mismatch")
    return {
        "status": "PASS",
        "identity": identity,
        "budget": 20,
        "step": STEPS,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_state_fingerprint": state_fingerprint,
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "source_state_fingerprint": manifest["source_state_fingerprint"],
        "target_test_opened_before_lock": False,
        "classifier_training_performed": False,
    }


def audit_r84_inputs() -> dict[str, Any]:
    if sha256_file(H1_MANIFEST) != EXPECTED_H1_SHA256:
        raise RuntimeError("FAIL_CLOSED H1 source pool manifest differs")
    alphas = {str(key): float(value) for key, value in FROZEN_ALPHAS["C5"].items()}
    if alphas != {"0": 1.0, "1": 0.01, "2": 10.0, "3": 0.1}:
        raise RuntimeError("FAIL_CLOSED C5 fixed alpha table differs")
    dataset = verify_dataset(DATA_ROOT)
    if dataset.get("status") != "PASS" or dataset.get("aggregate_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("FAIL_CLOSED canonical-v1 dataset differs")
    return {"status": "PASS", "h1_manifest": str(H1_MANIFEST.resolve()), "h1_sha256": EXPECTED_H1_SHA256, "alphas": alphas, "alpha_selection_performed": False, "dataset_aggregate_sha256": EXPECTED_DATASET_SHA256}


def verify_calibration_lock(lock_path: Path, model_path: Path) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    expected_alphas = {str(key): float(value) for key, value in FROZEN_ALPHAS["C5"].items()}
    observed_alphas = {str(key): float(value) for key, value in lock.get("fixed_alphas", {}).items()}
    if lock.get("status") != "SEALED_BEFORE_TARGET_TEST" or lock.get("target_test_opened") is not False or lock.get("alpha_selection_performed") is not False or observed_alphas != expected_alphas or not model_path.is_file() or sha256_file(model_path) != lock.get("r84_models_sha256"):
        raise RuntimeError("FAIL_CLOSED invalid Phase-3 calibration lock")
    return {"status": "PASS", "lock_sha256": sha256_file(lock_path), "r84_models_sha256": sha256_file(model_path)}


def _spec(classifier: dict[str, Any]) -> EndpointSpec:
    checkpoint = Path(classifier["checkpoint"])
    _cp, manifest, marker = _classifier_paths(classifier["identity"])
    return EndpointSpec(experiment_id=EXPERIMENT_ID, method="Posthoc-A0T", target="C5", checkpoint=checkpoint, checkpoint_sha256=classifier["checkpoint_sha256"], classification_manifest=manifest, completion_marker=marker)


def write_freeze(output: Path) -> dict[str, Any]:
    decision, metrics = _phase2_b20_metrics()
    selection = select_phase3_identity(decision, metrics)
    classifier = audit_selected_classifier(selection["identity"])
    payload = {"schema_version": "iotj.canonical_v1.method_breakthrough.phase3.freeze.v1", "status": "FROZEN", "freeze_commit": _git_head(), "selection": selection, "classifier": classifier, "r84": audit_r84_inputs(), "regression_profile": "R84_FED_H1_fixed_alpha", "classifier_training_performed": False, "alpha_selection_performed": False, "target_test_opened": False}
    path = Path(output) / "PRE_RUN_FREEZE.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("FAIL_CLOSED Phase-3 freeze differs")
    else:
        _json(path, payload)
        _json(Path(output) / "PHASE3_SELECTION.json", selection)
    return payload


def fit_calibration(output: Path, freeze: dict[str, Any], device: torch.device, batch_size: int) -> tuple[EndpointSpec, dict[int, Any]]:
    endpoint = Path(output) / "endpoint"
    if endpoint.exists():
        raise FileExistsError("FAIL_CLOSED Phase-3 endpoint already exists")
    endpoint.mkdir(parents=True)
    spec = _spec(freeze["classifier"])
    h1 = r84_common.load_h1()
    routes, classification = route_rows(
        spec.checkpoint,
        "C5",
        "calibration",
        device,
        batch_size,
        expected_endpoint=("step", STEPS),
    )
    if len(routes) != 320:
        raise RuntimeError("FAIL_CLOSED Phase-3 calibration count differs")
    oracle, deployment = prepare_rows("C5", "calibration", routes, h1)
    oracle_r84 = [r84_common.r84_row(row) for row in oracle]
    deployment_r84 = [r84_common.r84_row(row) for row in deployment]
    models = fit_fixed_alpha_models("C5", oracle_r84)
    model_path = endpoint / "r84_models.json"
    r84_common.write_json(model_path, _model_manifest(models))
    scopes = build_four_scopes(deployment_r84, oracle_r84, models)
    r84_common.write_csv(endpoint / "calibration_s_all.csv", _without_features(scopes["S_ALL"]))
    lock = {"schema_version": "iotj.canonical_v1.method_breakthrough.phase3.lock.v1", "status": "SEALED_BEFORE_TARGET_TEST", "experiment_id": EXPERIMENT_ID, "target_test_opened": False, "alpha_selection_performed": False, "fixed_alphas": FROZEN_ALPHAS["C5"], "calibration_N": len(routes), "classification_metrics": classification, "classifier": freeze["classifier"], "h1_sha256": EXPECTED_H1_SHA256, "r84_models_sha256": sha256_file(model_path)}
    _json(endpoint / "calibration_lock.json", lock)
    verify_calibration_lock(endpoint / "calibration_lock.json", model_path)
    return spec, models


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize_existing_evaluation(output: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    """Finalize an already-computed endpoint without reopening target test."""
    output = Path(output)
    endpoint = output / "endpoint"
    lock = verify_calibration_lock(endpoint / "calibration_lock.json", endpoint / "r84_models.json")
    sealed_open = json.loads((output / "SEALED_TEST_OPEN.json").read_text(encoding="utf-8"))
    endpoint_manifest = json.loads((endpoint / "endpoint_manifest.json").read_text(encoding="utf-8"))
    if sealed_open.get("status") != "OPENED_AFTER_PHASE3_CALIBRATION_LOCK":
        raise RuntimeError("FAIL_CLOSED Phase-3 sealed-test opening record invalid")
    if (
        endpoint_manifest.get("status") != "COMPLETE"
        or endpoint_manifest.get("experiment_id") != EXPERIMENT_ID
        or endpoint_manifest.get("checkpoint_sha256") != freeze["classifier"]["checkpoint_sha256"]
        or endpoint_manifest.get("target_test_used_for_selection") is not False
    ):
        raise RuntimeError("FAIL_CLOSED Phase-3 completed endpoint manifest invalid")

    summary = _read_csv(output / "POSTHOC_ARGMAX_BASELINE.csv")
    for required in ("POSTHOC_ARGMAX_PER_GAS.csv", "POSTHOC_ARGMAX_PER_CONCENTRATION.csv"):
        if not (output / required).is_file():
            raise RuntimeError(f"FAIL_CLOSED Phase-3 output missing: {required}")
    lookup = {row["scope"]: row for row in summary}
    required_scopes = ("S_ALL", "S_CC", "Oracle_ALL", "Oracle_CC")
    if set(lookup) != set(required_scopes):
        raise RuntimeError("FAIL_CLOSED Phase-3 scope summary incomplete")
    lines = "\n".join(
        f"| {scope} | {int(float(lookup[scope]['N']))} | {float(lookup[scope]['RMSE']):.6f} | "
        f"{float(lookup[scope]['MAE']):.6f} | {float(lookup[scope]['NRMSE_range']):.6f} | "
        f"{float(lookup[scope]['R2']):.6f} | {float(lookup[scope]['Bias']):.6f} |"
        for scope in required_scopes
    )
    classification = endpoint_manifest["test_classification"]
    (output / "POSTHOC_ARGMAX_BASELINE_REPORT.md").write_text(
        f"# Posthoc Argmax R84 baseline\n\nSelected identity: `{freeze['selection']['identity']}+B20` "
        f"by `{freeze['selection']['selection_rule']}`.\n\nC5 classification Accuracy/Macro-F1: "
        f"{classification['accuracy']:.6f}/{classification['macro_f1']:.6f}.\n\n"
        "| Scope | N | RMSE | MAE | NRMSE | R2 | Bias |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{lines}\n\nThe classifier was not retrained. R84 used the unchanged H1 source pool and fixed "
        "C5 alpha table; C5 test opened only after the calibration lock.\n",
        encoding="utf-8",
    )
    _json(
        endpoint / "fixed_endpoint_complete.json",
        {
            "status": "COMPLETE",
            "experiment_id": EXPERIMENT_ID,
            "classifier_checkpoint_sha256": freeze["classifier"]["checkpoint_sha256"],
            "r84_models_sha256": lock["r84_models_sha256"],
            "target_test_used_for_selection": False,
        },
    )
    manifest = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.phase3.protocol.v1",
        "status": "PASS",
        "selection": freeze["selection"],
        "classifier": freeze["classifier"],
        "r84": freeze["r84"],
        "classification": classification,
        "scope_metrics": summary,
        "target_test_selection": False,
    }
    _json(output / "protocol_manifest.json", manifest)
    excluded = {"sha256_index.json", "runner.pid", "runner.stdout.log", "runner.stderr.log"}
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in excluded)
    _json(output / "sha256_index.json", {str(path.relative_to(output)).replace("\\", "/"): sha256_file(path) for path in files})
    return manifest


def evaluate(output: Path, freeze: dict[str, Any], spec: EndpointSpec, models: dict[int, Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    endpoint = Path(output) / "endpoint"
    lock = verify_calibration_lock(endpoint / "calibration_lock.json", endpoint / "r84_models.json")
    _json(Path(output) / "SEALED_TEST_OPEN.json", {"status": "OPENED_AFTER_PHASE3_CALIBRATION_LOCK", "opened_at_utc": datetime.now(timezone.utc).isoformat(), "calibration_lock_sha256": lock["lock_sha256"], "target_test_selection": False})
    result = _evaluate_endpoint_test(
        spec,
        endpoint,
        r84_common.load_h1(),
        device,
        batch_size,
        {EXPERIMENT_ID: models},
        expected_classifier_endpoint=("step", STEPS),
    )
    r84_common.write_csv(Path(output) / "POSTHOC_ARGMAX_BASELINE.csv", result["summary"])
    r84_common.write_csv(Path(output) / "POSTHOC_ARGMAX_PER_GAS.csv", result["per_gas"])
    r84_common.write_csv(Path(output) / "POSTHOC_ARGMAX_PER_CONCENTRATION.csv", result["per_concentration"])
    return finalize_existing_evaluation(Path(output), freeze)


def run(output: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = write_freeze(output)
    spec, models = fit_calibration(output, freeze, device, batch_size)
    return evaluate(output, freeze, spec, models, device, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    if args.finalize_existing:
        freeze = json.loads((args.output / "PRE_RUN_FREEZE.json").read_text(encoding="utf-8"))
        result = finalize_existing_evaluation(args.output, freeze)
    else:
        result = run(args.output, device, args.batch_size)
    print(json.dumps({"status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
