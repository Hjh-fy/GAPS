"""Run the frozen canonical-v1 Q0 QC-necessity study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.canonical_qc_evaluation import (
    COVERAGE_GRID,
    RANDOM_REPETITIONS,
    RANDOM_SEED,
    aurc,
    audit_equal_mean_availability,
    classification_confidence_risk,
    decide_qc_necessity,
    grouped_model_dispersion,
    random_reference_metrics,
    risk_coverage_curve,
)
from gaps_flower.canonical_quantitative_features import load_feature_cache
from gaps_flower.canonical_r1_v1 import predict_ridge_model


STUDY_ID = "CAN-V1-CRRQ-Q0-QC-NECESSITY-20260812"
DATASET_SHA256 = "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"
R1_STUDY_ID = "CAN-V1-CRRQ-R1-CANONICAL-83D-R84-20260812"
R1_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812"
R2_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_r2_transfer_safe_v2_20260812"
R0_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812"
CANONICAL_RECORD_ROOT = ROOT / "results/iotj_canonical_v1_final_20260808/regression"
DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
DOC_ROOT = ROOT / "docs/experiments/iotj_canonical_v1_final/canonical_q0_qc_necessity_20260812"
FORMAL_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_q0_qc_necessity_20260812"
TARGETS = ("C3", "C4", "C5")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _validate_prerequisites():
    if read_json(R2_ROOT / "COMPLETE.json").get("status") != "PASS":
        raise RuntimeError("R2-v2 is incomplete")
    if read_json(R2_ROOT / "R2_DECISION.json").get("decision") != "RETAIN_R84_DEVICE_DEPENDENT":
        raise RuntimeError("Q0 requires frozen R84 backend")
    for target in TARGETS:
        for split in ("calibration", "test"):
            if not (CANONICAL_RECORD_ROOT / target / f"{split}_records.csv").is_file():
                raise FileNotFoundError(f"missing canonical classifier record: {target}/{split}")


def inspect():
    _validate_prerequisites()
    if FORMAL_ROOT.exists():
        raise FileExistsError("immutable Q0 formal root exists")
    return {
        "study_id": STUDY_ID,
        "regression_backend": "R84_CONCAT",
        "targets": list(TARGETS),
        "primary_target": "C5",
        "coverage_grid": [float(v) for v in COVERAGE_GRID],
        "random_repetitions": RANDOM_REPETITIONS,
        "random_seed": RANDOM_SEED,
        "q4_status": "Q4_CANONICAL_INPUTS_UNAVAILABLE",
        "formal_root_exists": False,
        "target_test_opened": False,
    }


def preflight(authorized_head):
    if authorized_head != head():
        raise RuntimeError("authorized freeze HEAD mismatch")
    payload = inspect()
    payload.update({
        "authorized_head": authorized_head,
        "r2_decision_sha256": sha256(R2_ROOT / "R2_DECISION.json"),
        "r1_prediction_sha256": sha256(R1_ROOT / "predictions.csv"),
        "protocol_sha256": sha256(DOC_ROOT / "protocol_manifest.json"),
    })
    return payload


def _source_models():
    lock = read_json(R0_ROOT / "model_lock.json")
    return {int(key): value["federated"] for key, value in lock["models"].items()}


def _source_prediction(models, h1):
    return np.column_stack([predict_ridge_model(models[gas], h1) for gas in range(4)])


def _confidence_by_identity(target, split):
    rows = read_csv(CANONICAL_RECORD_ROOT / target / f"{split}_records.csv")
    result = {}
    for row in rows:
        probs = [float(row[f"prob_class_{gas}"]) for gas in range(4)]
        result[row["physical_identity"]] = (float(row["confidence"]), probs)
    return result


def _target_inputs(target, split, source_models, target_models):
    client = int(target[1:])
    sensor, h1, identities, _ = load_feature_cache(
        R1_ROOT / f"canonical_feature_caches/{target}/{split}",
        expected_dataset_sha256=DATASET_SHA256,
        expected_study_id=R1_STUDY_ID,
    )
    physical = np.asarray([row["physical_identity"] for row in identities])
    groups = np.asarray([row["filename"] for row in identities])
    cls = np.load(DATA_ROOT / f"client_{client}/{split}_classification_labels.npy", allow_pickle=False).astype(int)
    truth4 = np.load(DATA_ROOT / f"client_{client}/{split}_regression_labels.npy", allow_pickle=False)
    source = _source_prediction(source_models, h1)
    x84 = [np.column_stack([sensor, source[:, gas]]) for gas in range(4)]
    route = np.column_stack([predict_ridge_model(target_models[target][str(gas)], x84[gas]) for gas in range(4)])
    confidence = _confidence_by_identity(target, split)
    missing = sorted(set(physical) - set(confidence))
    if missing:
        raise RuntimeError(f"classifier probability identity mismatch: {target}/{split}")
    probs = np.asarray([confidence[item][1] for item in physical])
    predicted = probs.argmax(axis=1)
    return {
        "sensor": sensor, "x84": x84, "physical": physical, "groups": groups,
        "true_class": cls, "truth": truth4[np.arange(len(cls)), cls],
        "route": route, "probabilities": probs, "predicted_class": predicted,
    }


def _random_curve(truth, prediction, ranges, coverages=COVERAGE_GRID):
    return [{"policy": "RANDOM_RETAIN", "target_coverage": float(coverage),
             "actual_coverage": float(coverage), **random_reference_metrics(
                 truth, prediction, ranges, float(coverage), RANDOM_REPETITIONS, RANDOM_SEED)}
            for coverage in coverages]


def run(authorized_head):
    receipt = preflight(authorized_head)
    FORMAL_ROOT.mkdir(parents=True)
    write_json(FORMAL_ROOT / "preflight_receipt.json", receipt)
    source_models = _source_models()
    target_lock = read_json(R1_ROOT / "target_model_lock.json")["models"]
    target_models = {target: target_lock[target]["R84_CONCAT"] for target in TARGETS}
    policy_lock = {"study_id": STUDY_ID, "backend": "R84_CONCAT", "coverage_grid": [float(v) for v in COVERAGE_GRID],
                   "random_repetitions": RANDOM_REPETITIONS, "random_seed": RANDOM_SEED,
                   "q4": audit_equal_mean_availability({"classification_uncertainty", "regression_uncertainty"}),
                   "target_test_opened": False}
    write_json(FORMAL_ROOT / "qc_policy_lock.json", policy_lock)
    release_sha = sha256(FORMAL_ROOT / "qc_policy_lock.json")

    all_curves = []
    random_rows = []
    uncertainty_audit = []
    pooled = []
    for target in TARGETS:
        calibration = _target_inputs(target, "calibration", source_models, target_models)
        test = _target_inputs(target, "test", source_models, target_models)
        predicted = test["predicted_class"]
        truth = test["truth"]
        prediction = test["route"][np.arange(len(predicted)), predicted]
        ranges = np.asarray([float(source_models[int(gas)]["clip_max"] - source_models[int(gas)]["clip_min"])
                             for gas in test["true_class"]])
        confidence_risk = classification_confidence_risk(test["probabilities"])
        uncertainty = np.empty(len(truth), dtype=np.float64)
        for gas in range(4):
            mask = calibration["true_class"] == gas
            score, audit = grouped_model_dispersion(
                calibration["x84"][gas][mask], calibration["truth"][mask], calibration["groups"][mask],
                test["x84"][gas], float(target_models[target][str(gas)]["alpha"]),
                float(source_models[gas]["clip_max"] - source_models[gas]["clip_min"]), 5,
            )
            routed = predicted == gas
            uncertainty[routed] = score[routed]
            uncertainty_audit.append({"target": target, "gas": gas, **audit})
        for policy, risk in (("CLASSIFICATION_CONFIDENCE_ONLY", confidence_risk),
                             ("REGRESSION_UNCERTAINTY_ONLY", uncertainty)):
            curves = risk_coverage_curve(truth, prediction, ranges, test["true_class"], predicted,
                                         risk, test["physical"], policy)
            for row in curves:
                row["scope"] = target
            all_curves.extend(curves)
        random = _random_curve(truth, prediction, ranges)
        for row in random:
            row["scope"] = target
        random_rows.extend(random)
        pooled.append((target, truth, prediction, ranges, test["true_class"], predicted,
                       confidence_risk, uncertainty, test["physical"]))
    # Target-stratified pooling preserves the same retained coverage within each target.
    for policy, risk_pos in (("CLASSIFICATION_CONFIDENCE_ONLY", 6), ("REGRESSION_UNCERTAINTY_ONLY", 7)):
        pooled_curve = []
        for coverage in COVERAGE_GRID:
            selected = []
            for item in pooled:
                from gaps_flower.canonical_qc_evaluation import retained_indices
                selected.append((item, retained_indices(item[risk_pos], item[8], float(coverage))))
            truth = np.concatenate([item[1][idx] for item, idx in selected])
            pred = np.concatenate([item[2][idx] for item, idx in selected])
            ranges = np.concatenate([item[3][idx] for item, idx in selected])
            tc = np.concatenate([item[4][idx] for item, idx in selected])
            pc = np.concatenate([item[5][idx] for item, idx in selected])
            error = pred - truth
            pooled_curve.append({"policy": policy, "scope": "POOLED", "target_coverage": float(coverage),
                                 "actual_coverage": len(truth) / sum(len(item[1]) for item in pooled), "retained_n": len(truth),
                                 "RMSE": float(np.sqrt(np.mean(error**2))), "NRMSE_range": float(np.sqrt(np.mean((error/ranges)**2))),
                                 "MAE": float(np.mean(np.abs(error))), "misroute_rate": float(np.mean(tc != pc)),
                                 "error_ge_40ppm_rate": float(np.mean(np.abs(error) >= 40)),
                                 "P90_absolute_error": float(np.percentile(np.abs(error), 90))})
        all_curves.extend(pooled_curve)
    write_json(FORMAL_ROOT / "target_test_release_receipt.json", {"qc_policy_lock_sha256": release_sha})
    write_csv(FORMAL_ROOT / "qc_risk_coverage_curves.csv", all_curves)
    write_csv(FORMAL_ROOT / "random_reference.csv", random_rows)
    write_csv(FORMAL_ROOT / "regression_uncertainty_audit.csv", uncertainty_audit)
    aurc_rows = []
    for scope in ("C3", "C4", "C5", "POOLED"):
        for policy in ("CLASSIFICATION_CONFIDENCE_ONLY", "REGRESSION_UNCERTAINTY_ONLY"):
            curve = [row for row in all_curves if row["scope"] == scope and row["policy"] == policy]
            aurc_rows.append({"scope": scope, "policy": policy, "AURC_RMSE": aurc(curve, "RMSE"),
                              "AURC_NRMSE": aurc(curve, "NRMSE_range")})
    write_csv(FORMAL_ROOT / "qc_aurc.csv", aurc_rows)
    c5 = {row["policy"]: row for row in aurc_rows if row["scope"] == "C5"}
    # Q4 is unavailable, so the registered result is explicit and does not silently promote another formula.
    decision = decide_qc_necessity(None, c5["CLASSIFICATION_CONFIDENCE_ONLY"]["AURC_NRMSE"],
                                   float("nan"), c5["REGRESSION_UNCERTAINTY_ONLY"]["AURC_NRMSE"])
    write_json(FORMAL_ROOT / "Q0_DECISION.json", {"study_id": STUDY_ID, "decision": decision,
               "q4_status": policy_lock["q4"]["decision"], "primary_target": "C5", "backend": "R84_CONCAT"})
    (FORMAL_ROOT / "QC_NECESSITY_REPORT.md").write_text(
        f"# Canonical Q0 QC necessity\n\nDecision: `{decision}`.\n\nCanonical Q4 status: `{policy_lock['q4']['decision']}`. "
        "No replacement multisignal formula was introduced. C5 is primary; C3/C4 and pooled are consistency evidence.\n",
        encoding="utf-8")
    index = {path.relative_to(FORMAL_ROOT).as_posix(): sha256(path) for path in FORMAL_ROOT.rglob("*")
             if path.is_file() and path.name not in {"sha256_index.json", "COMPLETE.json"}}
    write_json(FORMAL_ROOT / "sha256_index.json", index)
    write_json(FORMAL_ROOT / "COMPLETE.json", {"study_id": STUDY_ID, "status": "PASS", "decision": decision})
    return read_json(FORMAL_ROOT / "Q0_DECISION.json")


def audit():
    index = read_json(FORMAL_ROOT / "sha256_index.json")
    for name, digest in index.items():
        if sha256(FORMAL_ROOT / name) != digest:
            raise RuntimeError(f"Q0 hash mismatch: {name}")
    if read_json(FORMAL_ROOT / "target_test_release_receipt.json")["qc_policy_lock_sha256"] != sha256(FORMAL_ROOT / "qc_policy_lock.json"):
        raise RuntimeError("Q0 policy lock mismatch")
    return {"status": "PASS", "decision": read_json(FORMAL_ROOT / "Q0_DECISION.json")["decision"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "preflight", "run", "audit"))
    parser.add_argument("--authorized-freeze-commit", default="")
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect()
    elif args.command == "audit":
        result = audit()
    elif not args.authorized_freeze_commit:
        raise SystemExit("--authorized-freeze-commit required")
    elif args.command == "preflight":
        result = preflight(args.authorized_freeze_commit)
    else:
        result = run(args.authorized_freeze_commit)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
