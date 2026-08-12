"""Run the single registered canonical-v1 Q1 conformal-style QC study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.canonical_qc_evaluation import (  # noqa: E402
    COVERAGE_GRID,
    aurc,
    classification_confidence_risk,
    retained_indices,
    risk_coverage_curve,
)
from gaps_flower.canonical_r1_v1 import (  # noqa: E402
    assign_balanced_group_folds,
    fit_ridge_model,
    predict_ridge_model,
)
from scripts.run_iotj_canonical_q0_qc import (  # noqa: E402
    DATASET_SHA256,
    R0_ROOT,
    R1_ROOT,
    TARGETS,
    _source_models,
    _target_inputs,
)


STUDY_ID = "CAN-V1-CRRQ-Q1V2-CONFORMAL-QC-20260812"
Q0_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_q0_qc_necessity_20260812"
DOC_ROOT = ROOT / "docs/experiments/iotj_canonical_v1_final/canonical_q1_conformal_qc_v2_20260812"
FORMAL_ROOT = ROOT / "results/iotj_canonical_v1_final/canonical_q1_conformal_qc_v2_20260812"
NOMINAL_INTERVAL_COVERAGE = 0.90
SUPPORT_THRESHOLD = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def q1_trigger(q0_decision: str, c5_confidence: float, c5_regression: float,
               pooled_confidence: float, pooled_regression: float) -> bool:
    return (
        q0_decision == "MULTISIGNAL_QC_NOT_ESTABLISHED"
        and float(c5_regression) < float(c5_confidence)
        and float(pooled_regression) < float(pooled_confidence)
    )


def group_aware_split(groups):
    groups = np.asarray(groups, dtype=str)
    if len(np.unique(groups)) < 2:
        raise ValueError("at least two raw-file groups are required")
    folds = assign_balanced_group_folds(groups, n_folds=2)
    fit = folds == 0
    conformal = folds == 1
    if not fit.any() or not conformal.any() or set(groups[fit]) & set(groups[conformal]):
        raise RuntimeError("group-aware conformal split failed")
    return fit, conformal


def conformal_radius(absolute_residuals, coverage: float = NOMINAL_INTERVAL_COVERAGE) -> float:
    residuals = np.asarray(absolute_residuals, dtype=np.float64)
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("absolute residuals must be a finite non-empty vector")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must be in (0, 1)")
    rank = min(len(residuals), int(math.ceil((len(residuals) + 1) * float(coverage))))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def empirical_interval(prediction, radius):
    prediction = np.asarray(prediction, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    if not np.isfinite(radius).all() or np.any(radius < 0):
        raise ValueError("interval radius must be finite and non-negative")
    try:
        radius = np.broadcast_to(radius, prediction.shape)
    except ValueError as exc:
        raise ValueError("interval radius must be broadcast-compatible with prediction") from exc
    return prediction - radius, prediction + radius


def calibration_ecdf(calibration_values, values):
    calibration = np.sort(np.asarray(calibration_values, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    if calibration.ndim != 1 or not len(calibration) or not np.isfinite(calibration).all():
        raise ValueError("calibration CDF requires finite calibration values")
    return np.searchsorted(calibration, values, side="right") / float(len(calibration))


def equal_mean_risk(confidence_cdf, interval_width_cdf):
    confidence_cdf = np.asarray(confidence_cdf, dtype=np.float64)
    interval_width_cdf = np.asarray(interval_width_cdf, dtype=np.float64)
    if confidence_cdf.shape != interval_width_cdf.shape:
        raise ValueError("risk arrays differ in shape")
    return 0.5 * (confidence_cdf + interval_width_cdf)


def decide_q1(c5_confidence: float, c5_combined: float,
              pooled_confidence: float, pooled_combined: float) -> str:
    c5_gain = (float(c5_confidence) - float(c5_combined)) / float(c5_confidence)
    pooled_gain = (float(pooled_confidence) - float(pooled_combined)) / float(pooled_confidence)
    if c5_gain >= SUPPORT_THRESHOLD and pooled_gain >= SUPPORT_THRESHOLD:
        return "CONFORMAL_AUGMENTED_QC_SUPPORTED"
    return "CONFIDENCE_QC_FINAL"


def _q0_trigger_inputs():
    decision = _read_json(Q0_ROOT / "Q0_DECISION.json")["decision"]
    rows = _read_csv(Q0_ROOT / "qc_aurc.csv")
    keyed = {(row["scope"], row["policy"]): float(row["AURC_NRMSE"]) for row in rows}
    values = {
        "c5_confidence": keyed[("C5", "CLASSIFICATION_CONFIDENCE_ONLY")],
        "c5_regression": keyed[("C5", "REGRESSION_UNCERTAINTY_ONLY")],
        "pooled_confidence": keyed[("POOLED", "CLASSIFICATION_CONFIDENCE_ONLY")],
        "pooled_regression": keyed[("POOLED", "REGRESSION_UNCERTAINTY_ONLY")],
    }
    return decision, values


def _validate_prerequisites():
    completion = _read_json(Q0_ROOT / "COMPLETE.json")
    if completion.get("status") != "PASS":
        raise RuntimeError("Q0 is incomplete")
    decision, values = _q0_trigger_inputs()
    if not q1_trigger(decision, **values):
        raise RuntimeError("Q1 registered trigger is not satisfied")
    r2 = ROOT / "results/iotj_canonical_v1_final/canonical_r2_transfer_safe_v2_20260812/R2_DECISION.json"
    if _read_json(r2).get("decision") != "RETAIN_R84_DEVICE_DEPENDENT":
        raise RuntimeError("Q1 requires frozen R84 backend")
    return decision, values


def inspect():
    q0_decision, values = _validate_prerequisites()
    if FORMAL_ROOT.exists():
        raise FileExistsError("immutable Q1 formal root exists")
    return {
        "study_id": STUDY_ID,
        "q0_decision": q0_decision,
        "triggered": q1_trigger(q0_decision, **values),
        "regression_backend": "R84_CONCAT",
        "targets": list(TARGETS),
        "primary_target": "C5",
        "nominal_interval_coverage": NOMINAL_INTERVAL_COVERAGE,
        "coverage_grid": [float(value) for value in COVERAGE_GRID],
        "weight_search": False,
        "target_test_opened": False,
        "formal_root_exists": False,
    }


def preflight(authorized_head: str):
    if authorized_head != _head():
        raise RuntimeError("authorized freeze HEAD mismatch")
    payload = inspect()
    payload.update({
        "authorized_head": authorized_head,
        "q0_complete_sha256": _sha256(Q0_ROOT / "COMPLETE.json"),
        "q0_decision_sha256": _sha256(Q0_ROOT / "Q0_DECISION.json"),
        "q0_aurc_sha256": _sha256(Q0_ROOT / "qc_aurc.csv"),
        "r1_target_model_lock_sha256": _sha256(R1_ROOT / "target_model_lock.json"),
        "protocol_sha256": _sha256(DOC_ROOT / "protocol_manifest.json"),
    })
    return payload


def _build_calibration_lock(source_models, target_models):
    targets = {}
    calibration_arrays = {}
    for target in TARGETS:
        data = _target_inputs(target, "calibration", source_models, target_models)
        confidence_risk = classification_confidence_risk(data["probabilities"])
        width = np.empty(len(data["truth"]), dtype=np.float64)
        target_lock = {}
        for gas in range(4):
            gas_mask = data["true_class"] == gas
            fit_local, conformal_local = group_aware_split(data["groups"][gas_mask])
            gas_indices = np.flatnonzero(gas_mask)
            fit_indices = gas_indices[fit_local]
            conformal_indices = gas_indices[conformal_local]
            source = source_models[gas]
            model = fit_ridge_model(
                data["x84"][gas][fit_indices], data["truth"][fit_indices],
                float(target_models[target][str(gas)]["alpha"]),
                float(source["clip_min"]), float(source["clip_max"]),
            )
            conformal_prediction = predict_ridge_model(model, data["x84"][gas][conformal_indices])
            radius = conformal_radius(
                np.abs(conformal_prediction - data["truth"][conformal_indices]),
                NOMINAL_INTERVAL_COVERAGE,
            )
            width[gas_mask] = 2.0 * radius
            target_lock[str(gas)] = {
                "model": model,
                "radius": radius,
                "nominal_coverage": NOMINAL_INTERVAL_COVERAGE,
                "fit_groups": sorted(set(data["groups"][fit_indices].tolist())),
                "conformal_groups": sorted(set(data["groups"][conformal_indices].tolist())),
                "fit_n": int(len(fit_indices)),
                "conformal_n": int(len(conformal_indices)),
                "group_overlap": False,
            }
        targets[target] = target_lock
        calibration_arrays[target] = {
            "confidence_risk": confidence_risk,
            "interval_width": width,
        }
    return targets, calibration_arrays


def _coverage_rows(scope, truth, lower, upper, true_class, concentration):
    truth = np.asarray(truth, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    true_class = np.asarray(true_class, dtype=int)
    concentration = np.asarray(concentration, dtype=float)
    rows = []
    masks = [("ALL", None, None, np.ones(len(truth), dtype=bool))]
    for gas in sorted(set(true_class.tolist())):
        gas_mask = true_class == gas
        masks.append(("GAS", gas, None, gas_mask))
        for value in sorted(set(concentration[gas_mask].tolist())):
            masks.append(("CONCENTRATION", gas, value, gas_mask & (concentration == value)))
    for slice_name, gas, value, mask in masks:
        covered = (truth[mask] >= lower[mask]) & (truth[mask] <= upper[mask])
        widths = upper[mask] - lower[mask]
        rows.append({
            "scope": scope, "slice": slice_name, "gas_id": "" if gas is None else gas,
            "concentration": "" if value is None else value, "n": int(mask.sum()),
            "empirical_coverage": float(np.mean(covered)),
            "mean_width": float(np.mean(widths)), "median_width": float(np.median(widths)),
        })
    return rows


def _pooled_curve(items, policy, risk_key):
    rows = []
    for coverage in COVERAGE_GRID:
        selected = [(item, retained_indices(item[risk_key], item["physical"], float(coverage))) for item in items]
        truth = np.concatenate([item["truth"][idx] for item, idx in selected])
        prediction = np.concatenate([item["prediction"][idx] for item, idx in selected])
        ranges = np.concatenate([item["ranges"][idx] for item, idx in selected])
        true_class = np.concatenate([item["true_class"][idx] for item, idx in selected])
        predicted_class = np.concatenate([item["predicted_class"][idx] for item, idx in selected])
        error = prediction - truth
        rows.append({
            "policy": policy, "scope": "POOLED", "target_coverage": float(coverage),
            "actual_coverage": len(truth) / sum(len(item["truth"]) for item in items),
            "retained_n": len(truth), "RMSE": float(np.sqrt(np.mean(error**2))),
            "NRMSE_range": float(np.sqrt(np.mean((error / ranges) ** 2))),
            "MAE": float(np.mean(np.abs(error))),
            "misroute_rate": float(np.mean(true_class != predicted_class)),
            "error_ge_40ppm_rate": float(np.mean(np.abs(error) >= 40.0)),
            "P90_absolute_error": float(np.percentile(np.abs(error), 90)),
        })
    return rows


def run(authorized_head: str):
    receipt = preflight(authorized_head)
    FORMAL_ROOT.mkdir(parents=True)
    _write_json(FORMAL_ROOT / "preflight_receipt.json", receipt)

    source_models = _source_models()
    target_lock = _read_json(R1_ROOT / "target_model_lock.json")["models"]
    target_models = {target: target_lock[target]["R84_CONCAT"] for target in TARGETS}
    conformal_models, calibration = _build_calibration_lock(source_models, target_models)
    policy_lock = {
        "study_id": STUDY_ID,
        "backend": "R84_CONCAT",
        "nominal_interval_coverage": NOMINAL_INTERVAL_COVERAGE,
        "split": "deterministic_two_fold_raw_filename_group_aware",
        "interval": "absolute_residual_empirical_prediction_interval",
        "exact_iid_coverage_claim": False,
        "risk_policies": ["CLASSIFICATION_CONFIDENCE_ONLY", "INTERVAL_WIDTH_ONLY", "EQUAL_MEAN_CDF"],
        "combined_weights": {"confidence": 0.5, "interval_width": 0.5},
        "weight_search": False,
        "support_threshold_relative_nrmse_aurc": SUPPORT_THRESHOLD,
        "models": conformal_models,
        "calibration_cdf_references": {
            target: {
                name: np.sort(values).tolist()
                for name, values in references.items()
            }
            for target, references in calibration.items()
        },
        "target_test_opened": False,
    }
    _write_json(FORMAL_ROOT / "conformal_policy_lock.json", policy_lock)
    lock_sha = _sha256(FORMAL_ROOT / "conformal_policy_lock.json")
    _write_json(FORMAL_ROOT / "target_test_release_receipt.json", {"conformal_policy_lock_sha256": lock_sha})

    curves = []
    intervals = []
    coverage_rows = []
    pooled = []
    for target in TARGETS:
        test = _target_inputs(target, "test", source_models, target_models)
        predicted_class = test["predicted_class"]
        final_prediction = test["route"][np.arange(len(predicted_class)), predicted_class]
        split_prediction_matrix = np.column_stack([
            predict_ridge_model(conformal_models[target][str(gas)]["model"], test["x84"][gas])
            for gas in range(4)
        ])
        interval_center = split_prediction_matrix[np.arange(len(predicted_class)), predicted_class]
        radius = np.asarray([conformal_models[target][str(gas)]["radius"] for gas in predicted_class])
        lower, upper = empirical_interval(interval_center, radius)
        width = upper - lower
        confidence_risk = classification_confidence_risk(test["probabilities"])
        confidence_cdf = calibration_ecdf(calibration[target]["confidence_risk"], confidence_risk)
        width_cdf = calibration_ecdf(calibration[target]["interval_width"], width)
        combined = equal_mean_risk(confidence_cdf, width_cdf)
        ranges = np.asarray([
            float(source_models[int(gas)]["clip_max"] - source_models[int(gas)]["clip_min"])
            for gas in test["true_class"]
        ])
        for policy, risk in (
            ("CLASSIFICATION_CONFIDENCE_ONLY", confidence_risk),
            ("INTERVAL_WIDTH_ONLY", width),
            ("EQUAL_MEAN_CDF", combined),
        ):
            policy_curve = risk_coverage_curve(
                test["truth"], final_prediction, ranges, test["true_class"], predicted_class,
                risk, test["physical"], policy,
            )
            for row in policy_curve:
                row["scope"] = target
            curves.extend(policy_curve)
        coverage_rows.extend(_coverage_rows(
            target, test["truth"], lower, upper, test["true_class"], test["truth"]
        ))
        for index in range(len(test["truth"])):
            intervals.append({
                "target": target, "physical_identity": test["physical"][index],
                "raw_filename": test["groups"][index], "true_class": int(test["true_class"][index]),
                "predicted_class": int(predicted_class[index]), "truth": float(test["truth"][index]),
                "final_r84_prediction": float(final_prediction[index]),
                "interval_center": float(interval_center[index]), "lower": float(lower[index]),
                "upper": float(upper[index]), "interval_width": float(width[index]),
                "covered": bool(lower[index] <= test["truth"][index] <= upper[index]),
                "confidence_risk": float(confidence_risk[index]),
                "confidence_cdf": float(confidence_cdf[index]), "interval_width_cdf": float(width_cdf[index]),
                "equal_mean_cdf_risk": float(combined[index]),
            })
        pooled.append({
            "truth": test["truth"], "prediction": final_prediction, "ranges": ranges,
            "true_class": test["true_class"], "predicted_class": predicted_class,
            "physical": test["physical"], "confidence": confidence_risk,
            "width": width, "combined": combined,
        })
    for policy, key in (
        ("CLASSIFICATION_CONFIDENCE_ONLY", "confidence"),
        ("INTERVAL_WIDTH_ONLY", "width"),
        ("EQUAL_MEAN_CDF", "combined"),
    ):
        curves.extend(_pooled_curve(pooled, policy, key))

    _write_csv(FORMAL_ROOT / "conformal_intervals.csv", intervals)
    _write_csv(FORMAL_ROOT / "empirical_interval_coverage.csv", coverage_rows)
    _write_csv(FORMAL_ROOT / "q1_risk_coverage_curves.csv", curves)
    aurc_rows = []
    for scope in (*TARGETS, "POOLED"):
        for policy in ("CLASSIFICATION_CONFIDENCE_ONLY", "INTERVAL_WIDTH_ONLY", "EQUAL_MEAN_CDF"):
            curve = [row for row in curves if row["scope"] == scope and row["policy"] == policy]
            aurc_rows.append({
                "scope": scope, "policy": policy, "AURC_RMSE": aurc(curve, "RMSE"),
                "AURC_NRMSE": aurc(curve, "NRMSE_range"),
            })
    _write_csv(FORMAL_ROOT / "q1_aurc.csv", aurc_rows)
    same_count = [row for row in curves if float(row["target_coverage"]) in {0.90, 0.95}]
    _write_csv(FORMAL_ROOT / "q1_same_count_metrics.csv", same_count)
    keyed = {(row["scope"], row["policy"]): row for row in aurc_rows}
    decision = decide_q1(
        keyed[("C5", "CLASSIFICATION_CONFIDENCE_ONLY")]["AURC_NRMSE"],
        keyed[("C5", "EQUAL_MEAN_CDF")]["AURC_NRMSE"],
        keyed[("POOLED", "CLASSIFICATION_CONFIDENCE_ONLY")]["AURC_NRMSE"],
        keyed[("POOLED", "EQUAL_MEAN_CDF")]["AURC_NRMSE"],
    )
    decision_payload = {
        "study_id": STUDY_ID, "decision": decision, "backend": "R84_CONCAT",
        "c5_relative_nrmse_aurc_improvement": (
            keyed[("C5", "CLASSIFICATION_CONFIDENCE_ONLY")]["AURC_NRMSE"]
            - keyed[("C5", "EQUAL_MEAN_CDF")]["AURC_NRMSE"]
        ) / keyed[("C5", "CLASSIFICATION_CONFIDENCE_ONLY")]["AURC_NRMSE"],
        "pooled_relative_nrmse_aurc_improvement": (
            keyed[("POOLED", "CLASSIFICATION_CONFIDENCE_ONLY")]["AURC_NRMSE"]
            - keyed[("POOLED", "EQUAL_MEAN_CDF")]["AURC_NRMSE"]
        ) / keyed[("POOLED", "CLASSIFICATION_CONFIDENCE_ONLY")]["AURC_NRMSE"],
    }
    _write_json(FORMAL_ROOT / "Q1_DECISION.json", decision_payload)
    (FORMAL_ROOT / "Q1_CONFORMAL_QC_REPORT.md").write_text(
        "# Canonical Q1 conformal-style QC\n\n"
        f"Decision: `{decision}`.\n\n"
        "Intervals are raw-file-group-aware empirical prediction intervals at fixed nominal 90% coverage; "
        "no exact iid coverage guarantee is claimed. Confidence and interval width are normalized by "
        "calibration empirical CDFs and combined with fixed equal weights; no weight search was performed.\n",
        encoding="utf-8",
    )
    index = {
        path.relative_to(FORMAL_ROOT).as_posix(): _sha256(path)
        for path in FORMAL_ROOT.rglob("*")
        if path.is_file() and path.name not in {"sha256_index.json", "COMPLETE.json"}
    }
    _write_json(FORMAL_ROOT / "sha256_index.json", index)
    _write_json(FORMAL_ROOT / "COMPLETE.json", {
        "study_id": STUDY_ID, "status": "PASS", "decision": decision,
    })
    return decision_payload


def audit():
    index = _read_json(FORMAL_ROOT / "sha256_index.json")
    for name, digest in index.items():
        if _sha256(FORMAL_ROOT / name) != digest:
            raise RuntimeError(f"Q1 hash mismatch: {name}")
    lock = _read_json(FORMAL_ROOT / "conformal_policy_lock.json")
    if lock.get("weight_search") is not False or lock.get("combined_weights") != {"confidence": 0.5, "interval_width": 0.5}:
        raise RuntimeError("Q1 policy lock mismatch")
    for target in TARGETS:
        for gas in range(4):
            item = lock["models"][target][str(gas)]
            if set(item["fit_groups"]) & set(item["conformal_groups"]):
                raise RuntimeError("Q1 raw-file group leakage")
    receipt = _read_json(FORMAL_ROOT / "target_test_release_receipt.json")
    if receipt.get("conformal_policy_lock_sha256") != _sha256(FORMAL_ROOT / "conformal_policy_lock.json"):
        raise RuntimeError("Q1 test release lock mismatch")
    return {"status": "PASS", "decision": _read_json(FORMAL_ROOT / "Q1_DECISION.json")["decision"]}


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
