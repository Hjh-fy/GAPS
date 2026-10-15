"""Run the frozen Phase-4 expected downstream-cost router on C5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_regression_head_ablation import CLASS_NAMES
from scripts import run_gaps_cross_target_r84_full as r84_common
from scripts.run_iotj_a0t_vs_a4_regression import (
    DATA_ROOT,
    EXPECTED_DATASET_SHA256,
    EXPECTED_H1_SHA256,
    FROZEN_ALPHAS,
    summarize_scope,
)
from scripts.run_iotj_canonical_v1_r84 import enriched_oracle_rows, route_rows
from scripts.run_iotj_method_breakthrough_gate_c import cost_matrix_rows, load_frozen_r84_models
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


PHASE3_ROOT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/phase3_posthoc_argmax/retry3"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/phase4_cost_aware_routing"
EXPERIMENT_ID = "CAN-V1-MB-P4-COST-ROUTER-S42"
SEED = 42
BOOTSTRAP_REPLICATES = 2000
EXPECTED_CLASSIFIER_SHA256 = "857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def phase4_protocol() -> dict[str, Any]:
    return {
        "phase": "P4_EXPECTED_DOWNSTREAM_COST_ROUTING",
        "dataset": "canonical-v1",
        "target": "C5",
        "classifier": "immutable Phase3 I0+B20 step100",
        "regression": "immutable Phase3 R84_FED_H1_fixed_alpha",
        "cost_matrix_source": "C5 B20 calibration only",
        "cost_formula": "max(0, mean(SE_forced_j-SE_correct_c)); diagonal=0",
        "router": "argmin_j sum_c p(c|x) C(c,j)",
        "lambda": None,
        "threshold": None,
        "bootstrap_group": "raw_filename",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "target_test_used_for_cost_matrix": False,
        "target_test_checkpoint_selection": False,
        "hyperparameter_search": False,
        "decision_thresholds": {
            "supported_relative_rmse_improvement": 0.05,
            "modest_relative_rmse_improvement": 0.02,
            "maximum_macro_f1_drop": 0.005,
            "supported_probability_delta_negative": 0.5,
        },
    }


def expected_cost_routes(probabilities: np.ndarray, costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    if probabilities.ndim != 2 or costs.shape != (probabilities.shape[1], probabilities.shape[1]):
        raise ValueError("probability/cost shape mismatch")
    if not np.isfinite(probabilities).all() or not np.isfinite(costs).all():
        raise ValueError("non-finite probability/cost")
    risks = probabilities @ costs
    return np.argmin(risks, axis=1).astype(np.int64), risks


def decide_cost_router(
    *,
    argmax_rmse: float,
    cost_rmse: float,
    argmax_macro_f1: float,
    cost_macro_f1: float,
    bootstrap_probability_negative: float,
) -> dict[str, Any]:
    relative = (float(argmax_rmse) - float(cost_rmse)) / float(argmax_rmse)
    f1_drop = float(argmax_macro_f1) - float(cost_macro_f1)
    if relative >= 0.05 and f1_drop <= 0.005 and bootstrap_probability_negative > 0.5:
        decision = "COST_AWARE_ROUTING_SUPPORTED"
    elif 0.02 <= relative < 0.05 and f1_drop <= 0.005:
        decision = "COST_AWARE_ROUTING_MODEST"
    elif relative >= 0.02 and f1_drop > 0.005:
        decision = "QUANTITATIVE_GAIN_WITH_CLASSIFICATION_COST"
    else:
        decision = "COST_AWARE_ROUTING_NOT_SUPPORTED"
    return {
        "decision": decision,
        "relative_rmse_improvement": relative,
        "macro_f1_drop": f1_drop,
        "bootstrap_probability_delta_negative": float(bootstrap_probability_negative),
    }


def grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED
) -> list[dict[str, Any]]:
    filenames = sorted({str(row["filename"]) for row in rows})
    if len(filenames) < 2:
        raise RuntimeError("FAIL_CLOSED grouped bootstrap needs at least two raw filenames")
    groups = {name: [row for row in rows if str(row["filename"]) == name] for name in filenames}
    rng = np.random.RandomState(seed)
    output: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled_names = rng.choice(filenames, size=len(filenames), replace=True)
        sampled = [row for name in sampled_names for row in groups[str(name)]]
        truth = np.asarray([float(row["true_ppm"]) for row in sampled], dtype=np.float64)
        argmax = np.asarray([float(row["argmax_pred_ppm"]) for row in sampled], dtype=np.float64)
        cost = np.asarray([float(row["cost_pred_ppm"]) for row in sampled], dtype=np.float64)
        output.append(
            {
                "replicate": replicate,
                "sampled_raw_file_count": len(filenames),
                "sampled_window_count": len(sampled),
                "cost_minus_argmax_rmse": float(
                    np.sqrt(np.mean((cost - truth) ** 2)) - np.sqrt(np.mean((argmax - truth) ** 2))
                ),
            }
        )
    return output


def _hard_classification_metrics(true: Sequence[int], pred: Sequence[int]) -> dict[str, Any]:
    truth = np.asarray(true, dtype=np.int64)
    predicted = np.asarray(pred, dtype=np.int64)
    if truth.shape != predicted.shape or len(truth) == 0:
        raise ValueError("invalid hard classification arrays")
    confusion = np.zeros((4, 4), dtype=np.int64)
    for value, route in zip(truth, predicted):
        confusion[int(value), int(route)] += 1
    f1_values: list[float] = []
    for class_id in range(4):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "N": int(len(truth)),
        "accuracy": float(np.mean(truth == predicted)),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": confusion.tolist(),
    }


def _matrix_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.full((4, 4), np.nan, dtype=np.float64)
    for row in rows:
        matrix[int(row["true_class"]), int(row["forced_route"])] = float(row["primary_cost_squared_ppm"])
    if not np.isfinite(matrix).all() or not np.allclose(np.diag(matrix), 0.0):
        raise RuntimeError("FAIL_CLOSED expected cost matrix incomplete")
    return matrix


def _audit_phase3_inputs() -> dict[str, Any]:
    dataset = verify_dataset(DATA_ROOT)
    if dataset.get("status") != "PASS" or dataset.get("aggregate_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("FAIL_CLOSED canonical-v1 dataset differs")
    freeze = json.loads((PHASE3_ROOT / "PRE_RUN_FREEZE.json").read_text(encoding="utf-8"))
    lock = json.loads((PHASE3_ROOT / "endpoint/calibration_lock.json").read_text(encoding="utf-8"))
    marker = json.loads((PHASE3_ROOT / "endpoint/fixed_endpoint_complete.json").read_text(encoding="utf-8"))
    checkpoint = Path(freeze["classifier"]["checkpoint"])
    model_path = PHASE3_ROOT / "endpoint/r84_models.json"
    if (
        freeze.get("status") != "FROZEN"
        or freeze["selection"].get("identity") != "I0"
        or int(freeze["classifier"].get("step", -1)) != 100
        or freeze["classifier"].get("checkpoint_sha256") != EXPECTED_CLASSIFIER_SHA256
        or sha256_file(checkpoint) != EXPECTED_CLASSIFIER_SHA256
        or lock.get("target_test_opened") is not False
        or lock.get("r84_models_sha256") != sha256_file(model_path)
        or lock.get("h1_sha256") != EXPECTED_H1_SHA256
        or marker.get("status") != "COMPLETE"
        or marker.get("target_test_used_for_selection") is not False
    ):
        raise RuntimeError("FAIL_CLOSED Phase3 input provenance differs")
    return {
        "status": "PASS",
        "classifier_checkpoint": str(checkpoint.resolve()),
        "classifier_sha256": EXPECTED_CLASSIFIER_SHA256,
        "classifier_state_fingerprint": freeze["classifier"]["checkpoint_state_fingerprint"],
        "r84_models": str(model_path.resolve()),
        "r84_models_sha256": sha256_file(model_path),
        "h1_sha256": EXPECTED_H1_SHA256,
        "dataset_aggregate_sha256": EXPECTED_DATASET_SHA256,
        "phase3_fixed_marker_sha256": sha256_file(PHASE3_ROOT / "endpoint/fixed_endpoint_complete.json"),
    }


def write_pre_run_freeze(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED Phase4 output exists: {output}")
    inputs = _audit_phase3_inputs()
    output.mkdir(parents=True)
    freeze = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.phase4.freeze.v1",
        "status": "FROZEN_BEFORE_COST_MATRIX",
        "producer_commit": _git_head(),
        "protocol": phase4_protocol(),
        "inputs": inputs,
        "target_test_semantic_access": False,
        "hash_only_dataset_provenance_check": True,
    }
    _json(output / "PRE_RUN_FREEZE.json", freeze)
    return freeze


def _forced_predictions(
    split: str,
    routes: Sequence[Mapping[str, Any]],
    models: Mapping[int, Any],
    h1: Mapping[int, Any],
) -> list[dict[str, Any]]:
    base = enriched_oracle_rows("C5", split)
    if len(base) != len(routes):
        raise RuntimeError(f"FAIL_CLOSED C5/{split} route count differs")
    output: list[dict[str, Any]] = []
    for row, route in zip(base, routes):
        if int(row["sample_index"]) != int(route["sample_index"]):
            raise RuntimeError(f"FAIL_CLOSED C5/{split} sample alignment differs")
        for forced_route in range(4):
            item = {**dict(row), "pred_class": forced_route}
            item["H1_federated_source_ridge_ppm"] = float(h1[forced_route].predict(row["feature_dict"]))
            item = r84_common.r84_row(item)
            prediction = float(models[forced_route].predict([item])[0])
            output.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "physical_identity": row["physical_identity"],
                    "filename": row["filename"],
                    "true_class": int(row["true_class"]),
                    "true_gas": CLASS_NAMES[int(row["true_class"])],
                    "forced_route": forced_route,
                    "forced_route_gas": CLASS_NAMES[forced_route],
                    "true_ppm": float(row["true_ppm"]),
                    "pred_ppm": prediction,
                    "squared_error": (prediction - float(row["true_ppm"])) ** 2,
                }
            )
    return output


def build_and_lock_cost_matrix(output: Path, freeze: Mapping[str, Any], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    models = load_frozen_r84_models(Path(freeze["inputs"]["r84_models"]))
    checkpoint = Path(freeze["inputs"]["classifier_checkpoint"])
    routes, _metrics = route_rows(
        checkpoint, "C5", "calibration", device, batch_size, expected_endpoint=("step", 100)
    )
    if len(routes) != 320:
        raise RuntimeError("FAIL_CLOSED C5 B20 calibration count differs")
    forced = _forced_predictions("calibration", routes, models, r84_common.load_h1())
    matrix = cost_matrix_rows(forced)
    if len(forced) != 1280 or len(matrix) != 16:
        raise RuntimeError("FAIL_CLOSED Phase4 calibration matrix dimensions differ")
    forced_path = output / "CALIBRATION_FORCED_ROUTE_PREDICTIONS.csv"
    matrix_path = output / "EXPECTED_COST_MATRIX.csv"
    _csv(forced_path, forced)
    _csv(matrix_path, matrix)
    _json(
        output / "CALIBRATION_COST_MATRIX_LOCK.json",
        {
            "status": "LOCKED_BEFORE_TARGET_TEST",
            "target_test_opened": False,
            "calibration_N": 320,
            "forced_prediction_N": 1280,
            "cost_matrix_sha256": sha256_file(matrix_path),
            "forced_predictions_sha256": sha256_file(forced_path),
            "classifier_sha256": freeze["inputs"]["classifier_sha256"],
            "r84_models_sha256": freeze["inputs"]["r84_models_sha256"],
            "cost_formula": phase4_protocol()["cost_formula"],
            "router": phase4_protocol()["router"],
            "lambda": None,
            "threshold": None,
        },
    )
    return matrix


def require_cost_matrix_lock(output: Path) -> dict[str, Any]:
    lock_path = Path(output) / "CALIBRATION_COST_MATRIX_LOCK.json"
    matrix_path = Path(output) / "EXPECTED_COST_MATRIX.csv"
    if not lock_path.is_file() or not matrix_path.is_file():
        raise RuntimeError("FAIL_CLOSED cost matrix lock missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "LOCKED_BEFORE_TARGET_TEST"
        or lock.get("target_test_opened") is not False
        or lock.get("cost_matrix_sha256") != sha256_file(matrix_path)
    ):
        raise RuntimeError("FAIL_CLOSED cost matrix lock invalid")
    return {**lock, "status": "PASS", "lock_sha256": sha256_file(lock_path)}


def _policy_rows(
    routes: Sequence[Mapping[str, Any]], forced: Sequence[Mapping[str, Any]], matrix: np.ndarray
) -> list[dict[str, Any]]:
    forced_by_sample: dict[int, dict[int, Mapping[str, Any]]] = {}
    for row in forced:
        forced_by_sample.setdefault(int(row["sample_index"]), {})[int(row["forced_route"])] = row
    probabilities = np.asarray(
        [[float(row[f"prob_{class_id}"]) for class_id in range(4)] for row in routes], dtype=np.float64
    )
    cost_routes, risks = expected_cost_routes(probabilities, matrix)
    output: list[dict[str, Any]] = []
    for index, route in enumerate(routes):
        sample_index = int(route["sample_index"])
        candidates = forced_by_sample[sample_index]
        if set(candidates) != {0, 1, 2, 3}:
            raise RuntimeError("FAIL_CLOSED forced test routes incomplete")
        true_class = int(route["true_class"])
        argmax_route = int(route["pred_class"])
        cost_route = int(cost_routes[index])
        truth = float(candidates[0]["true_ppm"])
        argmax_pred = float(candidates[argmax_route]["pred_ppm"])
        cost_pred = float(candidates[cost_route]["pred_ppm"])
        item = {
            "sample_index": sample_index,
            "physical_identity": candidates[0]["physical_identity"],
            "filename": candidates[0]["filename"],
            "true_class": true_class,
            "true_gas": CLASS_NAMES[true_class],
            "true_ppm": truth,
            "argmax_route": argmax_route,
            "cost_route": cost_route,
            "argmax_route_correct": int(argmax_route == true_class),
            "cost_route_correct": int(cost_route == true_class),
            "argmax_pred_ppm": argmax_pred,
            "cost_pred_ppm": cost_pred,
            "argmax_abs_error": abs(argmax_pred - truth),
            "cost_abs_error": abs(cost_pred - truth),
            "argmax_squared_error": (argmax_pred - truth) ** 2,
            "cost_squared_error": (cost_pred - truth) ** 2,
        }
        for class_id in range(4):
            item[f"prob_class_{class_id}"] = float(probabilities[index, class_id])
            item[f"expected_risk_route_{class_id}"] = float(risks[index, class_id])
        output.append(item)
    return output


def _regression_summary(rows: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    converted = [
        {"true_ppm": row["true_ppm"], "true_class": row["true_class"], "pred_ppm": row[f"{policy}_pred_ppm"]}
        for row in rows
    ]
    return summarize_scope(converted)


def _concentration_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(int(row["true_class"]), float(row["true_ppm"])) for row in rows})
    for class_id, concentration in keys:
        selected = [
            row for row in rows if int(row["true_class"]) == class_id and float(row["true_ppm"]) == concentration
        ]
        for policy in ("argmax", "cost"):
            errors = np.asarray(
                [float(row[f"{policy}_pred_ppm"]) - float(row["true_ppm"]) for row in selected], dtype=np.float64
            )
            output.append(
                {
                    "policy": policy,
                    "class_id": class_id,
                    "gas": CLASS_NAMES[class_id],
                    "true_ppm": concentration,
                    "N": len(selected),
                    "RMSE": float(np.sqrt(np.mean(errors**2))),
                    "MAE": float(np.mean(np.abs(errors))),
                    "Bias": float(np.mean(errors)),
                }
            )
    return output


def evaluate_after_lock(
    output: Path,
    freeze: Mapping[str, Any],
    matrix_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    lock = require_cost_matrix_lock(output)
    _json(
        output / "SEALED_TEST_OPEN.json",
        {
            "status": "OPENED_AFTER_COST_MATRIX_LOCK",
            "cost_matrix_lock_sha256": lock["lock_sha256"],
            "target_test_checkpoint_selection": False,
        },
    )
    models = load_frozen_r84_models(Path(freeze["inputs"]["r84_models"]))
    checkpoint = Path(freeze["inputs"]["classifier_checkpoint"])
    routes, _classifier_metrics = route_rows(
        checkpoint, "C5", "test", device, batch_size, expected_endpoint=("step", 100)
    )
    if len(routes) != 1360:
        raise RuntimeError("FAIL_CLOSED C5 sealed-test count differs")
    forced = _forced_predictions("test", routes, models, r84_common.load_h1())
    rows = _policy_rows(routes, forced, _matrix_array(matrix_rows))
    _csv(output / "EXPECTED_COST_ROUTING_PREDICTIONS.csv", rows)

    truth_classes = [int(row["true_class"]) for row in rows]
    argmax_routes = [int(row["argmax_route"]) for row in rows]
    cost_routes = [int(row["cost_route"]) for row in rows]
    argmax_class = _hard_classification_metrics(truth_classes, argmax_routes)
    cost_class = _hard_classification_metrics(truth_classes, cost_routes)
    argmax_reg = _regression_summary(rows, "argmax")
    cost_reg = _regression_summary(rows, "cost")

    high = [row for row in rows if float(row["true_ppm"]) >= 40.0]
    high_rows: list[dict[str, Any]] = []
    for policy in ("argmax", "cost"):
        absolute = np.asarray([float(row[f"{policy}_abs_error"]) for row in high], dtype=np.float64)
        squared = np.asarray([float(row[f"{policy}_squared_error"]) for row in high], dtype=np.float64)
        high_rows.append(
            {
                "policy": policy,
                "threshold_ppm": 40.0,
                "N": len(high),
                "RMSE": float(np.sqrt(np.mean(squared))),
                "MAE": float(np.mean(absolute)),
                "P90_abs_error": float(np.quantile(absolute, 0.90)),
                "max_abs_error": float(np.max(absolute)),
                "SSE": float(np.sum(squared)),
            }
        )
    _csv(output / "HIGH_CONCENTRATION_ERROR.csv", high_rows)
    _csv(output / "CONCENTRATION_RMSE_CURVE.csv", _concentration_rows(rows))

    bootstrap = grouped_bootstrap(rows)
    _csv(output / "GROUPED_RAW_FILENAME_BOOTSTRAP.csv", bootstrap)
    deltas = np.asarray([float(row["cost_minus_argmax_rmse"]) for row in bootstrap], dtype=np.float64)
    bootstrap_summary = {
        "replicates": len(deltas),
        "seed": SEED,
        "mean_delta_rmse": float(np.mean(deltas)),
        "median_delta_rmse": float(np.median(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_delta_negative": float(np.mean(deltas < 0.0)),
    }
    _json(output / "GROUPED_BOOTSTRAP_SUMMARY.json", bootstrap_summary)
    decision = decide_cost_router(
        argmax_rmse=argmax_reg["RMSE"],
        cost_rmse=cost_reg["RMSE"],
        argmax_macro_f1=argmax_class["macro_f1"],
        cost_macro_f1=cost_class["macro_f1"],
        bootstrap_probability_negative=bootstrap_summary["probability_delta_negative"],
    )
    decision.update(
        {
            "argmax_classification": argmax_class,
            "cost_classification": cost_class,
            "argmax_regression": argmax_reg,
            "cost_regression": cost_reg,
            "cost_minus_argmax_rmse": float(cost_reg["RMSE"] - argmax_reg["RMSE"]),
            "high_concentration_cost_minus_argmax_rmse": float(high_rows[1]["RMSE"] - high_rows[0]["RMSE"]),
            "high_concentration_cost_minus_argmax_sse": float(high_rows[1]["SSE"] - high_rows[0]["SSE"]),
            "route_changes": int(sum(a != b for a, b in zip(argmax_routes, cost_routes))),
            "bootstrap": bootstrap_summary,
            "target_test_checkpoint_selection": False,
        }
    )
    _json(output / "PHASE4_DECISION.json", decision)
    _csv(
        output / "EXPECTED_COST_ROUTING_COMPARISON.csv",
        [
            {"policy": "Argmax", **argmax_class, **argmax_reg},
            {"policy": "ExpectedCost", **cost_class, **cost_reg},
        ],
    )
    return decision


def finalize(output: Path, freeze: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    report = f"""# Phase 4 Expected Downstream-Cost Routing

## Protocol

The 4x4 cost matrix was estimated only from the frozen 320-window C5 B20 calibration set. The policy is the parameter-free rule `argmin_j sum_c p(c|x) C(c,j)`. The matrix was hashed before the sealed C5 test opened. No lambda, threshold, checkpoint, or hyperparameter search was performed.

## Result

- Argmax RMSE / Macro-F1: {decision['argmax_regression']['RMSE']:.6f} ppm / {decision['argmax_classification']['macro_f1']:.6f}
- Expected-cost RMSE / Macro-F1: {decision['cost_regression']['RMSE']:.6f} ppm / {decision['cost_classification']['macro_f1']:.6f}
- Relative RMSE improvement: {decision['relative_rmse_improvement']:.6%}
- Macro-F1 drop: {decision['macro_f1_drop']:.6f}
- Route changes: {decision['route_changes']}
- Grouped raw-filename bootstrap P(delta RMSE < 0): {decision['bootstrap']['probability_delta_negative']:.6f}
- Bootstrap 95% interval: [{decision['bootstrap']['ci95_low']:.6f}, {decision['bootstrap']['ci95_high']:.6f}] ppm

## Decision

`{decision['decision']}`

The interval crossing zero is reported descriptively and is not an additional hard gate beyond the preregistered decision rule.
"""
    (output / "PHASE4_RESULT_ANALYSIS.md").write_text(report, encoding="utf-8")
    (output / "PHASE4_EXPERIMENT_AUDIT.md").write_text(
        "# Phase 4 experiment audit\n\n## Verdict: PASS\n\n"
        "- canonical-v1, classifier, H1, and R84 hashes match the frozen Phase 3 baseline.\n"
        "- The expected-cost matrix uses calibration true class/concentration only and was locked before semantic target-test access.\n"
        "- Test data was used once for fixed-policy evaluation only.\n"
        "- No model fitting, alpha search, threshold, lambda, checkpoint selection, QC, or algorithm search occurred.\n"
        "- Grouped bootstrap used complete raw filenames, 2000 replicates, and seed42.\n",
        encoding="utf-8",
    )
    _json(
        output / "fixed_endpoint_complete.json",
        {
            "status": "COMPLETE",
            "experiment_id": EXPERIMENT_ID,
            "decision": decision["decision"],
            "target_test_checkpoint_selection": False,
        },
    )
    manifest = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.phase4.protocol.v1",
        "status": "PASS",
        "producer_commit": freeze["producer_commit"],
        "protocol": phase4_protocol(),
        "inputs": freeze["inputs"],
        "cost_matrix_sha256": sha256_file(output / "EXPECTED_COST_MATRIX.csv"),
        "prediction_sha256": sha256_file(output / "EXPECTED_COST_ROUTING_PREDICTIONS.csv"),
        "decision": decision,
    }
    _json(output / "protocol_manifest.json", manifest)
    excluded = {"sha256_index.json", "runner.pid", "runner.stdout.log", "runner.stderr.log"}
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in excluded)
    _json(output / "sha256_index.json", {str(path.relative_to(output)).replace("\\", "/"): sha256_file(path) for path in files})


def run(output: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    output = output.resolve()
    freeze = write_pre_run_freeze(output)
    matrix = build_and_lock_cost_matrix(output, freeze, device, batch_size)
    decision = evaluate_after_lock(output, freeze, matrix, device, batch_size)
    finalize(output, freeze, decision)
    return {"status": "PASS", **decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    print(json.dumps(run(args.output, device, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
