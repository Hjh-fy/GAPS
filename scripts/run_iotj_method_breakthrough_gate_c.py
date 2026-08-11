"""Run the frozen calibration-only Gate C downstream routing-cost audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_regression_head_ablation import CLASS_NAMES, CLASS_RANGES, RidgeHead  # noqa: E402
from scripts import run_gaps_cross_target_r84_full as r84_common  # noqa: E402
from scripts.run_iotj_a0t_vs_a4_regression import (  # noqa: E402
    EXPECTED_DATASET_SHA256,
    EXPECTED_H1_SHA256,
    FROZEN_ALPHAS,
    audit_checkpoint,
    endpoint_specs,
)
from scripts.run_iotj_canonical_v1_r84 import enriched_oracle_rows  # noqa: E402
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset  # noqa: E402


DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
REGRESSION_ROOT = ROOT / "results/iotj_canonical_v1_final/a0t_vs_a4_regression/endpoints"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811/gate_c_downstream_route_cost"
METHOD_ROOT = ROOT / "results/iotj_canonical_v1_method_breakthrough_20260811"
METHOD_DIRS = {
    "A0T": REGRESSION_ROOT / "CAN-V1-REG-A0T-C5-S42",
    "A4": REGRESSION_ROOT / "CAN-V1-REG-A4-C5-S42",
}
BOOTSTRAP_REPLICATES = 2000
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def gate_c_protocol() -> dict[str, Any]:
    return {
        "gate": "C_DOWNSTREAM_ROUTING_COST_AUDIT",
        "dataset": "canonical-v1",
        "target": "C5",
        "cost_matrix_source": "C5_canonical_calibration_only",
        "primary_cost": "max_0_mean_incremental_squared_ppm",
        "secondary_cost": "mean_range_normalized_incremental_squared",
        "bootstrap_group": "raw_filename",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "target_test_used_for_cost_matrix": False,
        "hyperparameter_search": False,
        "decision_thresholds": {
            "minimum_positive_offdiagonal_cells": 2,
            "minimum_positive_cost_cv": 0.5,
            "minimum_positive_contribution_files": 2,
            "maximum_top_file_positive_share": 0.8,
            "minimum_actual_misroute_windows": 4,
        },
    }


def _ridge_from_json(payload: Mapping[str, Any]) -> RidgeHead:
    return RidgeHead(
        alpha=float(payload["alpha"]),
        feature_names=[str(value) for value in payload["feature_names"]],
        mean=np.asarray(payload["mean"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
        coef=np.asarray(payload["coef"], dtype=np.float64),
        clip_min=float(payload["clip_min"]),
        clip_max=float(payload["clip_max"]),
    )


def load_frozen_r84_models(path: Path) -> dict[int, RidgeHead]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"0", "1", "2", "3"}:
        raise RuntimeError("FAIL_CLOSED R84 model classes differ")
    models = {int(key): _ridge_from_json(value) for key, value in payload.items()}
    observed = {class_id: model.alpha for class_id, model in models.items()}
    if observed != FROZEN_ALPHAS["C5"]:
        raise RuntimeError("FAIL_CLOSED frozen C5 alpha differs")
    return models


def calibration_forced_predictions(models: Mapping[int, RidgeHead]) -> list[dict[str, Any]]:
    base = enriched_oracle_rows("C5", "calibration")
    if len(base) != 320:
        raise RuntimeError("FAIL_CLOSED C5 calibration count differs")
    h1 = r84_common.load_h1()
    output: list[dict[str, Any]] = []
    for row in base:
        true_class = int(row["true_class"])
        for forced_route in range(4):
            item = dict(row)
            item["pred_class"] = forced_route
            item["H1_federated_source_ridge_ppm"] = float(
                h1[forced_route].predict(row["feature_dict"])
            )
            item = r84_common.r84_row(item)
            prediction = float(models[forced_route].predict([item])[0])
            output.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "physical_identity": row["physical_identity"],
                    "filename": row["filename"],
                    "true_class": true_class,
                    "true_gas": CLASS_NAMES[true_class],
                    "forced_route": forced_route,
                    "forced_route_gas": CLASS_NAMES[forced_route],
                    "true_ppm": float(row["true_ppm"]),
                    "pred_ppm": prediction,
                    "squared_error": (prediction - float(row["true_ppm"])) ** 2,
                }
            )
    return output


def cost_matrix_rows(
    forced_rows: Sequence[Mapping[str, Any]], *, class_ranges: Mapping[int, float] = CLASS_RANGES
) -> list[dict[str, Any]]:
    correct: dict[tuple[int, int], float] = {}
    for row in forced_rows:
        true_class = int(row["true_class"])
        route = int(row["forced_route"])
        if route == true_class:
            correct[(true_class, int(row.get("sample_index", len(correct))))] = (
                float(row["pred_ppm"]) - float(row["true_ppm"])
            ) ** 2
    output: list[dict[str, Any]] = []
    true_classes = sorted({int(row["true_class"]) for row in forced_rows})
    for true_class in true_classes:
        routes = sorted(
            {int(row["forced_route"]) for row in forced_rows if int(row["true_class"]) == true_class}
        )
        for route in routes:
            selected = [
                row
                for row in forced_rows
                if int(row["true_class"]) == true_class and int(row["forced_route"]) == route
            ]
            increments = []
            for position, row in enumerate(selected):
                sample_index = int(row.get("sample_index", position))
                base = correct[(true_class, sample_index)]
                forced = (float(row["pred_ppm"]) - float(row["true_ppm"])) ** 2
                increments.append(forced - base)
            values = np.asarray(increments, dtype=np.float64)
            raw_mean = float(np.mean(values))
            diagonal = route == true_class
            output.append(
                {
                    "true_class": true_class,
                    "true_gas": CLASS_NAMES.get(true_class, str(true_class)),
                    "forced_route": route,
                    "forced_route_gas": CLASS_NAMES.get(route, str(route)),
                    "N": int(len(values)),
                    "mean_incremental_squared_ppm": 0.0 if diagonal else raw_mean,
                    "primary_cost_squared_ppm": 0.0 if diagonal else max(0.0, raw_mean),
                    "median_incremental_squared_ppm": 0.0 if diagonal else float(np.median(values)),
                    "p90_incremental_squared_ppm": 0.0 if diagonal else float(np.quantile(values, 0.90)),
                    "mean_range_normalized_incremental_squared": (
                        0.0 if diagonal else float(np.mean(values / float(class_ranges[true_class]) ** 2))
                    ),
                }
            )
    return output


def decide_gate_c(
    *,
    positive_offdiagonal_costs: Sequence[float],
    positive_contribution_files: int,
    top_file_share: float,
    actual_misroute_windows: int,
) -> dict[str, Any]:
    values = np.asarray(list(positive_offdiagonal_costs), dtype=np.float64)
    cv = float(np.std(values) / np.mean(values)) if len(values) and np.mean(values) > 0 else 0.0
    heterogeneous = bool(len(values) >= 2 and cv >= 0.5)
    distributed = bool(
        int(positive_contribution_files) >= 2
        and float(top_file_share) < 0.8
        and int(actual_misroute_windows) >= 4
    )
    decision = (
        "COST_AWARE_ROUTING_MOTIVATED"
        if heterogeneous and distributed
        else "COST_AWARE_ROUTING_NOT_SUPPORTED"
    )
    return {
        "decision": decision,
        "next_action": "GO_GATE_D" if decision == "COST_AWARE_ROUTING_MOTIVATED" else "STOP_COST_AWARE",
        "positive_offdiagonal_cells": int(len(values)),
        "positive_cost_cv": cv,
        "positive_contribution_files": int(positive_contribution_files),
        "top_file_positive_share": float(top_file_share),
        "actual_misroute_windows": int(actual_misroute_windows),
        "heterogeneous_costs": heterogeneous,
        "distributed_observed_contribution": distributed,
    }


def grouped_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED
) -> list[dict[str, Any]]:
    filenames = sorted({str(row["filename"]) for row in rows})
    if len(filenames) < 2:
        raise RuntimeError("FAIL_CLOSED grouped bootstrap needs at least two raw files")
    groups = {name: [row for row in rows if str(row["filename"]) == name] for name in filenames}
    rng = np.random.RandomState(seed)
    output: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled_names = rng.choice(filenames, size=len(filenames), replace=True)
        sampled = [row for name in sampled_names for row in groups[str(name)]]
        truth = np.asarray([float(row["true_ppm"]) for row in sampled])
        a0t = np.asarray([float(row["a0t_pred_ppm"]) for row in sampled])
        a4 = np.asarray([float(row["a4_pred_ppm"]) for row in sampled])
        a0t_excess = np.asarray([float(row["a0t_excess_se"]) for row in sampled])
        a4_excess = np.asarray([float(row["a4_excess_se"]) for row in sampled])
        output.append(
            {
                "replicate": replicate,
                "sampled_file_count": len(filenames),
                "sampled_window_count": len(sampled),
                "a4_minus_a0t_rmse": float(
                    np.sqrt(np.mean((a4 - truth) ** 2)) - np.sqrt(np.mean((a0t - truth) ** 2))
                ),
                "a4_minus_a0t_mean_excess_squared_error": float(
                    np.mean(a4_excess) - np.mean(a0t_excess)
                ),
            }
        )
    return output


def _bootstrap_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for metric in ("a4_minus_a0t_rmse", "a4_minus_a0t_mean_excess_squared_error"):
        values = np.asarray([float(row[metric]) for row in rows])
        output.append(
            {
                "metric": metric,
                "replicates": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_negative": float(np.mean(values < 0)),
            }
        )
    return output


def write_pre_run_freeze(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED Gate C output exists: {output}")
    dataset = verify_dataset(DATA_ROOT)
    if dataset.get("status") != "PASS" or dataset.get("aggregate_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("FAIL_CLOSED canonical-v1 hash differs")
    r84_hashes = {
        method: sha256_file(directory / "r84_models.json") for method, directory in METHOD_DIRS.items()
    }
    if len(set(r84_hashes.values())) != 1:
        raise RuntimeError("FAIL_CLOSED A0T/A4 frozen R84 models differ")
    endpoints: dict[str, Any] = {}
    specs = [spec for spec in endpoint_specs() if spec.target == "C5"]
    for spec in specs:
        endpoints[spec.method] = audit_checkpoint(spec)
        directory = METHOD_DIRS[spec.method]
        lock = json.loads((directory / "calibration_lock.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "endpoint_manifest.json").read_text(encoding="utf-8"))
        if lock.get("target_test_opened") is not False or lock.get("calibration_N") != 320:
            raise RuntimeError(f"FAIL_CLOSED invalid calibration lock: {spec.method}")
        if lock.get("r84_models_sha256") != r84_hashes[spec.method]:
            raise RuntimeError(f"FAIL_CLOSED R84 lock hash differs: {spec.method}")
        if manifest.get("target_test_used_for_selection") is not False:
            raise RuntimeError(f"FAIL_CLOSED target-test selection flag differs: {spec.method}")
    output.mkdir(parents=True)
    freeze = {
        "schema_version": "iotj.canonical_v1.method_breakthrough.gate_c.freeze.v1",
        "status": "FROZEN_BEFORE_COST_MATRIX",
        "producer_commit": _git_head(),
        "protocol": gate_c_protocol(),
        "dataset_aggregate_sha256": EXPECTED_DATASET_SHA256,
        "h1_sha256": EXPECTED_H1_SHA256,
        "calibration_manifest_sha256": sha256_file(
            DATA_ROOT / "client_5/calibration_experiment_info.json"
        ),
        "r84_models_sha256": r84_hashes,
        "classifier_checkpoints": endpoints,
        "target_test_data_used": False,
    }
    _json(output / "PRE_RUN_FREEZE.json", freeze)
    return freeze


def build_and_lock_cost_matrix(output: Path, freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    models = load_frozen_r84_models(METHOD_DIRS["A0T"] / "r84_models.json")
    forced = calibration_forced_predictions(models)
    matrix = cost_matrix_rows(forced)
    forced_path = output / "CALIBRATION_FORCED_ROUTE_PREDICTIONS.csv"
    matrix_path = output / "DOWNSTREAM_ROUTE_COST_MATRIX.csv"
    _csv(forced_path, forced)
    _csv(matrix_path, matrix)
    _json(
        output / "CALIBRATION_COST_MATRIX_LOCK.json",
        {
            "status": "LOCKED_BEFORE_TEST_DIAGNOSTIC",
            "target_test_opened": False,
            "calibration_N": 320,
            "forced_prediction_N": len(forced),
            "forced_predictions_sha256": sha256_file(forced_path),
            "cost_matrix_sha256": sha256_file(matrix_path),
            "r84_models_sha256": freeze["r84_models_sha256"],
            "h1_sha256": freeze["h1_sha256"],
            "primary_cost": gate_c_protocol()["primary_cost"],
        },
    )
    return matrix


def require_cost_matrix_lock(output: Path) -> dict[str, Any]:
    path = output / "CALIBRATION_COST_MATRIX_LOCK.json"
    if not path.is_file():
        raise RuntimeError("FAIL_CLOSED cost matrix lock missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    matrix = output / "DOWNSTREAM_ROUTE_COST_MATRIX.csv"
    forced = output / "CALIBRATION_FORCED_ROUTE_PREDICTIONS.csv"
    if (
        lock.get("status") != "LOCKED_BEFORE_TEST_DIAGNOSTIC"
        or lock.get("target_test_opened") is not False
        or not matrix.is_file()
        or not forced.is_file()
        or sha256_file(matrix) != lock.get("cost_matrix_sha256")
        or sha256_file(forced) != lock.get("forced_predictions_sha256")
    ):
        raise RuntimeError("FAIL_CLOSED cost matrix lock invalid")
    return lock


def _align_test_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_tables: dict[str, tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]] = {}
    for method, directory in METHOD_DIRS.items():
        manifest = json.loads((directory / "endpoint_manifest.json").read_text(encoding="utf-8"))
        s_all_path = directory / "test_s_all.csv"
        oracle_path = directory / "test_oracle_all.csv"
        if sha256_file(s_all_path) != manifest["prediction_sha256"]["S_ALL"]:
            raise RuntimeError(f"FAIL_CLOSED S_ALL prediction hash differs: {method}")
        if sha256_file(oracle_path) != manifest["prediction_sha256"]["Oracle_ALL"]:
            raise RuntimeError(f"FAIL_CLOSED Oracle prediction hash differs: {method}")
        s_all = {row["physical_identity"]: row for row in _read_csv(s_all_path)}
        oracle = {row["physical_identity"]: row for row in _read_csv(oracle_path)}
        if set(s_all) != set(oracle) or len(s_all) != 1360:
            raise RuntimeError(f"FAIL_CLOSED test identity mismatch: {method}")
        method_tables[method] = (s_all, oracle)
    identities = sorted(method_tables["A0T"][0])
    if set(identities) != set(method_tables["A4"][0]):
        raise RuntimeError("FAIL_CLOSED A0T/A4 test identities differ")
    paired: list[dict[str, Any]] = []
    misroutes: list[dict[str, Any]] = []
    for identity in identities:
        a0t, a0t_oracle = method_tables["A0T"][0][identity], method_tables["A0T"][1][identity]
        a4, a4_oracle = method_tables["A4"][0][identity], method_tables["A4"][1][identity]
        if float(a0t["true_ppm"]) != float(a4["true_ppm"]):
            raise RuntimeError("FAIL_CLOSED paired truth differs")
        if abs(float(a0t_oracle["pred_ppm"]) - float(a4_oracle["pred_ppm"])) > 1e-9:
            raise RuntimeError("FAIL_CLOSED shared Oracle prediction differs")
        oracle_se = float(a0t_oracle["squared_error"])
        paired.append(
            {
                "physical_identity": identity,
                "filename": a0t["filename"],
                "true_class": int(a0t["true_class"]),
                "true_ppm": float(a0t["true_ppm"]),
                "oracle_pred_ppm": float(a0t_oracle["pred_ppm"]),
                "a0t_pred_ppm": float(a0t["pred_ppm"]),
                "a4_pred_ppm": float(a4["pred_ppm"]),
                "a0t_excess_se": float(a0t["squared_error"]) - oracle_se,
                "a4_excess_se": float(a4["squared_error"]) - oracle_se,
            }
        )
        for method, routed, oracle in (("A0T", a0t, a0t_oracle), ("A4", a4, a4_oracle)):
            if int(routed["route_correct"]) == 0:
                routed_abs = float(routed["abs_error"])
                oracle_abs = float(oracle["abs_error"])
                misroutes.append(
                    {
                        "method": method,
                        "physical_identity": identity,
                        "filename": routed["filename"],
                        "true_class": int(routed["true_class"]),
                        "true_gas": CLASS_NAMES[int(routed["true_class"])],
                        "predicted_class": int(routed["pred_class"]),
                        "predicted_gas": CLASS_NAMES[int(routed["pred_class"])],
                        "true_ppm": float(routed["true_ppm"]),
                        "oracle_pred_ppm": float(oracle["pred_ppm"]),
                        "routed_pred_ppm": float(routed["pred_ppm"]),
                        "oracle_abs_error": oracle_abs,
                        "routed_abs_error": routed_abs,
                        "incremental_absolute_ppm_error": routed_abs - oracle_abs,
                        "incremental_squared_ppm_error": float(routed["squared_error"])
                        - float(oracle["squared_error"]),
                    }
                )
    return paired, misroutes


def analyze_test_after_lock(
    output: Path, matrix: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    require_cost_matrix_lock(output)
    paired, misroutes = _align_test_rows()
    _csv(output / "ACTUAL_TEST_MISROUTES.csv", misroutes)
    bootstrap = grouped_bootstrap(paired)
    _csv(output / "GROUPED_RAW_FILE_BOOTSTRAP.csv", bootstrap)
    bootstrap_summary = _bootstrap_summary(bootstrap)
    _csv(output / "GROUPED_RAW_FILE_BOOTSTRAP_SUMMARY.csv", bootstrap_summary)
    file_rows: list[dict[str, Any]] = []
    for filename in sorted({str(row["filename"]) for row in paired}):
        selected = [row for row in paired if str(row["filename"]) == filename]
        a0t_excess = float(np.sum([float(row["a0t_excess_se"]) for row in selected]))
        a4_excess = float(np.sum([float(row["a4_excess_se"]) for row in selected]))
        file_rows.append(
            {
                "filename": filename,
                "N": len(selected),
                "a0t_excess_squared_error_sum": a0t_excess,
                "a4_excess_squared_error_sum": a4_excess,
                "a0t_minus_a4_excess_squared_error": a0t_excess - a4_excess,
            }
        )
    positives = [max(0.0, float(row["a0t_minus_a4_excess_squared_error"])) for row in file_rows]
    positive_sum = float(sum(positives))
    top_share = max(positives) / positive_sum if positive_sum > 0 else 1.0
    positive_files = sum(value > 0 for value in positives)
    _csv(output / "MISROUTE_FILE_CONTRIBUTIONS.csv", file_rows)
    unique_misroute_ids = {str(row["physical_identity"]) for row in misroutes}
    positive_costs = [
        float(row["primary_cost_squared_ppm"])
        for row in matrix
        if int(row["true_class"]) != int(row["forced_route"])
        and float(row["primary_cost_squared_ppm"]) > 0
    ]
    decision = decide_gate_c(
        positive_offdiagonal_costs=positive_costs,
        positive_contribution_files=positive_files,
        top_file_share=top_share,
        actual_misroute_windows=len(unique_misroute_ids),
    )
    truth = np.asarray([float(row["true_ppm"]) for row in paired])
    a0t_pred = np.asarray([float(row["a0t_pred_ppm"]) for row in paired])
    a4_pred = np.asarray([float(row["a4_pred_ppm"]) for row in paired])
    observed = {
        "a0t_rmse": float(np.sqrt(np.mean((a0t_pred - truth) ** 2))),
        "a4_rmse": float(np.sqrt(np.mean((a4_pred - truth) ** 2))),
    }
    observed["a4_minus_a0t_rmse"] = observed["a4_rmse"] - observed["a0t_rmse"]
    top_two_share = sum(sorted(positives, reverse=True)[:2]) / positive_sum if positive_sum > 0 else 1.0
    result = {
        **decision,
        "observed": observed,
        "a0t_misroutes": sum(row["method"] == "A0T" for row in misroutes),
        "a4_misroutes": sum(row["method"] == "A4" for row in misroutes),
        "misroute_union_windows": len(unique_misroute_ids),
        "top_two_file_positive_share": float(top_two_share),
        "bootstrap": {row["metric"]: row for row in bootstrap_summary},
    }
    _json(output / "GATE_C_DECISION.json", result)
    return result


def finalize(output: Path, freeze: Mapping[str, Any], matrix: Sequence[Mapping[str, Any]], result: Mapping[str, Any]) -> None:
    rmse_boot = result["bootstrap"]["a4_minus_a0t_rmse"]
    report = f"""# Gate C Downstream Routing Cost Audit

## [Scientific Question]

Can equal classification-error counts produce materially different quantitative risk because true-gas to routed-gas errors have heterogeneous downstream concentration costs?

## [Protocol]

The 4x4 matrix was constructed only from the 320-window canonical-v1 C5 calibration set by forcing every sample through each frozen R84_FED_H1 gas route. The primary off-diagonal cost is max(0, mean incremental squared ppm error versus the correct route). The matrix was hashed and locked before existing C5 test predictions were opened for post-hoc explanation. Grouped bootstrap resampled complete raw filenames for 2000 seed42 replicates.

## [Primary Result]

- A0T C5 S_ALL RMSE: {result['observed']['a0t_rmse']:.6f} ppm
- A4 C5 S_ALL RMSE: {result['observed']['a4_rmse']:.6f} ppm
- A4 - A0T RMSE: {result['observed']['a4_minus_a0t_rmse']:.6f} ppm
- Positive off-diagonal cost CV: {result['positive_cost_cv']:.6f}
- A0T/A4 misroutes: {result['a0t_misroutes']}/{result['a4_misroutes']}
- Misroute union windows: {result['misroute_union_windows']}
- Positive-contribution raw files: {result['positive_contribution_files']}
- Top-one/top-two positive file shares: {result['top_file_positive_share']:.4f}/{result['top_two_file_positive_share']:.4f}
- Grouped-bootstrap A4-A0T RMSE 95% CI: [{float(rmse_boot['ci95_low']):.6f}, {float(rmse_boot['ci95_high']):.6f}] ppm; P(delta<0)={float(rmse_boot['probability_negative']):.4f}

## [Negative Result / Limitation]

The cost matrix is calibration-estimated on one target device and seed42. It motivates but does not validate a cost-aware decision rule. Test misroute decomposition is strictly post-hoc and cannot alter the matrix or routing policy.

## [Leakage Audit]

`CALIBRATION_COST_MATRIX_LOCK.json` was written before reading test prediction CSVs. No test row, label, error, probability, or filename entered cost construction, thresholding, model fitting, or hyperparameter selection.

## [Decision]

`{result['decision']}`.

## [Paper Implication]

Classification accuracy alone is insufficient to characterize quantitative routing risk when off-diagonal downstream costs are heterogeneous. A cost-aware router remains a separately gated future method, not a supported component yet.

## [Next Action]

`{result['next_action']}` is the registered scientific recommendation. This task stops here and does not execute Gate D/E/F.
"""
    (output / "GATE_C_REPORT.md").write_text(report, encoding="utf-8")
    audit = """# Gate C Experiment Audit

## Verdict: PASS

- canonical-v1, H1, A0T/A4 classifier checkpoints, R84 model files, endpoint prediction files, and calibration locks passed SHA/provenance checks.
- A0T and A4 use byte-identical frozen C5 R84 model files.
- The calibration-only 4x4 matrix was locked before test diagnostic access.
- The test stage was read-only, paired by physical identity, and grouped-bootstrap resampled raw filenames.
- No classifier/R84/QC training, refitting, threshold selection, or hyperparameter search occurred.
"""
    (output / "EXPERIMENT_AUDIT.md").write_text(audit, encoding="utf-8")
    _json(
        output / "protocol_manifest.json",
        {
            "status": "PASS",
            "producer_commit": freeze["producer_commit"],
            "protocol": gate_c_protocol(),
            "dataset_aggregate_sha256": freeze["dataset_aggregate_sha256"],
            "calibration_cost_matrix_sha256": sha256_file(output / "DOWNSTREAM_ROUTE_COST_MATRIX.csv"),
            "test_diagnostic_only": True,
            "decision": result,
        },
    )
    method_story = f"""# Method Breakthrough Decision

- Gate A: `SOURCE_DIVERSITY_SUPPORTED`; `SOURCE_DG_PROMISING` on seed42 C5 sensitivity only.
- Gate B: `FULL_ADAPTATION_REQUIRED` under the fixed 100-step lightweight study; historical B3 was excluded because it trained an inactive projection branch.
- Gate C: `{result['decision']}`; registered next action `{result['next_action']}`.

## Recommended final direction

Use multi-source federated knowledge formation as a supported sensitivity, retain full post-hoc target commissioning for the current validated lifecycle, and describe downstream-cost-aware routing only as motivated pending an independently executed Gate D. Do not claim cost-aware routing is already supported.

No Gate D/E/F experiment was started.
"""
    (METHOD_ROOT / "METHOD_BREAKTHROUGH_DECISION.md").write_text(method_story, encoding="utf-8")
    changelog = """# Manuscript Method Changelog

- Historical manuscript method: interleaved target-aware federated adaptation.
- Validated lifecycle candidate: source-only FL -> post-hoc full target commissioning -> frozen R84/FedRidge -> QC.
- Gate A adds a C5-only multi-source/source-DG sensitivity, not a replacement for the cross-target main table.
- Gate B does not support the tested lightweight endpoints at the fixed 100-step budget; full post-hoc adaptation remains the validated path.
- Gate C audits downstream route cost heterogeneity. Any cost-aware router remains unvalidated until Gate D is separately authorized and executed.
- Manuscript numeric claims are not automatically changed by this study.
"""
    (METHOD_ROOT / "MANUSCRIPT_METHOD_CHANGELOG.md").write_text(changelog, encoding="utf-8")
    excluded = {"sha256_index.json", "runner.stdout.log", "runner.stderr.log", "runner.pid"}
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in excluded)
    _json(
        output / "sha256_index.json",
        {str(path.relative_to(output)).replace("\\", "/"): sha256_file(path) for path in files},
    )


def run(output: Path) -> dict[str, Any]:
    output = output.resolve()
    freeze = write_pre_run_freeze(output)
    matrix = build_and_lock_cost_matrix(output, freeze)
    result = analyze_test_after_lock(output, matrix)
    finalize(output, freeze, matrix, result)
    return {"status": "PASS", "output": str(output), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate C calibration-only downstream routing-cost audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
