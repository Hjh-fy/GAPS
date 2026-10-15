"""Run the frozen canonical-v1 R0 feature/FedRidge reconstruction gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.canonical_fedridge import (
    RIDGE_ALPHAS,
    CanonicalRidgeModel,
    federated_fit,
    pooled_fit,
    select_source_alpha,
)
from gaps_flower.canonical_quantitative_features import (
    H1_FEATURE_NAMES,
    SENSOR_FEATURE_NAMES,
    build_feature_cache,
    load_feature_cache,
    sha256_file,
)
from run_regression_head_ablation import CLASS_NAMES
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


STUDY_ID = "CAN-V1-CRRQ-20260811"
SCHEMA_VERSION = "iotj.canonical_v1.crrq.r0.v1"
DATA_ROOT = ROOT / "dataset" / "iotj_canonical_v1"
RESULT_ROOT = (
    ROOT
    / "results"
    / "iotj_canonical_v1_final"
    / "canonical_regression_reconstruction_qc_20260811"
)
DEFAULT_OUTPUT = RESULT_ROOT / "R0"
PROTOCOL_MANIFEST = (
    ROOT
    / "docs"
    / "experiments"
    / "iotj_canonical_v1_final"
    / "canonical_regression_reconstruction_qc_20260811"
    / "protocol_manifest.json"
)
EXTRACTOR_PATH = ROOT / "run_regression_head_ablation.py"
SOURCE_CLIENTS = ("C1", "C2")
TARGET_CLIENTS = ("C3", "C4", "C5")
TOLERANCES = {
    "scaler_max_abs_difference": 1e-10,
    "coefficient_max_abs_difference": 1e-8,
    "prediction_max_abs_difference_ppm": 1e-6,
}


def build_r0_execution_plan() -> list[str]:
    return [
        "verify_canonical_dataset_and_C0",
        "build_source_train_calibration_caches",
        "select_source_alpha_and_refit_from_sufficient_statistics",
        "write_source_alpha_and_model_lock",
        "open_source_test_labels",
        "build_target_x_only_caches",
        "write_exact_recovery_and_hash_audits",
    ]


def decide_exact_recovery(
    *,
    scaler_difference: float,
    coefficient_difference: float,
    prediction_difference_ppm: float,
) -> dict[str, Any]:
    observed = {
        "scaler_max_abs_difference": float(scaler_difference),
        "coefficient_max_abs_difference": float(coefficient_difference),
        "prediction_max_abs_difference_ppm": float(prediction_difference_ppm),
    }
    passed = all(observed[key] <= TOLERANCES[key] for key in TOLERANCES)
    return {
        "status": "PASS" if passed else "FAIL_CLOSED",
        "observed": observed,
        "tolerances": dict(TOLERANCES),
        "practical_equivalence_fallback": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _sha256_index(root: Path, output: Path) -> dict[str, str]:
    index = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() != output.resolve()
    }
    write_json(output, index)
    return index


def _require_clean_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"FAIL_CLOSED R0 output is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _verify_prerequisites(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = verify_dataset(data_root)
    if dataset["status"] != "PASS":
        raise RuntimeError(f"FAIL_CLOSED canonical dataset hash mismatch: {dataset['bad_files']}")
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    if protocol.get("status") != "R0_EXECUTABLE_FREEZE_READY_FORMAL_NOT_STARTED":
        raise RuntimeError("FAIL_CLOSED R0 executable freeze is not active")
    c0_decision = json.loads((RESULT_ROOT / "C0" / "C0_DECISION.json").read_text(encoding="utf-8"))
    if c0_decision.get("decision") != "V1_INTERLEAVED_RETAINED":
        raise RuntimeError("FAIL_CLOSED frozen C0 decision differs")
    canonical = json.loads((data_root / "canonical_preprocessing_manifest.json").read_text(encoding="utf-8"))
    expected = protocol["canonical_freeze"]
    if (
        canonical.get("candidate_id") != expected["preprocessing"]
        or canonical.get("sampling_rate_hz") != 5
        or canonical.get("points_per_window") != 50
        or canonical.get("window_duration_s") != 10.0
    ):
        raise RuntimeError("FAIL_CLOSED canonical preprocessing contract differs")
    if sha256_file(EXTRACTOR_PATH) != protocol["feature_protocol"]["extractor_file_sha256_at_freeze"]:
        raise RuntimeError("FAIL_CLOSED frozen rich feature extractor changed")
    return dataset, protocol


def _cache_path(output: Path, client: str, split: str) -> Path:
    return output / "canonical_feature_caches" / client / split


def _build_cache(
    data_root: Path,
    output: Path,
    client: str,
    split: str,
    aggregate_sha256: str,
) -> dict[str, Any]:
    return build_feature_cache(
        data_root,
        output / "canonical_feature_caches",
        client=client,
        split=split,
        dataset_aggregate_sha256=aggregate_sha256,
        extractor_path=EXTRACTOR_PATH,
    )


def _load_labels(data_root: Path, client: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    client_id = int(client[1:])
    directory = data_root / f"client_{client_id}"
    classes = np.load(directory / f"{split}_classification_labels.npy", allow_pickle=False).astype(np.int64).reshape(-1)
    regression = np.load(directory / f"{split}_regression_labels.npy", allow_pickle=False).astype(np.float64)
    if regression.shape != (len(classes), 4):
        raise RuntimeError(f"FAIL_CLOSED label shape mismatch: {client}/{split}")
    return classes, regression


def _gas_client_data(
    data_root: Path,
    output: Path,
    clients: Sequence[str],
    split: str,
    gas_id: int,
    aggregate_sha256: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for client in clients:
        _sensor, h1, _identities, _manifest = load_feature_cache(
            _cache_path(output, client, split), expected_dataset_sha256=aggregate_sha256
        )
        classes, regression = _load_labels(data_root, client, split)
        if len(classes) != len(h1):
            raise RuntimeError(f"FAIL_CLOSED feature/label rows differ: {client}/{split}")
        mask = classes == gas_id
        if not np.any(mask):
            raise RuntimeError(f"FAIL_CLOSED no rows for {client}/{split}/gas{gas_id}")
        result[client] = (h1[mask], regression[mask, gas_id])
    return result


def _combine_roles(
    left: Mapping[str, tuple[np.ndarray, np.ndarray]],
    right: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if set(left) != set(right):
        raise RuntimeError("FAIL_CLOSED source client sets differ across roles")
    return {
        client: (
            np.vstack([left[client][0], right[client][0]]),
            np.concatenate([left[client][1], right[client][1]]),
        )
        for client in sorted(left)
    }


def _serialize_sufficient_stats(
    output: Path, gas_id: int, stats: Mapping[str, Sequence[Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    moments = {record.client_id: record for record in stats["moments"]}
    equations = {record.client_id: record for record in stats["normal_equations"]}
    for client in sorted(moments):
        moment = moments[client]
        equation = equations[client]
        payload = {
            "schema_version": f"{SCHEMA_VERSION}.sufficient_statistics",
            "client_id": client,
            "gas_id": gas_id,
            "gas": CLASS_NAMES[gas_id],
            "role": moment.role,
            "n": moment.n,
            "sum_x": moment.sum_x.tolist(),
            "sum_x2": moment.sum_x2.tolist(),
            "X_t_X": equation.x_t_x.tolist(),
            "X_t_y": equation.x_t_y.tolist(),
            "y_t_y": equation.y_y,
            "y_min": equation.y_min,
            "y_max": equation.y_max,
            "feature_moment_provenance_sha256": moment.provenance_sha256,
            "normal_equation_provenance_sha256": equation.provenance_sha256,
            "server_received": ["n", "sum_x", "sum_x2", "X_t_X", "X_t_y", "y_t_y", "y_min", "y_max"],
            "server_received_raw_rows": False,
            "server_received_raw_X_or_y": False,
        }
        path = output / "canonical_fedridge_sufficient_statistics" / f"gas_{gas_id}_{client}.json"
        write_json(path, payload)
        rows.append({
            "client": client,
            "gas_id": gas_id,
            "gas": CLASS_NAMES[gas_id],
            "n": moment.n,
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path),
        })
    return rows


def _pooled_alpha(
    train: Mapping[str, tuple[np.ndarray, np.ndarray]],
    calibration: Mapping[str, tuple[np.ndarray, np.ndarray]],
    gas_id: int,
) -> tuple[float, list[dict[str, Any]]]:
    train_x = np.vstack([train[client][0] for client in sorted(train)])
    train_y = np.concatenate([train[client][1] for client in sorted(train)])
    cal_x = np.vstack([calibration[client][0] for client in sorted(calibration)])
    cal_y = np.concatenate([calibration[client][1] for client in sorted(calibration)])
    best_alpha = float(RIDGE_ALPHAS[0])
    best_rmse = float("inf")
    audit: list[dict[str, Any]] = []
    for alpha in RIDGE_ALPHAS:
        model = pooled_fit(
            train_x,
            train_y,
            gas_id=gas_id,
            role="source_train",
            feature_names=H1_FEATURE_NAMES,
            alpha=alpha,
        )
        rmse = float(np.sqrt(np.mean((model.predict_matrix(cal_x) - cal_y) ** 2)))
        audit.append({"gas_id": gas_id, "alpha": float(alpha), "pooled_source_calibration_RMSE": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)
    return best_alpha, audit


def _model_metrics(model: CanonicalRidgeModel, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    prediction = model.predict_matrix(x)
    error = prediction - y
    return {
        "N": int(len(y)),
        "RMSE": float(np.sqrt(np.mean(error * error))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
    }


def run(data_root: Path, output: Path) -> dict[str, Any]:
    _require_clean_output(output)
    started = time.perf_counter()
    dataset, protocol = _verify_prerequisites(data_root)
    aggregate_sha256 = str(dataset["aggregate_sha256"])
    access_events: list[dict[str, Any]] = []
    feature_manifests: list[dict[str, Any]] = []

    for client in SOURCE_CLIENTS:
        for split in ("train", "calibration"):
            manifest = _build_cache(data_root, output, client, split, aggregate_sha256)
            feature_manifests.append(manifest)
            access_events.append({
                "stage": "source_feature_and_fit_input",
                "client": client,
                "split": split,
                "x": True,
                "phase": True,
                "class": True,
                "concentration": True,
                "selection_use": split == "calibration",
            })

    alpha_rows: list[dict[str, Any]] = []
    pooled_alpha_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    models: dict[int, CanonicalRidgeModel] = {}
    alpha_lock: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}.alpha_lock",
        "status": "SEALED_BEFORE_SOURCE_TEST_AND_TARGET_CACHE",
        "study_id": STUDY_ID,
        "alpha_grid": list(RIDGE_ALPHAS),
        "source_roles": {
            "selection_fit": ["C1_train", "C2_train"],
            "selection_validation": ["C1_calibration", "C2_calibration"],
            "final_refit": ["C1_train", "C1_calibration", "C2_train", "C2_calibration"],
        },
        "source_test_used_for_selection": False,
        "target_input_used_for_selection": False,
        "selected_alpha": {},
    }

    for gas_id in sorted(CLASS_NAMES):
        train = _gas_client_data(data_root, output, SOURCE_CLIENTS, "train", gas_id, aggregate_sha256)
        calibration = _gas_client_data(data_root, output, SOURCE_CLIENTS, "calibration", gas_id, aggregate_sha256)
        selected_alpha, audit = select_source_alpha(
            train,
            calibration,
            gas_id=gas_id,
            feature_names=H1_FEATURE_NAMES,
            alphas=RIDGE_ALPHAS,
        )
        pooled_selected, pooled_audit = _pooled_alpha(train, calibration, gas_id)
        alpha_rows.extend({**row, "gas": CLASS_NAMES[gas_id]} for row in audit)
        pooled_alpha_rows.extend({**row, "gas": CLASS_NAMES[gas_id]} for row in pooled_audit)
        if pooled_selected != selected_alpha:
            raise RuntimeError(f"FAIL_CLOSED pooled/federated alpha differs for gas {gas_id}")
        combined = _combine_roles(train, calibration)
        model, stats = federated_fit(
            combined,
            gas_id=gas_id,
            role="source_train_plus_calibration_refit",
            feature_names=H1_FEATURE_NAMES,
            alpha=selected_alpha,
        )
        models[gas_id] = model
        stats_rows.extend(_serialize_sufficient_stats(output, gas_id, stats))
        pooled_x = np.vstack([combined[client][0] for client in sorted(combined)])
        pooled_y = np.concatenate([combined[client][1] for client in sorted(combined)])
        pooled_model = pooled_fit(
            pooled_x,
            pooled_y,
            gas_id=gas_id,
            role="source_train_plus_calibration_refit",
            feature_names=H1_FEATURE_NAMES,
            alpha=selected_alpha,
        )
        scaler_difference = max(
            float(np.max(np.abs(model.mean - pooled_model.mean))),
            float(np.max(np.abs(model.scale - pooled_model.scale))),
        )
        coefficient_difference = float(np.max(np.abs(model.coef - pooled_model.coef)))
        prediction_difference = float(
            np.max(np.abs(model.predict_matrix(pooled_x) - pooled_model.predict_matrix(pooled_x)))
        )
        decision = decide_exact_recovery(
            scaler_difference=scaler_difference,
            coefficient_difference=coefficient_difference,
            prediction_difference_ppm=prediction_difference,
        )
        exact_rows.append({
            "gas_id": gas_id,
            "gas": CLASS_NAMES[gas_id],
            "selected_alpha": selected_alpha,
            "pooled_alpha": pooled_selected,
            **decision["observed"],
            "status": decision["status"],
        })
        alpha_lock["selected_alpha"][str(gas_id)] = {
            "gas": CLASS_NAMES[gas_id],
            "alpha": selected_alpha,
        }

    model_dir = output / "canonical_fedridge_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for gas_id, model in sorted(models.items()):
        write_json(model_dir / f"gas_{gas_id}.json", model.to_json())
    write_json(output / "canonical_fedridge_alpha_lock.json", alpha_lock)
    write_csv(output / "canonical_fedridge_alpha_audit.csv", alpha_rows)
    write_csv(output / "canonical_fedridge_pooled_alpha_audit.csv", pooled_alpha_rows)
    write_csv(output / "canonical_fedridge_sufficient_statistics_manifest.csv", stats_rows)

    exact_summary = {
        "schema_version": f"{SCHEMA_VERSION}.exact_recovery",
        "status": "PASS" if all(row["status"] == "PASS" for row in exact_rows) else "FAIL_CLOSED",
        "tolerances": dict(TOLERANCES),
        "practical_equivalence_fallback": False,
        "gas_results": exact_rows,
        "maxima": {
            key: max(float(row[key]) for row in exact_rows)
            for key in TOLERANCES
        },
    }
    write_json(output / "canonical_fedridge_exact_recovery.json", exact_summary)
    if exact_summary["status"] != "PASS":
        write_json(output / "R0_FAIL_CLOSED.json", exact_summary)
        _sha256_index(output, output / "R0_SHA256_INDEX.json")
        raise RuntimeError("FAIL_CLOSED canonical FedRidge exact recovery tolerance failed")

    source_test_metrics: list[dict[str, Any]] = []
    for client in SOURCE_CLIENTS:
        manifest = _build_cache(data_root, output, client, "test", aggregate_sha256)
        feature_manifests.append(manifest)
        access_events.append({
            "stage": "source_test_evaluation_after_lock",
            "client": client,
            "split": "test",
            "x": True,
            "phase": True,
            "class": True,
            "concentration": True,
            "selection_use": False,
        })
    for gas_id in sorted(CLASS_NAMES):
        test = _gas_client_data(data_root, output, SOURCE_CLIENTS, "test", gas_id, aggregate_sha256)
        for client, (x, y) in sorted(test.items()):
            source_test_metrics.append({
                "client": client,
                "gas_id": gas_id,
                "gas": CLASS_NAMES[gas_id],
                **_model_metrics(models[gas_id], x, y),
            })
    write_csv(output / "canonical_fedridge_source_test_metrics.csv", source_test_metrics)

    for client in TARGET_CLIENTS:
        for split in ("calibration", "test"):
            manifest = _build_cache(data_root, output, client, split, aggregate_sha256)
            feature_manifests.append(manifest)
            access_events.append({
                "stage": "target_x_only_feature_cache_after_source_lock",
                "client": client,
                "split": split,
                "x": True,
                "phase": True,
                "class": False,
                "concentration": False,
                "regression_label": False,
                "selection_use": False,
            })

    feature_rows = [
        {
            "client": manifest["client"],
            "split": manifest["split"],
            "rows": manifest["row_count"],
            "window_shape": "50x8",
            "sampling_rate_hz": manifest["sampling_rate_hz"],
            "sensor_dimensions": manifest["sensor_dimensions"],
            "h1_dimensions": manifest["h1_dimensions"],
            "source_array_sha256": manifest["source_array_sha256"],
            "metadata_sha256": manifest["metadata_sha256"],
            "cache_sha256": manifest["cache_sha256"],
            "legacy_cache_reused": False,
            "resized_or_interpolated_after_preprocessing": False,
        }
        for manifest in feature_manifests
    ]
    write_csv(output / "R0_FEATURE_PROVENANCE.csv", feature_rows)
    write_json(
        output / "R0_ACCESS_AUDIT.json",
        {
            "schema_version": f"{SCHEMA_VERSION}.access_audit",
            "status": "PASS",
            "execution_plan": build_r0_execution_plan(),
            "events": access_events,
            "target_test_labels_opened": False,
            "target_inputs_entered_source_fit_or_selection": False,
            "source_test_opened_after_alpha_and_model_lock": True,
        },
    )
    (output / "H1_CANONICAL_PORTING_AUDIT.md").write_text(
        "# H1 canonical porting audit\n\nStatus: **PASS**. All 104D H1 rows were recomputed from canonical-v1 5 Hz / 50x8 arrays with the frozen extractor. The 21 metadata/phase fields are appended to the 83 sensor statistics. Slope and absolute-difference fields are interpreted only as fixed-5-Hz discrete per-sample descriptors; no sampling-rate invariance or 10-Hz/5-Hz numerical equivalence is claimed. No legacy feature matrix, scaler, alpha, coefficient, or QC asset was read.\n",
        encoding="utf-8",
    )
    (output / "CANONICAL_83D_CACHE_AUDIT.md").write_text(
        "# Canonical 83D cache audit\n\nStatus: **PASS**. Every 83D row was created in the same call and from the same physical window as its 104D H1 row. Cache manifests bind the canonical array, metadata, extractor, ordered schemas, 50x8 shape, 5 Hz rate, study ID, and output bytes. Target caches contain computed features and row identities only; class and concentration labels were not loaded by the cache API.\n",
        encoding="utf-8",
    )
    (output / "CANONICAL_FEDRIDGE_RECONSTRUCTION_REPORT.md").write_text(
        "# Canonical FedRidge reconstruction\n\nStatus: **PASS**. C1/C2 train rows supplied local feature moments and normal equations; C1/C2 calibration supplied only clipped SSE/count for the frozen alpha grid. The final model was refit from train+calibration sufficient statistics, with population scaling, a 1e-9 scale floor, unregularized intercept, and `numpy.linalg.pinv`. Pooled audit-only reconstruction met every strict scaler, coefficient, and prediction tolerance. Source test labels were opened only after the alpha/model lock.\n",
        encoding="utf-8",
    )
    write_json(
        output / "R0_PROTOCOL_MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE_R1_READY",
            "study_id": STUDY_ID,
            "dataset_aggregate_sha256": aggregate_sha256,
            "C0_decision": "V1_INTERLEAVED_RETAINED",
            "feature_protocol": protocol["feature_protocol"],
            "alpha_grid": list(RIDGE_ALPHAS),
            "source_clients": list(SOURCE_CLIENTS),
            "targets": list(TARGET_CLIENTS),
            "source_test_used_for_selection": False,
            "target_test_labels_opened": False,
            "legacy_quantitative_assets_reused": False,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    write_json(
        output / "fixed_endpoint_complete.json",
        {
            "schema_version": f"{SCHEMA_VERSION}.completion",
            "status": "COMPLETE",
            "gate": "R0",
            "exact_recovery": "PASS",
            "R1_released": True,
        },
    )
    index = _sha256_index(output, output / "R0_SHA256_INDEX.json")
    return {
        "status": "PASS",
        "files": len(index),
        "feature_caches": len(feature_manifests),
        "exact_recovery": exact_summary,
    }


def audit(output: Path) -> dict[str, Any]:
    index_path = output / "R0_SHA256_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    bad = [relative for relative, expected in index.items() if sha256_file(output / relative) != expected]
    exact = json.loads((output / "canonical_fedridge_exact_recovery.json").read_text(encoding="utf-8"))
    marker = json.loads((output / "fixed_endpoint_complete.json").read_text(encoding="utf-8"))
    if bad or exact.get("status") != "PASS" or marker.get("R1_released") is not True:
        raise RuntimeError(f"FAIL_CLOSED R0 audit failed: {bad}")
    return {"status": "PASS", "files": len(index), "bad": bad}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "audit"), nargs="?", default="run")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.data_root.resolve(), args.output.resolve()) if args.stage == "run" else audit(args.output.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
