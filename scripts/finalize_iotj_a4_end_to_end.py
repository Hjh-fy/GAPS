"""Finalize the frozen A4 classifier and its downstream C5 replay evidence.

This module never trains the classifier.  Existing classification assets are
read-only and every downstream artifact is written to a new output root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_flower.state_fingerprint import checkpoint_provenance
from scripts import evaluate_iotj_b5_regression_multiseed as regression_base
from scripts.evaluate_iotj_feature_metadata_ablation import profile_feature_dict


SCHEMA_VERSION = "iotj.final_a4_end_to_end.v1"
FINAL_VARIANTS = {
    "R83_TARGET_ONLY": "pred_83d_ppm",
    "R84_FED_H1": "pred_84d_h1_ppm",
    "R86_ALL_PRIORS": "pred_86d_all_priors_ppm",
}
VARIANT_PRIORS = {
    "R83_TARGET_ONLY": (),
    "R84_FED_H1": (regression_base.PRIOR_KEYS[0],),
    "R86_ALL_PRIORS": regression_base.PRIOR_KEYS,
}
EXPECTED_VARIANT_DIMENSIONS = {
    "R83_TARGET_ONLY": 83,
    "R84_FED_H1": 84,
    "R86_ALL_PRIORS": 86,
}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"FAIL_CLOSED JSON object required: {path}")
    return payload


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    """Return serialization-independent state identity plus file provenance."""
    return checkpoint_provenance(Path(path))


def prepare_output_root(path: str | Path) -> Path:
    """Create a new output root and refuse to overwrite any existing content."""
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_output_subdir(root: str | Path, name: str) -> Path:
    output = Path(root) / name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def add_final_variant_features(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> list[dict[str, Any]]:
    """Project frozen 104-D rows onto the three registered final inputs."""
    if variant not in FINAL_VARIANTS:
        raise ValueError(f"unknown final regression variant: {variant}")
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        features = profile_feature_dict(item["feature_dict"], "M83_SENSOR")
        for prior in VARIANT_PRIORS[variant]:
            value = float(item[prior])
            if not math.isfinite(value):
                raise RuntimeError(f"FAIL_CLOSED non-finite source prior: {prior}")
            features[f"srcpred_{prior}"] = value
        item["feature_dict"] = features
        output.append(item)
    expected = EXPECTED_VARIANT_DIMENSIONS[variant]
    if any(len(row["feature_dict"]) != expected for row in output):
        raise RuntimeError(f"FAIL_CLOSED {variant} input dimension differs")
    return output


def canonicalize_route_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map the frozen evaluator's ``prob_k`` fields to final evidence names."""
    canonical: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for class_id in range(4):
            source = f"prob_{class_id}"
            target = f"prob_class_{class_id}"
            if target not in item:
                if source not in item:
                    raise RuntimeError("FAIL_CLOSED class probabilities are incomplete")
                item[target] = item[source]
            item.pop(source, None)
        canonical.append(item)
    return canonical


def validate_route_rows(
    rows: Sequence[Mapping[str, Any]], expected_n: int
) -> None:
    if len(rows) != expected_n:
        raise RuntimeError(
            f"FAIL_CLOSED route row count differs: {len(rows)} != {expected_n}"
        )
    indexes = [int(row.get("sample_index", -1)) for row in rows]
    if indexes != list(range(expected_n)):
        raise RuntimeError("FAIL_CLOSED route sample IDs are not canonical")
    probability_keys = [f"prob_class_{class_id}" for class_id in range(4)]
    for row in rows:
        if any(key not in row for key in probability_keys):
            raise RuntimeError("FAIL_CLOSED class probabilities are incomplete")
        probabilities = np.asarray(
            [float(row[key]) for key in probability_keys], dtype=np.float64
        )
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise RuntimeError("FAIL_CLOSED class probabilities are non-finite/negative")
        if not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5):
            raise RuntimeError("FAIL_CLOSED class probabilities do not sum to one")
        if int(row.get("pred_class", -1)) != int(np.argmax(probabilities)):
            raise RuntimeError("FAIL_CLOSED predicted class/probabilities disagree")


def fit_final_regressors(
    oracle: Sequence[Mapping[str, Any]],
    deployment: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[int, Any]], list[dict[str, Any]]]:
    """Select alphas and refit the three per-gas target Ridge variants."""
    deployment_by_id = {int(row["sample_index"]): row for row in deployment}
    if len(deployment_by_id) != len(deployment):
        raise RuntimeError("FAIL_CLOSED duplicate calibration sample IDs")
    models: dict[str, dict[int, Any]] = {}
    selection: list[dict[str, Any]] = []
    for variant in FINAL_VARIANTS:
        fit_features = add_final_variant_features(oracle, variant)
        deploy_features = add_final_variant_features(deployment, variant)
        deploy_by_id = {int(row["sample_index"]): row for row in deploy_features}
        feature_names = sorted(fit_features[0]["feature_dict"])
        models[variant] = {}
        for class_id, gas in sorted(regression_base.CLASS_NAMES.items()):
            class_rows = [
                row for row in fit_features if int(row["true_class"]) == class_id
            ]
            fit_rows, validation_seed_rows = regression_base.deterministic_train_val(
                class_rows, 0.25
            )
            validation_rows = [
                deploy_by_id[int(row["sample_index"])]
                for row in validation_seed_rows
            ]
            if (len(class_rows), len(fit_rows), len(validation_rows)) != (80, 60, 20):
                raise RuntimeError(
                    f"FAIL_CLOSED {variant}/{gas} calibration split is not 80/60/20"
                )
            truth = np.asarray(
                [float(row["true_ppm"]) for row in validation_rows],
                dtype=np.float64,
            )
            best_alpha = regression_base.RIDGE_ALPHAS[0]
            best_rmse = float("inf")
            grid: list[dict[str, float]] = []
            for alpha in regression_base.RIDGE_ALPHAS:
                candidate = regression_base.fit_ridge(fit_rows, feature_names, alpha)
                predictions = candidate.predict(validation_rows)
                score = float(np.sqrt(np.mean((predictions - truth) ** 2)))
                grid.append({"alpha": float(alpha), "validation_RMSE": score})
                if score < best_rmse:
                    best_alpha, best_rmse = alpha, score
            models[variant][class_id] = regression_base.fit_ridge(
                class_rows, feature_names, best_alpha
            )
            selection.append(
                {
                    "variant": variant,
                    "prediction_key": FINAL_VARIANTS[variant],
                    "class_id": class_id,
                    "gas": gas,
                    "calibration_fit_N": len(fit_rows),
                    "calibration_validation_N": len(validation_rows),
                    "target_input_dimension": len(feature_names),
                    "selected_alpha": float(best_alpha),
                    "calibration_validation_RMSE": best_rmse,
                    "alpha_grid_audit": json.dumps(grid, separators=(",", ":")),
                    "selection_split": "C5_calibration_internal_60_fit_20_validation",
                    "target_test_used_for_selection": False,
                }
            )
    return models, selection


def _classification_risk(route: Mapping[str, Any]) -> tuple[float, float, float]:
    probabilities = np.asarray(
        [float(route[f"prob_class_{class_id}"]) for class_id in range(4)],
        dtype=np.float64,
    )
    ranked = np.sort(probabilities)[::-1]
    confidence = float(ranked[0])
    margin = float(ranked[0] - ranked[1])
    entropy = float(
        -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
        / math.log(4.0)
    )
    return confidence, margin, entropy


def apply_final_regressors(
    deployment: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[int, Any]],
) -> list[dict[str, Any]]:
    validate_route_rows(route_rows, len(deployment))
    records: list[dict[str, Any]] = []
    for index, (row, route) in enumerate(zip(deployment, route_rows)):
        if int(row["sample_index"]) != index:
            raise RuntimeError("FAIL_CLOSED deployment rows are not canonical")
        if (
            int(row["true_class"]) != int(route["true_class"])
            or int(row["pred_class"]) != int(route["pred_class"])
        ):
            raise RuntimeError("FAIL_CLOSED route/deployment labels are misaligned")
        item: dict[str, Any] = {
            "sample_id": f"C5:{row.get('split', 'unknown')}:{index}",
            "sample_index": index,
            "client_id": "C5",
            "client": "C5",
            "split": str(row.get("split", "unknown")),
            "gas_true": regression_base.CLASS_NAMES[int(row["true_class"])],
            "gas_pred": regression_base.CLASS_NAMES[int(row["pred_class"])],
            "true_class": int(row["true_class"]),
            "pred_class": int(row["pred_class"]),
            "route_correct": int(int(row["true_class"]) == int(row["pred_class"])),
            "true_ppm": float(row["true_ppm"]),
        }
        for class_id in range(4):
            item[f"prob_class_{class_id}"] = float(route[f"prob_class_{class_id}"])
        confidence, margin, entropy = _classification_risk(route)
        item["class_confidence"] = confidence
        item["class_margin"] = margin
        item["class_entropy"] = entropy
        for variant, prediction_key in FINAL_VARIANTS.items():
            features = add_final_variant_features([row], variant)
            pred = float(models[variant][int(row["pred_class"])].predict(features)[0])
            if not math.isfinite(pred):
                raise RuntimeError(f"FAIL_CLOSED non-finite prediction: {variant}")
            item[prediction_key] = pred
        proposed = float(item["pred_84d_h1_ppm"])
        item["pred_ppm"] = proposed
        item["abs_error"] = abs(proposed - float(item["true_ppm"]))
        item["squared_error"] = (proposed - float(item["true_ppm"])) ** 2
        class_range = float(regression_base.CLASS_RANGES[int(row["pred_class"])])
        classification_uncertainty = max(1.0 - confidence, 1.0 - margin, entropy)
        regression_disagreement = abs(
            float(item["pred_84d_h1_ppm"]) - float(item["pred_83d_ppm"])
        ) / class_range
        source_priors = [float(row[key]) for key in regression_base.PRIOR_KEYS]
        source_prior_disagreement = (max(source_priors) - min(source_priors)) / class_range
        item["classification_uncertainty_risk"] = classification_uncertainty
        item["regression_disagreement_risk"] = regression_disagreement
        item["source_prior_disagreement_risk"] = source_prior_disagreement
        item["qc_risk_score"] = max(
            classification_uncertainty,
            regression_disagreement,
            source_prior_disagreement,
        )
        records.append(item)
    return records


def regression_metrics(
    rows: Sequence[Mapping[str, Any]],
    prediction_key: str,
    mask: Sequence[bool],
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if len(selected) != len(rows) or not selected.any():
        raise RuntimeError("FAIL_CLOSED regression metric mask is empty/misaligned")
    true = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    pred = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)
    classes = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise RuntimeError("FAIL_CLOSED non-finite regression values")
    error = pred - true
    ranges = np.asarray(
        [regression_base.CLASS_RANGES[int(value)] for value in classes],
        dtype=np.float64,
    )
    residual = float(np.sum(error[selected] ** 2))
    centered = true[selected] - float(np.mean(true[selected]))
    total = float(np.sum(centered**2))
    return {
        "N": int(selected.sum()),
        "RMSE": float(np.sqrt(np.mean(error[selected] ** 2))),
        "MAE": float(np.mean(np.abs(error[selected]))),
        "R2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "NRMSE": float(np.sqrt(np.mean((error[selected] / ranges[selected]) ** 2))),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def seal_calibration(
    output: Path,
    selection: Sequence[Mapping[str, Any]],
    models: Mapping[str, Mapping[int, Any]],
) -> Path:
    """Persist all choices and fitted heads before the target test is opened."""
    if not selection or any(
        bool(row.get("target_test_used_for_selection", True)) for row in selection
    ):
        raise RuntimeError("FAIL_CLOSED target test entered calibration selection")
    model_payload = {
        variant: {
            str(class_id): model.to_json()
            for class_id, model in sorted(class_models.items())
        }
        for variant, class_models in models.items()
    }
    write_json(output / "regression_models.json", model_payload)
    lock_path = output / "calibration_selection_lock.json"
    write_json(
        lock_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "SEALED_BEFORE_TARGET_TEST",
            "target_test_opened": False,
            "fixed_alpha_grid": list(regression_base.RIDGE_ALPHAS),
            "selection": list(selection),
            "models_file": "regression_models.json",
        },
    )
    return lock_path


def load_calibration_lock(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "SEALED_BEFORE_TARGET_TEST":
        raise RuntimeError("FAIL_CLOSED calibration is not sealed")
    if payload.get("target_test_opened") is not False:
        raise RuntimeError("FAIL_CLOSED target test was opened before lock validation")
    selection = payload.get("selection")
    if not isinstance(selection, list) or not selection or any(
        bool(row.get("target_test_used_for_selection", True)) for row in selection
    ):
        raise RuntimeError("FAIL_CLOSED target test entered calibration selection")
    return payload


def summarize_regression_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        raise RuntimeError("FAIL_CLOSED no regression records")
    route_correct = [bool(int(row["route_correct"])) for row in records]
    all_mask = [True] * len(records)
    main: list[dict[str, Any]] = []
    per_gas: list[dict[str, Any]] = []
    route: list[dict[str, Any]] = []
    for variant, prediction_key in FINAL_VARIANTS.items():
        common = {
            "variant": variant,
            "prediction_key": prediction_key,
            "input_dimension": EXPECTED_VARIANT_DIMENSIONS[variant],
        }
        for scope, mask in (("S_ALL", all_mask), ("S_CC", route_correct)):
            main.append(
                {**common, "evaluation_scope": scope, **regression_metrics(records, prediction_key, mask)}
            )
        for class_id, gas in sorted(regression_base.CLASS_NAMES.items()):
            gas_mask = [int(row["true_class"]) == class_id for row in records]
            gas_cc_mask = [a and b for a, b in zip(gas_mask, route_correct)]
            for scope, mask in (("S_ALL", gas_mask), ("S_CC", gas_cc_mask)):
                per_gas.append(
                    {
                        **common,
                        "evaluation_scope": scope,
                        "class_id": class_id,
                        "gas": gas,
                        **regression_metrics(records, prediction_key, mask),
                    }
                )
        for scope, mask in (
            ("ROUTE_CORRECT", route_correct),
            ("MISROUTED", [not value for value in route_correct]),
        ):
            if any(mask):
                metrics = regression_metrics(records, prediction_key, mask)
            else:
                metrics = {"N": 0, "RMSE": "", "MAE": "", "R2": "", "NRMSE": ""}
            route.append({**common, "evaluation_scope": scope, **metrics})
    return main, per_gas, route


def summarize_per_concentration(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = sorted({(int(row["true_class"]), float(row["true_ppm"])) for row in records})
    for variant, prediction_key in FINAL_VARIANTS.items():
        for class_id, concentration in groups:
            mask = [
                int(row["true_class"]) == class_id
                and math.isclose(float(row["true_ppm"]), concentration, abs_tol=1e-9)
                for row in records
            ]
            rows.append(
                {
                    "variant": variant,
                    "prediction_key": prediction_key,
                    "input_dimension": EXPECTED_VARIANT_DIMENSIONS[variant],
                    "class_id": class_id,
                    "gas": regression_base.CLASS_NAMES[class_id],
                    "true_ppm": concentration,
                    **regression_metrics(records, prediction_key, mask),
                }
            )
    return rows


def _value_after(argv: Sequence[str], flag: str) -> str:
    try:
        index = list(argv).index(flag)
    except ValueError as exc:
        raise RuntimeError(f"FAIL_CLOSED required protocol flag missing: {flag}") from exc
    if index + 1 >= len(argv):
        raise RuntimeError(f"FAIL_CLOSED protocol flag has no value: {flag}")
    return str(argv[index + 1])


def _bool_after(argv: Sequence[str], flag: str) -> bool:
    value = _value_after(argv, flag).strip().lower()
    if value not in {"true", "false"}:
        raise RuntimeError(f"FAIL_CLOSED invalid Boolean for {flag}: {value}")
    return value == "true"


def build_classifier_manifest(classification_root: str | Path) -> dict[str, Any]:
    """Build the fail-closed final classifier manifest from frozen A4 assets."""
    root = Path(classification_root).resolve()
    run_root = root / "FCL-E4-A4"
    locked = read_json(run_root / "locked_run_spec.json")
    completed = read_json(run_root / "fixed_endpoint_complete.json")
    evaluation = read_json(run_root / "final_evaluation_C5.json")

    protocol = locked.get("protocol")
    server = locked.get("server")
    client_c1 = locked.get("client_c1")
    client_c2 = locked.get("client_c2")
    if not all(isinstance(value, list) for value in (server, client_c1, client_c2)):
        raise RuntimeError("FAIL_CLOSED locked A4 command arrays are unavailable")
    if not isinstance(protocol, dict):
        raise RuntimeError("FAIL_CLOSED locked A4 protocol object is unavailable")

    checkpoint = run_root / "remote_server" / "server_round_025_adapted.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FAIL_CLOSED A4 round-25 checkpoint missing: {checkpoint}")
    target_metrics = evaluation.get("target_metrics")
    if not isinstance(target_metrics, dict):
        raise RuntimeError("FAIL_CLOSED C5 final target metrics missing")

    observed = {
        "rounds": int(protocol.get("rounds", -1)),
        "local_epochs": int(_value_after(client_c1, "--local-epochs")),
        "batch_size": int(protocol.get("batch_size", -1)),
        "seed": int(protocol.get("seed", -1)),
        "target_ce_weight": float(_value_after(server, "--da-lambda-target-ce")),
        "selective_aggregation": _bool_after(server, "--use-selective-agg"),
        "ablation_variant": _value_after(server, "--ablation-variant"),
        "target_information_method": _value_after(server, "--target-information-method"),
        "endpoint_round": int(completed.get("fixed_endpoint", {}).get("round", -1)),
    }
    expected = {
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "seed": 42,
        "target_ce_weight": 0.0,
        "selective_aggregation": False,
        "ablation_variant": "A4",
        "target_information_method": "a4",
        "endpoint_round": 25,
    }
    if observed != expected:
        raise RuntimeError(
            f"FAIL_CLOSED frozen A4 protocol differs: observed={observed}"
        )

    missing_reason = (
        "No immutable same-protocol server-centric A4 round-25 endpoint was "
        "found in the local result root or audited server result root; a "
        "full-GAPS endpoint is not an A4 substitute."
    )
    identity = checkpoint_identity(checkpoint)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "partial_complete_c5_only",
        "protocol": {
            "method": "server-centric A4",
            "rounds": 25,
            "local_epochs": 1,
            "batch_size": 32,
            "seed": 42,
            "source_clients": ["C1", "C2"],
            "target_ce_weight": 0.0,
            "selective_aggregation": False,
            "fixed_endpoint_only": True,
        },
        "targets": {
            "C3": {"status": "blocked", "checkpoint": None, "reason": missing_reason},
            "C4": {"status": "blocked", "checkpoint": None, "reason": missing_reason},
            "C5": {
                "status": "complete",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_identity": identity,
                "accuracy": float(target_metrics["accuracy"]),
                "macro_f1": float(target_metrics["macro_f1"]),
                "nll": float(target_metrics["nll"]),
                "ece": float(target_metrics["ece"]),
                "num_examples": int(target_metrics["num_examples"]),
                "selection_role": "none_fixed_endpoint_only",
            },
        },
        "classification_retrained": False,
        "full_gaps_endpoint_substituted_for_a4": False,
        "source_root": str(root),
    }


def write_classifier_freeze_outputs(output: Path, manifest: Mapping[str, Any]) -> None:
    rows = []
    for target, payload in manifest["targets"].items():
        rows.append(
            {
                "target": target,
                "method": manifest["protocol"]["method"],
                "status": payload["status"],
                "accuracy": payload.get("accuracy", ""),
                "macro_f1": payload.get("macro_f1", ""),
                "nll": payload.get("nll", ""),
                "ece": payload.get("ece", ""),
                "num_examples": payload.get("num_examples", ""),
                "checkpoint": payload.get("checkpoint") or "",
                "blocked_reason": payload.get("reason", ""),
            }
        )
    _write_csv(output / "final_classifier_cross_target.csv", rows)
    c5 = manifest["targets"]["C5"]
    text = f"""# Final classifier freeze

- Formal router: server-centric A4, round 25, seed 42, LE1, batch size 32.
- C5 is complete: accuracy={float(c5['accuracy']):.6f}, macro-F1={float(c5['macro_f1']):.6f}.
- C3/C4 are blocked because immutable same-protocol A4 endpoints are unavailable.
- No classifier was retrained and no full-GAPS checkpoint was substituted.
- Checkpoint equality uses ordered state-content fingerprint; whole-file SHA-256 is provenance only.
"""
    (output / "FINAL_CLASSIFIER_FREEZE.md").write_text(text, encoding="utf-8")


def run_regression_replay(
    output_root: Path,
    manifest: Mapping[str, Any],
    data_root: Path,
    runtime_contract: Path,
    h1_manifest: Path,
    device: str,
    batch_size: int,
) -> None:
    """Run calibration lock, then and only then open the C5 sealed test."""
    regression = prepare_output_subdir(output_root, "regression")
    checkpoint = Path(str(manifest["targets"]["C5"]["checkpoint"]))
    runtime = regression_base.C5H8Runtime.from_runtime_contract(
        runtime_contract.resolve(), device=device
    )
    h1, source_manifest = regression_base.load_source_heads(
        h1_manifest.resolve(), runtime
    )

    # Phase A: calibration only.  No test loader/API call is made above this lock.
    calibration_routes = canonicalize_route_rows(
        regression_base.classifier_routes(
            checkpoint, data_root.resolve(), "calibration", torch.device(device), batch_size
        )
    )
    validate_route_rows(calibration_routes, 320)
    calibration_oracle, calibration_deployment = regression_base.prepare_rows(
        data_root.resolve(), "calibration", calibration_routes, h1, runtime
    )
    models, selection = fit_final_regressors(
        calibration_oracle, calibration_deployment
    )
    lock_path = seal_calibration(regression, selection, models)
    _write_csv(regression / "calibration_alpha_selection.csv", selection)
    load_calibration_lock(lock_path)
    calibration_records = apply_final_regressors(
        calibration_deployment, calibration_routes, models
    )
    _write_csv(regression / "final_calibration_records.csv", calibration_records)

    # Phase B: the target test is opened only after the persisted lock validates.
    test_routes = canonicalize_route_rows(
        regression_base.classifier_routes(
            checkpoint, data_root.resolve(), "test", torch.device(device), batch_size
        )
    )
    validate_route_rows(test_routes, 1360)
    _test_oracle, test_deployment = regression_base.prepare_rows(
        data_root.resolve(), "test", test_routes, h1, runtime
    )
    test_records = apply_final_regressors(test_deployment, test_routes, models)
    _write_csv(regression / "final_test_records.csv", test_records)

    main, per_gas, route = summarize_regression_records(test_records)
    per_concentration = summarize_per_concentration(test_records)
    _write_csv(regression / "regression_main_summary.csv", main)
    _write_csv(regression / "regression_per_gas.csv", per_gas)
    _write_csv(regression / "regression_per_concentration.csv", per_concentration)
    _write_csv(regression / "regression_route_decomposition.csv", route)
    write_json(
        regression / "protocol_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "classifier": manifest["targets"]["C5"],
            "data_root": str(data_root.resolve()),
            "target_client": "C5",
            "calibration_N": 320,
            "test_N": 1360,
            "calibration_split": "60 fit + 20 validation per gas, then refit 80",
            "ridge_alpha_grid": list(regression_base.RIDGE_ALPHAS),
            "proposed_variant": "R84_FED_H1",
            "variants": {
                key: {
                    "prediction_key": FINAL_VARIANTS[key],
                    "input_dimension": EXPECTED_VARIANT_DIMENSIONS[key],
                    "source_priors": list(VARIANT_PRIORS[key]),
                }
                for key in FINAL_VARIANTS
            },
            "target_test_used_for_selection": False,
            "fixed_endpoint_only": True,
            "source_priors": source_manifest,
            "runtime_contract": str(runtime_contract.resolve()),
            "h1_manifest": str(h1_manifest.resolve()),
        },
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classification-root",
        default="results/iotj_final_classification_le1_20260804",
    )
    parser.add_argument(
        "--output-root",
        default="results/iotj_final_end_to_end_a4_20260804",
    )
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--data-root",
        default=(
            "dataset/client_data_c1234src_c5tgt_2080_"
            "timeaware_60_170_window_fullgrid"
        ),
    )
    parser.add_argument(
        "--runtime-contract",
        default=(
            "results/iotj_b5_c5_deployment_p1_20260722/"
            "c5_h8_runtime_contract_b5_v4/runtime_contract.json"
        ),
    )
    parser.add_argument(
        "--h1-manifest",
        default=(
            "results/iotj_h1_federated_ridge_equivalence_20260724/"
            "federated_h1_manifest.json"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_classifier_manifest(args.classification_root)
    output = Path(args.output_root)
    manifest_path = output / "final_classifier_manifest.json"
    if not output.exists() or not any(output.iterdir()):
        output = prepare_output_root(output)
        write_json(manifest_path, manifest)
        write_classifier_freeze_outputs(output, manifest)
    else:
        observed = read_json(manifest_path)
        if observed != manifest:
            raise RuntimeError("FAIL_CLOSED existing classifier manifest differs")
        if not (output / "final_classifier_cross_target.csv").exists():
            write_classifier_freeze_outputs(output, manifest)
    if not args.freeze_only:
        run_regression_replay(
            output,
            manifest,
            Path(args.data_root),
            Path(args.runtime_contract),
            Path(args.h1_manifest),
            args.device,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
