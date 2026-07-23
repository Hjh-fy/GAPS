"""Frozen RS0--RS4 source-regression topology experiment.

This is an experimental entry point.  It does not modify the B5/C5 runtime,
QC policy, or any frozen parity evidence.  The default action is a contract
check; a formal experiment requires the explicit ``--formal-run`` flag and a
new, empty output directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaps_flower.evaluate_regression_pipeline import (
    class_range,
    denormalize_by_class,
    load_split_arrays,
)
from gaps_flower.regression_task import (
    build_source_regression_loaders,
    create_regression_model,
    fedavg_regression_states,
    get_regression_state_keys,
    init_regression_branch_from_classifier,
    make_regression_config,
    train_regression_local,
)
from gaps_deploy.c5_h8_runtime import C5H8Runtime
from run_regression_head_ablation import (
    CLASS_NAMES,
    apply_client_models,
    build_oracle_rows,
    deterministic_train_val,
    fit_ridge,
    fit_select_refit,
    read_csv,
)

SCHEMA_VERSION = "iotj.federated_source_regression_prior.v1"
SEED = 42
SOURCE_CLIENTS = (1, 2)
TARGET_CLIENT = 5
MODEL_SELECTION_SPLIT = "calibration_validation"
RIDGE_ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
VARIANT_FEATURES = {
    "RS4_rich_only": (),
    "RS0_pooled_source": (
        "H1_source_ridge_ppm",
        "H2_source_per_gas_mlp_ppm",
        "H3_source_shared_mlp_ppm",
    ),
    "RS1_local_experts": ("pred_C1", "pred_C2"),
    "RS2_fedavg_prior": ("pred_FedAvg",),
    "RS3_local_plus_fedavg": ("pred_C1", "pred_C2", "pred_FedAvg"),
}
FORMAL_OUTPUT_FILES = (
    "protocol_manifest.json",
    "topology_audit.json",
    "source_local_model_manifest.json",
    "fedavg_model_manifest.json",
    "calibration_selection.csv",
    "test_predictions.csv",
    "regression_variant_summary.csv",
    "per_gas_summary.csv",
    "comparison_vs_pooled_h8.csv",
    "decision_gate.json",
    "source_prediction_mechanism.json",
    "README.md",
)
FROZEN_EVIDENCE = (
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/runtime_contract.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_contract_b5_v4/row_map_1360.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95_v1/parity_report.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc95_v1/runtime_rows.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc90_v1/parity_report.json",
    "results/iotj_b5_c5_deployment_p1_20260722/c5_h8_runtime_parity_hc90_v1/runtime_rows.csv",
)


@dataclass(frozen=True)
class Protocol:
    schema_version: str = SCHEMA_VERSION
    seed: int = SEED
    source_clients: tuple[int, int] = SOURCE_CLIENTS
    target_client: int = TARGET_CLIENT
    source_rounds: int = 3
    source_steps_per_client: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    target_validation_ratio: float = 0.25
    model_selection_split: str = MODEL_SELECTION_SPLIT
    route_source: str = "B5_seed42_predicted_class"
    qc_enabled: bool = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(
    state: Mapping[str, torch.Tensor], keys: Iterable[str] | None = None
) -> str:
    digest = hashlib.sha256()
    selected = sorted(keys if keys is not None else state)
    for key in selected:
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def training_strength(source_rounds: int, source_steps: int) -> dict[str, Any]:
    if source_rounds <= 0 or source_steps <= 0:
        raise ValueError("source_rounds and source_steps must both be positive")
    base_steps, extra_steps = divmod(source_steps, source_rounds)
    allocation = [
        base_steps + int(round_index <= extra_steps)
        for round_index in range(1, source_rounds + 1)
    ]
    if any(step <= 0 for step in allocation):
        raise ValueError(
            "Every source round must execute at least one optimizer step per client"
        )
    return {
        "source_steps_semantics": "total_optimizer_steps_per_client_distributed_across_rounds",
        "requested_source_rounds": int(source_rounds),
        "requested_source_steps_per_client_total": int(source_steps),
        "round_step_allocation_per_client": allocation,
        "actual_optimizer_steps_by_client": {
            "C1": int(sum(allocation)),
            "C2": int(sum(allocation)),
        },
        "actual_optimizer_steps_all_clients": int(2 * sum(allocation)),
        "fedavg_execution_count": int(source_rounds),
    }


def require_new_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment output: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def frozen_evidence_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in FROZEN_EVIDENCE:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen evidence is missing: {path}")
        result[relative] = sha256_file(path)
    return result


def feature_schema(rich_names: Sequence[str], variant: str) -> tuple[str, ...]:
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"Unknown variant: {variant}")
    return tuple(rich_names) + tuple(f"srcpred_{x}" for x in VARIANT_FEATURES[variant])


def assert_selection_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Model selection requires non-empty calibration-validation rows")
    bad = [row for row in rows if row.get("selection_split") != MODEL_SELECTION_SPLIT]
    if bad:
        raise ValueError(
            "Model selection accepts calibration-validation rows only; "
            f"rejected {len(bad)} rows"
        )


def selection_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash a frozen selection table without accepting test rows or labels."""
    assert_selection_rows(rows)
    normalized = [
        {
            key: row[key]
            for key in sorted(row)
            if key not in {"test_label", "test_labels", "test_RMSE", "test_MAE"}
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_state_contract(model: torch.nn.Module) -> dict[str, Any]:
    state = model.state_dict()
    keys = get_regression_state_keys(model)
    if not keys:
        raise RuntimeError("Regression aggregation key set is empty")
    missing = [key for key in keys if key not in state]
    if missing:
        raise RuntimeError(f"Aggregation keys missing from model state: {missing}")
    parameter_names = {name for name, _ in model.named_parameters()}
    aggregated_parameters = parameter_names.intersection(keys)
    prefixes = (
        "reg_proj.", "reg_transformer.", "reg_attn.", "reg_attn_linear.",
        "reg_stats_proj.", "reg_response_adapter.", "reg_tcn_adapter.",
        "reg_ratio_adapter.", "gas_embedding.", "reg_shared_trunk.",
        "reg_residual_heads.", "reg_heads.",
    )
    exact = {
        "proto_scale", "proto_bias", "proto_conc", "conc_directions",
        "conc_scale", "conc_bias",
    }
    expected_parameters = {
        name
        for name in parameter_names
        if name.startswith(prefixes) or name in exact
    }
    omitted = sorted(expected_parameters - aggregated_parameters)
    unexpected = sorted(aggregated_parameters - expected_parameters)
    if omitted or unexpected:
        raise RuntimeError(
            f"Regression aggregation scope mismatch: omitted={omitted}, "
            f"unexpected={unexpected}"
        )
    return {
        "model_class": type(model).__name__,
        "total_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "state_tensor_count": len(keys),
        "parameter_tensor_count": len(aggregated_parameters),
        "parameter_count": int(
            sum(dict(model.named_parameters())[name].numel()
                for name in aggregated_parameters)
        ),
        "state_keys": keys,
    }


def regression_config_snapshot(config: Any) -> dict[str, Any]:
    names = (
        "SEED", "NUM_CLASSES", "NUM_PHASES", "TCN_NORM",
        "USE_DUAL_PROJ", "REG_GRAD_DETACH", "REG_HEAD_DEPTH",
        "REG_OUTPUT_MODE", "REG_WINDOW_STATS", "REG_WINDOW_STATS_MODE",
        "REG_WINDOW_STATS_DIM", "REG_RESPONSE_BRANCH", "REG_DCT_K",
        "REG_DCT_GAMMA_INIT", "REG_DCT_DROPOUT", "REG_TCN_ADAPTER",
        "REG_USE_SHARED_TRUNK", "USE_REG_RATIO_BRANCH",
    )
    return {name: getattr(config, name) for name in names}


def load_b5_strict(
    model: torch.nn.Module, checkpoint: Path
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("model_state", payload)
    if not isinstance(source, dict):
        raise ValueError("B5 checkpoint does not contain a model state dictionary")
    missing, unexpected = model.load_state_dict(source, strict=False)
    allowed_missing = set(get_regression_state_keys(model))
    illegal_missing = sorted(set(missing) - allowed_missing)
    illegal_unexpected = sorted(unexpected)
    if illegal_missing or illegal_unexpected or set(missing) != allowed_missing:
        raise ValueError(
            "B5 checkpoint/model contract mismatch: "
            f"illegal_missing={illegal_missing}, unexpected={illegal_unexpected}, "
            f"allowed_missing_not_reported={sorted(allowed_missing - set(missing))}"
        )
    init_regression_branch_from_classifier(model)
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_state_tensors": len(source),
        "allowed_regression_missing_tensors": len(missing),
    }


def assert_identical_initialization(
    template: torch.nn.Module, copies: Mapping[int, torch.nn.Module]
) -> str:
    keys = get_regression_state_keys(template)
    expected = state_sha256(template.state_dict(), keys)
    hashes = {
        int(cid): state_sha256(model.state_dict(), keys)
        for cid, model in copies.items()
    }
    if any(value != expected for value in hashes.values()):
        raise RuntimeError(f"Local regression initialization mismatch: {hashes}")
    return expected


def traced_federated_train(
    model: torch.nn.Module,
    loaders: Mapping[int, Any],
    sample_counts: Mapping[int, int],
    *,
    device: torch.device,
    rounds: int,
    total_steps: int,
    learning_rate: float,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    if set(loaders) != set(SOURCE_CLIENTS):
        raise RuntimeError(f"Expected isolated C1/C2 loaders, got {sorted(loaders)}")
    keys = get_regression_state_keys(model)
    nonreg_keys = sorted(set(model.state_dict()) - set(keys))
    frozen_hash = state_sha256(model.state_dict(), nonreg_keys)
    base_steps, extra = divmod(total_steps, rounds)
    final_locals: dict[int, dict[str, torch.Tensor]] = {}
    trace: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        steps = base_steps + int(round_index <= extra)
        before = copy.deepcopy(model.state_dict())
        local_states: dict[int, dict[str, torch.Tensor]] = {}
        for client_id in SOURCE_CLIENTS:
            local = copy.deepcopy(model).to(device)
            local.load_state_dict(before, strict=True)
            train_regression_local(
                local,
                loaders[client_id],
                device,
                steps=steps,
                lr=learning_rate,
                stage_name=f"RS_client{client_id}_round{round_index}",
            )
            if state_sha256(local.state_dict(), nonreg_keys) != frozen_hash:
                raise RuntimeError(
                    f"Frozen classifier/backbone state changed on C{client_id}"
                )
            local_states[client_id] = {
                key: value.detach().cpu().clone()
                for key, value in local.state_dict().items()
            }
        averaged = fedavg_regression_states(
            local_states, dict(sample_counts), keys, device
        )
        updated = copy.deepcopy(before)
        for key, value in averaged.items():
            updated[key] = value.to(updated[key]).type_as(updated[key])
        model.load_state_dict(updated, strict=True)
        if state_sha256(model.state_dict(), nonreg_keys) != frozen_hash:
            raise RuntimeError("FedAvg changed frozen classifier/backbone state")
        trace.append(
            {
                "round": round_index,
                "steps_per_client": steps,
                "global_before_sha256": state_sha256(before, keys),
                "client_regression_sha256": {
                    str(cid): state_sha256(state, keys)
                    for cid, state in local_states.items()
                },
                "global_after_sha256": state_sha256(model.state_dict(), keys),
                "sample_counts": {str(k): int(v) for k, v in sample_counts.items()},
            }
        )
        final_locals = local_states
    return final_locals, trace


def predict_source_models(
    classifier: Any,
    models: Mapping[str, torch.nn.Module],
    dataset: Any,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if isinstance(classifier, torch.nn.Module):
        classifier.eval()
    for model in models.values():
        model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for x, true_cls, y_reg, phase, client_ids, row_ids in loader:
            x = x.to(device)
            phase = phase.to(device).long()
            if hasattr(classifier, "classify"):
                _logits, predicted = classifier.classify(
                    x.detach().cpu().numpy()
                )
                route = torch.from_numpy(
                    np.asarray(predicted, dtype=np.int64)
                ).to(device)
            else:
                logits, _, _ = classifier(x)
                route = logits.argmax(dim=1)
            columns: dict[str, np.ndarray] = {}
            for name, model in models.items():
                _, _, reg_feat = model(x)
                pred_norm = model.forward_reg(
                    reg_feat, y_cls=route, y_phase=phase
                )
                ppm = denormalize_by_class(pred_norm, route)
                values = ppm.detach().cpu().numpy().astype(float)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"{name} emitted NaN/Inf ppm")
                columns[name] = values
            for index in range(len(row_ids)):
                item = {
                    "client": f"C{int(client_ids[index])}",
                    "sample_index": int(row_ids[index]),
                    "pred_class": int(route[index].cpu()),
                    "route_class": int(route[index].cpu()),
                    "true_class": int(true_cls[index]),
                }
                item.update({key: float(value[index]) for key, value in columns.items()})
                records.append(item)
    return records


def attach_neural_predictions(
    rows: Sequence[dict[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    by_index = {int(row["sample_index"]): row for row in predictions}
    if len(by_index) != len(predictions):
        raise ValueError("Duplicate neural prediction row IDs")
    output: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["sample_index"])
        pred = by_index.get(index)
        if pred is None:
            raise ValueError(f"Missing neural prediction for {split} row {index}")
        item = dict(row)
        item["split"] = split
        item["pred_class"] = int(pred["pred_class"])
        item["route_class"] = int(pred["route_class"])
        for key in ("pred_C1", "pred_C2", "pred_FedAvg"):
            item[key] = float(pred[key])
        output.append(item)
    if len(output) != len(predictions):
        raise ValueError("Neural prediction/feature row count mismatch")
    return output


def add_variant_features(
    rows: Sequence[dict[str, Any]], variant: str
) -> tuple[list[dict[str, Any]], list[str]]:
    rich_names = sorted(rows[0]["feature_dict"])
    schema = list(feature_schema(rich_names, variant))
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        features = dict(item["feature_dict"])
        for name in VARIANT_FEATURES[variant]:
            features[f"srcpred_{name}"] = float(item[name])
        if sorted(features) != sorted(schema):
            raise RuntimeError(f"{variant} feature schema drift")
        item["feature_dict"] = features
        result.append(item)
    return result, schema


def fit_target_variant_calibration(
    calibration_rows: Sequence[dict[str, Any]],
    variant: str,
    val_ratio: float,
) -> tuple[
    dict[tuple[str, int], Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    list[str],
]:
    """Select on calibration-validation, then refit on full calibration.

    This function deliberately has no test-row argument.  The returned refit
    models may be applied to test only after ``decision_gate.json`` has been
    persisted.
    """
    calibration, schema = add_variant_features(calibration_rows, variant)
    refit_models: dict[tuple[str, int], Any] = {}
    selection_models: dict[tuple[str, int], Any] = {}
    selection: list[dict[str, Any]] = []
    fit_row_ids: set[int] = set()
    validation_row_ids: set[int] = set()
    validation_rows: list[dict[str, Any]] = []
    for class_id in sorted(CLASS_NAMES):
        class_rows = [
            row for row in calibration if int(row["true_class"]) == class_id
        ]
        fit_rows, val_rows = deterministic_train_val(class_rows, val_ratio)
        for row in val_rows:
            row["selection_split"] = MODEL_SELECTION_SPLIT
        assert_selection_rows(val_rows)
        if fit_row_ids.intersection(int(row["sample_index"]) for row in val_rows):
            raise RuntimeError("Calibration fit/validation overlap")
        fit_row_ids.update(int(row["sample_index"]) for row in fit_rows)
        validation_row_ids.update(int(row["sample_index"]) for row in val_rows)
        _refit_model, audit = fit_select_refit(
            fit_rows, val_rows, schema, RIDGE_ALPHAS
        )
        best_alpha = float(audit["best_alpha"])
        selection_models[("C5", class_id)] = fit_ridge(
            fit_rows, schema, best_alpha
        )
        refit_models[("C5", class_id)] = fit_ridge(
            class_rows, schema, best_alpha
        )
        validation_rows.extend(val_rows)
        selection.append(
            {
                "variant": variant,
                "client": "C5",
                "class_id": class_id,
                "gas": CLASS_NAMES[class_id],
                "selection_split": MODEL_SELECTION_SPLIT,
                "fit_n": len(fit_rows),
                "validation_n": len(val_rows),
                "best_alpha": best_alpha,
                "validation_RMSE": audit["best_val_RMSE"],
                "feature_dimension": len(schema),
            }
        )
    if fit_row_ids.intersection(validation_row_ids):
        raise RuntimeError("Calibration fit/validation row identity overlap")
    if len(validation_rows) != len(validation_row_ids):
        raise RuntimeError("Calibration-validation row IDs are not unique")
    predicted_validation = apply_client_models(
        validation_rows, selection_models, variant
    )
    for row in predicted_validation:
        row[f"{variant}_ppm"] = float(row[f"{variant}_ppm"])
    # RidgeHead.coef already includes the intercept as its first element.
    parameter_count = sum(len(model.coef) for model in refit_models.values())
    return (
        refit_models,
        predicted_validation,
        selection,
        parameter_count,
        schema,
    )


def apply_target_variant_test(
    models: Mapping[tuple[str, int], Any],
    test_rows: Sequence[dict[str, Any]],
    variant: str,
    expected_schema: Sequence[str],
) -> list[dict[str, Any]]:
    test, test_schema = add_variant_features(test_rows, variant)
    if list(expected_schema) != test_schema:
        raise RuntimeError(f"{variant} calibration/test feature schema mismatch")
    predicted = apply_client_models(list(test), dict(models), variant)
    values = np.asarray(
        [float(row[f"{variant}_ppm"]) for row in predicted], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"{variant} emitted non-finite C5 test predictions")
    return predicted


def metric_rows(
    rows: Sequence[Mapping[str, Any]], variant: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_key = f"{variant}_ppm"
    true = np.asarray([float(row["true_ppm"]) for row in rows])
    pred = np.asarray([float(row[pred_key]) for row in rows])
    classes = np.asarray([int(row["true_class"]) for row in rows])
    error = pred - true
    overall = {
        "variant": variant,
        "N": len(rows),
        "ALL_RMSE": float(np.sqrt(np.mean(error ** 2))),
        "ALL_MAE": float(np.mean(np.abs(error))),
        "NRMSE": float(
            np.sqrt(np.mean((error / np.asarray(
                [
                    max(
                        float(row.get("range_ppm", class_range(int(row["true_class"])))),
                        1e-12,
                    )
                    for row in rows
                ]
            )) ** 2))
        ),
        "S_CC_RMSE": float(
            np.sqrt(np.mean(error[np.asarray(
                [bool(int(row.get("route_class", -1)) == int(row["true_class"]))
                 for row in rows]
            )] ** 2))
        ),
        "CO_RMSE": float(np.sqrt(np.mean(error[classes == 1] ** 2))),
        "CO_high_200_250_RMSE": float(np.sqrt(np.mean(
            error[(classes == 1) & (true >= 200.0) & (true <= 250.0)] ** 2
        ))),
    }
    per_gas: list[dict[str, Any]] = []
    for class_id, gas in sorted(CLASS_NAMES.items()):
        mask = classes == class_id
        per_gas.append(
            {
                "variant": variant,
                "class_id": class_id,
                "gas": gas,
                "N": int(mask.sum()),
                "RMSE": float(np.sqrt(np.mean(error[mask] ** 2))),
                "MAE": float(np.mean(np.abs(error[mask]))),
            }
        )
    return overall, per_gas


def normalize_rs0_validation_rows(
    path: str | Path,
    expected_row_ids: set[int],
    observed_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_csv(path)
    if len(rows) != len(expected_row_ids):
        raise RuntimeError(
            f"RS0 calibration-validation row count mismatch: "
            f"{len(rows)} != {len(expected_row_ids)}"
        )
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in by_index:
            raise RuntimeError(f"Duplicate RS0 calibration-validation row: {index}")
        item = dict(row)
        item["selection_split"] = MODEL_SELECTION_SPLIT
        item["RS0_pooled_source_ppm"] = float(
            item["target_ridge_plus_source_preds_ppm"]
        )
        by_index[index] = item
    if set(by_index) != expected_row_ids:
        missing = sorted(expected_row_ids - set(by_index))
        extra = sorted(set(by_index) - expected_row_ids)
        raise RuntimeError(
            f"RS0 calibration-validation identity mismatch: "
            f"missing={missing}, extra={extra}"
        )
    normalized = [by_index[index] for index in sorted(by_index)]
    assert_selection_rows(normalized)
    observed_by_index = {
        int(row["sample_index"]): row for row in observed_rows
    }
    mismatches = [
        index
        for index, reference in by_index.items()
        if int(reference["pred_class"])
        != int(observed_by_index[index]["pred_class"])
    ]
    if mismatches:
        raise RuntimeError(
            "Canonical B5 calibration-validation route differs from frozen "
            f"RS0 reference at {len(mismatches)} rows"
        )
    return normalized


def freeze_decision_gate(
    validation_metrics: Mapping[str, Mapping[str, Any]],
    validation_per_gas: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    formal_commit: str,
) -> dict[str, Any]:
    required = set(VARIANT_FEATURES)
    if set(validation_metrics) != required:
        raise ValueError(
            f"Decision gate requires exactly RS0--RS4 metrics: "
            f"{sorted(set(validation_metrics))}"
        )
    rmse = {
        variant: float(validation_metrics[variant]["ALL_RMSE"])
        for variant in sorted(required)
    }
    if not all(np.isfinite(value) and value >= 0 for value in rmse.values()):
        raise ValueError(f"Invalid calibration-validation RMSE values: {rmse}")
    rs0 = rmse["RS0_pooled_source"]
    if rs0 <= 0:
        raise ValueError("RS0 calibration-validation RMSE must be positive")
    delta = 100.0 * (rmse["RS3_local_plus_fedavg"] / rs0 - 1.0)
    threshold_tolerance = 1e-12
    local_value = rmse["RS1_local_experts"] < rmse["RS4_rich_only"]
    fedavg_value = rmse["RS2_fedavg_prior"] < rmse["RS4_rich_only"]
    if delta < 0:
        selected_candidate = "RS3_local_plus_fedavg"
        selection_status = "candidate"
        selection_reason = "RS3 calibration-validation RMSE is better than RS0"
    elif delta <= 5.0 + threshold_tolerance:
        selected_candidate = "RS3_local_plus_fedavg"
        selection_status = "paper_preferred_candidate"
        selection_reason = (
            "RS3 calibration-validation degradation versus RS0 is at most 5%"
        )
    elif delta <= 10.0 + threshold_tolerance:
        selected_candidate = None
        selection_status = "inconclusive"
        selection_reason = (
            "RS3 calibration-validation degradation versus RS0 is between 5% and 10%"
        )
    else:
        selected_candidate = "RS0_pooled_source"
        selection_status = "no_promotion"
        selection_reason = (
            "RS3 calibration-validation degradation versus RS0 exceeds 10%"
        )
    per_gas = {
        variant: {
            str(row["gas"]): float(row["RMSE"])
            for row in validation_per_gas[variant]
        }
        for variant in sorted(required)
    }
    source_value_gases = [
        gas
        for gas in sorted(per_gas["RS4_rich_only"])
        if min(
            per_gas["RS1_local_experts"][gas],
            per_gas["RS2_fedavg_prior"][gas],
        )
        < per_gas["RS4_rich_only"][gas]
    ]
    single_gas_dependency = len(source_value_gases) == 1
    if (
        5.0 + threshold_tolerance
        < delta
        <= 10.0 + threshold_tolerance
    ) or (
        delta <= 5.0 + threshold_tolerance and single_gas_dependency
    ):
        recommendation = "INCONCLUSIVE"
    elif delta <= 5.0 + threshold_tolerance and (local_value or fedavg_value):
        recommendation = "PROMOTE_FOR_CONFIRMATION"
    else:
        recommendation = "STOP_FEDERATED_REGRESSION"
    return {
        "schema_version": "iotj.federated_source_regression_decision_gate.v1",
        "formal_run_commit": formal_commit,
        "selection_frozen_at": utc_now(),
        "selection_scope": "C5_calibration_validation_only",
        "selection_metric": "ALL_RMSE_ppm_lower_is_better",
        "calibration_validation_rmse": rmse,
        "calibration_validation_per_gas_rmse": per_gas,
        "RS0_calibration_val_RMSE": rmse["RS0_pooled_source"],
        "RS1_calibration_val_RMSE": rmse["RS1_local_experts"],
        "RS2_calibration_val_RMSE": rmse["RS2_fedavg_prior"],
        "RS3_calibration_val_RMSE": rmse["RS3_local_plus_fedavg"],
        "RS4_calibration_val_RMSE": rmse["RS4_rich_only"],
        "RS3_relative_delta_vs_RS0_percent": delta,
        "RS1_better_than_RS4": local_value,
        "RS2_better_than_RS4": fedavg_value,
        "source_prior_value_gases_vs_RS4": source_value_gases,
        "single_gas_dependency_on_calibration_validation": single_gas_dependency,
        "selected_candidate": selected_candidate,
        "selection_status": selection_status,
        "selection_reason": selection_reason,
        "next_stage_recommendation": recommendation,
        "test_opened_after_selection": False,
        "final_test_evaluation_timestamp": None,
        "test_metrics_used_for_selection": False,
    }


def source_prediction_mechanism(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    names = ("pred_C1", "pred_C2", "pred_FedAvg")
    values = np.asarray(
        [[float(row[name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid source prediction matrix for {scope}")
    correlation = np.corrcoef(values, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError(f"Non-finite source prediction correlation for {scope}")
    pairs: dict[str, Any] = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            diff = values[:, left] - values[:, right]
            pairs[f"{names[left]}__{names[right]}"] = {
                "pearson_correlation": float(correlation[left, right]),
                "mean_absolute_disagreement_ppm": float(np.mean(np.abs(diff))),
                "rms_disagreement_ppm": float(np.sqrt(np.mean(diff ** 2))),
                "p95_absolute_disagreement_ppm": float(
                    np.percentile(np.abs(diff), 95)
                ),
                "max_absolute_disagreement_ppm": float(np.max(np.abs(diff))),
            }
    spread = np.ptp(values, axis=1)
    return {
        "scope": scope,
        "N": int(values.shape[0]),
        "pairs": pairs,
        "three_model_spread_ppm": {
            "mean": float(np.mean(spread)),
            "p95": float(np.percentile(spread, 95)),
            "max": float(np.max(spread)),
        },
    }


def route_accuracy(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError("Route accuracy requires non-empty rows")
    return float(np.mean([
        int(row["pred_class"]) == int(row["true_class"]) for row in rows
    ]))


def protocol_payload(
    root: Path,
    protocol: Protocol,
    data_root: Path,
    classifier: Path,
    model_contract: Mapping[str, Any],
    frozen_before: Mapping[str, str],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for client_id, splits in ((1, ("train",)), (2, ("train",)), (5, ("calibration", "test"))):
        for split in splits:
            path = data_root / f"client_{client_id}" / f"{split}_features.npy"
            counts[f"C{client_id}_{split}"] = int(
                np.load(path, mmap_mode="r", allow_pickle=True).shape[0]
            )
    return {
        **asdict(protocol),
        "actual_training_strength": training_strength(
            protocol.source_rounds, protocol.source_steps_per_client
        ),
        "git_commit": git_commit(),
        "dataset_path": str(data_root.resolve()),
        "classifier_checkpoint": str(classifier.resolve()),
        "classifier_sha256": sha256_file(classifier),
        "roles": {"C1": "source_local", "C2": "source_local", "C5": "target"},
        "row_counts": counts,
        "variants": VARIANT_FEATURES,
        "ridge_alphas": RIDGE_ALPHAS,
        "regression_architecture": {
            "factory": "gaps_flower.regression_task.create_regression_model",
            "profile": "FLConfig defaults frozen at experiment manifest",
            **dict(model_contract),
        },
        "source_raw_rows_pooled": {
            "RS0_pooled_source": True,
            "RS1_local_experts": False,
            "RS2_fedavg_prior": False,
            "RS3_local_plus_fedavg": False,
            "RS4_rich_only": False,
        },
        "formal_output_files": FORMAL_OUTPUT_FILES,
        "frozen_evidence_sha256_before": dict(frozen_before),
        "runtime_or_qc_modified": False,
        "test_used_for_fit_select_or_refit": False,
    }


def run_contract_check(args: argparse.Namespace) -> dict[str, Any]:
    root = Path.cwd()
    data_root = Path(args.data_root)
    classifier = Path(args.classifier_checkpoint)
    frozen = frozen_evidence_hashes(root)
    config = make_regression_config(
        device=args.device,
        batch_size=args.batch_size,
        local_steps=args.source_steps,
        lr=args.learning_rate,
    )
    model = create_regression_model(config)
    b5 = load_b5_strict(model, classifier)
    model_contract = validate_state_contract(model)
    copies = {cid: copy.deepcopy(model) for cid in SOURCE_CLIENTS}
    init_hash = assert_identical_initialization(model, copies)
    canonical_classifier = C5H8Runtime.from_runtime_contract(
        Path(args.runtime_contract), device=args.device
    )
    result = {
        "status": "contract_verified",
        "b5_contract": b5,
        "regression_contract": {
            **model_contract,
            "config": regression_config_snapshot(config),
        },
        "initial_regression_sha256": init_hash,
        "canonical_classifier_route": {
            "runtime_contract": str(Path(args.runtime_contract).resolve()),
            "runtime_contract_sha256": sha256_file(args.runtime_contract),
            "model_class": type(canonical_classifier.model).__name__,
        },
        "feature_schemas": {
            name: list(feature_schema(("rich_feature_placeholder",), name))
            for name in ("RS1_local_experts", "RS2_fedavg_prior", "RS3_local_plus_fedavg")
        },
        "frozen_evidence_sha256": frozen,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_formal(args: argparse.Namespace, *, run_status: str = "formal") -> None:
    root = Path.cwd()
    output = Path(args.output_dir)
    require_new_empty_output(output)
    protocol = Protocol(
        source_rounds=args.source_rounds,
        source_steps_per_client=args.source_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    data_root = Path(args.data_root)
    classifier_path = Path(args.classifier_checkpoint)
    frozen_before = frozen_evidence_hashes(root)
    device = torch.device(args.device)
    config = make_regression_config(
        device=args.device,
        batch_size=args.batch_size,
        local_steps=args.source_steps,
        lr=args.learning_rate,
    )
    global_model = create_regression_model(config).to(device)
    b5_contract = load_b5_strict(global_model, classifier_path)
    state_contract = validate_state_contract(global_model)
    loaders, sample_counts = build_source_regression_loaders(
        str(data_root), list(SOURCE_CLIENTS), args.batch_size
    )
    if set(loaders) != set(SOURCE_CLIENTS):
        raise RuntimeError("C1/C2 isolated source loaders are both required")
    initial_copies = {cid: copy.deepcopy(global_model) for cid in SOURCE_CLIENTS}
    init_hash = assert_identical_initialization(global_model, initial_copies)
    local_states, trace = traced_federated_train(
        global_model,
        loaders,
        sample_counts,
        device=device,
        rounds=args.source_rounds,
        total_steps=args.source_steps,
        learning_rate=args.learning_rate,
    )
    local_models: dict[int, torch.nn.Module] = {}
    local_manifest: list[dict[str, Any]] = []
    for client_id in SOURCE_CLIENTS:
        model = create_regression_model(config).to(device)
        model.load_state_dict(local_states[client_id], strict=True)
        path = output / f"regression_source_client{client_id}_local.pth"
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "model_state": local_states[client_id],
                "client_id": client_id,
                "n_samples": int(sample_counts[client_id]),
                "classifier_sha256": b5_contract["checkpoint_sha256"],
                "initial_regression_sha256": init_hash,
            },
            path,
        )
        local_models[client_id] = model
        local_manifest.append(
            {
                "client_id": client_id,
                "data_scope": f"C{client_id}_train_only",
                "n_samples": int(sample_counts[client_id]),
                "checkpoint": str(path),
                "sha256": sha256_file(path),
            }
        )
    global_path = output / "regression_fedavg_global.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "model_state": {
                key: value.detach().cpu() for key, value in global_model.state_dict().items()
            },
            "sample_counts": sample_counts,
            "classifier_sha256": b5_contract["checkpoint_sha256"],
            "initial_regression_sha256": init_hash,
            "aggregation_trace": trace,
        },
        global_path,
    )
    canonical_classifier = C5H8Runtime.from_runtime_contract(
        Path(args.runtime_contract), device=args.device
    )
    prediction_models = {
        "pred_C1": local_models[1],
        "pred_C2": local_models[2],
        "pred_FedAvg": global_model,
    }

    # Phase 1: calibration-only fit/selection.  No C5 test file is loaded
    # before decision_gate.json is persisted.
    calibration_dataset = load_split_arrays(
        data_root, [TARGET_CLIENT], "calibration"
    )
    calibration_neural = predict_source_models(
        canonical_classifier,
        prediction_models,
        calibration_dataset,
        device,
        args.batch_size,
    )
    calibration_rows = attach_neural_predictions(
        build_oracle_rows(data_root, ["C5"], "calibration"),
        calibration_neural,
        "calibration",
    )
    if len(calibration_rows) != 320:
        raise RuntimeError(
            f"Frozen C5 calibration count mismatch: {len(calibration_rows)} != 320"
        )
    selection_rows: list[dict[str, Any]] = []
    refit_models: dict[str, dict[tuple[str, int], Any]] = {}
    variant_schemas: dict[str, list[str]] = {}
    target_parameter_counts: dict[str, int] = {}
    validation_metrics: dict[str, dict[str, Any]] = {}
    validation_per_gas: dict[str, list[dict[str, Any]]] = {}
    validation_row_ids: set[int] | None = None
    validation_prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in (
        "RS4_rich_only",
        "RS1_local_experts",
        "RS2_fedavg_prior",
        "RS3_local_plus_fedavg",
    ):
        (
            models,
            validation_predicted,
            selected,
            target_params,
            schema,
        ) = fit_target_variant_calibration(
            calibration_rows,
            variant,
            protocol.target_validation_ratio,
        )
        current_ids = {
            int(row["sample_index"]) for row in validation_predicted
        }
        if validation_row_ids is None:
            validation_row_ids = current_ids
        elif current_ids != validation_row_ids:
            raise RuntimeError(
                f"{variant} calibration-validation row identity drift"
            )
        selection_rows.extend(selected)
        refit_models[variant] = models
        variant_schemas[variant] = schema
        target_parameter_counts[variant] = target_params
        validation_prediction_rows[variant] = validation_predicted
        overall, gas_rows = metric_rows(validation_predicted, variant)
        validation_metrics[variant] = overall
        validation_per_gas[variant] = gas_rows
    if validation_row_ids is None or len(validation_row_ids) != 80:
        raise RuntimeError(
            f"Expected 80 calibration-validation rows, got "
            f"{0 if validation_row_ids is None else len(validation_row_ids)}"
        )
    rs0_validation = normalize_rs0_validation_rows(
        args.pooled_rs0_validation_predictions,
        validation_row_ids,
        calibration_rows,
    )
    rs0_validation_overall, rs0_validation_gas = metric_rows(
        rs0_validation, "RS0_pooled_source"
    )
    validation_metrics["RS0_pooled_source"] = rs0_validation_overall
    validation_per_gas["RS0_pooled_source"] = rs0_validation_gas
    validation_prediction_rows["RS0_pooled_source"] = rs0_validation

    formal_commit = git_commit()
    decision_gate = freeze_decision_gate(
        validation_metrics,
        validation_per_gas,
        formal_commit=formal_commit,
    )
    decision_gate["canonical_B5_calibration_validation_route_parity"] = {
        "mismatch_count": 0,
        "N": len(rs0_validation),
        "route_accuracy": route_accuracy(rs0_validation),
    }
    decision_gate_path = output / "decision_gate.json"
    write_json(decision_gate_path, decision_gate)
    if json.loads(decision_gate_path.read_text(encoding="utf-8"))[
        "test_opened_after_selection"
    ] is not False:
        raise RuntimeError("Decision gate was not frozen before C5 test opening")

    # Phase 2: one-time test generalization evaluation after selection freeze.
    test_dataset = load_split_arrays(data_root, [TARGET_CLIENT], "test")
    test_neural = predict_source_models(
        canonical_classifier,
        prediction_models,
        test_dataset,
        device,
        args.batch_size,
    )
    test_rows = attach_neural_predictions(
        build_oracle_rows(data_root, ["C5"], "test"),
        test_neural,
        "test",
    )
    if len(test_rows) != 1360:
        raise RuntimeError(
            f"Frozen C5 test count mismatch: {len(test_rows)} != 1360"
        )
    summary_rows: list[dict[str, Any]] = []
    per_gas_rows: list[dict[str, Any]] = []
    prediction_rows: dict[int, dict[str, Any]] = {
        int(row["sample_index"]): {
            "client": "C5",
            "split": "test",
            "sample_index": int(row["sample_index"]),
            "true_class": int(row["true_class"]),
            "true_ppm": float(row["true_ppm"]),
            "pred_class": int(row["pred_class"]),
            "route_class": int(row["route_class"]),
            "pred_C1": float(row["pred_C1"]),
            "pred_C2": float(row["pred_C2"]),
            "pred_FedAvg": float(row["pred_FedAvg"]),
        }
        for row in test_rows
    }
    for variant in (
        "RS4_rich_only",
        "RS1_local_experts",
        "RS2_fedavg_prior",
        "RS3_local_plus_fedavg",
    ):
        predicted = apply_target_variant_test(
            refit_models[variant],
            test_rows,
            variant,
            variant_schemas[variant],
        )
        for row in predicted:
            prediction_rows[int(row["sample_index"])][f"{variant}_ppm"] = float(
                row[f"{variant}_ppm"]
            )
        overall, gas_rows = metric_rows(predicted, variant)
        overall.update(
            {
                "source_parameter_count": (
                    0 if variant == "RS4_rich_only"
                    else state_contract["parameter_count"]
                ),
                "source_model_count": {
                    "RS4_rich_only": 0,
                    "RS1_local_experts": 2,
                    "RS2_fedavg_prior": 1,
                    "RS3_local_plus_fedavg": 3,
                }[variant],
                "total_source_parameter_instances": (
                    0
                    if variant == "RS4_rich_only"
                    else state_contract["parameter_count"]
                    * {
                        "RS1_local_experts": 2,
                        "RS2_fedavg_prior": 1,
                        "RS3_local_plus_fedavg": 3,
                    }[variant]
                ),
                "pooled_raw_source_required": False,
                "target_calibration_parameter_count": target_parameter_counts[variant],
                "inference_feature_dimension": len(variant_schemas[variant]),
            }
        )
        summary_rows.append(overall)
        per_gas_rows.extend(gas_rows)
    pooled_rows = read_csv(args.pooled_rs0_predictions)
    pooled_by_index = {int(row["sample_index"]): row for row in pooled_rows}
    if len(pooled_by_index) != 1360:
        raise RuntimeError("RS0 pooled reference must contain exactly 1360 unique rows")
    rs0_metric_input: list[dict[str, Any]] = []
    for index, output_row in sorted(prediction_rows.items()):
        reference = pooled_by_index.get(index)
        if reference is None:
            raise RuntimeError(f"RS0 reference missing test row {index}")
        if (
            int(reference["true_class"]) != int(output_row["true_class"])
            or abs(float(reference["true_ppm"]) - float(output_row["true_ppm"])) > 1e-9
        ):
            raise RuntimeError(
                "RS0/frozen C5 row identity mismatch at sample_index "
                f"{index}: experiment=({output_row['true_class']},"
                f"{output_row['true_ppm']}), reference="
                f"({reference['true_class']},{reference['true_ppm']})"
            )
        if int(reference["pred_class"]) != int(output_row["pred_class"]):
            raise RuntimeError(
                "Canonical B5 test route differs from frozen RS0 reference at "
                f"sample_index {index}"
            )
        value = float(reference["target_ridge_plus_source_preds_ppm"])
        output_row["RS0_pooled_source_ppm"] = value
        item = dict(output_row)
        item["range_ppm"] = float(reference.get("range_ppm", 1.0))
        rs0_metric_input.append(item)
    rs0_overall, rs0_gas = metric_rows(rs0_metric_input, "RS0_pooled_source")
    rs0_overall.update(
        {
            "source_parameter_count": "see_frozen_R4_policy",
            "source_model_count": 3,
            "total_source_parameter_instances": "see_frozen_R4_policy",
            "pooled_raw_source_required": True,
            "target_calibration_parameter_count": "see_frozen_R4_policy",
            "inference_feature_dimension": "see_frozen_R4_policy",
        }
    )
    summary_rows.insert(0, rs0_overall)
    per_gas_rows = rs0_gas + per_gas_rows
    rs0_rmse = float(rs0_overall["ALL_RMSE"])
    comparison = []
    for row in summary_rows:
        delta = 100.0 * (float(row["ALL_RMSE"]) / rs0_rmse - 1.0)
        comparison.append(
            {
                "variant": row["variant"],
                "ALL_RMSE": row["ALL_RMSE"],
                "RS0_ALL_RMSE": rs0_rmse,
                "delta_vs_RS0_percent": delta,
                "test_role": "generalization_evaluation_only",
                "selection_source": "pre_frozen_calibration_decision_gate",
                "preselected_candidate": decision_gate["selected_candidate"],
                "test_metrics_used_for_selection": False,
            }
        )
    decision_gate["test_opened_after_selection"] = True
    decision_gate["final_test_evaluation_timestamp"] = utc_now()
    decision_gate["generalization_test_rmse"] = {
        row["variant"]: float(row["ALL_RMSE"]) for row in summary_rows
    }
    decision_gate["canonical_B5_test_route_parity"] = {
        "mismatch_count": 0,
        "N": len(test_rows),
        "route_accuracy": route_accuracy(test_rows),
    }
    write_json(decision_gate_path, decision_gate)
    mechanism = {
        "schema_version": "iotj.federated_source_prediction_mechanism.v1",
        "selection_use": False,
        "calibration_full": source_prediction_mechanism(
            calibration_rows, scope="C5_calibration_full"
        ),
        "test": source_prediction_mechanism(
            test_rows, scope="C5_test_generalization_only"
        ),
    }
    write_json(output / "source_prediction_mechanism.json", mechanism)
    frozen_after = frozen_evidence_hashes(root)
    if frozen_after != frozen_before:
        raise RuntimeError("Frozen runtime v4/parity evidence changed during experiment")
    manifest = protocol_payload(
        root, protocol, data_root, classifier_path,
        {
            **state_contract,
            "config": regression_config_snapshot(config),
            "b5_contract": b5_contract,
            "initial_regression_sha256": init_hash,
        },
        frozen_before,
    )
    manifest["run_status"] = run_status
    manifest["advancement_eligible"] = run_status == "formal"
    manifest["canonical_classifier_route"] = {
        "loader": "gaps_deploy.c5_h8_runtime.C5H8Runtime.from_runtime_contract",
        "runtime_contract": str(Path(args.runtime_contract).resolve()),
        "runtime_contract_sha256": sha256_file(args.runtime_contract),
        "model_class": type(canonical_classifier.model).__name__,
        "route_source": "argmax_frozen_B5_runtime_logits",
    }
    expected_strength = training_strength(
        protocol.source_rounds, protocol.source_steps_per_client
    )
    trace_allocation = [int(row["steps_per_client"]) for row in trace]
    if (
        trace_allocation != expected_strength["round_step_allocation_per_client"]
        or len(trace) != expected_strength["fedavg_execution_count"]
    ):
        raise RuntimeError(
            "Observed optimizer-step/FedAvg trace does not match frozen protocol"
        )
    manifest["actual_training_strength"].update(
        {
            "trace_verified": True,
            "observed_round_step_allocation_per_client": trace_allocation,
            "observed_fedavg_execution_count": len(trace),
        }
    )
    manifest["decision_gate"] = {
        "path": str(decision_gate_path),
        "sha256": sha256_file(decision_gate_path),
        "selected_candidate": decision_gate["selected_candidate"],
        "next_stage_recommendation": decision_gate["next_stage_recommendation"],
        "test_opened_after_selection": decision_gate[
            "test_opened_after_selection"
        ],
    }
    manifest["frozen_evidence_sha256_after"] = frozen_after
    write_json(output / "protocol_manifest.json", manifest)
    write_json(
        output / "topology_audit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "C1_data_scope": "client_1/train_* only",
            "C2_data_scope": "client_2/train_* only",
            "pooled_source_rows_in_RS1_RS2_RS3": False,
            "aggregation_scope": state_contract,
            "classifier_backbone_frozen": True,
            "test_used_for_fit_select_or_refit": False,
            "test_opened_after_calibration_selection_freeze": True,
            "actual_training_strength": manifest["actual_training_strength"],
            "aggregation_trace": trace,
        },
    )
    write_json(output / "source_local_model_manifest.json", local_manifest)
    write_json(
        output / "fedavg_model_manifest.json",
        {
            "checkpoint": str(global_path),
            "sha256": sha256_file(global_path),
            "sample_counts": sample_counts,
            "aggregation": "sample_count_weighted_regression_only_FedAvg",
            "trace": trace,
        },
    )
    write_csv(
        output / "calibration_selection.csv",
        selection_rows,
        (
            "variant", "client", "class_id", "gas", "selection_split",
            "fit_n", "validation_n", "best_alpha", "validation_RMSE",
            "feature_dimension",
        ),
    )
    prediction_fields = (
        "client", "split", "sample_index", "true_class", "true_ppm",
        "pred_class", "route_class", "pred_C1", "pred_C2", "pred_FedAvg",
        "RS0_pooled_source_ppm", "RS1_local_experts_ppm",
        "RS2_fedavg_prior_ppm", "RS3_local_plus_fedavg_ppm",
        "RS4_rich_only_ppm",
    )
    write_csv(
        output / "test_predictions.csv",
        [prediction_rows[key] for key in sorted(prediction_rows)],
        prediction_fields,
    )
    summary_fields = (
        "variant", "N", "ALL_RMSE", "ALL_MAE", "NRMSE", "S_CC_RMSE",
        "CO_RMSE", "CO_high_200_250_RMSE", "source_parameter_count",
        "source_model_count", "total_source_parameter_instances",
        "pooled_raw_source_required",
        "target_calibration_parameter_count", "inference_feature_dimension",
    )
    write_csv(output / "regression_variant_summary.csv", summary_rows, summary_fields)
    write_csv(
        output / "per_gas_summary.csv",
        per_gas_rows,
        ("variant", "class_id", "gas", "N", "RMSE", "MAE"),
    )
    write_csv(
        output / "comparison_vs_pooled_h8.csv",
        comparison,
        (
            "variant", "ALL_RMSE", "RS0_ALL_RMSE",
            "delta_vs_RS0_percent", "test_role", "selection_source",
            "preselected_candidate", "test_metrics_used_for_selection",
        ),
    )
    (output / "README.md").write_text(
        "# IoT-J Federated Source Regression Prior\n\n"
        f"Status: {run_status} RS0–RS4 experiment output. "
        "QC and runtime are out of scope.\n\n"
        "Selection uses C5 calibration-validation only. C5 test is evaluated once "
        "after decision_gate.json is persisted. Test metrics are generalization-only "
        "and cannot change the selected candidate. RS0 is the immutable pooled-source "
        "R4 reference."
        + (
            "\n"
            if run_status == "formal"
            else "\n\nThis smoke output is not advancement-eligible evidence.\n"
        ),
        encoding="utf-8",
    )
    print(f"{run_status} RS0–RS4 output written to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default=(
            "results/iotj_ecs_c2_representative_20260720/raw/"
            "c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/"
            "training/server_round_025_adapted.pth"
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
        "--pooled-rs0-predictions",
        default=(
            "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/"
            "target_predictions_plus_source_preds.csv"
        ),
    )
    parser.add_argument(
        "--pooled-rs0-validation-predictions",
        default=(
            "results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/"
            "target_validation_plus_source_preds.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/iotj_federated_source_regression_prior_20260723",
    )
    parser.add_argument("--source-rounds", type=int, default=3)
    parser.add_argument("--source-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--formal-run", action="store_true")
    parser.add_argument("--smoke-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.formal_run and args.smoke_run:
        raise ValueError("--formal-run and --smoke-run are mutually exclusive")
    if args.smoke_run:
        if args.source_rounds != 1 or args.source_steps != 1:
            raise ValueError(
                "Smoke execution is fixed to --source-rounds 1 --source-steps 1"
            )
        run_formal(args, run_status="smoke_only")
    elif args.formal_run:
        run_formal(args, run_status="formal")
    else:
        run_contract_check(args)


if __name__ == "__main__":
    main()
