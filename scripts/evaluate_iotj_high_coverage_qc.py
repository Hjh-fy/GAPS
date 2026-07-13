"""Build and evaluate deployment-visible high-coverage C5 QC policies."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CLASS_RANGES = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}
RAW_COMPONENTS = (
    "raw_risk_confidence",
    "raw_risk_prototype",
    "raw_risk_support",
    "raw_risk_expert_disagreement",
    "raw_risk_source_spread",
)


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("split")), _to_int(row.get("sample_index"))


def _index_rows(
    rows: Sequence[dict[str, Any]], name: str
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {name} row: {key}")
        indexed[key] = row
    return indexed


def merge_aligned_streams(
    base_rows: Sequence[dict[str, Any]],
    h23_rows: Sequence[dict[str, Any]],
    h8_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Merge exact C5 rows, using H2.3 rows as the evaluated row set."""
    base = _index_rows(base_rows, "base")
    h8 = _index_rows(h8_rows, "H8")
    features = _index_rows(feature_rows, "feature")
    h23 = _index_rows(h23_rows, "H2.3")
    output: list[dict[str, Any]] = []
    for key in sorted(h23, key=lambda item: item[1]):
        if key[0] != split:
            continue
        if key not in base:
            raise ValueError(f"missing base row: {key}")
        if key not in h8:
            raise ValueError(f"missing H8 row: {key}")
        if key not in features:
            raise ValueError(f"missing feature row: {key}")
        item = dict(base[key])
        item.update(h23[key])
        item.update(h8[key])
        item.update(features[key])
        item["split"] = split
        item["sample_index"] = key[1]
        output.append(item)
    return output


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _feature_names(row: dict[str, Any]) -> tuple[str, ...]:
    names = tuple(sorted(key for key in row if key.startswith("cls_feat_")))
    if not names:
        raise ValueError("classification feature columns are required")
    return names


def _feature_vector(row: dict[str, Any], names: Sequence[str]) -> np.ndarray:
    vector = np.asarray([_to_float(row.get(name), np.nan) for name in names], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("classification feature vector contains non-finite values")
    return vector


def fit_feature_reference(calibration_fit_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fit class-phase feature cells from labeled target calibration-fit rows."""
    rows = list(calibration_fit_rows)
    if not rows:
        raise ValueError("calibration-fit rows are required")
    names = _feature_names(rows[0])
    features = np.stack([_feature_vector(row, names) for row in rows], axis=0)
    global_var = np.var(features, axis=0) + 1e-6

    def make_cell(indices: Sequence[int]) -> dict[str, Any]:
        selected = features[np.asarray(indices, dtype=np.int64)]
        cell_var = np.var(selected, axis=0)
        scale = np.sqrt(0.5 * cell_var + 0.5 * global_var + 1e-6)
        return {
            "mean": selected.mean(axis=0),
            "scale": scale,
            "support": selected,
            "n": int(len(selected)),
        }

    class_phase_indices: dict[tuple[int, int], list[int]] = {}
    class_indices: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        class_id = _to_int(row.get("true_class"))
        phase_id = _to_int(row.get("phase"))
        class_phase_indices.setdefault((class_id, phase_id), []).append(index)
        class_indices.setdefault(class_id, []).append(index)
    return {
        "schema_version": 1,
        "feature_names": names,
        "selection_split": "calibration_fit",
        "cells": {
            f"{class_id}:{phase_id}": make_cell(indices)
            for (class_id, phase_id), indices in class_phase_indices.items()
        },
        "classes": {
            str(class_id): make_cell(indices)
            for class_id, indices in class_indices.items()
        },
        "global": make_cell(list(range(len(rows)))),
    }


def _reference_cell(
    reference: dict[str, Any], route_class: int, phase: int
) -> dict[str, Any]:
    return (
        reference["cells"].get(f"{route_class}:{phase}")
        or reference["classes"].get(str(route_class))
        or reference["global"]
    )


def raw_deployment_risk(
    row: dict[str, Any], reference: dict[str, Any]
) -> dict[str, float]:
    """Compute raw risk components without reading any test truth field."""
    names = tuple(reference["feature_names"])
    feature = _feature_vector(row, names)
    route_class = _to_int(row.get("route_class", row.get("pred_class")))
    phase = _to_int(row.get("phase"))
    cell = _reference_cell(reference, route_class, phase)
    mean = np.asarray(cell["mean"], dtype=np.float64)
    scale = np.maximum(np.asarray(cell["scale"], dtype=np.float64), 1e-6)
    support = np.asarray(cell["support"], dtype=np.float64)
    prototype_distance = float(np.sqrt(np.mean(((feature - mean) / scale) ** 2)))
    support_distance = float(
        np.min(np.sqrt(np.mean(((support - feature) / scale) ** 2, axis=1)))
    )
    entropy_risk = _to_float(row.get("deployment_risk_classifier_entropy"))
    margin_risk = _to_float(row.get("deployment_risk_margin"))
    route_range = CLASS_RANGES.get(route_class, 1.0)
    h23 = _to_float(row.get("h23_plus_ppm"))
    h8 = _to_float(row.get("target_ridge_plus_source_preds_ppm"))
    expert_disagreement = abs(h23 - h8) / max(route_range, 1e-12)
    source_predictions = np.asarray(
        [
            _to_float(row.get("H1_source_ridge_ppm")),
            _to_float(row.get("H2_source_per_gas_mlp_ppm")),
            _to_float(row.get("H3_source_shared_mlp_ppm")),
        ],
        dtype=np.float64,
    )
    return {
        "raw_risk_confidence": float(max(entropy_risk, margin_risk)),
        "raw_risk_prototype": prototype_distance,
        "raw_risk_support": support_distance,
        "raw_risk_expert_disagreement": float(expert_disagreement),
        "raw_risk_source_spread": float(np.std(source_predictions) / max(route_range, 1e-12)),
    }


def fit_component_calibrator(raw_validation_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = list(raw_validation_rows)
    if not rows:
        raise ValueError("calibration-validation risk rows are required")
    distributions: dict[str, list[float]] = {}
    for key in RAW_COMPONENTS:
        values = sorted(_to_float(row.get(key), np.nan) for row in rows)
        if not values or not np.isfinite(values).all():
            raise ValueError(f"non-finite calibration risk component: {key}")
        distributions[key] = values
    return {
        "schema_version": 1,
        "selection_split": "calibration_validation",
        "component_distributions": distributions,
    }


def _percentile(value: float, distribution: Sequence[float]) -> float:
    values = np.asarray(distribution, dtype=np.float64)
    return float(np.searchsorted(values, float(value), side="right") / max(len(values), 1))


def score_deployment_rows(
    raw_rows: Sequence[dict[str, Any]], calibrator: dict[str, Any]
) -> list[dict[str, Any]]:
    distributions = calibrator["component_distributions"]
    output: list[dict[str, Any]] = []
    for row in raw_rows:
        item = dict(row)
        percentiles = {
            key: _percentile(_to_float(item.get(key)), distributions[key])
            for key in RAW_COMPONENTS
        }
        item["deployment_risk_confidence"] = percentiles["raw_risk_confidence"]
        item["deployment_risk_feature"] = 0.5 * (
            percentiles["raw_risk_prototype"] + percentiles["raw_risk_support"]
        )
        item["deployment_risk_disagreement"] = 0.5 * (
            percentiles["raw_risk_expert_disagreement"]
            + percentiles["raw_risk_source_spread"]
        )
        item["deployment_risk_full"] = (
            item["deployment_risk_confidence"]
            + item["deployment_risk_feature"]
            + item["deployment_risk_disagreement"]
        ) / 3.0
        output.append(item)
    return output


def _quantile_higher(values: Sequence[float], quantile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0 or not np.isfinite(ordered).all():
        raise ValueError("finite calibration risk values are required")
    index = max(0, min(int(math.ceil(float(quantile) * len(ordered))) - 1, len(ordered) - 1))
    return float(ordered[index])


def fit_workpoints(
    validation_rows: Sequence[dict[str, Any]], score_key: str
) -> dict[str, Any]:
    rows = list(validation_rows)
    values = [_to_float(row.get(score_key), np.nan) for row in rows]
    workpoints: dict[str, dict[str, Any]] = {
        "FULL": {
            "target_accept_coverage": 1.0,
            "target_nonreject_coverage": 1.0,
            "accept_threshold": None,
            "reject_threshold": None,
        }
    }
    for name, accept_coverage, nonreject_coverage in (
        ("HC95", 0.95, 0.9875),
        ("HC90", 0.90, 0.975),
    ):
        workpoints[name] = {
            "target_accept_coverage": accept_coverage,
            "target_nonreject_coverage": nonreject_coverage,
            "accept_threshold": _quantile_higher(values, accept_coverage),
            "reject_threshold": _quantile_higher(values, nonreject_coverage),
        }
    return {
        "schema_version": 1,
        "selection_split": "calibration_validation",
        "score_key": score_key,
        "workpoints": workpoints,
    }


def apply_workpoint(
    rows: Sequence[dict[str, Any]], policy: dict[str, Any], workpoint: str
) -> list[dict[str, Any]]:
    settings = policy["workpoints"][workpoint]
    score_key = str(policy["score_key"])
    accept_threshold = settings["accept_threshold"]
    reject_threshold = settings["reject_threshold"]
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        risk = _to_float(item.get(score_key), np.nan)
        if workpoint == "FULL":
            decision = "accept"
        elif not np.isfinite(risk):
            decision = "reject"
        elif risk <= float(accept_threshold):
            decision = "accept"
        elif risk > float(reject_threshold):
            decision = "reject"
        else:
            decision = "review"
        item["qc_workpoint"] = workpoint
        item["qc_score_key"] = score_key
        item["qc_decision"] = decision
        output.append(item)
    return output


def _regression_metrics(
    rows: Sequence[dict[str, Any]], pred_key: str
) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        return {
            "N": 0,
            "RMSE": None,
            "MAE": None,
            "NRMSE": None,
            "P90AE": None,
            "Bias": None,
            "R2": None,
        }
    truth = np.asarray([_to_float(row.get("true_ppm"), np.nan) for row in selected])
    pred = np.asarray([_to_float(row.get(pred_key), np.nan) for row in selected])
    classes = np.asarray([_to_int(row.get("true_class")) for row in selected])
    valid = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[valid]
    pred = pred[valid]
    classes = classes[valid]
    if truth.size == 0:
        return {
            "N": 0,
            "RMSE": None,
            "MAE": None,
            "NRMSE": None,
            "P90AE": None,
            "Bias": None,
            "R2": None,
        }
    error = pred - truth
    ranges = np.asarray([CLASS_RANGES.get(int(cls), 1.0) for cls in classes])
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "N": int(truth.size),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "NRMSE": float(np.sqrt(np.mean((error / ranges) ** 2))),
        "P90AE": float(np.percentile(np.abs(error), 90)),
        "Bias": float(np.mean(error)),
        "R2": float(1.0 - np.sum(error**2) / ss_tot) if ss_tot > 1e-12 else None,
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def evaluate_workpoint(
    rows: Sequence[dict[str, Any]],
    pred_key: str,
    *,
    n_random: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a frozen QC workpoint and matched random flagging controls."""
    selected = list(rows)
    total = len(selected)
    statuses = np.asarray(
        [str(row.get("qc_decision", "reject")) for row in selected], dtype=object
    )
    accept_mask = statuses == "accept"
    review_mask = statuses == "review"
    reject_mask = statuses == "reject"
    flagged_mask = ~accept_mask
    true_class = np.asarray([_to_int(row.get("true_class")) for row in selected])
    pred_class = np.asarray(
        [_to_int(row.get("pred_class", row.get("route_class"))) for row in selected]
    )
    route_wrong = true_class != pred_class
    truth = np.asarray([_to_float(row.get("true_ppm"), np.nan) for row in selected])
    pred = np.asarray([_to_float(row.get(pred_key), np.nan) for row in selected])
    ranges = np.asarray([CLASS_RANGES.get(int(cls), 1.0) for cls in true_class])
    normalized_error = np.abs(pred - truth) / np.maximum(ranges, 1e-12)
    high_error = np.isfinite(normalized_error) & (normalized_error >= 0.10)
    class_correct = ~route_wrong

    def take(mask: np.ndarray) -> list[dict[str, Any]]:
        return [row for row, keep in zip(selected, mask.tolist()) if keep]

    rng = np.random.default_rng(int(seed))
    indices = np.arange(total)
    n_flagged = int(flagged_mask.sum())
    random_rmse: list[float] = []
    random_route_recall: list[float] = []
    random_high_recall: list[float] = []
    for _ in range(max(0, int(n_random))):
        random_flags = np.zeros(total, dtype=bool)
        if n_flagged:
            random_flags[rng.choice(indices, size=n_flagged, replace=False)] = True
        random_metrics = _regression_metrics(take(~random_flags), pred_key)
        if random_metrics["RMSE"] is not None:
            random_rmse.append(float(random_metrics["RMSE"]))
        random_route_recall.append(
            float((random_flags & route_wrong).sum() / max(int(route_wrong.sum()), 1))
        )
        random_high_recall.append(
            float((random_flags & high_error).sum() / max(int(high_error.sum()), 1))
        )

    return {
        "N": total,
        "accept_N": int(accept_mask.sum()),
        "review_N": int(review_mask.sum()),
        "reject_N": int(reject_mask.sum()),
        "automatic_yield": float(accept_mask.mean()) if total else 0.0,
        "review_rate": float(review_mask.mean()) if total else 0.0,
        "reject_rate": float(reject_mask.mean()) if total else 0.0,
        "nonreject_coverage": float((accept_mask | review_mask).mean()) if total else 0.0,
        "full_metrics": _regression_metrics(selected, pred_key),
        "accept_metrics": _regression_metrics(take(accept_mask), pred_key),
        "review_metrics": _regression_metrics(take(review_mask), pred_key),
        "reject_metrics": _regression_metrics(take(reject_mask), pred_key),
        "route_wrong_total": int(route_wrong.sum()),
        "route_wrong_flagged": int((route_wrong & flagged_mask).sum()),
        "route_wrong_recall": float(
            (route_wrong & flagged_mask).sum() / max(int(route_wrong.sum()), 1)
        ),
        "high_error_total": int(high_error.sum()),
        "high_error_flagged": int((high_error & flagged_mask).sum()),
        "high_error_recall": float(
            (high_error & flagged_mask).sum() / max(int(high_error.sum()), 1)
        ),
        "class_correct_false_flag_rate": float(
            (class_correct & flagged_mask).sum() / max(int(class_correct.sum()), 1)
        ),
        "random_control": {
            "iterations": int(n_random),
            "flagged_N": n_flagged,
            "accept_RMSE": _numeric_summary(random_rmse),
            "route_wrong_recall": _numeric_summary(random_route_recall),
            "high_error_recall": _numeric_summary(random_high_recall),
        },
    }


def select_score_family(
    validation_rows: Sequence[dict[str, Any]],
    pred_key: str,
    *,
    candidates: Sequence[str] = (
        "deployment_risk_full",
        "deployment_risk_disagreement",
        "deployment_risk_feature",
        "deployment_risk_confidence",
    ),
) -> dict[str, Any]:
    """Select a predeclared risk family using calibration-validation only."""
    audits: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, float, float, int], dict[str, Any]]] = []
    for order, score_key in enumerate(candidates):
        policy = fit_workpoints(validation_rows, score_key)
        annotated = apply_workpoint(validation_rows, policy, "HC95")
        report = evaluate_workpoint(annotated, pred_key, n_random=0)
        p90 = report["accept_metrics"].get("P90AE")
        key = (
            -(float(report["route_wrong_recall"]) + float(report["high_error_recall"])),
            float(p90) if p90 is not None else float("inf"),
            float(report["class_correct_false_flag_rate"]),
            order,
        )
        audit = {
            "score_key": score_key,
            "selection_workpoint": "HC95",
            "selection_key": list(key),
            "policy": policy,
            "report": report,
        }
        audits.append(audit)
        ranked.append((key, audit))
    selected = min(ranked, key=lambda item: item[0])[1]
    return {
        "selection_split": "calibration_validation",
        "selected_score": selected["score_key"],
        "selected_policy": selected["policy"],
        "selected_report": selected["report"],
        "candidate_audit": audits,
    }


def evaluate_ranking_curve(
    rows: Sequence[dict[str, Any]],
    score_key: str,
    pred_key: str,
    *,
    coverages: Sequence[float] = (1.0, 0.98, 0.95, 0.925, 0.90),
    n_random: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate exact test ranking points; these are not deployable thresholds."""
    selected = list(rows)
    total = len(selected)
    ordered_indices = sorted(
        range(total),
        key=lambda index: (
            _to_float(selected[index].get(score_key), float("inf")),
            _to_int(selected[index].get("sample_index"), index),
        ),
    )
    output: list[dict[str, Any]] = []
    for coverage in coverages:
        if not 0.0 < float(coverage) <= 1.0:
            raise ValueError(f"coverage must be in (0,1], got {coverage}")
        n_accept = max(0, min(int(round(float(coverage) * total)), total))
        accept_indices = set(ordered_indices[:n_accept])
        annotated = []
        for index, row in enumerate(selected):
            item = dict(row)
            item["qc_decision"] = "accept" if index in accept_indices else "review"
            item["qc_workpoint"] = f"ranking_{coverage:.4f}"
            item["qc_score_key"] = score_key
            annotated.append(item)
        output.append(
            {
                "target_coverage": float(coverage),
                "realized_coverage": float(n_accept / max(total, 1)),
                "operational_threshold": False,
                "report": evaluate_workpoint(
                    annotated,
                    pred_key,
                    n_random=n_random,
                    seed=int(seed) + len(output),
                ),
            }
        )
    return output


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    selected = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in selected for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _attach_calibration_features(
    base_rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    excluded_keys: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    features = _index_rows(feature_rows, "calibration feature")
    output: list[dict[str, Any]] = []
    for row in base_rows:
        key = _row_key(row)
        if key[0] != "calibration" or key in excluded_keys:
            continue
        if key not in features:
            raise ValueError(f"missing calibration feature row: {key}")
        item = dict(row)
        item.update(features[key])
        item["split"] = "calibration"
        item["sample_index"] = key[1]
        output.append(item)
    return output


def run_high_coverage_qc(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    base_rows = _read_csv(args.target_inputs)
    h23_validation = _read_csv(args.h23_validation)
    h23_test = _read_csv(args.h23_test)
    h8_validation = _read_csv(args.h8_validation)
    h8_test = _read_csv(args.h8_test)
    backbone_calibration = _read_csv(args.backbone_calibration)
    backbone_test = _read_csv(args.backbone_test)

    validation_keys = {_row_key(row) for row in h23_validation}
    calibration_fit = _attach_calibration_features(
        base_rows, backbone_calibration, validation_keys
    )
    validation = merge_aligned_streams(
        base_rows,
        h23_validation,
        h8_validation,
        backbone_calibration,
        split="calibration",
    )
    test = merge_aligned_streams(
        base_rows,
        h23_test,
        h8_test,
        backbone_test,
        split="test",
    )
    if len(calibration_fit) != 240 or len(validation) != 80 or len(test) != 1360:
        raise ValueError(
            "expected calibration-fit/validation/test counts 240/80/1360; "
            f"got {len(calibration_fit)}/{len(validation)}/{len(test)}"
        )

    reference = fit_feature_reference(calibration_fit)
    raw_validation = []
    for row in validation:
        item = dict(row)
        item.update(raw_deployment_risk(item, reference))
        raw_validation.append(item)
    raw_test = []
    for row in test:
        item = dict(row)
        item.update(raw_deployment_risk(item, reference))
        raw_test.append(item)
    calibrator = fit_component_calibrator(raw_validation)
    validation_scored = score_deployment_rows(raw_validation, calibrator)
    test_scored = score_deployment_rows(raw_test, calibrator)
    selection = select_score_family(validation_scored, args.pred_key)
    policy = selection["selected_policy"]

    operational: dict[str, Any] = {}
    for index, workpoint in enumerate(("FULL", "HC95", "HC90")):
        annotated = apply_workpoint(test_scored, policy, workpoint)
        _write_csv(output_dir / f"test_{workpoint.lower()}_records.csv", annotated)
        operational[workpoint] = evaluate_workpoint(
            annotated,
            args.pred_key,
            n_random=args.n_random,
            seed=args.seed + index,
        )

    component_ablation: dict[str, Any] = {}
    for index, score_key in enumerate(
        (
            "deployment_risk_confidence",
            "deployment_risk_feature",
            "deployment_risk_disagreement",
            "deployment_risk_full",
        )
    ):
        candidate_policy = fit_workpoints(validation_scored, score_key)
        annotated = apply_workpoint(test_scored, candidate_policy, "HC95")
        component_ablation[score_key] = evaluate_workpoint(
            annotated,
            args.pred_key,
            n_random=args.n_random,
            seed=args.seed + 100 + index,
        )

    ranking_curve = evaluate_ranking_curve(
        test_scored,
        selection["selected_score"],
        args.pred_key,
        n_random=args.n_random,
        seed=args.seed + 200,
    )
    _write_csv(output_dir / "calibration_validation_scored.csv", validation_scored)
    _write_csv(output_dir / "test_scored.csv", test_scored)
    _write_json(output_dir / "feature_reference.json", reference)
    _write_json(output_dir / "component_calibrator.json", calibrator)
    _write_json(output_dir / "risk_selection.json", selection)
    _write_json(output_dir / "risk_policy.json", policy)
    _write_json(output_dir / "operational_summary.json", operational)
    _write_json(output_dir / "component_ablation.json", component_ablation)
    _write_json(output_dir / "ranking_curve.json", ranking_curve)
    manifest = {
        "schema_version": 1,
        "protocol": {"source_clients": [1, 2], "target_clients": [5]},
        "selection_split": "calibration_validation",
        "counts": {"calibration_fit": 240, "calibration_validation": 80, "test": 1360},
        "selected_score": selection["selected_score"],
        "pred_key": args.pred_key,
        "primary_workpoint": "HC95",
        "secondary_workpoint": "HC90",
        "ranking_curve_operational": False,
        "inputs": {
            key: getattr(args, key)
            for key in (
                "target_inputs",
                "h23_validation",
                "h23_test",
                "h8_validation",
                "h8_test",
                "backbone_calibration",
                "backbone_test",
            )
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {"manifest": manifest, "operational": operational}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate deployment-visible high-coverage C5 QC."
    )
    parser.add_argument("--target-inputs", required=True)
    parser.add_argument("--h23-validation", required=True)
    parser.add_argument("--h23-test", required=True)
    parser.add_argument("--h8-validation", required=True)
    parser.add_argument("--h8-test", required=True)
    parser.add_argument("--backbone-calibration", required=True)
    parser.add_argument("--backbone-test", required=True)
    parser.add_argument("--pred-key", default="target_ridge_plus_source_preds_ppm")
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_high_coverage_qc(args)
    print(json.dumps(_json_safe(result["manifest"]), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
