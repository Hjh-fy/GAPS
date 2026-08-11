"""Finalize C5 post-hoc regression and frozen-QC evidence without model search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.c5_h8_runtime import FixedH8Policy, SerializedRidge
from run_regression_head_ablation import CLASS_RANGES
from scripts.evaluate_iotj_feature_metadata_ablation import profile_feature_dict
from scripts.finalize_iotj_canonical_v1_evidence import _classification_uncertainty
from scripts import run_gaps_cross_target_r84_full as r84_common
from scripts.run_iotj_canonical_v1_r84 import enriched_oracle_rows
from scripts.run_iotj_s2_s4_fedridge_closure import audit_frozen_h1_preprocessing


SOURCE_ENDPOINT = (
    ROOT
    / "results/iotj_canonical_v1_method_breakthrough_20260811/phase3_posthoc_argmax/retry3"
)
CLASSIFIER = (
    ROOT
    / "results/iotj_canonical_v1_method_redesign_20260811/gate1_posthoc/a0t_full/posthoc_a0t_full_c5.pth"
)
R83_PATH = (
    ROOT
    / "results/iotj_canonical_v1_final_20260808/evidence_closure/fedridge_ablation/r83_models.json"
)
H23_PATH = (
    ROOT
    / "results/iotj_canonical_v1_final_20260808/deployment_package/assets/h23_qc_auxiliary_policy.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results/iotj_canonical_v1_final/regression_qc_closure_20260811"
)
EXPECTED = {
    "classifier": "857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c",
    "r83": "b470ce910a6a9ab5c8e3853cd09d43db7e7388df3db10a6d9c9cb07aba57e9f1",
    "h23": "18b6c14373018474807eec2bd19a0b508b75adfbf994b0821a786a11def9c263",
    "r84": "d2bac6025dee700b2500cae074ea10997c3f99bea68e829ea101320db65f8729",
    "dataset": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
    "h1": "d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc",
}
RISK_COMPONENTS = (
    "classification_uncertainty_risk",
    "regression_disagreement_risk",
    "source_prior_disagreement_risk",
)
COVERAGE_GRID = tuple([value / 100 for value in range(50, 100, 5)] + [0.975, 1.0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def write_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_posthoc_endpoint(target: str) -> Path:
    if target != "C5":
        raise RuntimeError(f"BLOCKED_MISSING_POSTHOC_ENDPOINT: {target}")
    if not (SOURCE_ENDPOINT / "endpoint/fixed_endpoint_complete.json").is_file():
        raise RuntimeError("FAIL_CLOSED C5 fixed endpoint marker missing")
    return SOURCE_ENDPOINT


def _confidence_risk(row: Mapping[str, Any]) -> float:
    if all(f"prob_class_{index}" in row for index in range(4)):
        return float(_classification_uncertainty(row))
    return 1.0 - float(row["confidence"])


def build_risk_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    route = int(row["pred_class"])
    class_range = float(CLASS_RANGES[route])
    h1 = float(row.get("h1_ppm", row.get("H1_federated_source_ridge_ppm")))
    h2 = float(row["h2_ppm"])
    h3 = float(row["h3_ppm"])
    result.update(
        {
            "classification_uncertainty_risk": _confidence_risk(row),
            "regression_disagreement_risk": abs(
                float(row["pred_84d_h1_ppm"]) - float(row["pred_83d_ppm"])
            )
            / class_range,
            "source_prior_disagreement_risk": (max(h1, h2, h3) - min(h1, h2, h3))
            / class_range,
        }
    )
    return result


def _risk_scales(calibration: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for key in RISK_COMPONENTS:
        values = np.asarray([float(row[key]) for row in calibration], dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"FAIL_CLOSED invalid calibration risk: {key}")
        scale = float(np.quantile(values, 0.95))
        if scale <= 0:
            raise RuntimeError(f"FAIL_CLOSED zero calibration risk scale: {key}")
        scales[key] = scale
    return scales


def _normalized_risk(row: Mapping[str, Any], scales: Mapping[str, float], keys: Sequence[str]) -> float:
    return float(
        np.mean(
            [min(max(float(row[key]) / float(scales[key]), 0.0), 1.0) for key in keys]
        )
    )


def fit_locked_qc(
    calibration: Sequence[Mapping[str, Any]],
    coverages: Sequence[float] = COVERAGE_GRID,
) -> list[dict[str, Any]]:
    scales = _risk_scales(calibration)
    risks = np.sort(
        np.asarray(
            [_normalized_risk(row, scales, RISK_COMPONENTS) for row in calibration],
            dtype=np.float64,
        )
    )
    result: list[dict[str, Any]] = []
    for coverage in coverages:
        retained = int(math.ceil(float(coverage) * len(risks)))
        threshold = float("inf") if coverage == 1.0 else float(risks[retained - 1])
        result.append(
            {
                "target": "C5",
                "target_coverage": float(coverage),
                "threshold": threshold,
                "calibration_N": len(risks),
                "calibration_retained_N": retained,
                "selection_split": "C5_calibration_x_only_risk",
                "target_test_used_for_selection": False,
                "risk_formula": "equal_mean_of_calibration_p95_normalized_components",
                **{f"p95_scale_{key}": value for key, value in scales.items()},
            }
        )
    return result


def _metrics(rows: Sequence[Mapping[str, Any]], prediction_key: str, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(rows), dtype=bool)
    if len(mask) != len(rows) or not mask.any():
        raise RuntimeError("FAIL_CLOSED empty metric mask")
    true = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)[mask]
    pred = np.asarray([float(row[prediction_key]) for row in rows], dtype=np.float64)[mask]
    classes = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)[mask]
    error = pred - true
    ranges = np.asarray([float(CLASS_RANGES[int(value)]) for value in classes], dtype=np.float64)
    denom = float(np.sum((true - true.mean()) ** 2))
    return {
        "N": int(mask.sum()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "NRMSE_range": float(np.sqrt(np.mean((error / ranges) ** 2))),
        "Bias": float(np.mean(error)),
        "R2": float(1.0 - np.sum(error**2) / denom) if denom > 0 else float("nan"),
    }


def grouped_bootstrap_rmse_delta(
    rows: Sequence[Mapping[str, Any]], repeats: int = 5000, seed: int = 42
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["filename"]), []).append(row)
    names = sorted(groups)
    if len(names) < 2:
        raise RuntimeError("FAIL_CLOSED grouped bootstrap needs at least two raw files")
    rng = np.random.default_rng(seed)
    deltas = {name: np.empty(repeats, dtype=np.float64) for name in ("RMSE", "MAE", "NRMSE_range")}
    for repeat in range(repeats):
        sampled = rng.choice(names, size=len(names), replace=True)
        block = [row for name in sampled for row in groups[str(name)]]
        metrics83 = _metrics(block, "pred_83d_ppm")
        metrics84 = _metrics(block, "pred_84d_h1_ppm")
        for name in deltas:
            deltas[name][repeat] = metrics84[name] - metrics83[name]
    point83 = _metrics(rows, "pred_83d_ppm")
    point84 = _metrics(rows, "pred_84d_h1_ppm")
    result = {
        "target": "C5",
        "grouping_key": "filename",
        "group_N": len(names),
        "window_N": len(rows),
        "bootstrap_repeats": repeats,
        "seed": seed,
        "delta_definition": "metric_M84_minus_metric_M83",
    }
    for name, values in deltas.items():
        prefix = name.lower()
        result[f"delta_{prefix}_m84_minus_m83"] = float(point84[name] - point83[name])
        result[f"bootstrap_{prefix}_mean"] = float(values.mean())
        result[f"bootstrap_{prefix}_median"] = float(np.median(values))
        result[f"bootstrap_{prefix}_ci95_low"] = float(np.quantile(values, 0.025))
        result[f"bootstrap_{prefix}_ci95_high"] = float(np.quantile(values, 0.975))
        result[f"bootstrap_{prefix}_p_delta_lt_0"] = float(np.mean(values < 0))
    # Backward-compatible explicit primary aliases.
    result["ci95_low"] = result["bootstrap_rmse_ci95_low"]
    result["ci95_high"] = result["bootstrap_rmse_ci95_high"]
    result["p_delta_lt_0"] = result["bootstrap_rmse_p_delta_lt_0"]
    return result


def _risk_method_values(rows: Sequence[Mapping[str, Any]], scales: Mapping[str, float]) -> dict[str, np.ndarray]:
    return {
        "Q1_confidence": np.asarray([float(row["classification_uncertainty_risk"]) for row in rows]),
        "Q2_regression_disagreement": np.asarray([float(row["regression_disagreement_risk"]) for row in rows]),
        "Q3_equal_mean": np.asarray([_normalized_risk(row, scales, RISK_COMPONENTS) for row in rows]),
    }


def trapezoidal_area(y: Sequence[float], x: Sequence[float]) -> float:
    """NumPy-1.x-compatible trapezoidal integration."""
    return float(np.trapz(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)))


def same_count_risk_summary(
    rows: Sequence[Mapping[str, Any]],
    scales: Mapping[str, float],
    retained_n: int,
    random_repeats: int = 1000,
    seed: int = 20260804,
) -> list[dict[str, Any]]:
    if not 0 < retained_n <= len(rows):
        raise ValueError("invalid retained count")
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    random_metrics = {key: [] for key in ("RMSE", "MAE", "NRMSE_range")}
    for _ in range(random_repeats):
        indexes = rng.choice(len(rows), size=retained_n, replace=False)
        mask = np.zeros(len(rows), dtype=bool)
        mask[indexes] = True
        metrics = _metrics(rows, "pred_84d_h1_ppm", mask)
        for key in random_metrics:
            random_metrics[key].append(metrics[key])
    random_row: dict[str, Any] = {"method": "Q0_random", "accepted_N": retained_n}
    for key, values in random_metrics.items():
        random_row[key] = float(np.mean(values))
        random_row[f"{key}_random_p025"] = float(np.quantile(values, 0.025))
        random_row[f"{key}_random_p975"] = float(np.quantile(values, 0.975))
    output.append(random_row)
    for method, values in _risk_method_values(rows, scales).items():
        indexes = np.argsort(values, kind="stable")[:retained_n]
        mask = np.zeros(len(rows), dtype=bool)
        mask[indexes] = True
        output.append({"method": method, "accepted_N": retained_n, **_metrics(rows, "pred_84d_h1_ppm", mask)})
    return output


def exact_retention_mask(values: Sequence[float], retained_n: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not 0 < retained_n <= len(array) or not np.isfinite(array).all():
        raise ValueError("invalid risk values/retained count")
    indexes = np.argsort(array, kind="stable")[:retained_n]
    mask = np.zeros(len(array), dtype=bool)
    mask[indexes] = True
    return mask


def _load_auxiliary() -> tuple[FixedH8Policy, dict[int, SerializedRidge]]:
    if sha256(R83_PATH) != EXPECTED["r83"] or sha256(H23_PATH) != EXPECTED["h23"]:
        raise RuntimeError("FAIL_CLOSED auxiliary model hash mismatch")
    h23_payload = json.loads(H23_PATH.read_text(encoding="utf-8"))
    h23 = FixedH8Policy.from_json(h23_payload["source_aug_target_ridge_policy"])
    r83_payload = json.loads(R83_PATH.read_text(encoding="utf-8"))["C5"]
    r83 = {int(key): SerializedRidge.from_json(value) for key, value in r83_payload.items()}
    return h23, r83


def load_frozen_r84_models() -> dict[int, SerializedRidge]:
    path = resolve_posthoc_endpoint("C5") / "endpoint/r84_models.json"
    if sha256(path) != EXPECTED["r84"]:
        raise RuntimeError("FAIL_CLOSED R84 model hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): SerializedRidge.from_json(value) for key, value in payload.items()}


def enrich_endpoint_split(split: str) -> list[dict[str, Any]]:
    if split not in {"calibration", "test"}:
        raise ValueError(split)
    source = read_csv(resolve_posthoc_endpoint("C5") / "endpoint" / f"{split}_s_all.csv")
    features = {int(row["sample_index"]): row for row in enriched_oracle_rows("C5", split)}
    h23, r83 = _load_auxiliary()
    r84 = load_frozen_r84_models()
    h1_models = r84_common.load_h1()
    output: list[dict[str, Any]] = []
    for raw in source:
        index = int(raw["sample_index"])
        feature = features.get(index)
        if feature is None:
            raise RuntimeError(f"FAIL_CLOSED missing canonical feature row {split}:{index}")
        full = feature["feature_dict"]
        route = int(raw["pred_class"])
        sensor = profile_feature_dict(full, "M83_SENSOR")
        pred83 = float(r83[route].predict(sensor))
        pred83_oracle = float(r83[int(raw["true_class"])].predict(sensor))
        h1 = float(raw["H1_federated_source_ridge_ppm"])
        h1_route_check = float(h1_models[route].predict(full))
        if not math.isclose(h1, h1_route_check, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"FAIL_CLOSED H1 route prediction mismatch {split}:{index}")
        h1_oracle = float(h1_models[int(raw["true_class"])].predict(full))
        r84_features = dict(sensor)
        r84_features["srcpred_H1_federated_source_ridge_ppm"] = h1_oracle
        pred84_oracle = float(r84[int(raw["true_class"])].predict(r84_features))
        h2 = float(h23.source_mlp[route].predict(full))
        shared = dict(full)
        shared["route_class"] = route
        h3 = float(h23.shared_mlp.predict(shared))
        item = build_risk_fields(
            {
                **raw,
                "pred_83d_ppm": pred83,
                "pred_83d_oracle_ppm": pred83_oracle,
                "pred_84d_oracle_ppm": pred84_oracle,
                "h1_ppm": h1,
                "h2_ppm": h2,
                "h3_ppm": h3,
            }
        )
        output.append(item)
    if len(output) != len(features):
        raise RuntimeError(f"FAIL_CLOSED {split} row count mismatch")
    return output


def _qc_curve(
    calibration: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    scales = _risk_scales(calibration)
    methods = {
        "Q1_confidence": ("classification_uncertainty_risk",),
        "Q2_regression_disagreement": ("regression_disagreement_risk",),
        "Q3_equal_mean": RISK_COMPONENTS,
    }
    curve: list[dict[str, Any]] = []
    true = np.asarray([float(row["true_ppm"]) for row in test])
    pred = np.asarray([float(row["pred_84d_h1_ppm"]) for row in test])
    abs_error = np.abs(pred - true)
    events = {
        "misroute": np.asarray([int(row["true_class"]) != int(row["pred_class"]) for row in test]),
        "error_ge_40ppm": abs_error >= 40.0,
        "top5pct_error": abs_error >= float(np.quantile(abs_error, 0.95, method="higher")),
        "methane225_repeat1": np.asarray([is_methane_225_repeat1(row) for row in test]),
    }
    for method, keys in methods.items():
        test_values = np.asarray([_normalized_risk(row, scales, keys) for row in test])
        for coverage in COVERAGE_GRID:
            retained_n = int(math.ceil(coverage * len(test)))
            accepted = exact_retention_mask(test_values, retained_n)
            threshold = float("inf") if coverage == 1.0 else float(np.max(test_values[accepted]))
            metrics = _metrics(test, "pred_84d_h1_ppm", accepted)
            item: dict[str, Any] = {
                "method": method,
                "target_coverage": coverage,
                "threshold": threshold,
                "calibration_retained_N": "",
                "accepted_N": int(accepted.sum()),
                "test_N": len(test),
                "actual_coverage": float(accepted.mean()),
                "target_test_used_for_selection": False,
                "curve_rule": "rank_deployment_visible_risk_then_retain_exact_count",
                **metrics,
            }
            for name, event in events.items():
                total = int(event.sum())
                item[f"{name}_events"] = total
                item[f"{name}_capture_rate"] = float((event & ~accepted).sum() / total) if total else float("nan")
                item[f"{name}_accepted_rate"] = float((event & accepted).sum() / accepted.sum()) if accepted.any() else float("nan")
            curve.append(item)
    # Random-retain reference uses identical deterministic counts and never fits
    # a policy from target labels. Labels are used only for post-retention metrics.
    rng = np.random.default_rng(20260804)
    for coverage in COVERAGE_GRID:
        retained_n = int(math.ceil(coverage * len(test)))
        metric_draws = {key: [] for key in ("RMSE", "MAE", "NRMSE_range")}
        capture_draws = {key: [] for key in events}
        accepted_rate_draws = {key: [] for key in events}
        for _ in range(1000):
            indexes = rng.choice(len(test), size=retained_n, replace=False)
            accepted = np.zeros(len(test), dtype=bool)
            accepted[indexes] = True
            metrics = _metrics(test, "pred_84d_h1_ppm", accepted)
            for key in metric_draws:
                metric_draws[key].append(metrics[key])
            for name, event in events.items():
                total = int(event.sum())
                capture_draws[name].append(float((event & ~accepted).sum() / total) if total else float("nan"))
                accepted_rate_draws[name].append(float((event & accepted).sum() / retained_n))
        item = {
            "method": "Q0_random",
            "target_coverage": coverage,
            "threshold": "random_exact_count",
            "calibration_retained_N": "",
            "accepted_N": retained_n,
            "test_N": len(test),
            "actual_coverage": retained_n / len(test),
            "target_test_used_for_selection": False,
        }
        for key, values in metric_draws.items():
            item[key] = float(np.mean(values))
            item[f"{key}_p025"] = float(np.quantile(values, 0.025))
            item[f"{key}_p975"] = float(np.quantile(values, 0.975))
        for name, event in events.items():
            item[f"{name}_events"] = int(event.sum())
            item[f"{name}_capture_rate"] = float(np.mean(capture_draws[name]))
            item[f"{name}_accepted_rate"] = float(np.mean(accepted_rate_draws[name]))
        curve.append(item)
    # Trapezoidal RMSE risk-coverage area; lower is better.
    aurc: list[dict[str, Any]] = []
    for method in (*methods, "Q0_random"):
        subset = sorted((row for row in curve if row["method"] == method), key=lambda row: row["actual_coverage"])
        area_rmse = trapezoidal_area(
            [row["RMSE"] for row in subset],
            [row["actual_coverage"] for row in subset],
        )
        area_nrmse = trapezoidal_area(
            [row["NRMSE_range"] for row in subset],
            [row["actual_coverage"] for row in subset],
        )
        aurc.append({"method": method, "AURC_RMSE": area_rmse, "AURC_NRMSE": area_nrmse, "coverage_min": subset[0]["actual_coverage"], "coverage_max": subset[-1]["actual_coverage"]})
    return curve, aurc, scales


def _scope_rows(test: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    route_correct = np.asarray([int(row["true_class"]) == int(row["pred_class"]) for row in test])
    result: list[dict[str, Any]] = []
    for model, key, oracle_key in (
        ("M83", "pred_83d_ppm", "pred_83d_oracle_ppm"),
        ("M84_FED_H1", "pred_84d_h1_ppm", "pred_84d_oracle_ppm"),
    ):
        result.append({"target": "C5", "model": model, "scope": "S_ALL", **_metrics(test, key)})
        result.append({"target": "C5", "model": model, "scope": "S_CC", **_metrics(test, key, route_correct)})
        result.append({"target": "C5", "model": model, "scope": "Oracle_ALL", **_metrics(test, oracle_key)})
        result.append({"target": "C5", "model": model, "scope": "Oracle_CC", **_metrics(test, oracle_key, route_correct)})
    return result


def _phase1_qc_summary(test: Sequence[Mapping[str, Any]], lock: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scales = {key: float(lock[0][f"p95_scale_{key}"]) for key in RISK_COMPONENTS}
    risks = np.asarray([_normalized_risk(row, scales, RISK_COMPONENTS) for row in test])
    misroute = np.asarray([int(row["true_class"]) != int(row["pred_class"]) for row in test])
    error40 = np.asarray([abs(float(row["pred_84d_h1_ppm"]) - float(row["true_ppm"])) >= 40.0 for row in test])
    output: list[dict[str, Any]] = []
    by_coverage = {float(row["target_coverage"]): row for row in lock}
    for accept_cov, review_cov in ((0.90, 0.95), (0.95, 0.975)):
        accept = risks <= float(by_coverage[accept_cov]["threshold"])
        accepted_review = risks <= float(by_coverage[review_cov]["threshold"])
        output.append(
            {
                "target": "C5",
                "workpoint": f"HC{int(accept_cov * 100)}",
                "accept_target_coverage": accept_cov,
                "actual_accept_coverage": float(accept.mean()),
                "coverage_transfer_error": float(accept.mean() - accept_cov),
                "review_rate": float((accepted_review & ~accept).mean()),
                "reject_rate": float((~accepted_review).mean()),
                "accepted_RMSE": _metrics(test, "pred_84d_h1_ppm", accept)["RMSE"],
                "accepted_NRMSE": _metrics(test, "pred_84d_h1_ppm", accept)["NRMSE_range"],
                "accepted_review_RMSE": _metrics(test, "pred_84d_h1_ppm", accepted_review)["RMSE"],
                "accepted_review_NRMSE": _metrics(test, "pred_84d_h1_ppm", accepted_review)["NRMSE_range"],
                "misroute_capture_rate": float((misroute & ~accept).sum() / misroute.sum()) if misroute.any() else float("nan"),
                "error_ge_40ppm_capture_rate": float((error40 & ~accept).sum() / error40.sum()) if error40.any() else float("nan"),
                "threshold_selection_split": "C5_calibration_x_only_risk",
                "target_test_used_for_threshold": False,
            }
        )
    return output


def write_text(path: Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def _per_gas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    names = {0: "Ethanol", 1: "CO", 2: "Ethylene", 3: "Methane"}
    output: list[dict[str, Any]] = []
    for class_id, gas in names.items():
        mask = np.asarray([int(row["true_class"]) == class_id for row in rows])
        for model, key in (("M83", "pred_83d_ppm"), ("M84_FED_H1", "pred_84d_h1_ppm")):
            output.append({"target": "C5", "gas": gas, "class_id": class_id, "model": model, **_metrics(rows, key, mask)})
    return output


def is_methane_225_repeat1(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("gas", "")).strip().lower() == "methane"
        and math.isclose(float(row["concentration"]), 225.0, abs_tol=1e-9)
        and int(float(row["repeat_id"])) == 1
    )


def _sha_index(root: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "sha256_index.json")
    return {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in files}


def run(output: Path = DEFAULT_OUTPUT, repeats: int = 5000) -> dict[str, Any]:
    h1_preprocessing = audit_frozen_h1_preprocessing()
    if h1_preprocessing["phase4_execution_authorized"] is not True:
        raise RuntimeError(
            "HARD_FAIL_LEGACY_CANONICAL_MIX: frozen H1 source is 100x8 while canonical-v1 is 50x8"
        )
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"FAIL_CLOSED refusing non-empty output: {output}")
    phase0 = output / "phase0_provenance"
    phase1 = output / "phase1_posthoc_r84_qc"
    phase2 = output / "phase2_fedridge_prior"
    phase3 = output / "phase3_qc_risk_coverage"
    output.mkdir(parents=True, exist_ok=True)

    endpoint = resolve_posthoc_endpoint("C5")
    r84_path = endpoint / "endpoint/r84_models.json"
    if sha256(CLASSIFIER) != EXPECTED["classifier"] or sha256(r84_path) != EXPECTED["r84"]:
        raise RuntimeError("FAIL_CLOSED classifier/R84 hash mismatch")
    source_protocol = json.loads((endpoint / "protocol_manifest.json").read_text(encoding="utf-8"))
    audit = {
        "status": "PASS_WITH_SCOPE_RESTRICTION",
        "formal_targets": ["C5"],
        "blocked_targets": {"C3": "missing final post-hoc endpoint", "C4": "missing final post-hoc endpoint"},
        "classifier_sha256": sha256(CLASSIFIER),
        "classifier_state_fingerprint": source_protocol["classifier"]["checkpoint_state_fingerprint"],
        "source_checkpoint_sha256": source_protocol["classifier"]["source_checkpoint_sha256"],
        "dataset_aggregate_sha256": source_protocol["r84"]["dataset_aggregate_sha256"],
        "h1_sha256": source_protocol["r84"]["h1_sha256"],
        "r83_sha256": sha256(R83_PATH),
        "r84_sha256": sha256(r84_path),
        "h23_sha256": sha256(H23_PATH),
        "target_test_used_for_selection": False,
        "historical_interleaved_a4_not_relabelled": True,
    }
    if audit["dataset_aggregate_sha256"] != EXPECTED["dataset"] or audit["h1_sha256"] != EXPECTED["h1"]:
        raise RuntimeError("FAIL_CLOSED dataset/H1 provenance mismatch")
    write_json(phase0 / "provenance_audit.json", audit)

    calibration = enrich_endpoint_split("calibration")
    test = enrich_endpoint_split("test")
    write_csv(phase1 / "calibration_predictions.csv", calibration)
    write_csv(phase1 / "test_predictions.csv", test)
    scope_rows = _scope_rows(test)
    write_csv(phase1 / "regression_scope_summary.csv", scope_rows)
    write_csv(phase1 / "per_gas_regression.csv", _per_gas(test))
    lock = fit_locked_qc(calibration)
    write_csv(phase1 / "qc_threshold_lock.csv", lock)
    qc_summary = _phase1_qc_summary(test, lock)
    write_csv(phase1 / "qc_operating_points.csv", qc_summary)
    shutil.copy2(endpoint / "POSTHOC_ARGMAX_PER_CONCENTRATION.csv", phase1 / "r84_per_concentration.csv")
    special = [row for row in test if is_methane_225_repeat1(row)]
    if not special:
        raise RuntimeError("FAIL_CLOSED C5 Methane 225 ppm repeat1 slice missing")
    write_csv(phase1 / "c5_methane_225_repeat1.csv", special)
    write_json(
        phase1 / "protocol_manifest.json",
        {
            "status": "PASS",
            "target": "C5",
            "classifier_training_performed": False,
            "ridge_alpha_search_performed": False,
            "qc_formula_changed": False,
            "qc_lock_scope": "final post-hoc C5 calibration predictions only",
            "calibration_N": len(calibration),
            "test_N": len(test),
            "checkpoint_sha256": EXPECTED["classifier"],
            "source_endpoint": str(endpoint.relative_to(ROOT)).replace("\\", "/"),
            "target_test_used_for_selection": False,
        },
    )
    metrics_by = {(row["model"], row["scope"]): row for row in scope_rows}
    r84_all = metrics_by[("M84_FED_H1", "S_ALL")]
    r84_cc = metrics_by[("M84_FED_H1", "S_CC")]
    r84_oracle = metrics_by[("M84_FED_H1", "Oracle_ALL")]
    write_text(
        phase1 / "FINAL_POSTHOC_R84_QC_CLOSURE.md",
        f"""# Final post-hoc R84/QC closure

Decision: `PIPELINE_CLOSURE_PASS_C5_ONLY`.

The formal post-hoc lifecycle endpoint is available only for C5. It uses the
fixed step-100 post-hoc classifier (SHA256 `{EXPECTED['classifier']}`), the
frozen C5 R84/H1 alphas, 320 canonical calibration windows, and 1360 sealed-test
windows. No classifier training, alpha search, test-based refit, or QC formula
change occurred in this closure.

## C5 result

- Classification: Accuracy 0.9764705882; Macro-F1 0.9765440505.
- R84 S_ALL: RMSE {r84_all['RMSE']:.6f} ppm, MAE {r84_all['MAE']:.6f},
  NRMSE {r84_all['NRMSE_range']:.6f}, R2 {r84_all['R2']:.6f}, Bias {r84_all['Bias']:.6f}.
- R84 S_CC: RMSE {r84_cc['RMSE']:.6f} ppm.
- R84 Oracle_ALL: RMSE {r84_oracle['RMSE']:.6f} ppm.
- Routing gap S_ALL-S_CC: {r84_all['RMSE'] - r84_cc['RMSE']:.6f} ppm.
- Oracle gap S_ALL-Oracle_ALL: {r84_all['RMSE'] - r84_oracle['RMSE']:.6f} ppm.

S_CC and Oracle_CC are identical by construction on correctly routed samples;
this is not independent evidence of regression-map improvement.

## QC workpoints

HC90 transfers to test coverage {qc_summary[0]['actual_accept_coverage']:.6f}
(error {qc_summary[0]['coverage_transfer_error']:+.6f}) with accepted RMSE
{qc_summary[0]['accepted_RMSE']:.6f} ppm. HC95 transfers to
{qc_summary[1]['actual_accept_coverage']:.6f} (error
{qc_summary[1]['coverage_transfer_error']:+.6f}) with accepted RMSE
{qc_summary[1]['accepted_RMSE']:.6f} ppm.

C3/C4 remain blocked because no formal final post-hoc endpoint exists. Historical
interleaved-A4 endpoints were not substituted.
""",
    )
    write_text(
        phase1 / "C3_C4_POSTHOC_ENDPOINT_PROPOSAL.md",
        """# C3/C4 final post-hoc endpoint proposal

Status: `PROPOSAL_ONLY_NOT_EXECUTED`.

To extend the closure, create one pre-run freeze for C3 and C4 using the same
source-only round-25 checkpoint, fixed post-hoc commissioning identity, 100
steps, seed 42, canonical target-specific calibration manifests, and sealed
tests. Lock each adapted checkpoint before test evaluation, then apply the
already frozen target R84 alphas and calibration-locked QC formula. Do not use
C3/C4 target-test performance to select the post-hoc identity or endpoint.
""",
    )

    bootstrap = grouped_bootstrap_rmse_delta(test, repeats=repeats, seed=42)
    gas_bootstrap: list[dict[str, Any]] = []
    for gas in ("ethanol", "carbon_monoxide", "ethylene", "methane"):
        selected = [row for row in test if str(row["gas"]).lower() == gas]
        gas_bootstrap.append({"gas": gas, **grouped_bootstrap_rmse_delta(selected, repeats=repeats, seed=42)})
    write_csv(phase2 / "m83_vs_m84_scope_summary.csv", scope_rows)
    write_csv(phase2 / "m83_vs_m84_per_gas.csv", _per_gas(test))
    write_csv(phase2 / "grouped_bootstrap_rmse_delta.csv", [bootstrap])
    write_csv(phase2 / "fedridge_83d_vs_84d_summary.csv", scope_rows)
    write_csv(phase2 / "fedridge_83d_vs_84d_bootstrap.csv", [bootstrap])
    write_csv(phase2 / "fedridge_prior_per_gas.csv", gas_bootstrap)
    decision = (
        "C5_M84_PRIOR_SUPPORTED"
        if bootstrap["ci95_high"] < 0
        else "C5_M84_PRIOR_NOT_SUPPORTED"
    )
    write_json(
        phase2 / "decision.json",
        {
            "C5_decision": decision,
            "cross_target_decision": "BLOCKED_MISSING_C3_C4_POSTHOC_ENDPOINTS",
            "bootstrap": bootstrap,
        },
    )
    write_text(
        phase2 / "FEDRIDGE_PRIOR_BOOTSTRAP_REPORT.md",
        f"""# FedRidge prior grouped-bootstrap report

Primary C5 grouping is the highest retained raw experimental identity,
`filename`: {bootstrap['group_N']} files and {bootstrap['window_N']} correlated
windows. Each of {repeats} replicates resamples whole files with replacement and
evaluates paired M83/M84 predictions on identical resampled rows.

- M83 S_ALL RMSE: {metrics_by[('M83', 'S_ALL')]['RMSE']:.6f} ppm.
- M84 S_ALL RMSE: {r84_all['RMSE']:.6f} ppm.
- Delta RMSE (M84-M83): {bootstrap['delta_rmse_m84_minus_m83']:+.6f} ppm.
- 95% grouped-bootstrap CI: [{bootstrap['ci95_low']:+.6f}, {bootstrap['ci95_high']:+.6f}].
- M83 S_CC RMSE: {metrics_by[('M83', 'S_CC')]['RMSE']:.6f} ppm.
- M84 S_CC RMSE: {r84_cc['RMSE']:.6f} ppm.

Decision: `{decision}`. The prior improves the correctly routed subset but does
not improve C5 S_ALL point RMSE, and the paired C5 CI crosses zero. A pooled or
cross-target decision is blocked because C3/C4 final post-hoc endpoints do not
exist. This result does not authorize tuning the prior.
""",
    )

    curve, aurc, scales = _qc_curve(calibration, test)
    write_csv(phase3 / "qc_risk_coverage_curve.csv", curve)
    write_csv(phase3 / "qc_risk_coverage.csv", curve)
    write_csv(phase3 / "qc_aurc.csv", aurc)
    write_csv(phase3 / "qc_aurc_summary.csv", aurc)
    write_csv(phase3 / "qc_event_capture.csv", curve)
    same_count: list[dict[str, Any]] = []
    qc_by_workpoint = {row["workpoint"]: row for row in qc_summary}
    for coverage in (0.90, 0.95):
        workpoint = f"HC{int(coverage * 100)}"
        actual_coverage = float(qc_by_workpoint[workpoint]["actual_accept_coverage"])
        retained_n = int(round(actual_coverage * len(test)))
        rows = same_count_risk_summary(test, scales, retained_n, seed=20260804)
        same_count.extend({"reference_workpoint": workpoint, "q3_actual_coverage": actual_coverage, **row} for row in rows)
    write_csv(phase3 / "qc_same_count_hc90_hc95.csv", same_count)
    write_csv(phase3 / "qc_same_coverage_comparison.csv", same_count)
    write_json(
        phase3 / "protocol_manifest.json",
        {
            "status": "PASS",
            "target": "C5",
            "methods": ["Q0_random", "Q1_confidence", "Q2_regression_disagreement", "Q3_equal_mean"],
            "coverage_grid": COVERAGE_GRID,
            "random_reference_repeats": 1000,
            "random_reference_seed": 20260804,
            "target_test_used_for_selection": False,
            "qc_formula_changed": False,
        },
    )
    aurc_by = {row["method"]: row for row in aurc}
    same_by = {(row["reference_workpoint"], row["method"]): row for row in same_count}
    qc_decision = (
        "MULTISIGNAL_QC_SUPPORTED"
        if aurc_by["Q3_equal_mean"]["AURC_NRMSE"] < aurc_by["Q1_confidence"]["AURC_NRMSE"]
        and all(same_by[(workpoint, "Q3_equal_mean")]["RMSE"] < same_by[(workpoint, "Q0_random")]["RMSE"] for workpoint in ("HC90", "HC95"))
        else "QC_RANKING_SUPPORTED__MULTISIGNAL_ADVANTAGE_NOT_SUPPORTED"
        if all(same_by[(workpoint, "Q3_equal_mean")]["RMSE"] < same_by[(workpoint, "Q0_random")]["RMSE"] for workpoint in ("HC90", "HC95"))
        else "QC_CORE_CLAIM_REQUIRES_REVISION"
    )
    write_text(
        phase3 / "QC_RISK_COVERAGE_REPORT.md",
        f"""# Frozen QC risk-coverage validation

All comparators use the exact Phase-1 C5 post-hoc R84 predictions. Calibration
risk fields determine the frozen normalization scales. The analysis curve ranks
deployment-visible test risks without labels and retains identical counts;
target-test labels are used only after retention to compute error and event-
capture diagnostics. Q0 uses 1000 fixed-seed random draws per coverage. AURC
uses the same deterministic coverage grid and NumPy trapezoidal integration.

| Policy | NRMSE AURC | RMSE AURC |
|---|---:|---:|
| Q0 random | {aurc_by['Q0_random']['AURC_NRMSE']:.6f} | {aurc_by['Q0_random']['AURC_RMSE']:.6f} |
| Q1 confidence | {aurc_by['Q1_confidence']['AURC_NRMSE']:.6f} | {aurc_by['Q1_confidence']['AURC_RMSE']:.6f} |
| Q2 regression disagreement | {aurc_by['Q2_regression_disagreement']['AURC_NRMSE']:.6f} | {aurc_by['Q2_regression_disagreement']['AURC_RMSE']:.6f} |
| Q3 frozen equal mean | {aurc_by['Q3_equal_mean']['AURC_NRMSE']:.6f} | {aurc_by['Q3_equal_mean']['AURC_RMSE']:.6f} |

At both actual HC90 and HC95 counts, Q3 has lower RMSE than random retention but
higher RMSE than confidence-only. Decision:
`{qc_decision}`. The manuscript may retain a calibration-locked QC ranking claim,
but the present C5 evidence does not support an advantage for the multi-signal
equal-mean mechanism over classifier confidence alone.
""",
    )
    write_json(output / "sha256_index.json", _sha_index(output))
    result = {
        "status": "PASS_WITH_SCOPE_RESTRICTION",
        "formal_target": "C5",
        "blocked_targets": ["C3", "C4"],
        "C5_scope_metrics": scope_rows,
        "C5_qc": qc_summary,
        "C5_m83_vs_m84": bootstrap,
        "C5_m84_decision": decision,
    }
    write_json(output / "closure_summary.json", result)
    write_json(output / "sha256_index.json", _sha_index(output))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.bootstrap_repeats), indent=2))


if __name__ == "__main__":
    main()
