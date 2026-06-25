"""Fair L1 evaluation: source lightweight heads + full target residual auto_v2.

This script is the fair follow-up to the lightweight source-head diagnostics.
It keeps the source-trained lightweight heads, then gives them the same kind of
target-side calibration opportunity:

1. Fit lightweight heads on C1/C2 source train.
2. Select source hyperparameters on C1/C2 source calibration.
3. Predict target calibration/test with the fixed-DA route.
4. Split target calibration into calibration-fit and calibration-validation.
5. Fit residual/affine candidates on calibration-fit.
6. Select per client/gas using calibration-validation only.
7. Refit selected candidates on full target calibration.
8. Report no-QC full-set target-test metrics.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    apply_models,
    build_oracle_rows,
    client_name,
    deterministic_train_val,
    fit_select_refit,
    fnum,
    inum,
    metrics,
    read_csv,
    summarize,
    write_csv,
)
from run_source_lightweight_regression_head_ablation import (
    apply_mlp_models,
    apply_shared_model,
    fit_select_refit_mlp,
    fit_select_refit_shared_mlp,
    parse_hidden_grid,
)


BASE_MODES = (
    "source_ridge",
    "source_per_gas_mlp",
    "source_shared_mlp",
)
RESIDUAL_MODES = (
    "identity",
    "affine",
    "ridge_basic",
    "ridge_phase",
    "piecewise_ridge",
)
BASIC_PREFIXES = (
    "mean_ch",
    "std_ch",
    "min_ch",
    "max_ch",
    "last_ch",
    "slope_ch",
    "range_ch",
    "early_mean_ch",
    "late_mean_ch",
    "late_minus_early_ch",
)
PHASE_PREFIXES = ("response_phase_", "phase_label_", "phase_id_")
METRIC_FIELDS = ("RMSE", "MAE", "NRMSE", "P90AE", "Bias", "Slope")


@dataclass
class IdentityCalibrator:
    clip_min: float
    clip_max: float

    def predict(self, rows: Sequence[dict[str, Any]], base_key: str) -> np.ndarray:
        values = np.asarray([fnum(row.get(base_key)) for row in rows], dtype=np.float64)
        return np.clip(values, self.clip_min, self.clip_max)

    def to_json(self) -> dict[str, Any]:
        return {"type": "identity", "clip_min": self.clip_min, "clip_max": self.clip_max}


@dataclass
class AffineResidualCalibrator:
    slope: float
    intercept: float
    clip_min: float
    clip_max: float

    def predict(self, rows: Sequence[dict[str, Any]], base_key: str) -> np.ndarray:
        x = np.asarray([fnum(row.get(base_key)) for row in rows], dtype=np.float64)
        return np.clip(self.slope * x + self.intercept, self.clip_min, self.clip_max)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "affine",
            "slope": self.slope,
            "intercept": self.intercept,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }


@dataclass
class RidgeResidualCalibrator:
    feature_names: list[str]
    alpha: float
    model: Any
    clip_min: float
    clip_max: float

    def predict(self, rows: Sequence[dict[str, Any]], base_key: str) -> np.ndarray:
        x = matrix_from_rows(rows, self.feature_names, base_key)
        pred = np.asarray(self.model.predict(x), dtype=np.float64)
        return np.clip(pred, self.clip_min, self.clip_max)

    def to_json(self) -> dict[str, Any]:
        ridge = self.model.named_steps["ridge"]
        scaler = self.model.named_steps["standardscaler"]
        return {
            "type": "ridge",
            "feature_names": self.feature_names,
            "alpha": self.alpha,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "coef": np.asarray(ridge.coef_, dtype=np.float64).reshape(-1).tolist(),
            "intercept": float(ridge.intercept_),
            "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64).reshape(-1).tolist(),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64).reshape(-1).tolist(),
        }


@dataclass
class PiecewiseRidgeCalibrator:
    low: RidgeResidualCalibrator
    high: RidgeResidualCalibrator
    threshold: float
    clip_min: float
    clip_max: float

    def predict(self, rows: Sequence[dict[str, Any]], base_key: str) -> np.ndarray:
        base = np.asarray([fnum(row.get(base_key)) for row in rows], dtype=np.float64)
        pred = np.empty(len(rows), dtype=np.float64)
        low_idxs = [idx for idx, value in enumerate(base) if value < self.threshold]
        high_idxs = [idx for idx, value in enumerate(base) if value >= self.threshold]
        if low_idxs:
            pred[low_idxs] = self.low.predict([rows[idx] for idx in low_idxs], base_key)
        if high_idxs:
            pred[high_idxs] = self.high.predict([rows[idx] for idx in high_idxs], base_key)
        return np.clip(pred, self.clip_min, self.clip_max)

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "piecewise_ridge",
            "threshold": self.threshold,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "low": self.low.to_json(),
            "high": self.high.to_json(),
        }


def feature_names_for(rows: Sequence[dict[str, Any]], mode: str) -> list[str]:
    all_names = sorted(rows[0]["feature_dict"].keys())
    if mode == "ridge_basic":
        return [name for name in all_names if name.startswith(BASIC_PREFIXES)]
    if mode in {"ridge_phase", "piecewise_ridge"}:
        return all_names
    raise ValueError(f"no features for mode {mode}")


def matrix_from_rows(rows: Sequence[dict[str, Any]], feature_names: Sequence[str], base_key: str) -> np.ndarray:
    x = np.zeros((len(rows), len(feature_names) + 1), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        x[row_idx, 0] = fnum(row.get(base_key))
        feats = row["feature_dict"]
        for col_idx, name in enumerate(feature_names, start=1):
            x[row_idx, col_idx] = fnum(feats.get(name), 0.0)
    return x


def fit_affine(rows: Sequence[dict[str, Any]], base_key: str) -> AffineResidualCalibrator:
    x = np.asarray([fnum(row.get(base_key)) for row in rows], dtype=np.float64)
    y = np.asarray([fnum(row.get("true_ppm")) for row in rows], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    clip_min = float(np.min(y)) if y.size else 0.0
    clip_max = float(np.max(y)) if y.size else 250.0
    if x.size < 2 or float(np.std(x)) < 1e-9:
        intercept = float(np.mean(y - x)) if y.size else 0.0
        return AffineResidualCalibrator(1.0, intercept, clip_min, clip_max)
    design = np.column_stack([x, np.ones_like(x)])
    reg = np.eye(2, dtype=np.float64) * 1e-6
    reg[1, 1] = 0.0
    coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ y
    return AffineResidualCalibrator(float(coef[0]), float(coef[1]), clip_min, clip_max)


def fit_ridge_direct(
    rows: Sequence[dict[str, Any]],
    base_key: str,
    feature_names: Sequence[str],
    alpha: float,
) -> RidgeResidualCalibrator:
    x = matrix_from_rows(rows, feature_names, base_key)
    y = np.asarray([fnum(row.get("true_ppm")) for row in rows], dtype=np.float64)
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
    return RidgeResidualCalibrator(
        feature_names=list(feature_names),
        alpha=float(alpha),
        model=model,
        clip_min=float(np.min(y)) if y.size else 0.0,
        clip_max=float(np.max(y)) if y.size else 250.0,
    )


def fit_select_ridge(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    base_key: str,
    feature_names: Sequence[str],
    alphas: Sequence[float],
) -> tuple[RidgeResidualCalibrator, dict[str, Any]]:
    y_val = np.asarray([fnum(row.get("true_ppm")) for row in val_rows], dtype=np.float64)
    best_score = float("inf")
    best_alpha = float(alphas[0])
    audit: list[dict[str, Any]] = []
    for alpha in alphas:
        model = fit_ridge_direct(train_rows, base_key, feature_names, alpha)
        pred = model.predict(val_rows, base_key)
        score = float(np.sqrt(np.mean((pred - y_val) ** 2))) if y_val.size else float("inf")
        audit.append({"alpha": float(alpha), "val_RMSE": score})
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return fit_ridge_direct(train_rows, base_key, feature_names, best_alpha), {
        "best_alpha": best_alpha,
        "best_val_RMSE": best_score,
        "alpha_audit": audit,
    }


def ppm_bin(value: float) -> str:
    return "high" if fnum(value) >= 125.0 else "low"


def fit_piecewise(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    base_key: str,
    feature_names: Sequence[str],
    alphas: Sequence[float],
) -> tuple[PiecewiseRidgeCalibrator, dict[str, Any]]:
    threshold = 125.0
    fallback_rows = list(train_rows)
    low_rows = [row for row in train_rows if ppm_bin(fnum(row.get(base_key))) == "low"] or fallback_rows
    high_rows = [row for row in train_rows if ppm_bin(fnum(row.get(base_key))) == "high"] or fallback_rows
    low, low_audit = fit_select_ridge(low_rows, val_rows, base_key, feature_names, alphas)
    high, high_audit = fit_select_ridge(high_rows, val_rows, base_key, feature_names, alphas)
    y_val = np.asarray([fnum(row.get("true_ppm")) for row in val_rows], dtype=np.float64)
    model = PiecewiseRidgeCalibrator(
        low=low,
        high=high,
        threshold=threshold,
        clip_min=float(np.min([fnum(row.get("true_ppm")) for row in train_rows])) if train_rows else 0.0,
        clip_max=float(np.max([fnum(row.get("true_ppm")) for row in train_rows])) if train_rows else 250.0,
    )
    pred = model.predict(val_rows, base_key)
    return model, {
        "best_alpha": f"low={low.alpha};high={high.alpha}",
        "best_val_RMSE": float(np.sqrt(np.mean((pred - y_val) ** 2))) if y_val.size else float("inf"),
        "low_audit": low_audit,
        "high_audit": high_audit,
    }


def parse_piecewise_alpha(text: Any) -> tuple[float, float]:
    low = 1.0
    high = 1.0
    for part in str(text).split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip() == "low":
            low = fnum(value, low)
        elif key.strip() == "high":
            high = fnum(value, high)
    return low, high


def fit_fixed_candidate(mode: str, rows: Sequence[dict[str, Any]], base_key: str, alpha_hint: Any) -> Any:
    if mode == "identity":
        y = np.asarray([fnum(row.get("true_ppm")) for row in rows], dtype=np.float64)
        return IdentityCalibrator(float(np.min(y)), float(np.max(y)))
    if mode == "affine":
        return fit_affine(rows, base_key)
    if mode in {"ridge_basic", "ridge_phase"}:
        names = feature_names_for(rows, mode)
        return fit_ridge_direct(rows, base_key, names, fnum(alpha_hint, 1.0))
    if mode == "piecewise_ridge":
        names = feature_names_for(rows, mode)
        low_alpha, high_alpha = parse_piecewise_alpha(alpha_hint)
        threshold = 125.0
        fallback_rows = list(rows)
        low_rows = [row for row in rows if ppm_bin(fnum(row.get(base_key))) == "low"] or fallback_rows
        high_rows = [row for row in rows if ppm_bin(fnum(row.get(base_key))) == "high"] or fallback_rows
        low = fit_ridge_direct(low_rows, base_key, names, low_alpha)
        high = fit_ridge_direct(high_rows, base_key, names, high_alpha)
        y = np.asarray([fnum(row.get("true_ppm")) for row in rows], dtype=np.float64)
        return PiecewiseRidgeCalibrator(
            low=low,
            high=high,
            threshold=threshold,
            clip_min=float(np.min(y)) if y.size else 0.0,
            clip_max=float(np.max(y)) if y.size else 250.0,
        )
    raise ValueError(f"unknown mode {mode}")


def fit_candidate(
    mode: str,
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    base_key: str,
    alphas: Sequence[float],
) -> tuple[Any, dict[str, Any]]:
    y_val = np.asarray([fnum(row.get("true_ppm")) for row in val_rows], dtype=np.float64)
    if mode == "identity":
        y_train = np.asarray([fnum(row.get("true_ppm")) for row in train_rows], dtype=np.float64)
        model = IdentityCalibrator(float(np.min(y_train)), float(np.max(y_train)))
        pred = model.predict(val_rows, base_key)
        return model, {"best_alpha": "", "best_val_RMSE": float(np.sqrt(np.mean((pred - y_val) ** 2)))}
    if mode == "affine":
        model = fit_affine(train_rows, base_key)
        pred = model.predict(val_rows, base_key)
        return model, {"best_alpha": "", "best_val_RMSE": float(np.sqrt(np.mean((pred - y_val) ** 2)))}
    if mode in {"ridge_basic", "ridge_phase"}:
        names = feature_names_for([*train_rows, *val_rows], mode)
        return fit_select_ridge(train_rows, val_rows, base_key, names, alphas)
    if mode == "piecewise_ridge":
        names = feature_names_for([*train_rows, *val_rows], mode)
        return fit_piecewise(train_rows, val_rows, base_key, names, alphas)
    raise ValueError(f"unknown mode {mode}")


def attach_source_head_predictions(
    rows: list[dict[str, Any]],
    source_models: dict[str, Any],
) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    ridge_rows = apply_models(out, source_models["ridge"], "source_ridge")
    for idx, row in enumerate(ridge_rows):
        out[idx]["source_ridge_ppm"] = fnum(row.get("source_ridge_ppm"))
    mlp_rows = apply_mlp_models(out, source_models["per_gas_mlp"], "source_per_gas_mlp")
    for idx, row in enumerate(mlp_rows):
        out[idx]["source_per_gas_mlp_ppm"] = fnum(row.get("source_per_gas_mlp_ppm"))
    shared_rows = apply_shared_model(out, source_models["shared_mlp"], "source_shared_mlp")
    for idx, row in enumerate(shared_rows):
        out[idx]["source_shared_mlp_ppm"] = fnum(row.get("source_shared_mlp_ppm"))
    return out


def fit_source_models(
    data_root: Path,
    source_clients: Sequence[str],
    alphas: Sequence[float],
    mlp_hiddens: Sequence[tuple[int, ...]],
    mlp_alphas: Sequence[float],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_train = build_oracle_rows(data_root, source_clients, "train")
    source_val = build_oracle_rows(data_root, source_clients, "calibration")
    source_test = build_oracle_rows(data_root, source_clients, "test")
    feature_names = sorted(source_train[0]["feature_dict"].keys())
    audit: list[dict[str, Any]] = []

    ridge_models = {}
    for cls_id in sorted(CLASS_NAMES):
        train_rows = [row for row in source_train if row["true_class"] == cls_id]
        val_rows = [row for row in source_val if row["true_class"] == cls_id]
        model, info = fit_select_refit(train_rows, val_rows, feature_names, alphas)
        ridge_models[cls_id] = model
        audit.append(
            {
                "base_mode": "source_ridge",
                "class_id": cls_id,
                "gas": CLASS_NAMES[cls_id],
                "best": info.get("best_alpha"),
                "best_val_RMSE": info.get("best_val_RMSE"),
            }
        )

    mlp_models = {}
    for cls_id in sorted(CLASS_NAMES):
        train_rows = [row for row in source_train if row["true_class"] == cls_id]
        val_rows = [row for row in source_val if row["true_class"] == cls_id]
        model, info = fit_select_refit_mlp(train_rows, val_rows, feature_names, mlp_hiddens, mlp_alphas, seed + cls_id)
        mlp_models[cls_id] = model
        audit.append(
            {
                "base_mode": "source_per_gas_mlp",
                "class_id": cls_id,
                "gas": CLASS_NAMES[cls_id],
                "best": str(info.get("best_hidden")) + f";alpha={info.get('best_alpha')}",
                "best_val_RMSE": info.get("best_val_RMSE"),
            }
        )

    shared_model, shared_info = fit_select_refit_shared_mlp(source_train, source_val, feature_names, mlp_hiddens, mlp_alphas, seed + 1000)
    audit.append(
        {
            "base_mode": "source_shared_mlp",
            "class_id": "all",
            "gas": "all",
            "best": str(shared_info.get("best_hidden")) + f";alpha={shared_info.get('best_alpha')}",
            "best_val_RMSE": shared_info.get("best_val_RMSE"),
        }
    )

    source_models = {"ridge": ridge_models, "per_gas_mlp": mlp_models, "shared_mlp": shared_model}
    source_pred = attach_source_head_predictions(source_test, source_models)
    source_summary: list[dict[str, Any]] = []
    for mode in BASE_MODES:
        source_summary.extend(summarize(source_pred, f"{mode}_ppm", f"{mode}_source_oracle", "test"))
    return source_models, source_summary, audit


def selected_key(base_mode: str, residual_mode: str) -> str:
    return f"{base_mode}_{residual_mode}_ppm"


def apply_calibrators(
    rows: Sequence[dict[str, Any]],
    base_mode: str,
    models: dict[tuple[str, int], Any],
    out_key: str,
) -> list[dict[str, Any]]:
    base_key = f"{base_mode}_ppm"
    out = [dict(row) for row in rows]
    for (client, cls_id), model in models.items():
        idxs = [
            idx for idx, row in enumerate(out)
            if str(row.get("client")) == client and inum(row.get("route_class")) == cls_id
        ]
        if not idxs:
            continue
        pred = model.predict([out[idx] for idx in idxs], base_key)
        for idx, value in zip(idxs, pred):
            out[idx][out_key] = float(value)
    return out


def fit_full_auto_v2_for_base(
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    base_mode: str,
    target_clients: Sequence[str],
    alphas: Sequence[float],
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_key = f"{base_mode}_ppm"
    selection_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    selected_modes: dict[tuple[str, int], str] = {}
    selected_alphas: dict[tuple[str, int], Any] = {}
    forced_alphas: dict[tuple[str, int, str], Any] = {}

    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                dict(row)
                for row in calibration_rows
                if row["client"] == client and inum(row["true_class"]) == cls_id
            ]
            if not cls_rows:
                continue
            fit_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            best_mode = ""
            best_score = float("inf")
            best_alpha: Any = ""
            for residual_mode in RESIDUAL_MODES:
                _model, info = fit_candidate(residual_mode, fit_rows, val_rows, base_key, alphas)
                score = fnum(info.get("best_val_RMSE"), float("inf"))
                forced_alphas[(client, cls_id, residual_mode)] = info.get("best_alpha", "")
                audit_rows.append(
                    {
                        "base_mode": base_mode,
                        "client": client,
                        "class_id": cls_id,
                        "gas": CLASS_NAMES[cls_id],
                        "residual_mode": residual_mode,
                        "n_fit": len(fit_rows),
                        "n_val": len(val_rows),
                        "best_alpha": info.get("best_alpha", ""),
                        "val_RMSE": score,
                    }
                )
                if score < best_score:
                    best_score = score
                    best_mode = residual_mode
                    best_alpha = info.get("best_alpha", "")
            selected_modes[(client, cls_id)] = best_mode
            selected_alphas[(client, cls_id)] = best_alpha
            selection_rows.append(
                {
                    "base_mode": base_mode,
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "selected_mode": best_mode,
                    "selected_alpha": best_alpha,
                    "selected_val_RMSE": best_score,
                    "n_calibration": len(cls_rows),
                    "n_fit": len(fit_rows),
                    "n_val": len(val_rows),
                }
            )

    candidate_rows: list[dict[str, Any]] = []
    forced_modes = ["identity", "affine", "ridge_phase", "piecewise_ridge"]
    candidates = [(f"{base_mode}_val_selected", selected_modes)]
    for mode in forced_modes:
        candidates.append((f"{base_mode}_forced_{mode}", {(client, cls_id): mode for client in target_clients for cls_id in sorted(CLASS_NAMES)}))

    for candidate, mode_by_key in candidates:
        models: dict[tuple[str, int], Any] = {}
        profile: dict[str, dict[str, str]] = {}
        for client in target_clients:
            profile[client] = {}
            for cls_id in sorted(CLASS_NAMES):
                cls_rows = [
                    dict(row)
                    for row in calibration_rows
                    if row["client"] == client and inum(row["true_class"]) == cls_id
                ]
                if not cls_rows:
                    continue
                mode = mode_by_key[(client, cls_id)]
                if candidate.endswith("_val_selected"):
                    alpha_hint = selected_alphas.get((client, cls_id), "")
                else:
                    alpha_hint = forced_alphas.get((client, cls_id, mode), "")
                model = fit_fixed_candidate(mode, cls_rows, base_key, alpha_hint)
                models[(client, cls_id)] = model
                profile[client][str(cls_id)] = mode
                audit_rows.append(
                    {
                        "base_mode": base_mode,
                        "candidate": candidate,
                        "client": client,
                        "class_id": cls_id,
                        "gas": CLASS_NAMES[cls_id],
                        "refit_mode": mode,
                        "alpha_hint": alpha_hint,
                        "n_refit": len(cls_rows),
                    }
                )
        rows = apply_calibrators(test_rows, base_mode, models, "candidate_ppm")
        for row in rows:
            row["candidate"] = candidate
            row["base_mode"] = base_mode
            row["selected_client_mode"] = profile.get(str(row.get("client")), {}).get(str(inum(row.get("route_class"))), "")
            row["corrected_ppm"] = fnum(row.get("candidate_ppm"))
        candidate_rows.extend(rows)

    return candidate_rows, selection_rows, audit_rows


def add_baseline_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_scope = {
        row["scope"]: row
        for row in summary_rows
        if row.get("mode") == "baseline_final_ppm" and row.get("split") == "test"
    }
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        item = dict(row)
        base = baseline_by_scope.get(str(row.get("scope")))
        if base:
            for field in METRIC_FIELDS:
                value = row.get(field)
                base_value = base.get(field)
                if value is not None and base_value is not None:
                    item[f"delta_{field}_vs_baseline"] = fnum(value) - fnum(base_value)
        out.append(item)
    return out


def value(summary: list[dict[str, Any]], mode: str, scope: str, field: str = "RMSE") -> float | None:
    for row in summary:
        if row.get("mode") == mode and row.get("scope") == scope and row.get("split") == "test":
            return None if row.get(field) in (None, "") else fnum(row.get(field))
    return None


def write_report(
    out: Path,
    summary_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    source_summary: list[dict[str, Any]],
    source_audit: list[dict[str, Any]],
) -> None:
    scopes = ["ALL", "C3-CO", "C4-CO", "C5-CO", "C3-CO_high_200_250", "C4-CO_high_200_250", "C5-CO_high_200_250", "nonCO_ALL"]
    modes = ["baseline_final_ppm"]
    for base in BASE_MODES:
        modes.extend(
            [
                f"{base}_forced_identity",
                f"{base}_forced_affine",
                f"{base}_forced_ridge_phase",
                f"{base}_forced_piecewise_ridge",
                f"{base}_val_selected",
            ]
        )
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for mode in modes:
        vals = []
        for scope in scopes:
            val = value(summary_rows, mode, scope)
            vals.append("" if val is None else f"{val:.2f}")
        lines.append("| " + mode + " | " + " | ".join(vals) + " |")

    select_lines = ["| base | client | gas | selected mode | val RMSE | n cal |", "|---|---|---|---|---:|---:|"]
    for row in selection_rows:
        select_lines.append(
            f"| {row['base_mode']} | {row['client']} | {row['gas']} | {row['selected_mode']} | "
            f"{fnum(row['selected_val_RMSE']):.2f} | {row['n_calibration']} |"
        )

    source_lines = ["| mode | ALL | C1-CO | C2-CO | nonCO_ALL |", "|---|---:|---:|---:|---:|"]
    for mode in [f"{base}_source_oracle" for base in BASE_MODES]:
        vals = []
        for scope in ["ALL", "C1-CO", "C2-CO", "nonCO_ALL"]:
            found = None
            for row in source_summary:
                if row.get("mode") == mode and row.get("scope") == scope:
                    found = row.get("RMSE")
                    break
            vals.append("" if found is None else f"{fnum(found):.2f}")
        source_lines.append("| " + mode + " | " + " | ".join(vals) + " |")

    text = f"""# Source Lightweight Heads + Full Residual auto_v2

Criterion: no-QC full-set target test final ppm. Test rows are not used for selection.

## Protocol

- Source heads are fitted on C1/C2 train.
- Source hyperparameters are selected on C1/C2 calibration.
- Target calibration is split deterministically into calibration-fit and calibration-validation.
- Residual candidates are selected per target client and gas using calibration-validation only.
- Selected residual candidates are refit on full target calibration.
- Target test is used only once for final reporting.

## Target Test RMSE

{chr(10).join(lines)}

## Calibration-val Selection

{chr(10).join(select_lines)}

## Source Oracle Test RMSE

{chr(10).join(source_lines)}

## Interpretation

- `forced_identity` is the source-lightweight direct-transfer baseline.
- `forced_affine` should broadly match the previous source-lightweight + target-affine diagnostic.
- `forced_ridge_phase` and `forced_piecewise_ridge` test whether rich residual calibration can rescue the source head.
- `val_selected` is the fair auto_v2-style result because the mode is chosen using calibration-validation only.
- A lightweight head should not replace R3aK16 unless `val_selected` approaches the original baseline and ideally the H2.3/H8 mainline.
"""
    (out / "source_lightweight_full_auto_v2_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair source lightweight head + full residual auto_v2 evaluation.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--source-clients", default="1,2")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-hidden-grid", default="16")
    parser.add_argument("--mlp-alphas", default="0.01,0.1")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/source_lightweight_full_auto_v2_20260625_lite")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    source_clients = [client_name(item.strip()) for item in args.source_clients.split(",") if item.strip()]
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    ridge_alphas = [float(item.strip()) for item in args.ridge_alphas.split(",") if item.strip()]
    mlp_hiddens = parse_hidden_grid(args.mlp_hidden_grid)
    mlp_alphas = [float(item.strip()) for item in args.mlp_alphas.split(",") if item.strip()]

    source_models, source_summary, source_audit = fit_source_models(
        data_root=data_root,
        source_clients=source_clients,
        alphas=ridge_alphas,
        mlp_hiddens=mlp_hiddens,
        mlp_alphas=mlp_alphas,
        seed=args.seed,
    )

    target_base_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    target_rows = add_target_features(target_base_rows, data_root)
    target_cal = [dict(row) for row in target_rows if row["split"] == "calibration"]
    target_test = [dict(row) for row in target_rows if row["split"] == "test"]
    for row in target_cal:
        row["route_class"] = row["true_class"]
    for row in target_test:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))

    target_cal = attach_source_head_predictions(target_cal, source_models)
    target_test = attach_source_head_predictions(target_test, source_models)

    all_candidate_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    all_audit_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(summarize(target_test, "baseline_final_ppm", "baseline_final_ppm", "test"))

    for base_mode in BASE_MODES:
        candidate_rows, selection_rows, audit_rows = fit_full_auto_v2_for_base(
            calibration_rows=target_cal,
            test_rows=target_test,
            base_mode=base_mode,
            target_clients=target_clients,
            alphas=ridge_alphas,
            val_ratio=args.val_ratio,
        )
        all_candidate_rows.extend(candidate_rows)
        all_selection_rows.extend(selection_rows)
        all_audit_rows.extend(audit_rows)
        for candidate in sorted({row["candidate"] for row in candidate_rows}):
            rows = [row for row in candidate_rows if row["candidate"] == candidate]
            summary_rows.extend(summarize(rows, "corrected_ppm", candidate, "test"))
    summary_rows = add_baseline_deltas(summary_rows)

    write_csv(out / "source_lightweight_full_auto_v2_summary.csv", summary_rows)
    write_csv(out / "source_lightweight_full_auto_v2_selection.csv", all_selection_rows)
    write_csv(out / "source_lightweight_full_auto_v2_audit.csv", all_audit_rows)
    write_csv(out / "source_lightweight_full_auto_v2_predictions.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in all_candidate_rows])
    write_csv(out / "source_lightweight_source_oracle_summary.csv", source_summary)
    write_csv(out / "source_lightweight_source_fit_audit.csv", source_audit)
    write_report(out, summary_rows, all_selection_rows, source_summary, source_audit)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "source_clients": source_clients,
                "target_clients": target_clients,
                "base_modes": BASE_MODES,
                "residual_modes": RESIDUAL_MODES,
                "ridge_alphas": ridge_alphas,
                "mlp_hidden_grid": [list(item) for item in mlp_hiddens],
                "mlp_alphas": mlp_alphas,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote source lightweight full auto_v2 evaluation to {out}")


if __name__ == "__main__":
    main()
