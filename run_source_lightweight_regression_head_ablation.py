"""Source-trained lightweight regression-head ablation.

This script answers a narrower question than the target direct-head work:

Can a lightweight source-domain regression head replace the original R3aK16
regression branch?

It freezes the existing fixed-DA route used in the target prediction CSV, fits
lightweight source heads on source clients with oracle gas labels, then evaluates
target test windows with the existing predicted gas route. QC is not used.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    apply_models,
    build_oracle_rows,
    client_name,
    fit_select_refit,
    fnum,
    inum,
    matrix_from_rows,
    read_csv,
    summarize,
    write_csv,
)


@dataclass
class MLPHead:
    hidden: tuple[int, ...]
    alpha: float
    model: Any
    clip_min: float
    clip_max: float
    feature_names: list[str]

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        x = matrix_from_rows(rows, self.feature_names)
        pred = np.asarray(self.model.predict(x), dtype=np.float64)
        return np.clip(pred, self.clip_min, self.clip_max)

    @property
    def param_count(self) -> int:
        mlp = self.model.named_steps["mlpregressor"]
        return int(
            sum(coef.size for coef in mlp.coefs_)
            + sum(intercept.size for intercept in mlp.intercepts_)
        )


@dataclass
class SharedMLPHead:
    hidden: tuple[int, ...]
    alpha: float
    model: Any
    clip_min: float
    clip_max: float
    feature_names: list[str]

    def _matrix(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        x = matrix_from_rows(rows, self.feature_names)
        gas = np.zeros((len(rows), len(CLASS_NAMES)), dtype=np.float64)
        for idx, row in enumerate(rows):
            cls_id = inum(row.get("route_class"))
            if cls_id in CLASS_NAMES:
                gas[idx, cls_id] = 1.0
        return np.concatenate([x, gas], axis=1)

    def predict(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        x = self._matrix(rows)
        pred = np.asarray(self.model.predict(x), dtype=np.float64)
        return np.clip(pred, self.clip_min, self.clip_max)

    @property
    def param_count(self) -> int:
        mlp = self.model.named_steps["mlpregressor"]
        return int(
            sum(coef.size for coef in mlp.coefs_)
            + sum(intercept.size for intercept in mlp.intercepts_)
        )


def parse_hidden_grid(text: str) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for raw in str(text).split(";"):
        raw = raw.strip()
        if not raw:
            continue
        out.append(tuple(int(v.strip()) for v in raw.strip("()").split(",") if v.strip()))
    if not out:
        raise ValueError("hidden grid is empty")
    return out


def fit_mlp(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    hidden: tuple[int, ...],
    alpha: float,
    seed: int,
) -> MLPHead:
    x = matrix_from_rows(rows, feature_names)
    y = np.asarray([fnum(row["true_ppm"]) for row in rows], dtype=np.float64)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="lbfgs",
        alpha=float(alpha),
        max_iter=800,
        random_state=int(seed),
    )
    model = make_pipeline(StandardScaler(), mlp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    return MLPHead(
        hidden=hidden,
        alpha=float(alpha),
        model=model,
        clip_min=float(np.min(y)),
        clip_max=float(np.max(y)),
        feature_names=list(feature_names),
    )


def fit_select_refit_mlp(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    hiddens: Sequence[tuple[int, ...]],
    alphas: Sequence[float],
    seed: int,
) -> tuple[MLPHead, dict[str, Any]]:
    y_val = np.asarray([fnum(row["true_ppm"]) for row in val_rows], dtype=np.float64)
    best_score = float("inf")
    best_hidden = tuple(hiddens[0])
    best_alpha = float(alphas[0])
    audit_rows: list[dict[str, Any]] = []
    for hidden in hiddens:
        for alpha in alphas:
            model = fit_mlp(train_rows, feature_names, hidden, float(alpha), seed)
            pred = model.predict(val_rows)
            score = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            audit_rows.append({"hidden": str(hidden), "alpha": float(alpha), "val_RMSE": score})
            if score < best_score:
                best_score = score
                best_hidden = tuple(hidden)
                best_alpha = float(alpha)
    model = fit_mlp([*train_rows, *val_rows], feature_names, best_hidden, best_alpha, seed)
    return model, {
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,
        "best_val_RMSE": best_score,
        "audit": audit_rows,
    }


def shared_matrix(rows: Sequence[dict[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    x = matrix_from_rows(rows, feature_names)
    gas = np.zeros((len(rows), len(CLASS_NAMES)), dtype=np.float64)
    for idx, row in enumerate(rows):
        cls_id = inum(row.get("route_class"))
        if cls_id in CLASS_NAMES:
            gas[idx, cls_id] = 1.0
    return np.concatenate([x, gas], axis=1)


def fit_shared_mlp(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    hidden: tuple[int, ...],
    alpha: float,
    seed: int,
) -> SharedMLPHead:
    x = shared_matrix(rows, feature_names)
    y = np.asarray([fnum(row["true_ppm"]) for row in rows], dtype=np.float64)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="lbfgs",
        alpha=float(alpha),
        max_iter=800,
        random_state=int(seed),
    )
    model = make_pipeline(StandardScaler(), mlp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x, y)
    return SharedMLPHead(
        hidden=hidden,
        alpha=float(alpha),
        model=model,
        clip_min=float(np.min(y)),
        clip_max=float(np.max(y)),
        feature_names=list(feature_names),
    )


def fit_select_refit_shared_mlp(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    hiddens: Sequence[tuple[int, ...]],
    alphas: Sequence[float],
    seed: int,
) -> tuple[SharedMLPHead, dict[str, Any]]:
    y_val = np.asarray([fnum(row["true_ppm"]) for row in val_rows], dtype=np.float64)
    best_score = float("inf")
    best_hidden = tuple(hiddens[0])
    best_alpha = float(alphas[0])
    audit_rows: list[dict[str, Any]] = []
    for hidden in hiddens:
        for alpha in alphas:
            model = fit_shared_mlp(train_rows, feature_names, hidden, float(alpha), seed)
            pred = model.predict(val_rows)
            score = float(np.sqrt(np.mean((pred - y_val) ** 2)))
            audit_rows.append({"hidden": str(hidden), "alpha": float(alpha), "val_RMSE": score})
            if score < best_score:
                best_score = score
                best_hidden = tuple(hidden)
                best_alpha = float(alpha)
    model = fit_shared_mlp([*train_rows, *val_rows], feature_names, best_hidden, best_alpha, seed)
    return model, {
        "best_hidden": best_hidden,
        "best_alpha": best_alpha,
        "best_val_RMSE": best_score,
        "audit": audit_rows,
    }


def apply_mlp_models(rows: list[dict[str, Any]], models: dict[int, MLPHead], prefix: str) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for cls_id, model in models.items():
        idxs = [idx for idx, row in enumerate(out) if inum(row.get("route_class")) == cls_id]
        if not idxs:
            continue
        pred_rows = [out[idx] for idx in idxs]
        pred = model.predict(pred_rows)
        for idx, value in zip(idxs, pred):
            out[idx][f"{prefix}_ppm"] = float(value)
            out[idx][f"{prefix}_delta_vs_final"] = float(value - fnum(out[idx].get("final_ppm"), value))
    return out


def apply_shared_model(rows: list[dict[str, Any]], model: SharedMLPHead, prefix: str) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    pred = model.predict(out)
    for row, value in zip(out, pred):
        row[f"{prefix}_ppm"] = float(value)
        row[f"{prefix}_delta_vs_final"] = float(value - fnum(row.get("final_ppm"), value))
    return out


def metric_value(summary: list[dict[str, Any]], mode: str, scope: str, metric: str = "RMSE") -> float | None:
    for row in summary:
        if row.get("mode") == mode and row.get("scope") == scope:
            value = row.get(metric)
            return None if value in (None, "") else float(value)
    return None


def write_report(out: Path, target_summary: list[dict[str, Any]], source_summary: list[dict[str, Any]], fit_audit: list[dict[str, Any]]) -> None:
    scopes = [
        "ALL",
        "C3-CO",
        "C4-CO",
        "C5-CO",
        "C3-CO_high_200_250",
        "C4-CO_high_200_250",
        "C5-CO_high_200_250",
        "nonCO_ALL",
    ]
    modes = [
        "baseline_final_ppm",
        "H1_source_ridge",
        "H2_source_per_gas_mlp",
        "H3_source_shared_mlp",
    ]
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for mode in modes:
        values = []
        for scope in scopes:
            value = metric_value(target_summary, mode, scope)
            values.append("" if value is None else f"{value:.2f}")
        lines.append("| " + mode + " | " + " | ".join(values) + " |")

    source_scopes = ["ALL", "C1-CO", "C2-CO", "C1-CO_high_200_250", "C2-CO_high_200_250", "nonCO_ALL"]
    source_lines = ["| mode | " + " | ".join(source_scopes) + " |", "|---|" + "|".join(["---:"] * len(source_scopes)) + "|"]
    for mode in ["H1_source_ridge_oracle", "H2_source_per_gas_mlp_oracle", "H3_source_shared_mlp_oracle"]:
        values = []
        for scope in source_scopes:
            value = metric_value(source_summary, mode, scope)
            values.append("" if value is None else f"{value:.2f}")
        source_lines.append("| " + mode + " | " + " | ".join(values) + " |")

    audit_lines = ["| experiment | class | gas | best config | val RMSE | params |", "|---|---:|---|---|---:|---:|"]
    for row in fit_audit:
        audit_lines.append(
            f"| {row['experiment']} | {row.get('class_id', '')} | {row.get('gas', '')} | "
            f"{row['best_config']} | {float(row['best_val_RMSE']):.2f} | {int(row['param_count'])} |"
        )

    text = f"""# Source Lightweight Regression Head Ablation

Question:

- Can lightweight source-domain regression heads replace the original R3aK16 source regression branch?

Protocol:

- Source fit: C1/C2 train.
- Source validation: C1/C2 calibration.
- Source test: C1/C2 test with oracle gas route.
- Target test: C3/C4/C5 test with existing fixed-DA predicted gas route.
- QC: not used.

## Target Test RMSE

{chr(10).join(lines)}

## Source Test RMSE

{chr(10).join(source_lines)}

## Fit Audit

{chr(10).join(audit_lines)}

## Reading Template

- A source-head replacement must beat or approach baseline target full-set metrics, not just source oracle metrics.
- If a head fits source well but fails target, the issue is cross-client concentration transfer rather than raw head capacity.
- H2/H3 are lightweight neural alternatives to H1 Ridge; if they still fail target, the current evidence favors keeping R3aK16 plus target-side direct heads.
"""
    (out / "source_lightweight_head_ablation_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source lightweight regression-head ablation.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--source-clients", default="1,2")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-hidden-grid", default="16;32")
    parser.add_argument("--mlp-alphas", default="0.001,0.01,0.1,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/source_lightweight_regression_head_ablation_20260625")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    source_clients = [client_name(item.strip()) for item in args.source_clients.split(",") if item.strip()]
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    ridge_alphas = [float(item.strip()) for item in args.ridge_alphas.split(",") if item.strip()]
    mlp_hiddens = parse_hidden_grid(args.mlp_hidden_grid)
    mlp_alphas = [float(item.strip()) for item in args.mlp_alphas.split(",") if item.strip()]

    source_train = build_oracle_rows(data_root, source_clients, "train")
    source_val = build_oracle_rows(data_root, source_clients, "calibration")
    source_test = build_oracle_rows(data_root, source_clients, "test")
    feature_names = sorted(source_train[0]["feature_dict"].keys())

    target_base_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    target_rows = add_target_features(target_base_rows, data_root)
    target_test = [row for row in target_rows if row["split"] == "test"]
    for row in target_test:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))

    fit_audit: list[dict[str, Any]] = []

    ridge_models = {}
    for cls_id in sorted(CLASS_NAMES):
        train_rows = [row for row in source_train if row["true_class"] == cls_id]
        val_rows = [row for row in source_val if row["true_class"] == cls_id]
        model, audit = fit_select_refit(train_rows, val_rows, feature_names, ridge_alphas)
        ridge_models[cls_id] = model
        fit_audit.append(
            {
                "experiment": "H1_source_ridge",
                "class_id": cls_id,
                "gas": CLASS_NAMES[cls_id],
                "best_config": f"alpha={audit['best_alpha']}",
                "best_val_RMSE": audit["best_val_RMSE"],
                "param_count": len(model.coef),
                "audit": json.dumps(audit["alpha_audit"], ensure_ascii=False),
            }
        )

    mlp_models = {}
    for cls_id in sorted(CLASS_NAMES):
        train_rows = [row for row in source_train if row["true_class"] == cls_id]
        val_rows = [row for row in source_val if row["true_class"] == cls_id]
        model, audit = fit_select_refit_mlp(
            train_rows,
            val_rows,
            feature_names,
            mlp_hiddens,
            mlp_alphas,
            args.seed + cls_id,
        )
        mlp_models[cls_id] = model
        fit_audit.append(
            {
                "experiment": "H2_source_per_gas_mlp",
                "class_id": cls_id,
                "gas": CLASS_NAMES[cls_id],
                "best_config": f"hidden={audit['best_hidden']}, alpha={audit['best_alpha']}",
                "best_val_RMSE": audit["best_val_RMSE"],
                "param_count": model.param_count,
                "audit": json.dumps(audit["audit"], ensure_ascii=False),
            }
        )

    shared_model, shared_audit = fit_select_refit_shared_mlp(
        source_train,
        source_val,
        feature_names,
        mlp_hiddens,
        mlp_alphas,
        args.seed + 1000,
    )
    fit_audit.append(
        {
            "experiment": "H3_source_shared_mlp",
            "class_id": "all",
            "gas": "all",
            "best_config": f"hidden={shared_audit['best_hidden']}, alpha={shared_audit['best_alpha']}",
            "best_val_RMSE": shared_audit["best_val_RMSE"],
            "param_count": shared_model.param_count,
            "audit": json.dumps(shared_audit["audit"], ensure_ascii=False),
        }
    )

    source_ridge = apply_models(source_test, ridge_models, "h1_source_ridge")
    for row in source_ridge:
        row["H1_source_ridge_oracle_ppm"] = row["h1_source_ridge_ppm"]
    source_mlp = apply_mlp_models(source_test, mlp_models, "H2_source_per_gas_mlp")
    for row in source_mlp:
        row["H2_source_per_gas_mlp_oracle_ppm"] = row["H2_source_per_gas_mlp_ppm"]
    source_shared = apply_shared_model(source_test, shared_model, "H3_source_shared_mlp")
    for row in source_shared:
        row["H3_source_shared_mlp_oracle_ppm"] = row["H3_source_shared_mlp_ppm"]

    target_ridge = apply_models(target_test, ridge_models, "h1_source_ridge")
    for row in target_ridge:
        row["H1_source_ridge_ppm"] = row["h1_source_ridge_ppm"]
    target_mlp = apply_mlp_models(target_test, mlp_models, "H2_source_per_gas_mlp")
    target_shared = apply_shared_model(target_test, shared_model, "H3_source_shared_mlp")

    target_summary: list[dict[str, Any]] = []
    target_summary.extend(summarize(target_test, "baseline_final_ppm", "baseline_final_ppm", "test"))
    target_summary.extend(summarize(target_ridge, "H1_source_ridge_ppm", "H1_source_ridge", "test"))
    target_summary.extend(summarize(target_mlp, "H2_source_per_gas_mlp_ppm", "H2_source_per_gas_mlp", "test"))
    target_summary.extend(summarize(target_shared, "H3_source_shared_mlp_ppm", "H3_source_shared_mlp", "test"))

    source_summary: list[dict[str, Any]] = []
    source_summary.extend(summarize(source_ridge, "H1_source_ridge_oracle_ppm", "H1_source_ridge_oracle", "test"))
    source_summary.extend(summarize(source_mlp, "H2_source_per_gas_mlp_oracle_ppm", "H2_source_per_gas_mlp_oracle", "test"))
    source_summary.extend(summarize(source_shared, "H3_source_shared_mlp_oracle_ppm", "H3_source_shared_mlp_oracle", "test"))

    write_csv(out / "target_summary.csv", target_summary)
    write_csv(out / "source_summary.csv", source_summary)
    write_csv(out / "fit_audit.csv", fit_audit)
    write_csv(out / "target_predictions_h1_source_ridge.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_ridge])
    write_csv(out / "target_predictions_h2_source_per_gas_mlp.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_mlp])
    write_csv(out / "target_predictions_h3_source_shared_mlp.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_shared])
    write_report(out, target_summary, source_summary, fit_audit)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "source_clients": source_clients,
                "target_clients": target_clients,
                "feature_count": len(feature_names),
                "ridge_alphas": ridge_alphas,
                "mlp_hidden_grid": [list(item) for item in mlp_hiddens],
                "mlp_alphas": mlp_alphas,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote source lightweight regression head ablation to {out}")


if __name__ == "__main__":
    main()
