"""Evaluate source-trained lightweight heads with target-domain calibration.

This is the fair follow-up to the source-only lightweight-head ablation:

1. Fit lightweight heads on source clients.
2. Use target calibration split to fit per-client/per-gas affine calibrators.
3. Evaluate target test with the fixed-DA predicted gas route.

QC is not used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    apply_models,
    build_oracle_rows,
    client_name,
    fit_select_refit,
    fnum,
    inum,
    read_csv,
    summarize,
    write_csv,
)
from run_source_lightweight_regression_head_ablation import (
    SharedMLPHead,
    apply_mlp_models,
    apply_shared_model,
    fit_select_refit_mlp,
    fit_select_refit_shared_mlp,
    parse_hidden_grid,
)


@dataclass
class AffineCalibrator:
    slope: float
    intercept: float
    clip_min: float
    clip_max: float
    n_fit: int

    def apply(self, pred: float) -> float:
        value = self.slope * float(pred) + self.intercept
        return float(np.clip(value, self.clip_min, self.clip_max))


def fit_affine(x: Sequence[float], y: Sequence[float], ridge: float = 1e-6) -> AffineCalibrator:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2 or float(np.std(x_arr)) < 1e-9:
        intercept = float(np.mean(y_arr - x_arr)) if x_arr.size else 0.0
        return AffineCalibrator(
            slope=1.0,
            intercept=intercept,
            clip_min=float(np.min(y_arr)) if y_arr.size else 0.0,
            clip_max=float(np.max(y_arr)) if y_arr.size else 250.0,
            n_fit=int(x_arr.size),
        )
    design = np.column_stack([x_arr, np.ones_like(x_arr)])
    reg = np.eye(2, dtype=np.float64) * float(ridge)
    reg[1, 1] = 0.0
    coef = np.linalg.pinv(design.T @ design + reg) @ design.T @ y_arr
    return AffineCalibrator(
        slope=float(coef[0]),
        intercept=float(coef[1]),
        clip_min=float(np.min(y_arr)),
        clip_max=float(np.max(y_arr)),
        n_fit=int(x_arr.size),
    )


def fit_target_affine_calibrators(
    calibration_rows: list[dict[str, Any]],
    pred_key: str,
    target_clients: Sequence[str],
) -> tuple[dict[tuple[str, int], AffineCalibrator], list[dict[str, Any]]]:
    calibrators: dict[tuple[str, int], AffineCalibrator] = {}
    audit: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id, gas in CLASS_NAMES.items():
            rows = [
                row for row in calibration_rows
                if row["client"] == client and inum(row["true_class"]) == cls_id
            ]
            cal = fit_affine(
                [fnum(row.get(pred_key)) for row in rows],
                [fnum(row.get("true_ppm")) for row in rows],
            )
            calibrators[(client, cls_id)] = cal
            audit.append(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": gas,
                    "pred_key": pred_key,
                    "n_fit": cal.n_fit,
                    "slope": cal.slope,
                    "intercept": cal.intercept,
                    "clip_min": cal.clip_min,
                    "clip_max": cal.clip_max,
                }
            )
    return calibrators, audit


def apply_target_affine(
    rows: list[dict[str, Any]],
    pred_key: str,
    out_key: str,
    calibrators: dict[tuple[str, int], AffineCalibrator],
) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for row in out:
        client = str(row["client"])
        cls_id = inum(row.get("route_class"))
        cal = calibrators.get((client, cls_id))
        base = fnum(row.get(pred_key))
        row[out_key] = cal.apply(base) if cal is not None and np.isfinite(base) else base
    return out


def metric_value(summary: list[dict[str, Any]], mode: str, scope: str, metric: str = "RMSE") -> float | None:
    for row in summary:
        if row.get("mode") == mode and row.get("scope") == scope:
            value = row.get(metric)
            return None if value in (None, "") else float(value)
    return None


def write_report(out: Path, summary: list[dict[str, Any]], audit: list[dict[str, Any]]) -> None:
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
        "H1_source_ridge_target_affine",
        "H2_source_per_gas_mlp",
        "H2_source_per_gas_mlp_target_affine",
        "H3_source_shared_mlp",
        "H3_source_shared_mlp_target_affine",
    ]
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for mode in modes:
        values = []
        for scope in scopes:
            value = metric_value(summary, mode, scope)
            values.append("" if value is None else f"{value:.2f}")
        lines.append("| " + mode + " | " + " | ".join(values) + " |")

    audit_lines = ["| pred key | client | gas | n | slope | intercept | clip |", "|---|---|---|---:|---:|---:|---|"]
    for row in audit:
        audit_lines.append(
            f"| {row['pred_key']} | {row['client']} | {row['gas']} | {row['n_fit']} | "
            f"{float(row['slope']):.4f} | {float(row['intercept']):.2f} | "
            f"[{float(row['clip_min']):.1f}, {float(row['clip_max']):.1f}] |"
        )

    text = f"""# Source Lightweight Heads + Target Affine Calibration

Question:

- If lightweight heads are source-trained first, can target calibration rescue the cross-client ppm mapping?

Protocol:

- Source fit: C1/C2 train.
- Source model selection: C1/C2 calibration.
- Target calibration: C3/C4/C5 calibration, per client and true gas.
- Target test: C3/C4/C5 test with fixed-DA predicted gas route.
- QC: not used.

## Target Test RMSE

{chr(10).join(lines)}

## Target Affine Calibrators

{chr(10).join(audit_lines)}

## Reading

- Compare each source-only mode against its target-affine version.
- Compare target-affine versions against `baseline_final_ppm` and the already confirmed target-only direct heads.
- If target-affine improves but remains below target-only direct heads, source pretraining is not adding enough value.
"""
    (out / "source_lightweight_target_calibrated_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate source lightweight heads with target affine calibration.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--source-clients", default="1,2")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-hidden-grid", default="16")
    parser.add_argument("--mlp-alphas", default="0.01,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/source_lightweight_target_calibrated_20260625_lite")
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
    feature_names = sorted(source_train[0]["feature_dict"].keys())

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

    ridge_models = {}
    for cls_id in sorted(CLASS_NAMES):
        train_rows = [row for row in source_train if row["true_class"] == cls_id]
        val_rows = [row for row in source_val if row["true_class"] == cls_id]
        model, _audit = fit_select_refit(train_rows, val_rows, feature_names, ridge_alphas)
        ridge_models[cls_id] = model

    mlp_models = {}
    for cls_id in sorted(CLASS_NAMES):
        train_rows = [row for row in source_train if row["true_class"] == cls_id]
        val_rows = [row for row in source_val if row["true_class"] == cls_id]
        model, _audit = fit_select_refit_mlp(
            train_rows,
            val_rows,
            feature_names,
            mlp_hiddens,
            mlp_alphas,
            args.seed + cls_id,
        )
        mlp_models[cls_id] = model

    shared_model, _shared_audit = fit_select_refit_shared_mlp(
        source_train,
        source_val,
        feature_names,
        mlp_hiddens,
        mlp_alphas,
        args.seed + 1000,
    )

    cal_ridge = apply_models(target_cal, ridge_models, "h1_source_ridge")
    for row in cal_ridge:
        row["H1_source_ridge_ppm"] = row["h1_source_ridge_ppm"]
    test_ridge = apply_models(target_test, ridge_models, "h1_source_ridge")
    for row in test_ridge:
        row["H1_source_ridge_ppm"] = row["h1_source_ridge_ppm"]

    cal_mlp = apply_mlp_models(target_cal, mlp_models, "H2_source_per_gas_mlp")
    test_mlp = apply_mlp_models(target_test, mlp_models, "H2_source_per_gas_mlp")

    cal_shared = apply_shared_model(target_cal, shared_model, "H3_source_shared_mlp")
    test_shared = apply_shared_model(target_test, shared_model, "H3_source_shared_mlp")

    all_audit: list[dict[str, Any]] = []
    ridge_cals, audit = fit_target_affine_calibrators(cal_ridge, "H1_source_ridge_ppm", target_clients)
    all_audit.extend(audit)
    mlp_cals, audit = fit_target_affine_calibrators(cal_mlp, "H2_source_per_gas_mlp_ppm", target_clients)
    all_audit.extend(audit)
    shared_cals, audit = fit_target_affine_calibrators(cal_shared, "H3_source_shared_mlp_ppm", target_clients)
    all_audit.extend(audit)

    test_ridge_cal = apply_target_affine(
        test_ridge,
        "H1_source_ridge_ppm",
        "H1_source_ridge_target_affine_ppm",
        ridge_cals,
    )
    test_mlp_cal = apply_target_affine(
        test_mlp,
        "H2_source_per_gas_mlp_ppm",
        "H2_source_per_gas_mlp_target_affine_ppm",
        mlp_cals,
    )
    test_shared_cal = apply_target_affine(
        test_shared,
        "H3_source_shared_mlp_ppm",
        "H3_source_shared_mlp_target_affine_ppm",
        shared_cals,
    )

    summary: list[dict[str, Any]] = []
    summary.extend(summarize(target_test, "baseline_final_ppm", "baseline_final_ppm", "test"))
    summary.extend(summarize(test_ridge, "H1_source_ridge_ppm", "H1_source_ridge", "test"))
    summary.extend(summarize(test_ridge_cal, "H1_source_ridge_target_affine_ppm", "H1_source_ridge_target_affine", "test"))
    summary.extend(summarize(test_mlp, "H2_source_per_gas_mlp_ppm", "H2_source_per_gas_mlp", "test"))
    summary.extend(summarize(test_mlp_cal, "H2_source_per_gas_mlp_target_affine_ppm", "H2_source_per_gas_mlp_target_affine", "test"))
    summary.extend(summarize(test_shared, "H3_source_shared_mlp_ppm", "H3_source_shared_mlp", "test"))
    summary.extend(summarize(test_shared_cal, "H3_source_shared_mlp_target_affine_ppm", "H3_source_shared_mlp_target_affine", "test"))

    write_csv(out / "target_summary.csv", summary)
    write_csv(out / "target_affine_audit.csv", all_audit)
    write_csv(out / "target_predictions_h1_source_ridge_target_affine.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in test_ridge_cal])
    write_csv(out / "target_predictions_h2_source_per_gas_mlp_target_affine.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in test_mlp_cal])
    write_csv(out / "target_predictions_h3_source_shared_mlp_target_affine.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in test_shared_cal])
    write_report(out, summary, all_audit)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "source_clients": source_clients,
                "target_clients": target_clients,
                "target_calibration": "per-client/per-true-gas affine on target calibration split",
                "feature_count": len(feature_names),
                "mlp_hidden_grid": [list(item) for item in mlp_hiddens],
                "mlp_alphas": mlp_alphas,
                "ridge_alphas": ridge_alphas,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote source lightweight target-calibrated evaluation to {out}")


if __name__ == "__main__":
    main()
