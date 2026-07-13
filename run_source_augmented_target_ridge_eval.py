"""Target Ridge heads augmented with source-lightweight head predictions.

This experiment checks whether source-trained lightweight heads add useful
information once target calibration is allowed.

For each target client/gas, we fit a target-calibrated Ridge head on rich window
features plus source-head ppm predictions. Target test still uses the fixed-DA
predicted gas route. QC is not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    apply_client_models,
    apply_models,
    build_oracle_rows,
    client_name,
    deterministic_train_val,
    fit_ridge,
    fit_select_refit,
    fnum,
    inum,
    read_csv,
    summarize,
    write_csv,
)
from run_formal_target_ridge_auto_v2_eval import (
    apply_c4_rescue,
    attach_response_phase,
    selected_c4_gate,
)
from run_source_lightweight_regression_head_ablation import (
    apply_mlp_models,
    apply_shared_model,
    fit_select_refit_mlp,
    fit_select_refit_shared_mlp,
    parse_hidden_grid,
)


def add_pred_features(rows: list[dict[str, Any]], pred_keys: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        feature_dict = dict(item["feature_dict"])
        for key in pred_keys:
            feature_dict[f"srcpred_{key}"] = fnum(item.get(key), 0.0)
        item["feature_dict"] = feature_dict
        out.append(item)
    return out


def fit_source_heads(
    data_root: Path,
    source_clients: list[str],
    ridge_alphas: list[float],
    mlp_hiddens: list[tuple[int, ...]],
    mlp_alphas: list[float],
    seed: int,
) -> tuple[dict[int, Any], dict[int, Any], Any, list[str]]:
    source_train = build_oracle_rows(data_root, source_clients, "train")
    source_val = build_oracle_rows(data_root, source_clients, "calibration")
    feature_names = sorted(source_train[0]["feature_dict"].keys())

    ridge_models: dict[int, Any] = {}
    mlp_models: dict[int, Any] = {}
    for cls_id in sorted(CLASS_NAMES):
        train_rows = [row for row in source_train if row["true_class"] == cls_id]
        val_rows = [row for row in source_val if row["true_class"] == cls_id]
        ridge_model, _ = fit_select_refit(train_rows, val_rows, feature_names, ridge_alphas)
        ridge_models[cls_id] = ridge_model
        mlp_model, _ = fit_select_refit_mlp(
            train_rows,
            val_rows,
            feature_names,
            mlp_hiddens,
            mlp_alphas,
            seed + cls_id,
        )
        mlp_models[cls_id] = mlp_model
    shared_model, _ = fit_select_refit_shared_mlp(
        source_train,
        source_val,
        feature_names,
        mlp_hiddens,
        mlp_alphas,
        seed + 1000,
    )
    return ridge_models, mlp_models, shared_model, feature_names


def attach_source_predictions(
    rows: list[dict[str, Any]],
    ridge_models: dict[int, Any],
    mlp_models: dict[int, Any],
    shared_model: Any,
) -> list[dict[str, Any]]:
    ridge_rows = apply_models(rows, ridge_models, "h1_source_ridge")
    for row in ridge_rows:
        row["H1_source_ridge_ppm"] = row["h1_source_ridge_ppm"]
    mlp_rows = apply_mlp_models(ridge_rows, mlp_models, "H2_source_per_gas_mlp")
    shared_rows = apply_shared_model(mlp_rows, shared_model, "H3_source_shared_mlp")
    return shared_rows


def metric_value(summary: list[dict[str, Any]], mode: str, scope: str, metric: str = "RMSE") -> float | None:
    for row in summary:
        if row.get("mode") == mode and row.get("scope") == scope:
            value = row.get(metric)
            return None if value in (None, "") else float(value)
    return None


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        client_name(row.get("client") or row.get("client_id")),
        str(row.get("split", "calibration")),
        inum(row.get("sample_index")),
    )


def attach_prediction_column(
    rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    prediction_key: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in prediction_rows:
        key = row_key(row)
        if key in by_key:
            raise ValueError(f"duplicate prediction row for {prediction_key}: {key}")
        by_key[key] = row
    output: list[dict[str, Any]] = []
    for row in rows:
        key = row_key(row)
        prediction_row = by_key.pop(key, None)
        if prediction_row is None or prediction_key not in prediction_row:
            raise ValueError(f"missing prediction {prediction_key}: {key}")
        item = dict(row)
        item[prediction_key] = fnum(prediction_row[prediction_key])
        output.append(item)
    if by_key:
        raise ValueError(f"unmatched prediction rows for {prediction_key}: {len(by_key)}")
    return output


def fit_target_ridge_holdout_predictions(
    training_feature_rows: list[dict[str, Any]],
    validation_feature_rows: list[dict[str, Any]],
    target_clients: list[str],
    feature_names: list[str],
    alphas: list[float],
    val_ratio: float,
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit on calibration-train rows and predict deterministic holdout rows.

    ``training_feature_rows`` and ``validation_feature_rows`` may contain
    different feature_dict values for the same physical row. This lets source
    features be oracle-routed for fitting while validation rows use the
    deployment-visible route.
    """
    validation_by_key = {row_key(row): row for row in validation_feature_rows}
    train_models: dict[tuple[str, int], Any] = {}
    val_rows_all: list[dict[str, Any]] = []
    fit_audit: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row for row in training_feature_rows
                if client_name(row.get("client")) == client and inum(row.get("true_class")) == cls_id
            ]
            if not cls_rows:
                continue
            train_rows, val_seed_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            val_rows = [dict(validation_by_key[row_key(row)]) for row in val_seed_rows]
            for row in val_rows:
                row["route_class"] = inum(row.get("pred_class"))
            _refit_model, audit = fit_select_refit(train_rows, val_rows, feature_names, alphas)
            best_alpha = fnum(audit["best_alpha"])
            train_models[(client, cls_id)] = fit_ridge(train_rows, feature_names, best_alpha)
            val_rows_all.extend(val_rows)
            fit_audit.append(
                {
                    "family": prefix,
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_alpha": best_alpha,
                    "best_val_RMSE": audit["best_val_RMSE"],
                }
            )
    return apply_client_models(val_rows_all, train_models, prefix), fit_audit


def write_report(out: Path, summary: list[dict[str, Any]], fit_audit: list[dict[str, Any]]) -> None:
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
        "target_ridge_rich_only",
        "target_ridge_rich_only_plus_c4_rescue",
        "target_ridge_plus_source_preds",
        "target_ridge_plus_source_preds_plus_c4_rescue",
    ]
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for mode in modes:
        values = []
        for scope in scopes:
            value = metric_value(summary, mode, scope)
            values.append("" if value is None else f"{value:.2f}")
        lines.append("| " + mode + " | " + " | ".join(values) + " |")

    audit_lines = ["| feature set | client | gas | train N | val N | alpha | val RMSE |", "|---|---|---|---:|---:|---:|---:|"]
    for row in fit_audit:
        audit_lines.append(
            f"| {row['feature_set']} | {row['client']} | {row['gas']} | {row['train_N']} | {row['val_N']} | "
            f"{float(row['best_alpha']):.6g} | {float(row['best_val_RMSE']):.2f} |"
        )

    text = f"""# Source-Augmented Target Ridge Evaluation

Question:

- Do source-trained lightweight heads add useful information to target-calibrated direct heads?

Protocol:

- Source heads: trained on C1/C2 train, selected on C1/C2 calibration.
- Target heads: per-client/per-gas Ridge, selected on target calibration holdout.
- Target test: fixed-DA predicted gas route.
- QC: not used.

Feature sets:

- `target_ridge_rich_only`: rich target window statistics only.
- `target_ridge_plus_source_preds`: rich target stats plus source Ridge/MLP/shared-MLP ppm predictions.
- `*_plus_c4_rescue`: same prediction plus the existing calibration-selected C4 route-rescue gate.

## Target Test RMSE

{chr(10).join(lines)}

## Fit Audit

{chr(10).join(audit_lines)}

## Reading

- If source predictions add useful transferable information, `target_ridge_plus_source_preds` should beat `target_ridge_rich_only`.
- If not, the target-only direct-head path is simpler and should remain the mainline.
"""
    (out / "source_augmented_target_ridge_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate source-prediction-augmented target Ridge heads.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--source-clients", default="1,2")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-hidden-grid", default="16")
    parser.add_argument("--mlp-alphas", default="0.01,0.1")
    parser.add_argument("--route-rescue-artifact", default="results/deployment_candidates_20260624/c12_c345_a1_rich_residual_plus_c4_rescue.json")
    parser.add_argument("--disable-c4-rescue", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/source_augmented_target_ridge_20260625_lite")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    source_clients = [client_name(item.strip()) for item in args.source_clients.split(",") if item.strip()]
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    ridge_alphas = [float(item.strip()) for item in args.ridge_alphas.split(",") if item.strip()]
    mlp_hiddens = parse_hidden_grid(args.mlp_hidden_grid)
    mlp_alphas = [float(item.strip()) for item in args.mlp_alphas.split(",") if item.strip()]

    ridge_models, mlp_models, shared_model, _source_feature_names = fit_source_heads(
        data_root,
        source_clients,
        ridge_alphas,
        mlp_hiddens,
        mlp_alphas,
        args.seed,
    )

    target_base_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    target_rows = add_target_features(target_base_rows, data_root)
    target_cal_route = [dict(row) for row in target_rows if row["split"] == "calibration"]
    target_cal = [dict(row) for row in target_cal_route]
    target_test = [dict(row) for row in target_rows if row["split"] == "test"]
    for row in target_cal:
        row["route_class"] = row["true_class"]
    for row in target_test:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))

    target_cal_with_src = attach_source_predictions(target_cal, ridge_models, mlp_models, shared_model)
    target_cal_route_with_src = attach_source_predictions(target_cal_route, ridge_models, mlp_models, shared_model)
    target_test_with_src = attach_source_predictions(target_test, ridge_models, mlp_models, shared_model)
    pred_keys = ["H1_source_ridge_ppm", "H2_source_per_gas_mlp_ppm", "H3_source_shared_mlp_ppm"]
    target_cal_aug = add_pred_features(target_cal_with_src, pred_keys)
    target_cal_route_aug = add_pred_features(target_cal_route_with_src, pred_keys)
    target_test_aug = add_pred_features(target_test_with_src, pred_keys)

    rich_feature_names = sorted(target_cal[0]["feature_dict"].keys())
    aug_feature_names = sorted(target_cal_aug[0]["feature_dict"].keys())

    fit_audit: list[dict[str, Any]] = []
    rich_models: dict[tuple[str, int], Any] = {}
    aug_models: dict[tuple[str, int], Any] = {}
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            rich_rows = [row for row in target_cal if row["client"] == client and inum(row["true_class"]) == cls_id]
            rich_train, rich_val = deterministic_train_val(rich_rows, val_ratio=0.25)
            rich_model, audit = fit_select_refit(rich_train, rich_val, rich_feature_names, ridge_alphas)
            rich_models[(client, cls_id)] = rich_model
            fit_audit.append(
                {
                    "feature_set": "rich_only",
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(rich_train),
                    "val_N": len(rich_val),
                    "best_alpha": audit["best_alpha"],
                    "best_val_RMSE": audit["best_val_RMSE"],
                }
            )

            aug_rows = [row for row in target_cal_aug if row["client"] == client and inum(row["true_class"]) == cls_id]
            aug_train, aug_val = deterministic_train_val(aug_rows, val_ratio=0.25)
            aug_model, audit = fit_select_refit(aug_train, aug_val, aug_feature_names, ridge_alphas)
            aug_models[(client, cls_id)] = aug_model
            fit_audit.append(
                {
                    "feature_set": "rich_plus_source_preds",
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(aug_train),
                    "val_N": len(aug_val),
                    "best_alpha": audit["best_alpha"],
                    "best_val_RMSE": audit["best_val_RMSE"],
                }
            )

    target_rich = apply_client_models(target_test, rich_models, "target_ridge_rich_only")
    target_aug = apply_client_models(target_test_aug, aug_models, "target_ridge_plus_source_preds")
    validation_rich, validation_rich_audit = fit_target_ridge_holdout_predictions(
        target_cal_route,
        target_cal_route,
        target_clients,
        rich_feature_names,
        ridge_alphas,
        0.25,
        "target_ridge_rich_only",
    )
    validation_aug, validation_aug_audit = fit_target_ridge_holdout_predictions(
        target_cal_aug,
        target_cal_route_aug,
        target_clients,
        aug_feature_names,
        ridge_alphas,
        0.25,
        "target_ridge_plus_source_preds",
    )
    target_aug = attach_prediction_column(
        target_aug,
        target_rich,
        "target_ridge_rich_only_ppm",
    )
    validation_aug = attach_prediction_column(
        validation_aug,
        validation_rich,
        "target_ridge_rich_only_ppm",
    )

    gate = None
    target_rich_rescue: list[dict[str, Any]] = []
    target_aug_rescue: list[dict[str, Any]] = []
    validation_rich_rescue: list[dict[str, Any]] = []
    validation_aug_rescue: list[dict[str, Any]] = []
    if not args.disable_c4_rescue:
        gate = selected_c4_gate(args.route_rescue_artifact)
        target_rich_phase = attach_response_phase(target_rich, data_root)
        target_aug_phase = attach_response_phase(target_aug, data_root)
        target_rich_rescue = apply_c4_rescue(
            target_rich_phase,
            "target_ridge_rich_only_ppm",
            "target_ridge_rich_only_plus_c4_rescue_ppm",
            gate,
        )
        target_aug_rescue = apply_c4_rescue(
            target_aug_phase,
            "target_ridge_plus_source_preds_ppm",
            "target_ridge_plus_source_preds_plus_c4_rescue_ppm",
            gate,
        )
        validation_rich_rescue = apply_c4_rescue(
            attach_response_phase(validation_rich, data_root),
            "target_ridge_rich_only_ppm",
            "target_ridge_rich_only_plus_c4_rescue_ppm",
            gate,
        )
        validation_aug_rescue = apply_c4_rescue(
            attach_response_phase(validation_aug, data_root),
            "target_ridge_plus_source_preds_ppm",
            "target_ridge_plus_source_preds_plus_c4_rescue_ppm",
            gate,
        )

    summary: list[dict[str, Any]] = []
    summary.extend(summarize(target_test, "baseline_final_ppm", "baseline_final_ppm", "test"))
    summary.extend(summarize(target_rich, "target_ridge_rich_only_ppm", "target_ridge_rich_only", "test"))
    summary.extend(summarize(target_aug, "target_ridge_plus_source_preds_ppm", "target_ridge_plus_source_preds", "test"))
    if not args.disable_c4_rescue:
        summary.extend(summarize(target_rich_rescue, "target_ridge_rich_only_plus_c4_rescue_ppm", "target_ridge_rich_only_plus_c4_rescue", "test"))
        summary.extend(summarize(target_aug_rescue, "target_ridge_plus_source_preds_plus_c4_rescue_ppm", "target_ridge_plus_source_preds_plus_c4_rescue", "test"))

    write_csv(out / "target_summary.csv", summary)
    write_csv(out / "fit_audit.csv", fit_audit)
    write_csv(out / "validation_fit_audit.csv", [*validation_rich_audit, *validation_aug_audit])
    write_csv(out / "target_predictions_rich_only.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_rich])
    write_csv(out / "target_predictions_plus_source_preds.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_aug])
    write_csv(out / "target_validation_rich_only.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in validation_rich])
    write_csv(out / "target_validation_plus_source_preds.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in validation_aug])
    if not args.disable_c4_rescue:
        write_csv(out / "target_predictions_rich_only_plus_c4_rescue.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_rich_rescue])
        write_csv(out / "target_predictions_plus_source_preds_plus_c4_rescue.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in target_aug_rescue])
        write_csv(out / "target_validation_rich_only_plus_c4_rescue.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in validation_rich_rescue])
        write_csv(out / "target_validation_plus_source_preds_plus_c4_rescue.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in validation_aug_rescue])
    write_report(out, summary, fit_audit)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "source_clients": source_clients,
                "target_clients": target_clients,
                "ridge_alphas": ridge_alphas,
                "mlp_hidden_grid": [list(item) for item in mlp_hiddens],
                "mlp_alphas": mlp_alphas,
                "source_prediction_features": pred_keys,
                "route_rescue_artifact": None if args.disable_c4_rescue else args.route_rescue_artifact,
                "c4_rescue_enabled": not args.disable_c4_rescue,
                "selected_c4_gate": gate,
                "rich_feature_count": len(rich_feature_names),
                "augmented_feature_count": len(aug_feature_names),
                "seed": args.seed,
                "validation_outputs": [
                    "target_validation_rich_only.csv",
                    "target_validation_plus_source_preds.csv",
                    *(
                        []
                        if args.disable_c4_rescue
                        else [
                            "target_validation_rich_only_plus_c4_rescue.csv",
                            "target_validation_plus_source_preds_plus_c4_rescue.csv",
                        ]
                    ),
                    "validation_fit_audit.csv",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote source-augmented target Ridge evaluation to {out}")


if __name__ == "__main__":
    main()
