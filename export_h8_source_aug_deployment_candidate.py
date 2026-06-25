"""Export H8 source-augmented CO-specialist as a runtime artifact.

The exported artifact extends the existing H2.3 hybrid MLP/Ridge artifact with
``source_aug_target_ridge_policy``. Runtime should then use:

- C4 route rescue first;
- H8 source-aug target Ridge on enabled predicted-CO rows;
- existing H2.3 target MLP/Ridge fallback for all other rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from export_hybrid_mlp_ridge_deployment_candidate import serialize_mlp_head
from run_formal_target_ridge_auto_v2_eval import selected_c4_gate
from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    client_name,
    deterministic_train_val,
    fit_select_refit,
    fnum,
    inum,
    read_csv,
)
from run_source_augmented_target_ridge_eval import (
    add_pred_features,
    attach_source_predictions,
    fit_source_heads,
)


PRED_KEYS = ["H1_source_ridge_ppm", "H2_source_per_gas_mlp_ppm", "H3_source_shared_mlp_ppm"]


def load_enabled_clients(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(v) for v in payload.get("switch_rule", {}).get("enabled_clients", [])]


def target_calibration_rows(target_predictions: Path, data_root: Path, target_clients: list[str]) -> list[dict[str, Any]]:
    raw_rows = [
        row
        for row in read_csv(target_predictions)
        if str(row.get("split")) == "calibration"
        and client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    rows = add_target_features(raw_rows, data_root)
    for row in rows:
        row["route_class"] = row["true_class"]
    return rows


def serialize_source_heads(
    ridge_models: dict[int, Any],
    mlp_models: dict[int, Any],
    shared_model: Any,
) -> dict[str, Any]:
    ridge = []
    mlp = []
    for cls_id in sorted(CLASS_NAMES):
        r = ridge_models[cls_id].to_json()
        r.update({"class_id": cls_id, "gas": CLASS_NAMES[cls_id]})
        ridge.append(r)
        m = serialize_mlp_head(mlp_models[cls_id])
        m.update({"class_id": cls_id, "gas": CLASS_NAMES[cls_id]})
        mlp.append(m)
    shared = serialize_mlp_head(shared_model)
    shared.update({"gas": "shared"})
    return {
        "ridge_per_gas": ridge,
        "mlp_per_gas": mlp,
        "shared_mlp": shared,
    }


def fit_target_augmented_models(
    target_cal_aug: list[dict[str, Any]],
    target_clients: list[str],
    feature_names: list[str],
    ridge_alphas: list[float],
    enabled_clients: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row
                for row in target_cal_aug
                if row["client"] == client and inum(row["true_class"]) == cls_id
            ]
            train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=0.25)
            model, audit = fit_select_refit(train_rows, val_rows, feature_names, ridge_alphas)
            payload = model.to_json()
            payload.update(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "enabled": client in enabled_clients and cls_id == 1,
                    "selected_mode": "source_aug_target_ridge" if client in enabled_clients and cls_id == 1 else "baseline_h2_3",
                    "n_fit": len(cls_rows),
                }
            )
            models.append(payload)
            audit_rows.append(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_alpha": audit["best_alpha"],
                    "best_val_RMSE": audit["best_val_RMSE"],
                    "enabled": int(client in enabled_clients and cls_id == 1),
                }
            )
    return models, audit_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export H8 source-augmented CO-specialist artifact.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--source-clients", default="1,2")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-hidden-grid", default="16")
    parser.add_argument("--mlp-alphas", default="0.01,0.1")
    parser.add_argument("--selector-profile", default="results/h8_calibration_selector_20260625/h8_pred_co_source_aug_selector_profile.json")
    parser.add_argument("--base-artifact", default="results/deployment_h2_3_mlp_ridge_candidate_20260624/rich_residual_candidate.json")
    parser.add_argument("--route-rescue-artifact", default="results/deployment_candidates_20260624/c12_c345_a1_rich_residual_plus_c4_rescue.json")
    parser.add_argument("--output", default="results/deployment_candidates_20260624/c12_c345_h8_source_aug_co_specialist.json")
    parser.add_argument("--audit-output", default="results/h8_source_aug_runtime_export_20260625/h8_source_aug_export_fit_audit.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    source_clients = [client_name(item.strip()) for item in args.source_clients.split(",") if item.strip()]
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    ridge_alphas = [float(item.strip()) for item in args.ridge_alphas.split(",") if item.strip()]
    from run_source_lightweight_regression_head_ablation import parse_hidden_grid

    mlp_hiddens = parse_hidden_grid(args.mlp_hidden_grid)
    mlp_alphas = [float(item.strip()) for item in args.mlp_alphas.split(",") if item.strip()]
    enabled_clients = set(load_enabled_clients(Path(args.selector_profile)))

    ridge_models, mlp_models, shared_model, source_feature_names = fit_source_heads(
        data_root,
        source_clients,
        ridge_alphas,
        mlp_hiddens,
        mlp_alphas,
        args.seed,
    )
    target_cal = target_calibration_rows(Path(args.target_predictions), data_root, target_clients)
    target_cal_with_src = attach_source_predictions(target_cal, ridge_models, mlp_models, shared_model)
    target_cal_aug = add_pred_features(target_cal_with_src, PRED_KEYS)
    aug_feature_names = sorted(target_cal_aug[0]["feature_dict"].keys())
    target_aug_models, audit_rows = fit_target_augmented_models(
        target_cal_aug,
        target_clients,
        aug_feature_names,
        ridge_alphas,
        enabled_clients,
    )

    base_artifact = json.loads(Path(args.base_artifact).read_text(encoding="utf-8"))
    base_artifact["schema"] = "gaps_hybrid_mlp_ridge_policy.v1+h8_source_aug.v1"
    base_artifact["candidate_name"] = "c12_c345_h8_pred_co_source_aug_else_h23"
    base_artifact["source_aug_target_ridge_policy"] = {
        "schema": "source_aug_target_ridge.v1",
        "base_profile": "h2_3",
        "switch_rule": {"type": "pred_class_equals", "class_id": 1, "enabled_clients": sorted(enabled_clients)},
        "source_prediction_keys": PRED_KEYS,
        "source_feature_names": source_feature_names,
        "source_heads": serialize_source_heads(ridge_models, mlp_models, shared_model),
        "feature_names": aug_feature_names,
        "models": target_aug_models,
        "selector_profile": args.selector_profile,
    }
    base_artifact["route_rescue_policy"] = {
        "schema": "c4_route_rescue_policy.v1",
        "selected_gate": selected_c4_gate(args.route_rescue_artifact),
    }
    base_artifact.setdefault("source_files", {})
    base_artifact["source_files"].update(
        {
            "h8_export_script": Path(__file__).name,
            "target_predictions": args.target_predictions,
            "selector_profile": args.selector_profile,
            "base_artifact": args.base_artifact,
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base_artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(Path(args.audit_output), audit_rows)
    print(
        json.dumps(
            {
                "output": str(out),
                "audit_output": args.audit_output,
                "enabled_clients": sorted(enabled_clients),
                "source_feature_count": len(source_feature_names),
                "augmented_feature_count": len(aug_feature_names),
                "target_aug_models": len(target_aug_models),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
