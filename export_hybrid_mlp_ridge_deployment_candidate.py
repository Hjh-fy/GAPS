"""Export hybrid target MLP/Ridge candidates as deployment artifacts.

The exported artifact stays compatible with the existing deployment bundle:
it is copied as ``rich_residual_candidate.json`` and loaded by
``RichResidualPolicy``. The runtime then applies, in order:

1. formal C4 route-rescue;
2. target MLP direct head where selected;
3. target Ridge direct head where selected;
4. older residual policy if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_formal_target_mlp_auto_v2_eval import fit_mlp
from run_formal_target_ridge_auto_v2_eval import selected_c4_gate
from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    client_name,
    fit_ridge,
    fnum,
    inum,
    read_csv,
)


DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "h2_2": {
        "candidate_name": "c12_c345_h2_2_mlp_c3_ridge_c4c5_plus_c4_rescue",
        "mlp_modes": {"C3": "mlp_direct", "C4": "baseline_final", "C5": "baseline_final"},
        "ridge_modes": {"C3": "baseline_final", "C4": "ridge_direct", "C5": "ridge_direct"},
        "c5_grid_mlp_clients": [],
    },
    "h2_3": {
        "candidate_name": "c12_c345_h2_3_mlp_c3_ridge_c4_c5grid_plus_c4_rescue",
        "mlp_modes": {"C3": "mlp_direct", "C4": "baseline_final", "C5": "mlp_direct"},
        "ridge_modes": {"C3": "baseline_final", "C4": "ridge_direct", "C5": "baseline_final"},
        "c5_grid_mlp_clients": ["C5"],
    },
}


def parse_hidden(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in str(text).strip("()").split(",") if v.strip())


def serialize_mlp_head(model: Any) -> dict[str, Any]:
    scaler = model.model.named_steps["standardscaler"]
    mlp = model.model.named_steps["mlpregressor"]
    return {
        "hidden": list(model.hidden),
        "alpha": float(model.alpha),
        "feature_names": list(model.feature_names),
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "coefs": [coef.astype(float).tolist() for coef in mlp.coefs_],
        "intercepts": [intercept.astype(float).tolist() for intercept in mlp.intercepts_],
        "activation": str(mlp.activation),
        "out_activation": str(mlp.out_activation_),
        "clip_min": float(model.clip_min),
        "clip_max": float(model.clip_max),
    }


def load_fit_audit(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    return {
        (str(row["client"]), int(float(row["class_id"]))): row
        for row in read_csv(path)
    }


def fit_rows(
    target_predictions: str,
    data_root: Path,
    target_clients: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_rows = [
        row for row in read_csv(target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
        and str(row.get("split")) == "calibration"
    ]
    rows = add_target_features(raw_rows, data_root)
    for row in rows:
        row["route_class"] = row["true_class"]
    return rows, sorted(rows[0]["feature_dict"].keys())


def load_profile(candidate: str, profile_json: str = "") -> dict[str, Any]:
    if profile_json:
        profile = json.loads(Path(profile_json).read_text(encoding="utf-8"))
    else:
        if candidate not in DEFAULT_PROFILES:
            raise ValueError(f"Unknown candidate: {candidate}")
        profile = dict(DEFAULT_PROFILES[candidate])
    for key in ["candidate_name", "mlp_modes", "ridge_modes"]:
        if key not in profile:
            raise ValueError(f"Profile missing required key: {key}")
    profile.setdefault("c5_grid_mlp_clients", [])
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Export H2 hybrid MLP/Ridge deployment candidate.")
    parser.add_argument("--candidate", choices=sorted(DEFAULT_PROFILES), default="h2_3")
    parser.add_argument("--profile-json", default="", help="Optional deployment profile JSON overriding --candidate modes")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--ridge-formal-dir", default="results/formal_target_ridge_auto_v2_20260624")
    parser.add_argument("--mlp-formal-dir", default="results/formal_target_mlp_auto_v2_20260624")
    parser.add_argument("--c5-mlp-formal-dir", default="results/formal_target_mlp_c5_grid_20260624")
    parser.add_argument("--route-rescue-artifact", default="results/deployment_candidates_20260624/c12_c345_a1_rich_residual_plus_c4_rescue.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_clients = ["C3", "C4", "C5"]
    profile = load_profile(args.candidate, args.profile_json)
    mlp_modes = {str(k): str(v) for k, v in profile["mlp_modes"].items()}
    ridge_modes = {str(k): str(v) for k, v in profile["ridge_modes"].items()}
    candidate_name = str(profile["candidate_name"])
    c5_grid_mlp_clients = {str(v) for v in profile.get("c5_grid_mlp_clients", [])}
    out = Path(args.output or f"results/deployment_candidates_20260624/{candidate_name}.json")
    rows, feature_names = fit_rows(args.target_predictions, Path(args.data_root), target_clients)

    ridge_audit = load_fit_audit(Path(args.ridge_formal_dir) / "formal_target_ridge_fit_audit.csv")
    mlp_audit = load_fit_audit(Path(args.mlp_formal_dir) / "formal_target_mlp_fit_audit.csv")
    c5_mlp_audit = load_fit_audit(Path(args.c5_mlp_formal_dir) / "formal_target_mlp_fit_audit.csv")

    ridge_models: list[dict[str, Any]] = []
    mlp_models: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row for row in rows
                if row["client"] == client and inum(row["true_class"]) == cls_id
            ]

            ridge_alpha = fnum(ridge_audit[(client, cls_id)]["best_alpha"])
            ridge = fit_ridge(cls_rows, feature_names, ridge_alpha).to_json()
            ridge.update(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "enabled": ridge_modes.get(client) == "ridge_direct",
                    "selected_mode": ridge_modes.get(client, "baseline_final"),
                    "n_fit": len(cls_rows),
                }
            )
            ridge_models.append(ridge)

            use_c5_grid = client in c5_grid_mlp_clients
            audit_source = c5_mlp_audit if use_c5_grid else mlp_audit
            fit_row = audit_source[(client, cls_id)]
            hidden = parse_hidden(fit_row["best_hidden"])
            alpha = fnum(fit_row["best_alpha"])
            mlp = serialize_mlp_head(
                fit_mlp(
                    cls_rows,
                    feature_names,
                    hidden,
                    alpha,
                    args.seed + cls_id + 100 * int(client[1:]),
                )
            )
            mlp.update(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "enabled": mlp_modes.get(client) == "mlp_direct",
                    "selected_mode": mlp_modes.get(client, "baseline_final"),
                    "n_fit": len(cls_rows),
                    "fit_audit_source": str((Path(args.c5_mlp_formal_dir) if use_c5_grid else Path(args.mlp_formal_dir)) / "formal_target_mlp_fit_audit.csv"),
                }
            )
            mlp_models.append(mlp)

    artifact = {
        "schema": "gaps_hybrid_mlp_ridge_policy.v1",
        "candidate_name": candidate_name,
        "direction": "C12_to_C345",
        "target_mlp_policy": {
            "schema": "target_mlp_direct.v1",
            "selection_rule": candidate_name,
            "selected_modes": mlp_modes,
            "feature_names": feature_names,
            "models": mlp_models,
        },
        "target_ridge_policy": {
            "schema": "target_ridge_direct.v1",
            "selection_rule": candidate_name,
            "selected_modes": ridge_modes,
            "feature_names": feature_names,
            "models": ridge_models,
        },
        "route_rescue_policy": {
            "schema": "c4_route_rescue_policy.v1",
            "selected_gate": selected_c4_gate(args.route_rescue_artifact),
        },
        "source_files": {
            "target_predictions": args.target_predictions,
            "ridge_formal_dir": args.ridge_formal_dir,
            "mlp_formal_dir": args.mlp_formal_dir,
            "c5_mlp_formal_dir": args.c5_mlp_formal_dir,
            "route_rescue_artifact": args.route_rescue_artifact,
            "profile_json": args.profile_json,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out),
                "candidate_name": candidate_name,
                "mlp_modes": mlp_modes,
                "ridge_modes": ridge_modes,
                "mlp_models": len(mlp_models),
                "ridge_models": len(ridge_models),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
