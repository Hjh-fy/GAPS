"""Formal target-calibrated shallow MLP direct-head candidate evaluation.

H2 in the regression-head matrix:

- fixed classifier/backbone/route;
- target-client calibration only;
- one shallow MLP per target client and gas;
- calibration-validation model/selector;
- target test for final reporting only.
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

from run_formal_target_ridge_auto_v2_eval import (
    apply_c4_rescue,
    attach_response_phase,
    selected_c4_gate,
    test_gate_audit,
    write_report,
)
from run_regression_head_ablation import (
    CLASS_NAMES,
    CO_CLASS,
    add_target_features,
    client_name,
    deterministic_train_val,
    fnum,
    inum,
    matrix_from_rows,
    metrics,
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


def fit_mlp(rows: Sequence[dict[str, Any]], feature_names: Sequence[str], hidden: tuple[int, ...], alpha: float, seed: int) -> MLPHead:
    x = matrix_from_rows(rows, feature_names)
    y = np.asarray([fnum(row["true_ppm"]) for row in rows], dtype=np.float64)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="lbfgs",
        alpha=float(alpha),
        max_iter=800,
        early_stopping=False,
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
    return model, {"best_hidden": best_hidden, "best_alpha": best_alpha, "best_val_RMSE": best_score, "audit": audit_rows}


def apply_client_mlp(rows: list[dict[str, Any]], models: dict[tuple[str, int], MLPHead], prefix: str) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for (client, cls_id), model in models.items():
        idxs = [
            idx for idx, row in enumerate(out)
            if str(row.get("client")) == client and inum(row.get("route_class")) == cls_id
        ]
        if not idxs:
            continue
        pred_rows = [out[idx] for idx in idxs]
        pred = model.predict(pred_rows)
        for idx, value in zip(idxs, pred):
            out[idx][f"{prefix}_ppm"] = float(value)
            out[idx][f"{prefix}_delta_vs_final"] = float(value - fnum(out[idx].get("final_ppm"), value))
    return out


def client_scope_rmse(rows: Sequence[dict[str, Any]], pred_key: str, client: str, co_only: bool = False, nonco_only: bool = False) -> float:
    selected = [row for row in rows if row["client"] == client]
    if co_only:
        selected = [row for row in selected if inum(row["true_class"]) == CO_CLASS]
    if nonco_only:
        selected = [row for row in selected if inum(row["true_class"]) != CO_CLASS]
    return fnum(metrics(selected, pred_key).get("RMSE"), float("inf"))


def build_selection_table(
    val_rows: list[dict[str, Any]],
    target_clients: Sequence[str],
    max_nonco_delta: float,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str], dict[str, str]]:
    table: list[dict[str, Any]] = []
    conservative: dict[str, str] = {}
    coaware: dict[str, str] = {}
    hybrid: dict[str, str] = {}
    for client in target_clients:
        base_all = client_scope_rmse(val_rows, "baseline_final_ppm", client)
        mlp_all = client_scope_rmse(val_rows, "mlp_direct_val_ppm", client)
        base_co = client_scope_rmse(val_rows, "baseline_final_ppm", client, co_only=True)
        mlp_co = client_scope_rmse(val_rows, "mlp_direct_val_ppm", client, co_only=True)
        base_nonco = client_scope_rmse(val_rows, "baseline_final_ppm", client, nonco_only=True)
        mlp_nonco = client_scope_rmse(val_rows, "mlp_direct_val_ppm", client, nonco_only=True)
        d_all = mlp_all - base_all
        d_co = mlp_co - base_co
        d_nonco = mlp_nonco - base_nonco
        cons_pass = d_all < 0 and d_nonco <= max_nonco_delta
        coaware_score = d_co + 0.25 * max(0.0, d_nonco)
        conservative[client] = "mlp_direct" if cons_pass else "baseline_final"
        coaware[client] = "mlp_direct" if coaware_score < 0 else "baseline_final"
        hybrid[client] = "mlp_direct" if cons_pass or coaware_score < 0 else "baseline_final"
        table.append(
            {
                "client": client,
                "selected_mode": conservative[client],
                "coaware_selected_mode": coaware[client],
                "hybrid_selected_mode": hybrid[client],
                "passes_constraints": int(cons_pass),
                "baseline_ALL_RMSE": base_all,
                "mlp_ALL_RMSE": mlp_all,
                "delta_ALL_RMSE": d_all,
                "baseline_CO_RMSE": base_co,
                "mlp_CO_RMSE": mlp_co,
                "delta_CO_RMSE": d_co,
                "baseline_nonCO_RMSE": base_nonco,
                "mlp_nonCO_RMSE": mlp_nonco,
                "delta_nonCO_RMSE": d_nonco,
                "coaware_score": coaware_score,
                "max_nonco_delta": max_nonco_delta,
            }
        )
    return table, conservative, coaware, hybrid


def apply_selection(rows: list[dict[str, Any]], selected: dict[str, str], output_key: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        mode = selected.get(str(item["client"]), "baseline_final")
        item["selected_mlp_mode"] = mode
        item[output_key] = fnum(item["mlp_direct_ppm"]) if mode == "mlp_direct" else fnum(item["baseline_final_ppm"])
        out.append(item)
    return out


def report_table(summary_rows: list[dict[str, Any]], modes: Sequence[str], scopes: Sequence[str]) -> str:
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for mode in modes:
        vals = []
        for scope in scopes:
            value = None
            for row in summary_rows:
                if row["mode"] == mode and row["split"] == "test" and row["scope"] == scope:
                    value = row.get("RMSE")
                    break
            vals.append("" if value in (None, "") else f"{fnum(value):.2f}")
        lines.append("| " + mode + " | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_mlp_report(out: Path, selection_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> None:
    modes = [
        "A0_baseline_final",
        "H2_forced_target_mlp_direct",
        "H2_val_select_target_mlp_direct",
        "H2_coaware_select_target_mlp_direct",
        "H2_hybrid_select_target_mlp_direct",
        "H2_forced_target_mlp_plus_c4_rescue",
        "H2_val_select_target_mlp_plus_c4_rescue",
        "H2_coaware_select_target_mlp_plus_c4_rescue",
        "H2_hybrid_select_target_mlp_plus_c4_rescue",
    ]
    scopes = ["ALL", "C3-CO", "C4-CO", "C5-CO", "C3-CO_high_200_250", "C4-CO_high_200_250", "C5-CO_high_200_250", "nonCO_ALL"]
    select_lines = [
        "| client | conservative | co-aware | hybrid | dALL | dCO | dnonCO | co-aware score |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in selection_rows:
        select_lines.append(
            "| {client} | {selected_mode} | {coaware_selected_mode} | {hybrid_selected_mode} | "
            "{delta_ALL_RMSE:+.2f} | {delta_CO_RMSE:+.2f} | {delta_nonCO_RMSE:+.2f} | {coaware_score:+.2f} |".format(**row)
        )
    audit_lines = ["| candidate | gated | true CO high | nonCO |", "|---|---:|---:|---:|"]
    for row in audit_rows:
        audit_lines.append(f"| {row['candidate']} | {row['gated_N']} | {row['gated_true_CO_high_N']} | {row['gated_nonCO_N']} |")
    text = f"""# Formal Target MLP auto_v2 Evaluation

H2 shallow per-gas MLP over the same rich statistics used by H1 Ridge.

Selection uses target calibration only. Test split is used only for final reporting.

## Calibration Selection

{chr(10).join(select_lines)}

## Target Test RMSE

{report_table(summary_rows, modes, scopes)}

## C4 Rescue Test Audit

{chr(10).join(audit_lines)}
"""
    (out / "formal_target_mlp_auto_v2_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal target-calibrated shallow MLP candidate.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--route-rescue-artifact", default="results/deployment_candidates_20260624/c12_c345_a1_rich_residual_plus_c4_rescue.json")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--hidden-grid", default="32")
    parser.add_argument("--alphas", default="0.001,0.01")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--max-nonco-delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/formal_target_mlp_auto_v2_20260624")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    hiddens = [tuple(int(v.strip()) for v in item.split(",") if v.strip()) for item in args.hidden_grid.split(";") if item.strip()]
    gate = selected_c4_gate(args.route_rescue_artifact)

    raw_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    rows = attach_response_phase(add_target_features(raw_rows, Path(args.data_root)), Path(args.data_root))
    for row in rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    test_rows = [row for row in rows if row["split"] == "test"]
    feature_names = sorted(rows[0]["feature_dict"].keys())

    pre_models: dict[tuple[str, int], MLPHead] = {}
    fit_audit: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [row for row in calibration_rows if row["client"] == client and inum(row["true_class"]) == cls_id]
            train_rows, heldout = deterministic_train_val(cls_rows, val_ratio=args.val_ratio)
            model, audit = fit_select_refit_mlp(train_rows, heldout, feature_names, hiddens, alphas, args.seed + cls_id + 100 * int(client[1:]))
            pre_models[(client, cls_id)] = model
            for row in heldout:
                val_item = dict(row)
                val_item["route_class"] = val_item["pred_class"]
                val_rows.append(val_item)
            fit_audit.append(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(train_rows),
                    "val_N": len(heldout),
                    "best_hidden": str(audit["best_hidden"]),
                    "best_alpha": audit["best_alpha"],
                    "best_val_RMSE": audit["best_val_RMSE"],
                    "grid_audit": json.dumps(audit["audit"], ensure_ascii=False),
                }
            )
    val_pred = apply_client_mlp(val_rows, pre_models, "mlp_direct_val")
    for row in val_pred:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["mlp_direct_val_ppm"] = fnum(row.get("mlp_direct_val_ppm"))
    selection_rows, conservative_modes, coaware_modes, hybrid_modes = build_selection_table(val_pred, target_clients, args.max_nonco_delta)

    # Refit final models on full calibration with the chosen per-head hyperparams.
    full_models: dict[tuple[str, int], MLPHead] = {}
    fit_lookup = {(row["client"], int(row["class_id"])): row for row in fit_audit}
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [row for row in calibration_rows if row["client"] == client and inum(row["true_class"]) == cls_id]
            fit_row = fit_lookup[(client, cls_id)]
            hidden = tuple(int(v.strip()) for v in str(fit_row["best_hidden"]).strip("()").split(",") if v.strip())
            full_models[(client, cls_id)] = fit_mlp(cls_rows, feature_names, hidden, float(fit_row["best_alpha"]), args.seed + cls_id + 100 * int(client[1:]))

    forced_rows = apply_client_mlp(test_rows, full_models, "mlp_direct")
    for row in forced_rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["mlp_direct_ppm"] = fnum(row.get("mlp_direct_ppm"))
    val_select_rows = apply_selection(forced_rows, conservative_modes, "val_select_mlp_ppm")
    coaware_rows = apply_selection(forced_rows, coaware_modes, "coaware_select_mlp_ppm")
    hybrid_rows = apply_selection(forced_rows, hybrid_modes, "hybrid_select_mlp_ppm")

    forced_rescue = apply_c4_rescue(forced_rows, "mlp_direct_ppm", "mlp_direct_plus_c4_rescue_ppm", gate)
    val_rescue = apply_c4_rescue(val_select_rows, "val_select_mlp_ppm", "val_select_mlp_plus_c4_rescue_ppm", gate)
    coaware_rescue = apply_c4_rescue(coaware_rows, "coaware_select_mlp_ppm", "coaware_select_mlp_plus_c4_rescue_ppm", gate)
    hybrid_rescue = apply_c4_rescue(hybrid_rows, "hybrid_select_mlp_ppm", "hybrid_select_mlp_plus_c4_rescue_ppm", gate)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(summarize(forced_rows, "baseline_final_ppm", "A0_baseline_final", "test"))
    summary_rows.extend(summarize(forced_rows, "mlp_direct_ppm", "H2_forced_target_mlp_direct", "test"))
    summary_rows.extend(summarize(val_select_rows, "val_select_mlp_ppm", "H2_val_select_target_mlp_direct", "test"))
    summary_rows.extend(summarize(coaware_rows, "coaware_select_mlp_ppm", "H2_coaware_select_target_mlp_direct", "test"))
    summary_rows.extend(summarize(hybrid_rows, "hybrid_select_mlp_ppm", "H2_hybrid_select_target_mlp_direct", "test"))
    summary_rows.extend(summarize(forced_rescue, "mlp_direct_plus_c4_rescue_ppm", "H2_forced_target_mlp_plus_c4_rescue", "test"))
    summary_rows.extend(summarize(val_rescue, "val_select_mlp_plus_c4_rescue_ppm", "H2_val_select_target_mlp_plus_c4_rescue", "test"))
    summary_rows.extend(summarize(coaware_rescue, "coaware_select_mlp_plus_c4_rescue_ppm", "H2_coaware_select_target_mlp_plus_c4_rescue", "test"))
    summary_rows.extend(summarize(hybrid_rescue, "hybrid_select_mlp_plus_c4_rescue_ppm", "H2_hybrid_select_target_mlp_plus_c4_rescue", "test"))

    output_rows = []
    row_maps = {
        "forced": {(row["client"], row["sample_index"]): row for row in forced_rows},
        "val": {(row["client"], row["sample_index"]): row for row in val_select_rows},
        "coaware": {(row["client"], row["sample_index"]): row for row in coaware_rows},
        "hybrid": {(row["client"], row["sample_index"]): row for row in hybrid_rows},
        "forced_rescue": {(row["client"], row["sample_index"]): row for row in forced_rescue},
        "val_rescue": {(row["client"], row["sample_index"]): row for row in val_rescue},
        "coaware_rescue": {(row["client"], row["sample_index"]): row for row in coaware_rescue},
        "hybrid_rescue": {(row["client"], row["sample_index"]): row for row in hybrid_rescue},
    }
    for key, row in row_maps["forced"].items():
        item = {k: v for k, v in row.items() if k != "feature_dict"}
        item["val_select_mlp_ppm"] = row_maps["val"][key]["val_select_mlp_ppm"]
        item["coaware_select_mlp_ppm"] = row_maps["coaware"][key]["coaware_select_mlp_ppm"]
        item["hybrid_select_mlp_ppm"] = row_maps["hybrid"][key]["hybrid_select_mlp_ppm"]
        item["mlp_direct_plus_c4_rescue_ppm"] = row_maps["forced_rescue"][key]["mlp_direct_plus_c4_rescue_ppm"]
        item["val_select_mlp_plus_c4_rescue_ppm"] = row_maps["val_rescue"][key]["val_select_mlp_plus_c4_rescue_ppm"]
        item["coaware_select_mlp_plus_c4_rescue_ppm"] = row_maps["coaware_rescue"][key]["coaware_select_mlp_plus_c4_rescue_ppm"]
        item["hybrid_select_mlp_plus_c4_rescue_ppm"] = row_maps["hybrid_rescue"][key]["hybrid_select_mlp_plus_c4_rescue_ppm"]
        output_rows.append(item)

    audit_rows = []
    audit_rows.extend(test_gate_audit(forced_rescue, "H2_forced_target_mlp_plus_c4_rescue"))
    audit_rows.extend(test_gate_audit(val_rescue, "H2_val_select_target_mlp_plus_c4_rescue"))
    audit_rows.extend(test_gate_audit(coaware_rescue, "H2_coaware_select_target_mlp_plus_c4_rescue"))
    audit_rows.extend(test_gate_audit(hybrid_rescue, "H2_hybrid_select_target_mlp_plus_c4_rescue"))

    write_csv(out / "formal_target_mlp_predictions.csv", output_rows)
    write_csv(out / "formal_target_mlp_summary.csv", summary_rows)
    write_csv(out / "formal_target_mlp_selection_table.csv", selection_rows)
    write_csv(out / "formal_target_mlp_fit_audit.csv", fit_audit)
    write_csv(out / "formal_target_mlp_c4_rescue_audit.csv", audit_rows)
    write_mlp_report(out, selection_rows, summary_rows, audit_rows)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "route_rescue_artifact": args.route_rescue_artifact,
                "target_clients": target_clients,
                "hiddens": [list(h) for h in hiddens],
                "alphas": alphas,
                "val_ratio": args.val_ratio,
                "max_nonco_delta": args.max_nonco_delta,
                "seed": args.seed,
                "selected_modes": conservative_modes,
                "coaware_selected_modes": coaware_modes,
                "hybrid_selected_modes": hybrid_modes,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote formal target MLP auto_v2 evaluation to {out}")


if __name__ == "__main__":
    main()
