"""Formal target-calibrated Ridge direct-head candidate evaluation.

This script turns the H1 diagnostic into an auto_v2-style candidate:

1. Split each target client's calibration split into fit/validation.
2. Train per-client, per-gas Ridge heads on calibration-fit only.
3. Select per-client baseline vs Ridge direct using calibration-validation only.
4. Refit selected Ridge heads on full calibration.
5. Report target test metrics for baseline, forced Ridge, val-selected Ridge,
   and versions with the previously formalized C4 route-rescue gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from run_regression_head_ablation import (
    CLASS_NAMES,
    CO_CLASS,
    add_target_features,
    apply_client_models,
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


def selected_c4_gate(artifact_path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    gate = data.get("route_rescue_policy", {}).get("selected_gate")
    if not gate:
        raise ValueError(f"No route_rescue_policy.selected_gate in {artifact_path}")
    return gate


def parse_pred_classes(text: Any) -> set[int]:
    return {int(float(part)) for part in str(text).split(",") if str(part).strip()}


def c4_gate_hit(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    if str(row.get("client")) != "C4":
        return False
    if inum(row.get("pred_class")) not in parse_pred_classes(gate.get("pred_classes")):
        return False
    # Keep the selected route-rescue semantics: gate on base auto_v2 final_ppm.
    if fnum(row.get("final_ppm")) >= fnum(gate.get("max_ppm")):
        return False
    if fnum(row.get("risk_score"), 0.0) < fnum(gate.get("risk_threshold"), 0.0):
        return False
    phase = str(gate.get("phase", "any"))
    if phase != "any" and str(row.get("response_phase")) != phase:
        return False
    return True


def attach_response_phase(rows: list[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    meta_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        client = str(item["client"])
        split = str(item["split"])
        key = (client, split)
        if key not in meta_cache:
            meta_path = data_root / f"client_{int(client[1:])}" / f"{split}_experiment_info.json"
            meta_cache[key] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
        idx = inum(item.get("sample_index"))
        meta = meta_cache[key][idx] if 0 <= idx < len(meta_cache[key]) else {}
        item["response_phase"] = str(meta.get("response_phase", "unknown"))
        item["phase_label"] = str(meta.get("phase_label", "unknown"))
        item["filename"] = str(meta.get("filename", ""))
        item["repeat_id"] = meta.get("repeat_id", "")
        out.append(item)
    return out


def apply_c4_rescue(rows: list[dict[str, Any]], source_key: str, output_key: str, gate: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rescue_ppm = fnum(gate.get("rescue_ppm"))
    for row in rows:
        item = dict(row)
        hit = c4_gate_hit(item, gate)
        item["c4_rescue_applied"] = int(hit)
        item[output_key] = rescue_ppm if hit else fnum(item.get(source_key))
        out.append(item)
    return out


def client_scope_rmse(rows: Sequence[dict[str, Any]], pred_key: str, client: str, co_only: bool = False, nonco_only: bool = False) -> float:
    selected = [row for row in rows if row["client"] == client]
    if co_only:
        selected = [row for row in selected if inum(row["true_class"]) == CO_CLASS]
    if nonco_only:
        selected = [row for row in selected if inum(row["true_class"]) != CO_CLASS]
    value = metrics(selected, pred_key).get("RMSE")
    return fnum(value, float("inf"))


def fit_client_models(
    calibration_rows: list[dict[str, Any]],
    target_clients: Sequence[str],
    feature_names: Sequence[str],
    alphas: Sequence[float],
    val_ratio: float,
) -> tuple[dict[tuple[str, int], Any], list[dict[str, Any]], list[dict[str, Any]]]:
    models: dict[tuple[str, int], Any] = {}
    fit_audit: list[dict[str, Any]] = []
    val_rows_all: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row for row in calibration_rows
                if row["client"] == client and inum(row["true_class"]) == cls_id
            ]
            train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            model, audit = fit_select_refit(train_rows, val_rows, feature_names, alphas)
            models[(client, cls_id)] = model
            for row in val_rows:
                val_item = dict(row)
                val_item["route_class"] = val_item["pred_class"]
                val_rows_all.append(val_item)
            fit_audit.append(
                {
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_alpha": audit["best_alpha"],
                    "best_val_RMSE": audit["best_val_RMSE"],
                    "alpha_audit": json.dumps(audit["alpha_audit"], ensure_ascii=False),
                }
            )
    return models, fit_audit, val_rows_all


def refit_full_calibration(
    calibration_rows: list[dict[str, Any]],
    target_clients: Sequence[str],
    feature_names: Sequence[str],
    selected_alphas: dict[tuple[str, int], float],
) -> dict[tuple[str, int], Any]:
    from run_regression_head_ablation import fit_ridge

    models: dict[tuple[str, int], Any] = {}
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row for row in calibration_rows
                if row["client"] == client and inum(row["true_class"]) == cls_id
            ]
            models[(client, cls_id)] = fit_ridge(cls_rows, feature_names, selected_alphas[(client, cls_id)])
    return models


def build_selection_table(
    val_rows: list[dict[str, Any]],
    target_clients: Sequence[str],
    max_nonco_delta: float,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str], dict[str, str]]:
    table: list[dict[str, Any]] = []
    selected: dict[str, str] = {}
    coaware_selected: dict[str, str] = {}
    hybrid_selected: dict[str, str] = {}
    for client in target_clients:
        base_all = client_scope_rmse(val_rows, "baseline_final_ppm", client)
        ridge_all = client_scope_rmse(val_rows, "ridge_direct_val_ppm", client)
        base_nonco = client_scope_rmse(val_rows, "baseline_final_ppm", client, nonco_only=True)
        ridge_nonco = client_scope_rmse(val_rows, "ridge_direct_val_ppm", client, nonco_only=True)
        base_co = client_scope_rmse(val_rows, "baseline_final_ppm", client, co_only=True)
        ridge_co = client_scope_rmse(val_rows, "ridge_direct_val_ppm", client, co_only=True)
        passes = ridge_all < base_all and ridge_nonco <= base_nonco + max_nonco_delta
        mode = "ridge_direct" if passes else "baseline_final"
        selected[client] = mode
        co_delta = ridge_co - base_co
        nonco_delta = ridge_nonco - base_nonco
        coaware_score = co_delta + 0.25 * max(0.0, nonco_delta)
        coaware_mode = "ridge_direct" if coaware_score < 0.0 else "baseline_final"
        coaware_selected[client] = coaware_mode
        hybrid_mode = "ridge_direct" if passes or coaware_score < 0.0 else "baseline_final"
        hybrid_selected[client] = hybrid_mode
        table.append(
            {
                "client": client,
                "selected_mode": mode,
                "coaware_selected_mode": coaware_mode,
                "hybrid_selected_mode": hybrid_mode,
                "passes_constraints": int(passes),
                "baseline_ALL_RMSE": base_all,
                "ridge_ALL_RMSE": ridge_all,
                "delta_ALL_RMSE": ridge_all - base_all,
                "baseline_CO_RMSE": base_co,
                "ridge_CO_RMSE": ridge_co,
                "delta_CO_RMSE": ridge_co - base_co,
                "baseline_nonCO_RMSE": base_nonco,
                "ridge_nonCO_RMSE": ridge_nonco,
                "delta_nonCO_RMSE": nonco_delta,
                "coaware_score": coaware_score,
                "max_nonco_delta": max_nonco_delta,
            }
        )
    return table, selected, coaware_selected, hybrid_selected


def apply_val_selection(rows: list[dict[str, Any]], selected: dict[str, str], output_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        mode = selected.get(str(item["client"]), "baseline_final")
        item["selected_ridge_mode"] = mode
        item[output_key] = fnum(item["ridge_direct_ppm"]) if mode == "ridge_direct" else fnum(item["baseline_final_ppm"])
        out.append(item)
    return out


def test_gate_audit(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    gated = [row for row in rows if inum(row.get("c4_rescue_applied")) == 1]
    return [
        {
            "candidate": key,
            "split": "test",
            "gated_N": len(gated),
            "gated_true_CO_high_N": sum(inum(row["true_class"]) == CO_CLASS and fnum(row["true_ppm"]) >= 200.0 for row in gated),
            "gated_nonCO_N": sum(inum(row["true_class"]) != CO_CLASS for row in gated),
        }
    ]


def report_table(summary_rows: list[dict[str, Any]], modes: Sequence[str], scopes: Sequence[str]) -> str:
    lines = ["| mode | " + " | ".join(scopes) + " |", "|---|" + "|".join(["---:"] * len(scopes)) + "|"]
    for mode in modes:
        values = []
        for scope in scopes:
            value = None
            for row in summary_rows:
                if row["mode"] == mode and row["split"] == "test" and row["scope"] == scope:
                    value = row.get("RMSE")
                    break
            values.append("" if value in (None, "") else f"{fnum(value):.2f}")
        lines.append("| " + mode + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out: Path,
    selection_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> None:
    modes = [
        "A0_baseline_final",
        "H1_forced_target_ridge_direct",
        "H1_val_select_target_ridge_direct",
        "H1_coaware_select_target_ridge_direct",
        "H1_hybrid_select_target_ridge_direct",
        "H1_forced_target_ridge_plus_c4_rescue",
        "H1_val_select_target_ridge_plus_c4_rescue",
        "H1_coaware_select_target_ridge_plus_c4_rescue",
        "H1_hybrid_select_target_ridge_plus_c4_rescue",
    ]
    scopes = ["ALL", "C3-CO", "C4-CO", "C5-CO", "C3-CO_high_200_250", "C4-CO_high_200_250", "C5-CO_high_200_250", "nonCO_ALL"]
    select_lines = [
        "| client | conservative | co-aware | hybrid | base ALL | ridge ALL | dALL | base CO | ridge CO | dCO | base nonCO | ridge nonCO | dnonCO | co-aware score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selection_rows:
        select_lines.append(
            "| {client} | {selected_mode} | {coaware_selected_mode} | {hybrid_selected_mode} | {baseline_ALL_RMSE:.2f} | {ridge_ALL_RMSE:.2f} | {delta_ALL_RMSE:+.2f} | "
            "{baseline_CO_RMSE:.2f} | {ridge_CO_RMSE:.2f} | {delta_CO_RMSE:+.2f} | "
            "{baseline_nonCO_RMSE:.2f} | {ridge_nonCO_RMSE:.2f} | {delta_nonCO_RMSE:+.2f} | {coaware_score:+.2f} |".format(**row)
        )
    audit_lines = ["| candidate | gated | true CO high | nonCO |", "|---|---:|---:|---:|"]
    for row in audit_rows:
        audit_lines.append(
            f"| {row['candidate']} | {row['gated_N']} | {row['gated_true_CO_high_N']} | {row['gated_nonCO_N']} |"
        )
    text = f"""# Formal Target Ridge auto_v2 Evaluation

Selection uses target calibration only. Test split is used only for final reporting.

## Calibration Selection

{chr(10).join(select_lines)}

## Target Test RMSE

{report_table(summary_rows, modes, scopes)}

## C4 Rescue Test Audit

{chr(10).join(audit_lines)}

## Reading

- This is the formal version of the H1 target-client calibrated Ridge direct-head diagnostic.
- Per-client baseline vs Ridge selection is based on calibration-validation only.
- The C4 route-rescue gate is the previously formalized calibration-selected gate, applied with the same base `final_ppm` gate semantics.
"""
    (out / "formal_target_ridge_auto_v2_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal target-calibrated Ridge direct auto_v2 candidate.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--route-rescue-artifact", default="results/deployment_candidates_20260624/c12_c345_a1_rich_residual_plus_c4_rescue.json")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--max-nonco-delta", type=float, default=1.0)
    parser.add_argument("--output-dir", default="results/formal_target_ridge_auto_v2_20260624")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    gate = selected_c4_gate(args.route_rescue_artifact)

    raw_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    rows = attach_response_phase(add_target_features(raw_rows, data_root), data_root)
    for row in rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    test_rows = [row for row in rows if row["split"] == "test"]
    feature_names = sorted(rows[0]["feature_dict"].keys())

    pre_models, fit_audit, val_rows = fit_client_models(calibration_rows, target_clients, feature_names, alphas, args.val_ratio)
    val_pred = apply_client_models(val_rows, pre_models, "ridge_direct_val")
    for row in val_pred:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["ridge_direct_val_ppm"] = fnum(row.get("ridge_direct_val_ppm"))
    selection_rows, selected_modes, coaware_modes, hybrid_modes = build_selection_table(val_pred, target_clients, args.max_nonco_delta)
    selected_alphas = {(row["client"], int(row["class_id"])): fnum(row["best_alpha"]) for row in fit_audit}
    full_models = refit_full_calibration(calibration_rows, target_clients, feature_names, selected_alphas)

    forced_rows = apply_client_models(test_rows, full_models, "ridge_direct")
    for row in forced_rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["ridge_direct_ppm"] = fnum(row.get("ridge_direct_ppm"))
    selected_rows = apply_val_selection(forced_rows, selected_modes, "val_select_ridge_ppm")
    coaware_rows = apply_val_selection(forced_rows, coaware_modes, "coaware_select_ridge_ppm")
    hybrid_rows = apply_val_selection(forced_rows, hybrid_modes, "hybrid_select_ridge_ppm")

    forced_rescue = apply_c4_rescue(forced_rows, "ridge_direct_ppm", "ridge_direct_plus_c4_rescue_ppm", gate)
    selected_rescue = apply_c4_rescue(selected_rows, "val_select_ridge_ppm", "val_select_ridge_plus_c4_rescue_ppm", gate)
    coaware_rescue = apply_c4_rescue(coaware_rows, "coaware_select_ridge_ppm", "coaware_select_ridge_plus_c4_rescue_ppm", gate)
    hybrid_rescue = apply_c4_rescue(hybrid_rows, "hybrid_select_ridge_ppm", "hybrid_select_ridge_plus_c4_rescue_ppm", gate)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(summarize(forced_rows, "baseline_final_ppm", "A0_baseline_final", "test"))
    summary_rows.extend(summarize(forced_rows, "ridge_direct_ppm", "H1_forced_target_ridge_direct", "test"))
    summary_rows.extend(summarize(selected_rows, "val_select_ridge_ppm", "H1_val_select_target_ridge_direct", "test"))
    summary_rows.extend(summarize(coaware_rows, "coaware_select_ridge_ppm", "H1_coaware_select_target_ridge_direct", "test"))
    summary_rows.extend(summarize(hybrid_rows, "hybrid_select_ridge_ppm", "H1_hybrid_select_target_ridge_direct", "test"))
    summary_rows.extend(summarize(forced_rescue, "ridge_direct_plus_c4_rescue_ppm", "H1_forced_target_ridge_plus_c4_rescue", "test"))
    summary_rows.extend(summarize(selected_rescue, "val_select_ridge_plus_c4_rescue_ppm", "H1_val_select_target_ridge_plus_c4_rescue", "test"))
    summary_rows.extend(summarize(coaware_rescue, "coaware_select_ridge_plus_c4_rescue_ppm", "H1_coaware_select_target_ridge_plus_c4_rescue", "test"))
    summary_rows.extend(summarize(hybrid_rescue, "hybrid_select_ridge_plus_c4_rescue_ppm", "H1_hybrid_select_target_ridge_plus_c4_rescue", "test"))

    output_rows: list[dict[str, Any]] = []
    by_key_forced = {(row["client"], row["sample_index"]): row for row in forced_rows}
    by_key_selected = {(row["client"], row["sample_index"]): row for row in selected_rows}
    by_key_coaware = {(row["client"], row["sample_index"]): row for row in coaware_rows}
    by_key_hybrid = {(row["client"], row["sample_index"]): row for row in hybrid_rows}
    by_key_forced_rescue = {(row["client"], row["sample_index"]): row for row in forced_rescue}
    by_key_selected_rescue = {(row["client"], row["sample_index"]): row for row in selected_rescue}
    by_key_coaware_rescue = {(row["client"], row["sample_index"]): row for row in coaware_rescue}
    by_key_hybrid_rescue = {(row["client"], row["sample_index"]): row for row in hybrid_rescue}
    for key, row in by_key_forced.items():
        item = {k: v for k, v in row.items() if k != "feature_dict"}
        item["val_select_ridge_ppm"] = by_key_selected[key]["val_select_ridge_ppm"]
        item["selected_ridge_mode"] = by_key_selected[key]["selected_ridge_mode"]
        item["coaware_select_ridge_ppm"] = by_key_coaware[key]["coaware_select_ridge_ppm"]
        item["coaware_ridge_mode"] = by_key_coaware[key]["selected_ridge_mode"]
        item["hybrid_select_ridge_ppm"] = by_key_hybrid[key]["hybrid_select_ridge_ppm"]
        item["hybrid_ridge_mode"] = by_key_hybrid[key]["selected_ridge_mode"]
        item["ridge_direct_plus_c4_rescue_ppm"] = by_key_forced_rescue[key]["ridge_direct_plus_c4_rescue_ppm"]
        item["forced_c4_rescue_applied"] = by_key_forced_rescue[key]["c4_rescue_applied"]
        item["val_select_ridge_plus_c4_rescue_ppm"] = by_key_selected_rescue[key]["val_select_ridge_plus_c4_rescue_ppm"]
        item["selected_c4_rescue_applied"] = by_key_selected_rescue[key]["c4_rescue_applied"]
        item["coaware_select_ridge_plus_c4_rescue_ppm"] = by_key_coaware_rescue[key]["coaware_select_ridge_plus_c4_rescue_ppm"]
        item["coaware_c4_rescue_applied"] = by_key_coaware_rescue[key]["c4_rescue_applied"]
        item["hybrid_select_ridge_plus_c4_rescue_ppm"] = by_key_hybrid_rescue[key]["hybrid_select_ridge_plus_c4_rescue_ppm"]
        item["hybrid_c4_rescue_applied"] = by_key_hybrid_rescue[key]["c4_rescue_applied"]
        output_rows.append(item)

    audit_rows = []
    audit_rows.extend(test_gate_audit(forced_rescue, "H1_forced_target_ridge_plus_c4_rescue"))
    audit_rows.extend(test_gate_audit(selected_rescue, "H1_val_select_target_ridge_plus_c4_rescue"))
    audit_rows.extend(test_gate_audit(coaware_rescue, "H1_coaware_select_target_ridge_plus_c4_rescue"))
    audit_rows.extend(test_gate_audit(hybrid_rescue, "H1_hybrid_select_target_ridge_plus_c4_rescue"))

    write_csv(out / "formal_target_ridge_predictions.csv", output_rows)
    write_csv(out / "formal_target_ridge_summary.csv", summary_rows)
    write_csv(out / "formal_target_ridge_selection_table.csv", selection_rows)
    write_csv(out / "formal_target_ridge_fit_audit.csv", fit_audit)
    write_csv(out / "formal_target_ridge_c4_rescue_audit.csv", audit_rows)
    write_report(out, selection_rows, summary_rows, audit_rows)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "route_rescue_artifact": args.route_rescue_artifact,
                "target_clients": target_clients,
                "alphas": alphas,
                "val_ratio": args.val_ratio,
                "max_nonco_delta": args.max_nonco_delta,
                "selected_modes": selected_modes,
                "coaware_selected_modes": coaware_modes,
                "hybrid_selected_modes": hybrid_modes,
                "coaware_score": "delta_CO_RMSE + 0.25 * max(0, delta_nonCO_RMSE)",
                "hybrid_rule": "conservative_pass OR coaware_score < 0",
                "selected_c4_gate": gate,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote formal target Ridge auto_v2 evaluation to {out}")


if __name__ == "__main__":
    main()
