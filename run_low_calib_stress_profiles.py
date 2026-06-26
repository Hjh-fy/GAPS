"""Low-calibration stress test for the current target profiles.

This script reruns the target-side profile fitting with stratified subsets of
the target calibration split.  It is intentionally model-capability-facing: QC
is not used, and the test split is only evaluated after calibration-only fitting
and profile decisions.

Profiles:

* B0: original R3aK16 + auto_v2 final_ppm.
* H2.3: C3 MLP, C4 Ridge, C5 expanded-grid MLP.
* H8+C4 selector: H8 source-aug CO switch on top of H2.3, plus the
  calibration-selected C4 route-rescue gate when calibration guardrails pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_formal_c4_route_rescue_selector import candidate_grid, score_gate, select_gate
from run_formal_target_mlp_auto_v2_eval import (
    MLPHead,
    apply_client_mlp,
    apply_selection as apply_mlp_selection,
    build_selection_table as build_mlp_selection_table,
    fit_select_refit_mlp,
)
from run_formal_target_ridge_auto_v2_eval import (
    apply_c4_rescue,
    apply_val_selection as apply_ridge_selection,
    attach_response_phase,
    build_selection_table as build_ridge_selection_table,
    fit_client_models as fit_ridge_client_models,
    refit_full_calibration as refit_ridge_full_calibration,
)
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
    read_csv,
    summarize,
    write_csv,
)
from run_source_augmented_target_ridge_eval import add_pred_features, attach_source_predictions, fit_source_heads
from run_source_lightweight_regression_head_ablation import parse_hidden_grid


TARGET_PREDICTIONS = "results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv"
DATA_ROOT = "dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def group_key(row: dict[str, Any]) -> tuple[str, int, float]:
    return (str(row["client"]), inum(row["true_class"]), round(fnum(row["true_ppm"]), 6))


def stratified_subset(rows: list[dict[str, Any]], fraction: float, seed: int) -> list[dict[str, Any]]:
    if fraction >= 0.999:
        return list(rows)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: inum(row.get("sample_index")))
        count = max(1, int(round(len(group) * fraction)))
        idx = np.arange(len(group))
        rng.shuffle(idx)
        selected = sorted(idx[:count].tolist())
        out.extend(group[i] for i in selected)
    return sorted(out, key=lambda row: (str(row["client"]), inum(row["true_class"]), fnum(row["true_ppm"]), inum(row["sample_index"])))


def subset_audit(rows: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for key in sorted({group_key(row) for row in rows}):
        selected = [row for row in rows if group_key(row) == key]
        audit.append(
            {
                "calib_ratio": ratio,
                "client": key[0],
                "true_class": key[1],
                "true_ppm": key[2],
                "N": len(selected),
            }
        )
    return audit


def fit_ridge_models(
    rows: list[dict[str, Any]],
    target_clients: list[str],
    feature_names: list[str],
    alphas: list[float],
    val_ratio: float,
) -> tuple[dict[tuple[str, int], Any], list[dict[str, Any]]]:
    models: dict[tuple[str, int], Any] = {}
    audit: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [row for row in rows if row["client"] == client and inum(row["true_class"]) == cls_id]
            train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            model, fit_audit = fit_select_refit(train_rows, val_rows, feature_names, alphas)
            models[(client, cls_id)] = model
            audit.append(
                {
                    "model": "ridge",
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "fit_N": len(cls_rows),
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_alpha": fit_audit["best_alpha"],
                    "best_val_RMSE": fit_audit["best_val_RMSE"],
                }
            )
    return models, audit


def fit_mlp_models(
    rows: list[dict[str, Any]],
    target_clients: list[str],
    feature_names: list[str],
    hiddens_by_client: dict[str, list[tuple[int, ...]]],
    alphas_by_client: dict[str, list[float]],
    val_ratio: float,
    seed: int,
) -> tuple[dict[tuple[str, int], MLPHead], list[dict[str, Any]]]:
    models: dict[tuple[str, int], MLPHead] = {}
    audit: list[dict[str, Any]] = []
    for client in target_clients:
        hiddens = hiddens_by_client[client]
        alphas = alphas_by_client[client]
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [row for row in rows if row["client"] == client and inum(row["true_class"]) == cls_id]
            train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            model, fit_audit = fit_select_refit_mlp(
                train_rows,
                val_rows,
                feature_names,
                hiddens,
                alphas,
                seed + cls_id + 100 * int(client[1:]),
            )
            models[(client, cls_id)] = model
            audit.append(
                {
                    "model": "mlp",
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "fit_N": len(cls_rows),
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_hidden": str(fit_audit["best_hidden"]),
                    "best_alpha": fit_audit["best_alpha"],
                    "best_val_RMSE": fit_audit["best_val_RMSE"],
                }
            )
    return models, audit


def fit_source_aug_models(
    cal_rows_aug: list[dict[str, Any]],
    target_clients: list[str],
    feature_names: list[str],
    alphas: list[float],
    val_ratio: float,
) -> tuple[dict[tuple[str, int], Any], list[dict[str, Any]]]:
    models: dict[tuple[str, int], Any] = {}
    audit: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [row for row in cal_rows_aug if row["client"] == client and inum(row["true_class"]) == cls_id]
            train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            model, fit_audit = fit_select_refit(train_rows, val_rows, feature_names, alphas)
            models[(client, cls_id)] = model
            audit.append(
                {
                    "model": "source_aug_target_ridge",
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "fit_N": len(cls_rows),
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_alpha": fit_audit["best_alpha"],
                    "best_val_RMSE": fit_audit["best_val_RMSE"],
                }
            )
    return models, audit


def with_deployment_route(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["route_class"] = item.get("pred_class")
        out.append(item)
    return out


def fit_apply_ridge_hybrid(
    cal_rows: list[dict[str, Any]],
    apply_rows: list[dict[str, Any]],
    target_clients: list[str],
    feature_names: list[str],
    alphas: list[float],
    val_ratio: float,
    max_nonco_delta: float,
    output_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pre_models, fit_audit, val_rows = fit_ridge_client_models(
        cal_rows,
        target_clients,
        feature_names,
        alphas,
        val_ratio,
    )
    val_pred = apply_client_models(val_rows, pre_models, "ridge_direct_val")
    for row in val_pred:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["ridge_direct_val_ppm"] = fnum(row.get("ridge_direct_val_ppm"))
    selection_rows, _selected, _coaware, hybrid_modes = build_ridge_selection_table(
        val_pred,
        target_clients,
        max_nonco_delta,
    )
    selected_alphas = {(row["client"], int(row["class_id"])): fnum(row["best_alpha"]) for row in fit_audit}
    full_models = refit_ridge_full_calibration(cal_rows, target_clients, feature_names, selected_alphas)
    forced_rows = apply_client_models(with_deployment_route(apply_rows), full_models, "ridge_direct")
    for row in forced_rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["ridge_direct_ppm"] = fnum(row.get("ridge_direct_ppm"))
    hybrid_rows = apply_ridge_selection(forced_rows, hybrid_modes, output_key)
    for row in hybrid_rows:
        row["ridge_lowcal_ppm"] = fnum(row[output_key])
    for row in fit_audit:
        row["model"] = "ridge"
        row["fit_N"] = int(row.get("train_N", 0)) + int(row.get("val_N", 0))
    for row in selection_rows:
        row["model"] = "ridge"
    return hybrid_rows, fit_audit, selection_rows


def fit_apply_mlp_hybrid(
    cal_rows: list[dict[str, Any]],
    apply_rows: list[dict[str, Any]],
    target_clients: list[str],
    feature_names: list[str],
    hiddens_by_client: dict[str, list[tuple[int, ...]]],
    alphas_by_client: dict[str, list[float]],
    val_ratio: float,
    max_nonco_delta: float,
    seed: int,
    output_key: str,
    output_alias: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    models: dict[tuple[str, int], MLPHead] = {}
    fit_audit: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for client in target_clients:
        hiddens = hiddens_by_client[client]
        alphas = alphas_by_client[client]
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [row for row in cal_rows if row["client"] == client and inum(row["true_class"]) == cls_id]
            train_rows, heldout = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            model, audit = fit_select_refit_mlp(
                train_rows,
                heldout,
                feature_names,
                hiddens,
                alphas,
                seed + cls_id + 100 * int(client[1:]),
            )
            models[(client, cls_id)] = model
            for row in heldout:
                val_item = dict(row)
                val_item["route_class"] = val_item["pred_class"]
                val_rows.append(val_item)
            fit_audit.append(
                {
                    "model": "mlp",
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "fit_N": len(cls_rows),
                    "train_N": len(train_rows),
                    "val_N": len(heldout),
                    "best_hidden": str(audit["best_hidden"]),
                    "best_alpha": audit["best_alpha"],
                    "best_val_RMSE": audit["best_val_RMSE"],
                }
            )

    val_pred = apply_client_mlp(val_rows, models, "mlp_direct_val")
    for row in val_pred:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["mlp_direct_val_ppm"] = fnum(row.get("mlp_direct_val_ppm"))
    selection_rows, _selected, _coaware, hybrid_modes = build_mlp_selection_table(
        val_pred,
        target_clients,
        max_nonco_delta,
    )
    forced_rows = apply_client_mlp(with_deployment_route(apply_rows), models, "mlp_direct")
    for row in forced_rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["mlp_direct_ppm"] = fnum(row.get("mlp_direct_ppm"))
    hybrid_rows = apply_mlp_selection(forced_rows, hybrid_modes, output_key)
    for row in hybrid_rows:
        row[output_alias] = fnum(row[output_key])
    for row in selection_rows:
        row["model"] = "mlp"
    return hybrid_rows, fit_audit, selection_rows


def select_lowcal_c4_gate(cal_rows: list[dict[str, Any]], ratio: float, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pandas as pd

    cal_phase = attach_response_phase(cal_rows, Path(DATA_ROOT))
    df = pd.DataFrame(cal_phase)
    scores = [score_gate(df, spec) for spec in candidate_grid()]
    selected = select_gate(scores)
    selected["low_calib_ratio"] = ratio
    write_csv(out_dir / f"c4_gate_scores_ratio_{ratio:g}.csv", scores)
    return selected, scores


def apply_gate_to(rows: list[dict[str, Any]], source_key: str, output_key: str, gate: dict[str, Any]) -> list[dict[str, Any]]:
    phased = attach_response_phase(rows, Path(DATA_ROOT))
    pred_classes = {int(float(part)) for part in str(gate.get("pred_classes", "")).split(",") if str(part).strip()}
    max_final = fnum(gate.get("max_final", gate.get("max_ppm")), float("inf"))
    min_risk = fnum(gate.get("min_risk", gate.get("risk_threshold")), 0.0)
    max_conf_margin = fnum(gate.get("max_conf_margin"), 1.0)
    phase = str(gate.get("phase", "any"))
    rescue_ppm = fnum(gate.get("rescue_ppm"))
    out: list[dict[str, Any]] = []
    for row in phased:
        item = dict(row)
        hit = str(item.get("client")) == "C4"
        hit = hit and inum(item.get("pred_class")) in pred_classes
        hit = hit and fnum(item.get("final_ppm")) < max_final
        hit = hit and fnum(item.get("risk_score"), 0.0) >= min_risk
        hit = hit and (phase == "any" or str(item.get("response_phase")) == phase)
        hit = hit and fnum(item.get("confidence_margin"), 1.0) <= max_conf_margin
        item["c4_rescue_applied"] = int(hit)
        item[output_key] = rescue_ppm if hit else fnum(item.get(source_key))
        out.append(item)
    return out


def metric_lookup(summary: list[dict[str, Any]], mode: str, scope: str, field: str = "RMSE") -> float:
    for row in summary:
        if row.get("mode") == mode and row.get("scope") == scope and row.get("split") == "test":
            return fnum(row.get(field))
    return float("nan")


def calibration_decision(
    cal_eval: list[dict[str, Any]],
    gate: dict[str, Any],
    h23_key: str,
    h8_key: str,
) -> dict[str, Any]:
    pred_co = [row for row in cal_eval if inum(row.get("pred_class")) == CO_CLASS]
    true_co = [row for row in pred_co if inum(row.get("true_class")) == CO_CLASS]
    false_co = len(pred_co) - len(true_co)
    precision = len(true_co) / len(pred_co) if pred_co else 0.0
    fp_rate = false_co / len(pred_co) if pred_co else 1.0
    h23_co = metric_for_rows([row for row in cal_eval if inum(row.get("true_class")) == CO_CLASS], h23_key)
    h8_co = metric_for_rows([row for row in cal_eval if inum(row.get("true_class")) == CO_CLASS], h8_key)
    h23_nonco = metric_for_rows([row for row in cal_eval if inum(row.get("true_class")) != CO_CLASS], h23_key)
    h8_nonco = metric_for_rows([row for row in cal_eval if inum(row.get("true_class")) != CO_CLASS], h8_key)
    h8_enabled = bool(precision >= 0.95 and fp_rate <= 0.05 and h8_co <= h23_co)
    gate_enabled = bool(int(gate.get("false_hits", gate.get("calibration_false_hits", 0))) == 0 and int(gate.get("true_c4_high_hits", 0)) > 0)
    return {
        "pred_co_N": len(pred_co),
        "pred_co_precision": precision,
        "pred_co_false_positive_rate": fp_rate,
        "calib_h2_3_CO_RMSE": h23_co,
        "calib_h8_CO_RMSE": h8_co,
        "calib_h2_3_nonCO_RMSE": h23_nonco,
        "calib_h8_nonCO_RMSE": h8_nonco,
        "h8_enabled": int(h8_enabled),
        "gate_enabled": int(gate_enabled),
        "selected_profile": "H8_plus_C4_lowcal" if h8_enabled and gate_enabled else "H2_3_lowcal",
    }


def metric_for_rows(rows: list[dict[str, Any]], pred_key: str) -> float:
    if not rows:
        return float("inf")
    pred = np.asarray([fnum(row[pred_key]) for row in rows], dtype=np.float64)
    true = np.asarray([fnum(row["true_ppm"]) for row in rows], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def no_gate_rows(rows: list[dict[str, Any]], source_key: str, output_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["c4_rescue_applied"] = 0
        item[output_key] = fnum(item.get(source_key))
        out.append(item)
    return out


def gate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [row for row in rows if inum(row.get("c4_rescue_applied")) == 1]
    true_c4_high_co = [
        row for row in hits
        if str(row.get("client")) == "C4"
        and inum(row.get("true_class")) == CO_CLASS
        and fnum(row.get("true_ppm")) >= 200.0
    ]
    nonco_hits = [row for row in hits if inum(row.get("true_class")) != CO_CLASS]
    return {
        "test_gate_hit_N": len(hits),
        "test_gate_true_c4_high_hits": len(true_c4_high_co),
        "test_gate_false_hits": len(hits) - len(true_c4_high_co),
        "test_gate_nonCO_hits": len(nonco_hits),
    }


def add_h2_h8_predictions(
    base_rows: list[dict[str, Any]],
    ridge_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    c5_grid_rows: list[dict[str, Any]],
    source_aug_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ridge_by_key = {(row["client"], row["split"], row["sample_index"]): row for row in ridge_rows}
    mlp_by_key = {(row["client"], row["split"], row["sample_index"]): row for row in mlp_rows}
    c5_by_key = {(row["client"], row["split"], row["sample_index"]): row for row in c5_grid_rows}
    aug_by_key = {(row["client"], row["split"], row["sample_index"]): row for row in source_aug_rows}
    out: list[dict[str, Any]] = []
    for row in base_rows:
        key = (row["client"], row["split"], row["sample_index"])
        client = str(row["client"])
        item = {k: v for k, v in row.items() if k != "feature_dict"}
        item["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        item["ridge_lowcal_ppm"] = fnum(ridge_by_key[key]["ridge_lowcal_ppm"])
        item["mlp_lowcal_ppm"] = fnum(mlp_by_key[key]["mlp_lowcal_ppm"])
        item["c5_grid_lowcal_ppm"] = fnum(c5_by_key[key]["c5_grid_lowcal_ppm"]) if key in c5_by_key else item["mlp_lowcal_ppm"]
        item["source_aug_lowcal_ppm"] = fnum(aug_by_key[key]["source_aug_lowcal_ppm"])
        if client == "C3":
            h23 = item["mlp_lowcal_ppm"]
        elif client == "C4":
            h23 = item["ridge_lowcal_ppm"]
        elif client == "C5":
            h23 = item["c5_grid_lowcal_ppm"]
        else:
            h23 = item["baseline_final_ppm"]
        item["h2_3_lowcal_ppm"] = h23
        item["h8_lowcal_ppm"] = item["source_aug_lowcal_ppm"] if inum(item.get("pred_class")) == CO_CLASS else h23
        out.append(item)
    return out


def run_ratio(
    ratio: float,
    rows: list[dict[str, Any]],
    target_clients: list[str],
    source_models: tuple[dict[int, Any], dict[int, Any], Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_ratio = float(args.base_calib_ratio)
    fraction = min(1.0, float(ratio) / base_ratio)
    calibration_all = [row for row in rows if row["split"] == "calibration"]
    test_rows = [row for row in rows if row["split"] == "test"]
    cal_subset = stratified_subset(calibration_all, fraction, seed=args.seed + int(ratio * 100))
    feature_names = sorted(rows[0]["feature_dict"].keys())
    alphas = [float(item.strip()) for item in args.ridge_alphas.split(",") if item.strip()]

    gate, gate_scores = select_lowcal_c4_gate(cal_subset, ratio, out_dir)
    gate.setdefault("false_hits", gate.get("calibration_false_hits", 0))

    ridge_test, ridge_audit, ridge_selection = fit_apply_ridge_hybrid(
        cal_subset,
        test_rows,
        target_clients,
        feature_names,
        alphas,
        args.val_ratio,
        args.max_nonco_delta,
        "ridge_lowcal_hybrid_ppm",
    )
    ridge_cal, _ridge_cal_audit, _ridge_cal_selection = fit_apply_ridge_hybrid(
        cal_subset,
        cal_subset,
        target_clients,
        feature_names,
        alphas,
        args.val_ratio,
        args.max_nonco_delta,
        "ridge_lowcal_hybrid_ppm",
    )

    standard_hiddens = parse_hidden_grid(args.mlp_hidden_grid)
    standard_alphas = [float(item.strip()) for item in args.mlp_alphas.split(",") if item.strip()]
    c5_hiddens = parse_hidden_grid(args.c5_mlp_hidden_grid)
    c5_alphas = [float(item.strip()) for item in args.c5_mlp_alphas.split(",") if item.strip()]
    mlp_test, mlp_audit, mlp_selection = fit_apply_mlp_hybrid(
        cal_subset,
        test_rows,
        target_clients,
        feature_names,
        hiddens_by_client={client: standard_hiddens for client in target_clients},
        alphas_by_client={client: standard_alphas for client in target_clients},
        val_ratio=args.val_ratio,
        max_nonco_delta=args.max_nonco_delta,
        seed=args.seed,
        output_key="mlp_lowcal_hybrid_ppm",
        output_alias="mlp_lowcal_ppm",
    )
    c5_test, c5_audit, c5_selection = fit_apply_mlp_hybrid(
        [row for row in cal_subset if row["client"] == "C5"],
        [row for row in test_rows if row["client"] == "C5"],
        ["C5"],
        feature_names,
        hiddens_by_client={"C5": c5_hiddens},
        alphas_by_client={"C5": c5_alphas},
        val_ratio=args.val_ratio,
        max_nonco_delta=args.max_nonco_delta,
        seed=args.seed + 5000,
        output_key="c5_grid_lowcal_hybrid_ppm",
        output_alias="c5_grid_lowcal_ppm",
    )
    mlp_cal, _mlp_cal_audit, _mlp_cal_selection = fit_apply_mlp_hybrid(
        cal_subset,
        cal_subset,
        target_clients,
        feature_names,
        hiddens_by_client={client: standard_hiddens for client in target_clients},
        alphas_by_client={client: standard_alphas for client in target_clients},
        val_ratio=args.val_ratio,
        max_nonco_delta=args.max_nonco_delta,
        seed=args.seed,
        output_key="mlp_lowcal_hybrid_ppm",
        output_alias="mlp_lowcal_ppm",
    )
    c5_cal, _c5_cal_audit, _c5_cal_selection = fit_apply_mlp_hybrid(
        [row for row in cal_subset if row["client"] == "C5"],
        [row for row in cal_subset if row["client"] == "C5"],
        ["C5"],
        feature_names,
        hiddens_by_client={"C5": c5_hiddens},
        alphas_by_client={"C5": c5_alphas},
        val_ratio=args.val_ratio,
        max_nonco_delta=args.max_nonco_delta,
        seed=args.seed + 5000,
        output_key="c5_grid_lowcal_hybrid_ppm",
        output_alias="c5_grid_lowcal_ppm",
    )

    ridge_src, mlp_src, shared_src = source_models
    pred_keys = ["H1_source_ridge_ppm", "H2_source_per_gas_mlp_ppm", "H3_source_shared_mlp_ppm"]
    cal_with_src = add_pred_features(attach_source_predictions(cal_subset, ridge_src, mlp_src, shared_src), pred_keys)
    test_with_src = add_pred_features(attach_source_predictions(test_rows, ridge_src, mlp_src, shared_src), pred_keys)
    aug_feature_names = sorted(cal_with_src[0]["feature_dict"].keys())
    aug_models, aug_audit = fit_source_aug_models(cal_with_src, target_clients, aug_feature_names, alphas, args.val_ratio)
    aug_test = apply_client_models(with_deployment_route(test_with_src), aug_models, "source_aug_lowcal")
    aug_cal = apply_client_models(with_deployment_route(cal_with_src), aug_models, "source_aug_lowcal")

    test_pred = add_h2_h8_predictions(test_rows, ridge_test, mlp_test, c5_test, aug_test)
    cal_pred = add_h2_h8_predictions(cal_subset, ridge_cal, mlp_cal, c5_cal, aug_cal)
    decision = calibration_decision(cal_pred, gate, "h2_3_lowcal_ppm", "h8_lowcal_ppm")

    if decision["gate_enabled"]:
        h23_gate = apply_gate_to(test_pred, "h2_3_lowcal_ppm", "h2_3_lowcal_plus_c4_ppm", gate)
        h8_gate = apply_gate_to(test_pred, "h8_lowcal_ppm", "h8_lowcal_plus_c4_ppm", gate)
    else:
        h23_gate = no_gate_rows(test_pred, "h2_3_lowcal_ppm", "h2_3_lowcal_plus_c4_ppm")
        h8_gate = no_gate_rows(test_pred, "h8_lowcal_ppm", "h8_lowcal_plus_c4_ppm")
    selected_key = "h8_lowcal_plus_c4_ppm" if decision["selected_profile"] == "H8_plus_C4_lowcal" else "h2_3_lowcal_plus_c4_ppm"
    selected_rows = []
    for row in h8_gate:
        item = dict(row)
        # h23_gate and h8_gate have the same row order from the same base list.
        h23_item = h23_gate[len(selected_rows)]
        item["h2_3_lowcal_plus_c4_ppm"] = h23_item["h2_3_lowcal_plus_c4_ppm"]
        item["selector_lowcal_ppm"] = fnum(item[selected_key])
        item["selector_selected_profile"] = decision["selected_profile"]
        selected_rows.append(item)

    summary: list[dict[str, Any]] = []
    for pred_key, mode in [
        ("baseline_final_ppm", "B0_baseline_final"),
        ("h2_3_lowcal_plus_c4_ppm", "H2_3_lowcal"),
        ("h8_lowcal_plus_c4_ppm", "H8_plus_C4_forced_lowcal"),
        ("selector_lowcal_ppm", "H8_C4_selector_lowcal"),
    ]:
        summary.extend(summarize(selected_rows, pred_key, mode, "test"))
    for row in summary:
        row["calib_ratio"] = ratio
        row["calibration_subset_fraction_of_20pct"] = fraction

    audit_rows = []
    for row in [*ridge_audit, *mlp_audit, *c5_audit, *aug_audit]:
        item = dict(row)
        item["calib_ratio"] = ratio
        audit_rows.append(item)
    selection_audit = []
    for row in [*ridge_selection, *mlp_selection, *c5_selection]:
        item = dict(row)
        item["calib_ratio"] = ratio
        selection_audit.append(item)
    decision_row = dict(decision)
    decision_row.update(
        {
            "calib_ratio": ratio,
            "calibration_subset_N": len(cal_subset),
            "calibration_fraction_of_20pct": fraction,
            "gate_hit_N": gate.get("hit_N"),
            "gate_true_c4_high_hits": gate.get("true_c4_high_hits"),
            "gate_false_hits": gate.get("false_hits"),
            "gate_recall": gate.get("calib_c4_high_recall"),
            "gate": json.dumps(gate, ensure_ascii=False),
            **gate_audit(h23_gate),
        }
    )
    for row in selected_rows:
        row["calib_ratio"] = ratio
    return summary, [decision_row], [*audit_rows, *selection_audit], subset_audit(cal_subset, ratio)


def fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def write_report(out_dir: Path, summary_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> None:
    modes = ["B0_baseline_final", "H2_3_lowcal", "H8_plus_C4_forced_lowcal", "H8_C4_selector_lowcal"]
    scopes = ["ALL", "C3-CO", "C4-CO", "C5-CO", "C4-CO_high_200_250", "C5-CO_high_200_250", "nonCO_ALL"]
    lines = [
        "# Low-Calibration Stress Profiles",
        "",
        "This stress test refits target-side calibration/profile heads using stratified subsets of the target calibration split.",
        "QC is not used. Test metrics are reported after calibration-only fitting and selector decisions.",
        "",
        "## Selector Decisions",
        "",
        "| calib_ratio | subset_N | selected_profile | h8_enabled | gate_enabled | pred_CO_precision | h2_CO | h8_CO | calib_gate_false | calib_gate_hits | test_gate_false | test_gate_hits |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decision_rows:
        lines.append(
            "| {ratio} | {n} | {profile} | {h8} | {gate_enabled} | {precision} | {h2co} | {h8co} | {false} | {hits} | {test_false} | {test_hits} |".format(
                ratio=fmt(row["calib_ratio"], 0),
                n=row["calibration_subset_N"],
                profile=row["selected_profile"],
                h8=row["h8_enabled"],
                gate_enabled=row["gate_enabled"],
                precision=fmt(row["pred_co_precision"], 3),
                h2co=fmt(row["calib_h2_3_CO_RMSE"]),
                h8co=fmt(row["calib_h8_CO_RMSE"]),
                false=row["gate_false_hits"],
                hits=row["gate_hit_N"],
                test_false=row.get("test_gate_false_hits", ""),
                test_hits=row.get("test_gate_hit_N", ""),
            )
        )
    lines.extend(["", "## Test RMSE", ""])
    for ratio in sorted({float(row["calib_ratio"]) for row in summary_rows}, reverse=True):
        lines.extend(
            [
                f"### Calibration {ratio:g}%",
                "",
                "| mode | " + " | ".join(scopes) + " |",
                "|---|" + "|".join(["---:"] * len(scopes)) + "|",
            ]
        )
        for mode in modes:
            values = []
            for scope in scopes:
                value = ""
                for row in summary_rows:
                    if float(row["calib_ratio"]) == ratio and row["mode"] == mode and row["scope"] == scope:
                        value = fmt(row["RMSE"])
                        break
                values.append(value)
            lines.append("| " + mode + " | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## Reading",
            "",
            "- `H8_C4_selector_lowcal` uses calibration-only stress rules and can fall back to H2.3.",
            "- This is a profile-refit stress test, not a new exported runtime bundle.",
            "- If low-ratio gates lose support or produce false hits, the co-priority specialist should be restricted to the full 20% calibration setting.",
            "",
        ]
    )
    (out_dir / "low_calib_stress_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--target-predictions", default=TARGET_PREDICTIONS)
    parser.add_argument("--source-clients", default="1,2")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ratios", default="20,10,5")
    parser.add_argument("--base-calib-ratio", type=float, default=20.0)
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-hidden-grid", default="32")
    parser.add_argument("--mlp-alphas", default="0.001,0.01")
    parser.add_argument("--c5-mlp-hidden-grid", default="16;32;64;32,16")
    parser.add_argument("--c5-mlp-alphas", default="0.001,0.01,0.1,1")
    parser.add_argument("--source-mlp-hidden-grid", default="16")
    parser.add_argument("--source-mlp-alphas", default="0.01,0.1")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--max-nonco-delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/low_calib_stress_profiles_20260626")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clients = [client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()]
    source_clients = [client_name(item.strip()) for item in args.source_clients.split(",") if item.strip()]
    ratios = [float(item.strip()) for item in args.ratios.split(",") if item.strip()]
    ridge_alphas = [float(item.strip()) for item in args.ridge_alphas.split(",") if item.strip()]
    source_hiddens = parse_hidden_grid(args.source_mlp_hidden_grid)
    source_alphas = [float(item.strip()) for item in args.source_mlp_alphas.split(",") if item.strip()]

    raw_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    rows = add_target_features(raw_rows, Path(args.data_root))
    for row in rows:
        row["client"] = client_name(row.get("client") or row.get("client_id"))
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        if row["split"] == "calibration":
            row["route_class"] = row["true_class"]

    source_models = fit_source_heads(
        Path(args.data_root),
        source_clients,
        ridge_alphas,
        source_hiddens,
        source_alphas,
        args.seed,
    )[:3]

    all_summary: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    all_fit_audit: list[dict[str, Any]] = []
    all_subset_audit: list[dict[str, Any]] = []
    for ratio in ratios:
        summary, decisions, fit_audit, subset_rows = run_ratio(
            ratio,
            rows,
            target_clients,
            source_models,
            args,
            out_dir,
        )
        all_summary.extend(summary)
        all_decisions.extend(decisions)
        all_fit_audit.extend(fit_audit)
        all_subset_audit.extend(subset_rows)

    write_csv(out_dir / "low_calib_stress_summary.csv", all_summary)
    write_csv(out_dir / "low_calib_selector_decisions.csv", all_decisions)
    write_csv(out_dir / "low_calib_fit_audit.csv", all_fit_audit)
    write_csv(out_dir / "low_calib_subset_audit.csv", all_subset_audit)
    write_report(out_dir, all_summary, all_decisions)
    write_json(
        out_dir / "manifest.json",
        {
            "data_root": args.data_root,
            "target_predictions": args.target_predictions,
            "source_clients": source_clients,
            "target_clients": target_clients,
            "ratios": ratios,
            "base_calib_ratio": args.base_calib_ratio,
            "outputs": [
                "low_calib_stress_summary.csv",
                "low_calib_selector_decisions.csv",
                "low_calib_fit_audit.csv",
                "low_calib_subset_audit.csv",
                "low_calib_stress_report.md",
            ],
        },
    )
    print(json.dumps({"output_dir": str(out_dir), "ratios": ratios}, indent=2))


if __name__ == "__main__":
    main()
