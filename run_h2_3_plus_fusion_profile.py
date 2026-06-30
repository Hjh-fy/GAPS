"""Run P2 H2.3+ balanced fusion profile experiments.

The profile keeps the existing H2.3 route/rescue semantics as the anchor and
uses the P1-positive A3 rich+reg_feat Ridge head only as a weak, calibration
selected blend candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from run_formal_target_mlp_auto_v2_eval import MLPHead, apply_client_mlp, fit_mlp
from run_formal_target_ridge_auto_v2_eval import apply_c4_rescue, attach_response_phase, selected_c4_gate
from run_h2_3_backbone_feature_ablation import build_feature_groups, load_feature_rows, merge_backbone_features
from run_regression_head_ablation import (
    CLASS_NAMES,
    CO_CLASS,
    add_target_features,
    apply_client_models,
    client_name,
    client_num,
    deterministic_train_val,
    fit_ridge,
    fnum,
    inum,
    metrics,
    read_csv,
    rmse,
    summarize,
    write_csv,
)


MODE_ORDER = [
    "A0_baseline_final",
    "H2_3_direct_only_r25_refit",
    "H2_3_current_r25_refit_anchor",
    "H2_3_plus_reg_feat_ridge_rescue",
    "H2_3_plus_blend_r25",
    "H2_3_current_r25_reference",
]


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        client_name(row.get("client") or row.get("client_id")),
        str(row.get("split")),
        inum(row.get("sample_index")),
    )


def parse_clients(text: str) -> list[str]:
    return [client_name(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_hidden_grid(text: str) -> list[tuple[int, ...]]:
    return [
        tuple(int(value.strip()) for value in item.split(",") if value.strip())
        for item in text.split(";")
        if item.strip()
    ]


def blend_value(anchor: Any, candidate: Any, weight: float) -> float:
    anchor_f = fnum(anchor)
    candidate_f = fnum(candidate, anchor_f)
    return float(anchor_f + float(weight) * (candidate_f - anchor_f))


def rows_with_blend(
    rows: Sequence[dict[str, Any]],
    *,
    anchor_key: str,
    candidate_key: str,
    output_key: str,
    weight: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item[output_key] = blend_value(item.get(anchor_key), item.get(candidate_key), weight)
        out.append(item)
    return out


def scoped_rmse(rows: Sequence[dict[str, Any]], pred_key: str, *, nonco_only: bool = False) -> float:
    selected = list(rows)
    if nonco_only:
        selected = [row for row in selected if inum(row.get("true_class")) != CO_CLASS]
    if not selected:
        return float("nan")
    return fnum(metrics(selected, pred_key).get("RMSE"), float("inf"))


def select_client_blend_weights(
    val_rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    weight_grid: Sequence[float],
    *,
    anchor_key: str,
    candidate_key: str,
    max_nonco_delta: float,
    min_all_delta: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Choose per-client convex blend weights using validation rows only."""
    selected: dict[str, float] = {}
    audit: list[dict[str, Any]] = []
    grid = sorted({float(weight) for weight in weight_grid})
    if 0.0 not in grid:
        grid = [0.0, *grid]

    for client in target_clients:
        client_rows = [row for row in val_rows if client_name(row.get("client")) == client]
        anchor_all = scoped_rmse(client_rows, anchor_key)
        anchor_nonco = scoped_rmse(client_rows, anchor_key, nonco_only=True)
        best_weight = 0.0
        best_rmse = anchor_all
        client_audit: list[dict[str, Any]] = []
        for weight in grid:
            blended = rows_with_blend(
                client_rows,
                anchor_key=anchor_key,
                candidate_key=candidate_key,
                output_key="_candidate_blend_ppm",
                weight=weight,
            )
            blend_all = scoped_rmse(blended, "_candidate_blend_ppm")
            blend_nonco = scoped_rmse(blended, "_candidate_blend_ppm", nonco_only=True)
            nonco_guard = (not np.isfinite(anchor_nonco)) or blend_nonco <= anchor_nonco + max_nonco_delta
            all_improves = np.isfinite(blend_all) and blend_all < anchor_all - min_all_delta
            passes = bool(all_improves and nonco_guard)
            if passes and blend_all < best_rmse:
                best_rmse = blend_all
                best_weight = float(weight)
            client_audit.append(
                {
                    "client": client,
                    "weight": float(weight),
                    "anchor_ALL_RMSE": anchor_all,
                    "blend_ALL_RMSE": blend_all,
                    "delta_ALL_RMSE": blend_all - anchor_all,
                    "anchor_nonCO_RMSE": anchor_nonco,
                    "blend_nonCO_RMSE": blend_nonco,
                    "delta_nonCO_RMSE": blend_nonco - anchor_nonco if np.isfinite(anchor_nonco) else "",
                    "passes_guard": int(passes),
                    "selected": 0,
                }
            )
        selected[client] = best_weight
        for row in client_audit:
            row["selected"] = int(abs(float(row["weight"]) - best_weight) < 1e-12)
            audit.append(row)
    return selected, audit


def apply_client_blends(
    rows: Sequence[dict[str, Any]],
    weights_by_client: dict[str, float],
    *,
    anchor_key: str,
    candidate_key: str,
    output_key: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        client = client_name(item.get("client"))
        weight = float(weights_by_client.get(client, 0.0))
        item["blend_weight"] = weight
        item[output_key] = blend_value(item.get(anchor_key), item.get(candidate_key), weight)
        out.append(item)
    return out


def add_group_features(rows: Sequence[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["feature_dict"] = build_feature_groups(row)[group_name]
        item["feature_group"] = group_name
        out.append(item)
    return out


def fit_ridge_family(
    calibration_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    feature_names: Sequence[str],
    alphas: Sequence[float],
    val_ratio: float,
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    val_rows_all: list[dict[str, Any]] = []
    train_models: dict[tuple[str, int], Any] = {}
    final_models: dict[tuple[str, int], Any] = {}
    fit_audit: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row for row in calibration_rows
                if client_name(row.get("client")) == client and inum(row.get("true_class")) == cls_id
            ]
            train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            y_val = np.asarray([fnum(row["true_ppm"]) for row in val_rows], dtype=np.float64)
            best_alpha = float(alphas[0])
            best_score = float("inf")
            alpha_audit: list[dict[str, Any]] = []
            for alpha in alphas:
                model = fit_ridge(train_rows, feature_names, float(alpha))
                pred = model.predict(val_rows, clip=True)
                score = rmse(y_val, pred)
                alpha_audit.append({"alpha": float(alpha), "val_RMSE": score})
                if score < best_score:
                    best_score = score
                    best_alpha = float(alpha)
            train_models[(client, cls_id)] = fit_ridge(train_rows, feature_names, best_alpha)
            final_models[(client, cls_id)] = fit_ridge(cls_rows, feature_names, best_alpha)
            for row in val_rows:
                val_item = dict(row)
                val_item["route_class"] = val_item["pred_class"]
                val_rows_all.append(val_item)
            fit_audit.append(
                {
                    "family": prefix,
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_alpha": best_alpha,
                    "best_val_RMSE": best_score,
                    "alpha_audit": json.dumps(alpha_audit, ensure_ascii=False),
                }
            )
    return (
        apply_client_models(val_rows_all, train_models, prefix),
        apply_client_models(list(test_rows), final_models, prefix),
        fit_audit,
    )


def fit_mlp_family(
    calibration_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    feature_names: Sequence[str],
    hiddens: Sequence[tuple[int, ...]],
    alphas: Sequence[float],
    val_ratio: float,
    seed: int,
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    val_rows_all: list[dict[str, Any]] = []
    train_models: dict[tuple[str, int], MLPHead] = {}
    final_models: dict[tuple[str, int], MLPHead] = {}
    fit_audit: list[dict[str, Any]] = []
    for client in target_clients:
        for cls_id in sorted(CLASS_NAMES):
            cls_rows = [
                row for row in calibration_rows
                if client_name(row.get("client")) == client and inum(row.get("true_class")) == cls_id
            ]
            train_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            y_val = np.asarray([fnum(row["true_ppm"]) for row in val_rows], dtype=np.float64)
            best_score = float("inf")
            best_hidden = tuple(hiddens[0])
            best_alpha = float(alphas[0])
            grid_audit: list[dict[str, Any]] = []
            model_seed = int(seed) + cls_id + 100 * client_num(client)
            for hidden in hiddens:
                for alpha in alphas:
                    model = fit_mlp(train_rows, feature_names, tuple(hidden), float(alpha), model_seed)
                    pred = model.predict(val_rows)
                    score = rmse(y_val, pred)
                    grid_audit.append({"hidden": str(tuple(hidden)), "alpha": float(alpha), "val_RMSE": score})
                    if score < best_score:
                        best_score = score
                        best_hidden = tuple(hidden)
                        best_alpha = float(alpha)
            train_models[(client, cls_id)] = fit_mlp(train_rows, feature_names, best_hidden, best_alpha, model_seed)
            final_models[(client, cls_id)] = fit_mlp(cls_rows, feature_names, best_hidden, best_alpha, model_seed)
            for row in val_rows:
                val_item = dict(row)
                val_item["route_class"] = val_item["pred_class"]
                val_rows_all.append(val_item)
            fit_audit.append(
                {
                    "family": prefix,
                    "client": client,
                    "class_id": cls_id,
                    "gas": CLASS_NAMES[cls_id],
                    "train_N": len(train_rows),
                    "val_N": len(val_rows),
                    "best_hidden": str(best_hidden),
                    "best_alpha": best_alpha,
                    "best_val_RMSE": best_score,
                    "grid_audit": json.dumps(grid_audit, ensure_ascii=False),
                }
            )
    return (
        apply_client_mlp(val_rows_all, train_models, prefix),
        apply_client_mlp(list(test_rows), final_models, prefix),
        fit_audit,
    )


def by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {row_key(row): row for row in rows}


def combine_h2_3_rows(
    *,
    c3_mlp_rows: Sequence[dict[str, Any]],
    c4_ridge_rows: Sequence[dict[str, Any]],
    c5_grid_rows: Sequence[dict[str, Any]],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for rows, pred_key, expected_client in [
        (c3_mlp_rows, "h2_c3_mlp_ppm", "C3"),
        (c4_ridge_rows, "h2_c4_ridge_ppm", "C4"),
        (c5_grid_rows, "h2_c5_grid_mlp_ppm", "C5"),
    ]:
        for row in rows:
            if client_name(row.get("client")) != expected_client or pred_key not in row:
                continue
            key = row_key(row)
            if key in seen:
                raise ValueError(f"Duplicate H2.3 anchor row for key {key}")
            seen.add(key)
            item = {k: v for k, v in row.items() if k != "feature_dict"}
            item["h2_3_direct_only_ppm"] = fnum(item.get(pred_key))
            out.append(item)
    return apply_c4_rescue(out, "h2_3_direct_only_ppm", "h2_3_current_ppm", gate)


def merge_prediction_sets(
    base_rows: Sequence[dict[str, Any]],
    extra_rows: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    extra_by_key = by_key(extra_rows)
    out: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []
    for row in base_rows:
        key = row_key(row)
        extra = extra_by_key.get(key)
        item = dict(row)
        if extra is None:
            missing.append(key)
        else:
            for value_key in keys:
                item[value_key] = extra.get(value_key)
        out.append(item)
    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise ValueError(f"Missing prediction rows for {len(missing)} keys; first keys: {preview}")
    return out


def client_nrmse_rows(summary_rows: Sequence[dict[str, Any]], target_clients: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    modes = [mode for mode in MODE_ORDER if any(row["mode"] == mode for row in summary_rows)]
    for mode in modes:
        selected = [row for row in summary_rows if row["mode"] == mode and row["scope"] in target_clients]
        values = [fnum(row.get("NRMSE")) for row in selected]
        rmses = [fnum(row.get("RMSE")) for row in selected]
        item: dict[str, Any] = {
            "mode": mode,
            "macro_client_NRMSE": float(np.nanmean(values)) if values else "",
            "macro_client_RMSE": float(np.nanmean(rmses)) if rmses else "",
        }
        for client in target_clients:
            row = next((entry for entry in selected if entry["scope"] == client), None)
            item[f"{client}_NRMSE"] = "" if row is None else row.get("NRMSE")
            item[f"{client}_RMSE"] = "" if row is None else row.get("RMSE")
        out.append(item)
    return out


def find_metric(summary_rows: Sequence[dict[str, Any]], mode: str, scope: str, key: str) -> float:
    for row in summary_rows:
        if row["mode"] == mode and row["scope"] == scope:
            return fnum(row.get(key))
    return float("nan")


def find_macro(per_client_rows: Sequence[dict[str, Any]], mode: str, key: str) -> float:
    for row in per_client_rows:
        if row["mode"] == mode:
            return fnum(row.get(key))
    return float("nan")


def format_float(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    value_f = fnum(value)
    return "" if not np.isfinite(value_f) else f"{value_f:.{digits}f}"


def build_reading(summary_rows: Sequence[dict[str, Any]], per_client_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    anchor = "H2_3_current_r25_refit_anchor"
    blend = "H2_3_plus_blend_r25"
    direct = "H2_3_direct_only_r25_refit"
    return {
        "anchor_ALL_NRMSE": find_metric(summary_rows, anchor, "ALL", "NRMSE"),
        "blend_ALL_NRMSE": find_metric(summary_rows, blend, "ALL", "NRMSE"),
        "direct_ALL_NRMSE": find_metric(summary_rows, direct, "ALL", "NRMSE"),
        "anchor_macro_client_NRMSE": find_macro(per_client_rows, anchor, "macro_client_NRMSE"),
        "blend_macro_client_NRMSE": find_macro(per_client_rows, blend, "macro_client_NRMSE"),
        "direct_macro_client_NRMSE": find_macro(per_client_rows, direct, "macro_client_NRMSE"),
        "anchor_nonCO_RMSE": find_metric(summary_rows, anchor, "nonCO_ALL", "RMSE"),
        "blend_nonCO_RMSE": find_metric(summary_rows, blend, "nonCO_ALL", "RMSE"),
    }


def write_report(
    out_dir: Path,
    summary_rows: Sequence[dict[str, Any]],
    per_client_rows: Sequence[dict[str, Any]],
    selection_rows: Sequence[dict[str, Any]],
    reading: dict[str, Any],
) -> None:
    scopes = [
        "ALL",
        "C3",
        "C4",
        "C5",
        "C3-CO",
        "C4-CO",
        "C5-CO",
        "C3-CO_high_200_250",
        "C4-CO_high_200_250",
        "C5-CO_high_200_250",
        "nonCO_ALL",
    ]
    lines = [
        "# H2.3+ Balanced Fusion Profile P2",
        "",
        "Selection uses calibration-validation only. Test metrics are reported after the per-client blend weights are fixed.",
        "",
        "## Selected Blend Weights",
        "",
        "| client | selected weight | anchor val ALL RMSE | blend val ALL RMSE | anchor val nonCO RMSE | blend val nonCO RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selection_rows:
        if inum(row.get("selected")) != 1:
            continue
        lines.append(
            "| {client} | {weight:.2f} | {anchor_all} | {blend_all} | {anchor_nonco} | {blend_nonco} |".format(
                client=row["client"],
                weight=fnum(row["weight"]),
                anchor_all=format_float(row.get("anchor_ALL_RMSE"), 2),
                blend_all=format_float(row.get("blend_ALL_RMSE"), 2),
                anchor_nonco=format_float(row.get("anchor_nonCO_RMSE"), 2),
                blend_nonco=format_float(row.get("blend_nonCO_RMSE"), 2),
            )
        )

    lines.extend(
        [
            "",
            "## Test RMSE",
            "",
            "| mode | " + " | ".join(scopes) + " | macro-client NRMSE |",
            "|---|" + "|".join(["---:"] * (len(scopes) + 1)) + "|",
        ]
    )
    for mode in MODE_ORDER:
        if not any(row["mode"] == mode for row in summary_rows):
            continue
        values = [format_float(find_metric(summary_rows, mode, scope, "RMSE"), 2) for scope in scopes]
        values.append(format_float(find_macro(per_client_rows, mode, "macro_client_NRMSE"), 4))
        lines.append("| " + mode + " | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Test NRMSE",
            "",
            "| mode | ALL NRMSE | macro-client NRMSE | C3 NRMSE | C4 NRMSE | C5 NRMSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    per_by_mode = {row["mode"]: row for row in per_client_rows}
    for mode in MODE_ORDER:
        if not any(row["mode"] == mode for row in summary_rows):
            continue
        row = per_by_mode.get(mode, {})
        lines.append(
            "| {mode} | {all_nrmse} | {macro} | {c3} | {c4} | {c5} |".format(
                mode=mode,
                all_nrmse=format_float(find_metric(summary_rows, mode, "ALL", "NRMSE"), 4),
                macro=format_float(row.get("macro_client_NRMSE"), 4),
                c3=format_float(row.get("C3_NRMSE"), 4),
                c4=format_float(row.get("C4_NRMSE"), 4),
                c5=format_float(row.get("C5_NRMSE"), 4),
            )
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- Anchor ALL NRMSE: {format_float(reading['anchor_ALL_NRMSE'], 4)}",
            f"- Blend ALL NRMSE: {format_float(reading['blend_ALL_NRMSE'], 4)}",
            f"- Direct-only ALL NRMSE: {format_float(reading['direct_ALL_NRMSE'], 4)}",
            f"- Anchor macro-client NRMSE: {format_float(reading['anchor_macro_client_NRMSE'], 4)}",
            f"- Blend macro-client NRMSE: {format_float(reading['blend_macro_client_NRMSE'], 4)}",
            f"- Anchor nonCO_ALL RMSE: {format_float(reading['anchor_nonCO_RMSE'], 2)}",
            f"- Blend nonCO_ALL RMSE: {format_float(reading['blend_nonCO_RMSE'], 2)}",
            "- H2.3+ can be considered for balanced only if the blend improves ALL or macro-client NRMSE without hurting nonCO_ALL.",
            "",
        ]
    )
    (out_dir / "fusion_profile_report.md").write_text("\n".join(lines), encoding="utf-8")


def load_reference_rows(path: str | Path, base_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ref_path = Path(path)
    if not ref_path.exists():
        return []
    ref_by_key = by_key(read_csv(ref_path))
    out: list[dict[str, Any]] = []
    for row in base_rows:
        ref = ref_by_key.get(row_key(row))
        if ref is None:
            continue
        item = dict(row)
        item["reference_h2_3_current_ppm"] = fnum(ref.get("A1_h2_3_current_ppm"))
        out.append(item)
    return out


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clients = parse_clients(args.target_clients)
    ridge_alphas = parse_float_grid(args.ridge_alphas)
    mlp_alphas = parse_float_grid(args.mlp_alphas)
    c5_grid_alphas = parse_float_grid(args.c5_grid_alphas)
    blend_weights = parse_float_grid(args.blend_weights)
    hiddens = parse_hidden_grid(args.hidden_grid)
    c5_hiddens = parse_hidden_grid(args.c5_hidden_grid)
    gate = selected_c4_gate(args.route_rescue_artifact)

    raw_rows = [
        row for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    rows = attach_response_phase(add_target_features(raw_rows, data_root), data_root)
    for row in rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["route_class"] = inum(row.get("pred_class"))

    feature_rows = load_feature_rows(args.backbone_calibration, args.backbone_test)
    rows = merge_backbone_features(rows, feature_rows)
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    test_rows = [row for row in rows if row["split"] == "test"]
    rich_feature_names = sorted(calibration_rows[0]["feature_dict"].keys())

    c3_val, c3_test, c3_audit = fit_mlp_family(
        calibration_rows,
        test_rows,
        ["C3"],
        rich_feature_names,
        hiddens,
        mlp_alphas,
        args.val_ratio,
        args.seed,
        "h2_c3_mlp",
    )
    c4_val, c4_test, c4_audit = fit_ridge_family(
        calibration_rows,
        test_rows,
        ["C4"],
        rich_feature_names,
        ridge_alphas,
        args.val_ratio,
        "h2_c4_ridge",
    )
    c5_val, c5_test, c5_audit = fit_mlp_family(
        calibration_rows,
        test_rows,
        ["C5"],
        rich_feature_names,
        c5_hiddens,
        c5_grid_alphas,
        args.val_ratio,
        args.seed,
        "h2_c5_grid_mlp",
    )
    anchor_val = combine_h2_3_rows(c3_mlp_rows=c3_val, c4_ridge_rows=c4_val, c5_grid_rows=c5_val, gate=gate)
    anchor_test = combine_h2_3_rows(c3_mlp_rows=c3_test, c4_ridge_rows=c4_test, c5_grid_rows=c5_test, gate=gate)

    regfeat_rows = add_group_features(rows, "A3_rich_plus_reg_feat")
    regfeat_calibration = [row for row in regfeat_rows if row["split"] == "calibration"]
    regfeat_test = [row for row in regfeat_rows if row["split"] == "test"]
    regfeat_feature_names = sorted(regfeat_calibration[0]["feature_dict"].keys())
    reg_val, reg_test, reg_audit = fit_ridge_family(
        regfeat_calibration,
        regfeat_test,
        target_clients,
        regfeat_feature_names,
        ridge_alphas,
        args.val_ratio,
        "regfeat_ridge",
    )
    reg_val = apply_c4_rescue(reg_val, "regfeat_ridge_ppm", "regfeat_ridge_plus_c4_rescue_ppm", gate)
    reg_test = apply_c4_rescue(reg_test, "regfeat_ridge_ppm", "regfeat_ridge_plus_c4_rescue_ppm", gate)

    val_merged = merge_prediction_sets(
        anchor_val,
        reg_val,
        ["regfeat_ridge_ppm", "regfeat_ridge_plus_c4_rescue_ppm"],
    )
    test_merged = merge_prediction_sets(
        anchor_test,
        reg_test,
        ["regfeat_ridge_ppm", "regfeat_ridge_plus_c4_rescue_ppm"],
    )

    selected_weights, selection_rows = select_client_blend_weights(
        val_merged,
        target_clients,
        blend_weights,
        anchor_key="h2_3_current_ppm",
        candidate_key="regfeat_ridge_plus_c4_rescue_ppm",
        max_nonco_delta=args.max_nonco_delta,
        min_all_delta=args.min_all_delta,
    )
    val_blended = apply_client_blends(
        val_merged,
        selected_weights,
        anchor_key="h2_3_current_ppm",
        candidate_key="regfeat_ridge_plus_c4_rescue_ppm",
        output_key="h2_3_plus_blend_ppm",
    )
    test_blended = apply_client_blends(
        test_merged,
        selected_weights,
        anchor_key="h2_3_current_ppm",
        candidate_key="regfeat_ridge_plus_c4_rescue_ppm",
        output_key="h2_3_plus_blend_ppm",
    )

    summary_rows: list[dict[str, Any]] = []
    for pred_key, mode in [
        ("baseline_final_ppm", "A0_baseline_final"),
        ("h2_3_direct_only_ppm", "H2_3_direct_only_r25_refit"),
        ("h2_3_current_ppm", "H2_3_current_r25_refit_anchor"),
        ("regfeat_ridge_plus_c4_rescue_ppm", "H2_3_plus_reg_feat_ridge_rescue"),
        ("h2_3_plus_blend_ppm", "H2_3_plus_blend_r25"),
    ]:
        summary_rows.extend(summarize(test_blended, pred_key, mode, "test"))

    reference_rows = load_reference_rows(args.reference_h2_3_predictions, test_blended)
    if reference_rows:
        summary_rows.extend(summarize(reference_rows, "reference_h2_3_current_ppm", "H2_3_current_r25_reference", "test"))

    per_client_rows = client_nrmse_rows(summary_rows, target_clients)
    reading = build_reading(summary_rows, per_client_rows)
    fit_audit = [*c3_audit, *c4_audit, *c5_audit, *reg_audit]

    write_csv(out_dir / "fusion_profile_predictions.csv", test_blended)
    write_csv(out_dir / "fusion_profile_validation_predictions.csv", val_blended)
    write_csv(out_dir / "fusion_profile_summary.csv", summary_rows)
    write_csv(out_dir / "fusion_profile_per_client.csv", per_client_rows)
    write_csv(out_dir / "fusion_profile_selection.csv", selection_rows)
    write_csv(out_dir / "fusion_profile_fit_audit.csv", fit_audit)
    write_report(out_dir, summary_rows, per_client_rows, selection_rows, reading)
    manifest = {
        "data_root": str(data_root),
        "target_predictions": args.target_predictions,
        "backbone_calibration": args.backbone_calibration,
        "backbone_test": args.backbone_test,
        "route_rescue_artifact": args.route_rescue_artifact,
        "reference_h2_3_predictions": args.reference_h2_3_predictions,
        "target_clients": target_clients,
        "ridge_alphas": ridge_alphas,
        "mlp_alphas": mlp_alphas,
        "hidden_grid": [list(hidden) for hidden in hiddens],
        "c5_grid_alphas": c5_grid_alphas,
        "c5_hidden_grid": [list(hidden) for hidden in c5_hiddens],
        "blend_weights": blend_weights,
        "selected_weights": selected_weights,
        "val_ratio": args.val_ratio,
        "max_nonco_delta": args.max_nonco_delta,
        "min_all_delta": args.min_all_delta,
        "seed": args.seed,
        "reading": reading,
        "outputs": [
            "fusion_profile_predictions.csv",
            "fusion_profile_validation_predictions.csv",
            "fusion_profile_summary.csv",
            "fusion_profile_per_client.csv",
            "fusion_profile_selection.csv",
            "fusion_profile_fit_audit.csv",
            "fusion_profile_report.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "selected_weights": selected_weights, "reading": reading}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/f6_c12_c345_strong_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--backbone-calibration", default="results/f6_r25_backbone_feature_export_20260630/backbone_features_calibration.csv")
    parser.add_argument("--backbone-test", default="results/f6_r25_backbone_feature_export_20260630/backbone_features_test.csv")
    parser.add_argument("--route-rescue-artifact", default="results/deployment_candidates_20260624/c12_c345_a1_rich_residual_plus_c4_rescue.json")
    parser.add_argument("--reference-h2-3-predictions", default="results/f6_h2_3_no_b0_feature_ablation_20260629/c12_c345/h2_3_no_b0_feature_ablation_predictions.csv")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-alphas", default="0.001,0.01")
    parser.add_argument("--hidden-grid", default="32")
    parser.add_argument("--c5-grid-alphas", default="0.001,0.01,0.1,1")
    parser.add_argument("--c5-hidden-grid", default="16;32;64;32,16")
    parser.add_argument("--blend-weights", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--max-nonco-delta", type=float, default=1.0)
    parser.add_argument("--min-all-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/h2_3_plus_fusion_profile_20260630/r25_balanced")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
