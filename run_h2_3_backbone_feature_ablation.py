"""Run H2.3 feature-fusion ablations with exported backbone features.

This script answers the first regression-aware fusion question: does the
official F6 r25 classification backbone provide useful direct-head regression
features beyond the existing rich response statistics and B0/source priors?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from run_formal_target_ridge_auto_v2_eval import fit_client_models, refit_full_calibration
from run_regression_head_ablation import (
    CO_CLASS,
    add_target_features,
    apply_client_models,
    client_name,
    client_num,
    fnum,
    inum,
    read_csv,
    summarize,
    write_csv,
)


FEATURE_GROUP_ORDER = [
    "A0_rich_only",
    "A1_rich_plus_confidence",
    "A2_rich_plus_cls_feat",
    "A3_rich_plus_reg_feat",
    "A4_rich_plus_b0",
    "A5_rich_plus_source_priors",
    "A6_rich_plus_all_backbone",
    "A7_rich_plus_all_priors",
]

B0_PRIOR_KEYS = [
    "final_ppm",
    "baseline_final_ppm",
    "base_r3ak16_raw_ppm",
]

SOURCE_PRIOR_KEYS = [
    "pred_ppm",
    "calibrated_ppm",
    "routed_pred_ppm",
    "raw_ppm",
    "auto_v2_ppm",
]


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        client_name(row.get("client") or row.get("client_id")),
        str(row.get("split")),
        inum(row.get("sample_index")),
    )


def numeric_items(row: dict[str, Any], keys: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        out[key] = fnum(row.get(key), 0.0)
    return out


def merge_backbone_features(
    rows: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    *,
    require_all: bool = True,
) -> list[dict[str, Any]]:
    features_by_key: dict[tuple[str, str, int], dict[str, float]] = {}
    for feature_row in feature_rows:
        key = row_key(feature_row)
        values: dict[str, float] = {}
        for name, value in feature_row.items():
            if name in {"client", "client_id", "split", "sample_index"}:
                continue
            values[name] = fnum(value, 0.0)
        features_by_key[key] = values

    merged: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []
    for row in rows:
        key = row_key(row)
        item = dict(row)
        item["backbone_feature_dict"] = dict(features_by_key.get(key, {}))
        if require_all and not item["backbone_feature_dict"]:
            missing.append(key)
        merged.append(item)
    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise ValueError(f"Missing backbone features for {len(missing)} rows; first keys: {preview}")
    return merged


def confidence_features(backbone: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in backbone.items():
        if key in {"confidence", "margin", "entropy"} or key.startswith("prob_"):
            out[key] = fnum(value, 0.0)
        if key.startswith("pred_class_"):
            pred_class = inum(value)
            out[key] = float(pred_class)
            for cls_id in range(4):
                out[f"{key}_is_{cls_id}"] = float(pred_class == cls_id)
    return out


def prefixed_features(source: dict[str, float], prefixes: Sequence[str]) -> dict[str, float]:
    return {
        key: fnum(value, 0.0)
        for key, value in source.items()
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def build_feature_groups(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    rich = {str(key): fnum(value, 0.0) for key, value in dict(row.get("feature_dict", {})).items()}
    backbone = {str(key): fnum(value, 0.0) for key, value in dict(row.get("backbone_feature_dict", {})).items()}
    conf = confidence_features(backbone)
    cls_feat = prefixed_features(backbone, ["cls_feat_"])
    reg_feat = prefixed_features(backbone, ["reg_feat_"])
    b0 = numeric_items(row, B0_PRIOR_KEYS)
    source = numeric_items(row, SOURCE_PRIOR_KEYS)

    all_backbone = {**conf, **cls_feat, **reg_feat}
    all_priors = {**all_backbone, **b0, **source}
    return {
        "A0_rich_only": dict(rich),
        "A1_rich_plus_confidence": {**rich, **conf},
        "A2_rich_plus_cls_feat": {**rich, **cls_feat},
        "A3_rich_plus_reg_feat": {**rich, **reg_feat},
        "A4_rich_plus_b0": {**rich, **b0},
        "A5_rich_plus_source_priors": {**rich, **source},
        "A6_rich_plus_all_backbone": {**rich, **all_backbone},
        "A7_rich_plus_all_priors": {**rich, **all_priors},
    }


def c5_nonco_wrong_route_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    c5_nonco = [
        row
        for row in rows
        if client_name(row.get("client") or row.get("client_id")) == "C5"
        and inum(row.get("true_class")) != CO_CLASS
    ]
    wrong_as_co = [row for row in c5_nonco if inum(row.get("pred_class")) == CO_CLASS]
    total = len(c5_nonco)
    return {
        "C5_nonCO_N": total,
        "C5_nonCO_pred_CO_N": len(wrong_as_co),
        "C5_nonCO_pred_CO_rate": float(len(wrong_as_co) / total) if total else 0.0,
    }


def parse_clients(text: str) -> list[str]:
    return [client_name(item.strip()) for item in text.split(",") if item.strip()]


def load_feature_rows(*paths: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_csv(path))
    return rows


def add_group_features(rows: Sequence[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        groups = build_feature_groups(row)
        item = dict(row)
        item["feature_dict"] = groups[group_name]
        item["feature_group"] = group_name
        out.append(item)
    return out


def client_nrmse_rows(summary_rows: Sequence[dict[str, Any]], target_clients: Sequence[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    modes = [mode for mode in FEATURE_GROUP_ORDER if any(row["mode"] == mode for row in summary_rows)]
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


def classify_reading(
    summary_rows: Sequence[dict[str, Any]],
    per_client_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    base = "A0_rich_only"
    base_all = find_metric(summary_rows, base, "ALL", "NRMSE")
    base_macro = find_macro(per_client_rows, base, "macro_client_NRMSE")
    base_nonco = find_metric(summary_rows, base, "nonCO_ALL", "RMSE")
    embedding_modes = ["A2_rich_plus_cls_feat", "A3_rich_plus_reg_feat", "A6_rich_plus_all_backbone", "A7_rich_plus_all_priors"]
    prior_modes = ["A4_rich_plus_b0", "A5_rich_plus_source_priors", "A7_rich_plus_all_priors"]

    def improves(mode: str) -> bool:
        all_nrmse = find_metric(summary_rows, mode, "ALL", "NRMSE")
        macro_nrmse = find_macro(per_client_rows, mode, "macro_client_NRMSE")
        nonco_rmse = find_metric(summary_rows, mode, "nonCO_ALL", "RMSE")
        return (
            (np.isfinite(all_nrmse) and all_nrmse < base_all)
            or (np.isfinite(macro_nrmse) and macro_nrmse < base_macro)
        ) and (not np.isfinite(nonco_rmse) or nonco_rmse <= base_nonco + 1.0)

    backbone_positive = [mode for mode in embedding_modes if improves(mode)]
    prior_positive = [mode for mode in prior_modes if improves(mode)]
    if backbone_positive:
        label = "backbone-positive"
    elif prior_positive:
        label = "prior-positive"
    else:
        label = "negative"
    return {
        "classification": label,
        "backbone_positive_modes": backbone_positive,
        "prior_positive_modes": prior_positive,
        "base_ALL_NRMSE": base_all,
        "base_macro_client_NRMSE": base_macro,
        "base_nonCO_RMSE": base_nonco,
    }


def format_float(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return ""
    value_f = fnum(value)
    return "" if not np.isfinite(value_f) else f"{value_f:.{digits}f}"


def write_report(
    path: Path,
    summary_rows: Sequence[dict[str, Any]],
    per_client_rows: Sequence[dict[str, Any]],
    wrong_route: dict[str, Any],
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
        "# H2.3 Backbone Feature Ablation",
        "",
        "Official line: F6 r25 final adapted checkpoint. No test metrics are used for feature selection.",
        "",
        "## Test RMSE",
        "",
        "| mode | " + " | ".join(scopes) + " | macro-client NRMSE |",
        "|---|" + "|".join(["---:"] * (len(scopes) + 1)) + "|",
    ]
    for mode in FEATURE_GROUP_ORDER:
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
    by_mode = {row["mode"]: row for row in per_client_rows}
    for mode in FEATURE_GROUP_ORDER:
        row = by_mode.get(mode, {})
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
            "## C5 nonCO Wrong-Route Audit",
            "",
            f"- C5 nonCO N: {wrong_route['C5_nonCO_N']}",
            f"- C5 nonCO predicted as CO N: {wrong_route['C5_nonCO_pred_CO_N']}",
            f"- C5 nonCO predicted as CO rate: {wrong_route['C5_nonCO_pred_CO_rate']:.4f}",
            "",
            "## Reading",
            "",
            f"- Classification: `{reading['classification']}`",
            f"- Backbone-positive modes: {reading['backbone_positive_modes']}",
            f"- Prior-positive modes: {reading['prior_positive_modes']}",
            "- If classification is `backbone-positive` or `prior-positive`, proceed to H2.3+ fusion profile.",
            "- If classification is `negative`, keep current H2.3 and defer regression-aware encoder work.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clients = parse_clients(args.target_clients)
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    feature_rows = load_feature_rows(args.backbone_calibration, args.backbone_test)

    raw_rows = [
        row
        for row in read_csv(args.target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in set(target_clients)
    ]
    rows = add_target_features(raw_rows, data_root)
    for row in rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["route_class"] = inum(row.get("pred_class"))
    rows = merge_backbone_features(rows, feature_rows)

    all_summary_rows: list[dict[str, Any]] = []
    all_fit_rows: list[dict[str, Any]] = []
    prediction_rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    test_rows_base = [row for row in rows if row["split"] == "test"]

    for group_name in FEATURE_GROUP_ORDER:
        group_rows = add_group_features(rows, group_name)
        calibration_rows = [row for row in group_rows if row["split"] == "calibration"]
        test_rows = [row for row in group_rows if row["split"] == "test"]
        feature_names = sorted(calibration_rows[0]["feature_dict"].keys())
        _pre_models, fit_audit, _val_rows = fit_client_models(
            calibration_rows,
            target_clients,
            feature_names,
            alphas,
            args.val_ratio,
        )
        selected_alphas = {(row["client"], int(row["class_id"])): fnum(row["best_alpha"]) for row in fit_audit}
        full_models = refit_full_calibration(calibration_rows, target_clients, feature_names, selected_alphas)
        pred_rows = apply_client_models(test_rows, full_models, group_name)
        pred_key = f"{group_name}_ppm"
        all_summary_rows.extend(summarize(pred_rows, pred_key, group_name, "test"))
        for fit_row in fit_audit:
            item = dict(fit_row)
            item["mode"] = group_name
            item["feature_count"] = len(feature_names)
            all_fit_rows.append(item)
        for row in pred_rows:
            key = row_key(row)
            target = prediction_rows_by_key.setdefault(
                key,
                {k: v for k, v in row.items() if k not in {"feature_dict", "backbone_feature_dict"}},
            )
            target[pred_key] = fnum(row.get(pred_key))

    per_client_rows = client_nrmse_rows(all_summary_rows, target_clients)
    wrong_route = c5_nonco_wrong_route_audit(test_rows_base)
    reading = classify_reading(all_summary_rows, per_client_rows)
    write_csv(out_dir / "feature_ablation_predictions.csv", prediction_rows_by_key.values())
    write_csv(out_dir / "feature_ablation_summary.csv", all_summary_rows)
    write_csv(out_dir / "feature_ablation_per_client.csv", per_client_rows)
    write_csv(out_dir / "feature_ablation_fit_audit.csv", all_fit_rows)
    write_csv(out_dir / "feature_ablation_wrong_route_audit.csv", [wrong_route])
    write_report(out_dir / "feature_ablation_report.md", all_summary_rows, per_client_rows, wrong_route, reading)

    manifest = {
        "data_root": str(data_root),
        "target_predictions": str(args.target_predictions),
        "backbone_calibration": str(args.backbone_calibration),
        "backbone_test": str(args.backbone_test),
        "target_clients": target_clients,
        "feature_groups": FEATURE_GROUP_ORDER,
        "alphas": alphas,
        "val_ratio": float(args.val_ratio),
        "reading": reading,
        "wrong_route_audit": wrong_route,
        "outputs": [
            "feature_ablation_predictions.csv",
            "feature_ablation_summary.csv",
            "feature_ablation_per_client.csv",
            "feature_ablation_fit_audit.csv",
            "feature_ablation_wrong_route_audit.csv",
            "feature_ablation_report.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "reading": reading, "wrong_route_audit": wrong_route}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", required=True)
    parser.add_argument("--backbone-calibration", required=True)
    parser.add_argument("--backbone-test", required=True)
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", default="results/h2_3_backbone_feature_ablation_20260630/r25")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
