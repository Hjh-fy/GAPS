"""Fit the C5-only H2.3 anchor and H2.3+ blend without C4 rescue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_formal_target_ridge_auto_v2_eval import attach_response_phase
from run_h2_3_backbone_feature_ablation import load_feature_rows, merge_backbone_features
from run_h2_3_plus_fusion_profile import (
    add_group_features,
    apply_client_blends,
    fit_mlp_family,
    fit_ridge_family,
    merge_prediction_sets,
    parse_float_grid,
    parse_hidden_grid,
    row_key,
    select_client_blend_weights,
)
from run_regression_head_ablation import (
    add_target_features,
    client_name,
    fnum,
    inum,
    read_csv,
    summarize,
    write_csv,
)


EXPECTED_COUNTS = {"calibration": 320, "test": 1360}
TARGET_CLIENTS = ("C5",)


def build_c5_anchor_rows(
    rows: Sequence[dict[str, Any]],
    source_key: str = "h2_c5_grid_mlp_ppm",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if client_name(row.get("client") or row.get("client_id")) != "C5":
            raise ValueError("C5 H2.3 anchor received a non-C5 row")
        if source_key not in row:
            raise KeyError(f"missing C5 anchor prediction: {source_key}")
        item = {key: value for key, value in row.items() if key != "feature_dict"}
        item["h23_anchor_ppm"] = fnum(item[source_key])
        item["h2_3_direct_only_ppm"] = item["h23_anchor_ppm"]
        item["h2_3_current_ppm"] = item["h23_anchor_ppm"]
        output.append(item)
    return output


def validate_c5_rows(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    keys: set[tuple[str, str, int]] = set()
    counts = {"calibration": 0, "test": 0}
    for row in rows:
        if client_name(row.get("client") or row.get("client_id")) != "C5":
            raise ValueError("primary H2.3+ rows must be C5 only")
        split = str(row.get("split"))
        if split not in counts:
            raise ValueError(f"unexpected split: {split}")
        key = row_key(row)
        if key in keys:
            raise ValueError(f"duplicate row: {key}")
        keys.add(key)
        counts[split] += 1
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"expected C5 counts {EXPECTED_COUNTS}; got {counts}")
    return counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ridge_alphas = parse_float_grid(args.ridge_alphas)
    mlp_alphas = parse_float_grid(args.mlp_alphas)
    hidden_grid = parse_hidden_grid(args.hidden_grid)
    blend_weights = parse_float_grid(args.blend_weights)

    raw_rows = read_csv(args.target_predictions)
    filtered = [
        row
        for row in raw_rows
        if client_name(row.get("client") or row.get("client_id")) == "C5"
    ]
    if len(filtered) != len(raw_rows):
        raise ValueError("target prediction input contains non-C5 rows")
    counts = validate_c5_rows(filtered)
    rows = attach_response_phase(add_target_features(filtered, data_root), data_root)
    for row in rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
        row["route_class"] = inum(row.get("pred_class"))

    feature_rows = load_feature_rows(args.backbone_calibration, args.backbone_test)
    rows = merge_backbone_features(rows, feature_rows)
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    test_rows = [row for row in rows if row["split"] == "test"]
    rich_feature_names = sorted(calibration_rows[0]["feature_dict"].keys())

    mlp_val, mlp_test, mlp_audit = fit_mlp_family(
        calibration_rows,
        test_rows,
        TARGET_CLIENTS,
        rich_feature_names,
        hidden_grid,
        mlp_alphas,
        args.val_ratio,
        args.seed,
        "h2_c5_grid_mlp",
    )
    anchor_val = build_c5_anchor_rows(mlp_val)
    anchor_test = build_c5_anchor_rows(mlp_test)

    regfeat_rows = add_group_features(rows, "A3_rich_plus_reg_feat")
    regfeat_calibration = [row for row in regfeat_rows if row["split"] == "calibration"]
    regfeat_test = [row for row in regfeat_rows if row["split"] == "test"]
    regfeat_names = sorted(regfeat_calibration[0]["feature_dict"].keys())
    ridge_val, ridge_test, ridge_audit = fit_ridge_family(
        regfeat_calibration,
        regfeat_test,
        TARGET_CLIENTS,
        regfeat_names,
        ridge_alphas,
        args.val_ratio,
        "regfeat_ridge",
    )

    val_merged = merge_prediction_sets(anchor_val, ridge_val, ["regfeat_ridge_ppm"])
    test_merged = merge_prediction_sets(anchor_test, ridge_test, ["regfeat_ridge_ppm"])
    selected_weights, selection_rows = select_client_blend_weights(
        val_merged,
        TARGET_CLIENTS,
        blend_weights,
        anchor_key="h2_3_current_ppm",
        candidate_key="regfeat_ridge_ppm",
        max_nonco_delta=args.max_nonco_delta,
        min_all_delta=args.min_all_delta,
    )
    val_blended = apply_client_blends(
        val_merged,
        selected_weights,
        anchor_key="h2_3_current_ppm",
        candidate_key="regfeat_ridge_ppm",
        output_key="h2_3_plus_blend_ppm",
    )
    test_blended = apply_client_blends(
        test_merged,
        selected_weights,
        anchor_key="h2_3_current_ppm",
        candidate_key="regfeat_ridge_ppm",
        output_key="h2_3_plus_blend_ppm",
    )
    for row in (*val_blended, *test_blended):
        row["h23_weak_ridge_ppm"] = fnum(row.get("regfeat_ridge_ppm"))
        row["h23_plus_ppm"] = fnum(row.get("h2_3_plus_blend_ppm"))
        row["c4_rescue_applied"] = 0

    summary_rows: list[dict[str, Any]] = []
    for pred_key, mode in (
        ("baseline_final_ppm", "R0_source_R3aK16_reference"),
        ("h23_anchor_ppm", "R2_C5_H2.3_MLP_anchor"),
        ("h23_weak_ridge_ppm", "R3_component_C5_regfeat_Ridge"),
        ("h23_plus_ppm", "R3_C5_H2.3_plus"),
    ):
        summary_rows.extend(summarize(test_blended, pred_key, mode, "test"))

    write_csv(output_dir / "c5_h23_plus_test_predictions.csv", test_blended)
    write_csv(output_dir / "c5_h23_plus_validation_predictions.csv", val_blended)
    write_csv(output_dir / "c5_h23_plus_summary.csv", summary_rows)
    write_csv(output_dir / "c5_h23_plus_selection.csv", selection_rows)
    write_csv(output_dir / "c5_h23_plus_fit_audit.csv", [*mlp_audit, *ridge_audit])
    manifest = {
        "schema_version": 1,
        "protocol": {"source_clients": [1, 2], "target_clients": [5]},
        "data_root": str(data_root),
        "target_predictions": args.target_predictions,
        "backbone_calibration": args.backbone_calibration,
        "backbone_test": args.backbone_test,
        "counts": counts,
        "ridge_alphas": ridge_alphas,
        "mlp_alphas": mlp_alphas,
        "hidden_grid": [list(hidden) for hidden in hidden_grid],
        "blend_weights": blend_weights,
        "selected_weights": selected_weights,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "c4_rescue_enabled": False,
        "selection_uses_test_labels": False,
        "outputs": {
            "test_predictions": str(output_dir / "c5_h23_plus_test_predictions.csv"),
            "validation_predictions": str(output_dir / "c5_h23_plus_validation_predictions.csv"),
            "summary": str(output_dir / "c5_h23_plus_summary.csv"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", required=True)
    parser.add_argument("--backbone-calibration", required=True)
    parser.add_argument("--backbone-test", required=True)
    parser.add_argument("--ridge-alphas", default="0,0.01,0.1,1,10,100,1000")
    parser.add_argument("--mlp-alphas", default="0.001,0.01,0.1,1")
    parser.add_argument("--hidden-grid", default="16;32;64;32,16")
    parser.add_argument("--blend-weights", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--max-nonco-delta", type=float, default=1.0)
    parser.add_argument("--min-all-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
