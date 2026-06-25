"""L2 unified selector for lightweight and target-direct regression heads.

This is a selection-only experiment. It does not retrain any head. It combines
existing calibration-validation scores and test predictions from:

- B0 baseline final_ppm;
- target Ridge direct head;
- target shallow MLP direct head;
- L1 source-lightweight + full residual auto_v2 heads.

The selector chooses one candidate per target client and gas using
calibration-validation RMSE only, then applies that profile to target test rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_regression_head_ablation import (
    CLASS_NAMES,
    add_target_features,
    client_name,
    deterministic_train_val,
    fnum,
    inum,
    metrics,
    read_csv,
    summarize,
    write_csv,
)


KEY_FIELDS = ("client", "split", "sample_index")
SCOPES = [
    "ALL",
    "C3-CO",
    "C4-CO",
    "C5-CO",
    "C3-CO_high_200_250",
    "C4-CO_high_200_250",
    "C5-CO_high_200_250",
    "nonCO_ALL",
]


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def load_target_rows(target_predictions: Path, data_root: Path, target_clients: set[str]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_csv(target_predictions)
        if client_name(row.get("client") or row.get("client_id")) in target_clients
    ]
    return add_target_features(rows, data_root)


def baseline_val_scores(target_rows: list[dict[str, Any]], val_ratio: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    calibration_rows = [dict(row) for row in target_rows if row["split"] == "calibration"]
    for row in calibration_rows:
        row["route_class"] = row["true_class"]
    clients = sorted({str(row["client"]) for row in calibration_rows})
    for client in clients:
        for cls_id, gas in CLASS_NAMES.items():
            cls_rows = [
                row
                for row in calibration_rows
                if str(row["client"]) == client and inum(row["true_class"]) == cls_id
            ]
            if not cls_rows:
                continue
            _fit_rows, val_rows = deterministic_train_val(cls_rows, val_ratio=val_ratio)
            score = metrics(val_rows, "final_ppm").get("RMSE")
            out.append(
                {
                    "candidate": "baseline_final_ppm",
                    "client": client,
                    "class_id": cls_id,
                    "gas": gas,
                    "val_RMSE": fnum(score),
                    "n_val": len(val_rows),
                    "source": "target_predictions_calibration",
                }
            )
    return out


def direct_head_val_scores(path: Path, candidate: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    out: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        out.append(
            {
                "candidate": candidate,
                "client": str(row["client"]),
                "class_id": int(row["class_id"]),
                "gas": str(row["gas"]),
                "val_RMSE": fnum(row["best_val_RMSE"]),
                "n_val": int(row["val_N"]),
                "source": str(path),
            }
        )
    return out


def l1_val_scores(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    name_map = {
        "source_ridge": "l1_source_ridge_full_auto_v2",
        "source_per_gas_mlp": "l1_source_per_gas_mlp_full_auto_v2",
        "source_shared_mlp": "l1_source_shared_mlp_full_auto_v2",
    }
    out: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        out.append(
            {
                "candidate": name_map[str(row["base_mode"])],
                "client": str(row["client"]),
                "class_id": int(row["class_id"]),
                "gas": str(row["gas"]),
                "val_RMSE": fnum(row["selected_val_RMSE"]),
                "n_val": int(row["n_val"]),
                "source": str(path),
                "selected_l1_mode": row.get("selected_mode", ""),
                "selected_l1_alpha": row.get("selected_alpha", ""),
            }
        )
    return out


def select_profile(val_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in val_rows:
        groups.setdefault((str(row["client"]), int(row["class_id"])), []).append(row)
    profile: list[dict[str, Any]] = []
    for (client, cls_id), rows in sorted(groups.items()):
        best = min(rows, key=lambda row: fnum(row.get("val_RMSE"), float("inf")))
        item = dict(best)
        item["selected_candidate"] = item.pop("candidate")
        item["all_candidates_json"] = json.dumps(
            sorted(
                [
                    {
                        "candidate": row["candidate"],
                        "val_RMSE": fnum(row["val_RMSE"]),
                        "n_val": int(row["n_val"]),
                    }
                    for row in rows
                ],
                key=lambda row: row["val_RMSE"],
            ),
            ensure_ascii=False,
        )
        profile.append(item)
    return profile


def select_client_profile(val_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in val_rows:
        groups.setdefault((str(row["client"]), str(row["candidate"])), []).append(row)

    by_client: dict[str, list[dict[str, Any]]] = {}
    for (client, candidate), rows in groups.items():
        total_n = sum(int(row["n_val"]) for row in rows)
        if total_n <= 0:
            continue
        rmse = (
            sum(int(row["n_val"]) * fnum(row["val_RMSE"]) ** 2 for row in rows)
            / total_n
        ) ** 0.5
        by_client.setdefault(client, []).append(
            {
                "client": client,
                "candidate": candidate,
                "client_val_RMSE": rmse,
                "n_val": total_n,
                "all_candidates_json": json.dumps(
                    sorted(
                        [
                            {
                                "gas": row["gas"],
                                "class_id": int(row["class_id"]),
                                "val_RMSE": fnum(row["val_RMSE"]),
                                "n_val": int(row["n_val"]),
                            }
                            for row in rows
                        ],
                        key=lambda row: row["class_id"],
                    ),
                    ensure_ascii=False,
                ),
            }
        )

    profile: list[dict[str, Any]] = []
    for client, rows in sorted(by_client.items()):
        best = min(rows, key=lambda row: fnum(row["client_val_RMSE"], float("inf")))
        item = dict(best)
        item["selected_candidate"] = item.pop("candidate")
        item["candidate_ranking_json"] = json.dumps(
            sorted(
                [
                    {
                        "candidate": row["candidate"],
                        "client_val_RMSE": fnum(row["client_val_RMSE"]),
                        "n_val": int(row["n_val"]),
                    }
                    for row in rows
                ],
                key=lambda row: row["client_val_RMSE"],
            ),
            ensure_ascii=False,
        )
        profile.append(item)
    return profile


def load_test_predictions(
    target_rows: list[dict[str, Any]],
    ridge_path: Path,
    mlp_path: Path,
    l1_path: Path,
) -> dict[str, dict[tuple[str, str, str], float]]:
    test_base = [dict(row) for row in target_rows if row["split"] == "test"]
    predictions: dict[str, dict[tuple[str, str, str], float]] = {
        "baseline_final_ppm": {row_key(row): fnum(row.get("final_ppm")) for row in test_base},
    }

    ridge = pd.read_csv(ridge_path)
    predictions["target_ridge_direct"] = {
        row_key(row): fnum(row.get("ridge_direct_ppm"))
        for row in ridge.to_dict("records")
    }
    mlp = pd.read_csv(mlp_path)
    predictions["target_mlp_direct"] = {
        row_key(row): fnum(row.get("mlp_direct_ppm"))
        for row in mlp.to_dict("records")
    }

    l1 = pd.read_csv(l1_path)
    l1_map = {
        "source_ridge_val_selected": "l1_source_ridge_full_auto_v2",
        "source_per_gas_mlp_val_selected": "l1_source_per_gas_mlp_full_auto_v2",
        "source_shared_mlp_val_selected": "l1_source_shared_mlp_full_auto_v2",
    }
    for source_candidate, out_candidate in l1_map.items():
        sub = l1[l1["candidate"] == source_candidate]
        predictions[out_candidate] = {
            row_key(row): fnum(row.get("corrected_ppm"))
            for row in sub.to_dict("records")
        }
    return predictions


def build_candidate_rows(
    target_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    prediction_maps: dict[str, dict[tuple[str, str, str], float]],
) -> list[dict[str, Any]]:
    profile = {
        (str(row["client"]), int(row["class_id"])): str(row["selected_candidate"])
        for row in profile_rows
    }
    out: list[dict[str, Any]] = []
    for row in target_rows:
        if row["split"] != "test":
            continue
        item = dict(row)
        cls_id = inum(item.get("route_class"))
        candidate = profile.get((str(item["client"]), cls_id), "baseline_final_ppm")
        key = row_key(item)
        value = prediction_maps.get(candidate, prediction_maps["baseline_final_ppm"]).get(key)
        if value is None:
            value = prediction_maps["baseline_final_ppm"].get(key, fnum(item.get("final_ppm")))
            item["l2_fallback_used"] = 1
        else:
            item["l2_fallback_used"] = 0
        item["candidate"] = "L2_unified_val_selector"
        item["selected_candidate"] = candidate
        item["corrected_ppm"] = fnum(value)
        out.append(item)
    return out


def build_client_candidate_rows(
    target_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    prediction_maps: dict[str, dict[tuple[str, str, str], float]],
) -> list[dict[str, Any]]:
    profile = {str(row["client"]): str(row["selected_candidate"]) for row in profile_rows}
    out: list[dict[str, Any]] = []
    for row in target_rows:
        if row["split"] != "test":
            continue
        item = dict(row)
        candidate = profile.get(str(item["client"]), "baseline_final_ppm")
        key = row_key(item)
        value = prediction_maps.get(candidate, prediction_maps["baseline_final_ppm"]).get(key)
        if value is None:
            value = prediction_maps["baseline_final_ppm"].get(key, fnum(item.get("final_ppm")))
            item["l2_fallback_used"] = 1
        else:
            item["l2_fallback_used"] = 0
        item["candidate"] = "L2_client_val_selector"
        item["selected_candidate"] = candidate
        item["corrected_ppm"] = fnum(value)
        out.append(item)
    return out


def selected_table(summary_rows: list[dict[str, Any]], modes: list[str]) -> str:
    by_mode_scope = {(row["mode"], row["scope"]): row for row in summary_rows}
    lines = ["| mode | " + " | ".join(SCOPES) + " |", "|---|" + "|".join(["---:"] * len(SCOPES)) + "|"]
    for mode in modes:
        values = [mode]
        for scope in SCOPES:
            row = by_mode_scope.get((mode, scope), {})
            val = row.get("RMSE")
            values.append("" if val is None else f"{fnum(val):.2f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    out: Path,
    summary_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    client_profile_rows: list[dict[str, Any]],
) -> None:
    modes = [
        "baseline_final_ppm",
        "target_ridge_direct",
        "target_mlp_direct",
        "l1_source_ridge_full_auto_v2",
        "l1_source_per_gas_mlp_full_auto_v2",
        "l1_source_shared_mlp_full_auto_v2",
        "L2_unified_val_selector",
        "L2_client_val_selector",
    ]
    counts = pd.Series([row["selected_candidate"] for row in profile_rows]).value_counts().to_dict()
    count_lines = ["| selected candidate | count |", "|---|---:|"]
    for key, value in counts.items():
        count_lines.append(f"| {key} | {value} |")

    profile_lines = ["| client | gas | selected | val RMSE |", "|---|---|---|---:|"]
    for row in profile_rows:
        profile_lines.append(
            f"| {row['client']} | {row['gas']} | {row['selected_candidate']} | {fnum(row['val_RMSE']):.2f} |"
        )

    client_lines = ["| client | selected | client val RMSE |", "|---|---|---:|"]
    for row in client_profile_rows:
        client_lines.append(
            f"| {row['client']} | {row['selected_candidate']} | {fnum(row['client_val_RMSE']):.2f} |"
        )

    text = f"""# L2 Unified Lightweight Selector

Scope: C12 -> C345 target test, no-QC full-set.

This experiment selects one candidate per target client/gas using calibration-validation RMSE only.

## Test RMSE

{selected_table(summary_rows, modes)}

## Selection Counts

{chr(10).join(count_lines)}

## Selected Profile

{chr(10).join(profile_lines)}

## Client-Level Conservative Profile

{chr(10).join(client_lines)}

## Reading

- If L2 selects lightweight candidates frequently and improves test metrics, lightweight source heads provide useful target signal beyond direct target heads.
- If L2 mostly selects target direct heads, lightweight heads remain deployment-lite candidates rather than performance-mainline candidates.
- `L2_unified_val_selector` selects per client/gas and may overfit small calibration-validation cells.
- `L2_client_val_selector` is the conservative variant: one candidate per client based on aggregate client-level validation RMSE.
- Test metrics here do not include C4 route rescue or H8 CO-specialist switching; compare against H2.3/H8 mainline reports separately.
"""
    (out / "l2_unified_selector_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L2 unified lightweight selector.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--target-predictions", default="results/timeaware_2080_c12src_c345tgt_fixed_da_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv")
    parser.add_argument("--target-clients", default="3,4,5")
    parser.add_argument("--ridge-audit", default="results/formal_target_ridge_auto_v2_20260624/formal_target_ridge_fit_audit.csv")
    parser.add_argument("--ridge-predictions", default="results/formal_target_ridge_auto_v2_20260624/formal_target_ridge_predictions.csv")
    parser.add_argument("--mlp-audit", default="results/formal_target_mlp_auto_v2_20260624/formal_target_mlp_fit_audit.csv")
    parser.add_argument("--mlp-predictions", default="results/formal_target_mlp_auto_v2_20260624/formal_target_mlp_predictions.csv")
    parser.add_argument("--l1-selection", default="results/source_lightweight_full_auto_v2_20260625_fair/source_lightweight_full_auto_v2_selection.csv")
    parser.add_argument("--l1-predictions", default="results/source_lightweight_full_auto_v2_20260625_fair/source_lightweight_full_auto_v2_predictions.csv")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", default="results/lightweight_l2_unified_selector_20260625")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_clients = {client_name(item.strip()) for item in args.target_clients.split(",") if item.strip()}
    target_rows = load_target_rows(Path(args.target_predictions), Path(args.data_root), target_clients)

    val_rows = []
    val_rows.extend(baseline_val_scores(target_rows, args.val_ratio))
    val_rows.extend(direct_head_val_scores(Path(args.ridge_audit), "target_ridge_direct"))
    val_rows.extend(direct_head_val_scores(Path(args.mlp_audit), "target_mlp_direct"))
    val_rows.extend(l1_val_scores(Path(args.l1_selection)))
    profile_rows = select_profile(val_rows)
    client_profile_rows = select_client_profile(val_rows)

    prediction_maps = load_test_predictions(
        target_rows=target_rows,
        ridge_path=Path(args.ridge_predictions),
        mlp_path=Path(args.mlp_predictions),
        l1_path=Path(args.l1_predictions),
    )
    candidate_rows = build_candidate_rows(target_rows, profile_rows, prediction_maps)
    client_candidate_rows = build_client_candidate_rows(target_rows, client_profile_rows, prediction_maps)

    summary_rows: list[dict[str, Any]] = []
    test_rows = [dict(row) for row in target_rows if row["split"] == "test"]
    for row in test_rows:
        row["baseline_final_ppm"] = fnum(row.get("final_ppm"))
    summary_rows.extend(summarize(test_rows, "baseline_final_ppm", "baseline_final_ppm", "test"))
    for candidate, pred_map in prediction_maps.items():
        if candidate == "baseline_final_ppm":
            continue
        rows = []
        for row in test_rows:
            item = dict(row)
            item["candidate_ppm"] = pred_map.get(row_key(item), fnum(item.get("final_ppm")))
            rows.append(item)
        summary_rows.extend(summarize(rows, "candidate_ppm", candidate, "test"))
    summary_rows.extend(summarize(candidate_rows, "corrected_ppm", "L2_unified_val_selector", "test"))
    summary_rows.extend(summarize(client_candidate_rows, "corrected_ppm", "L2_client_val_selector", "test"))

    write_csv(out / "l2_unified_selector_val_scores.csv", val_rows)
    write_csv(out / "l2_unified_selector_profile.csv", profile_rows)
    write_csv(out / "l2_client_selector_profile.csv", client_profile_rows)
    write_csv(out / "l2_unified_selector_predictions.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in candidate_rows])
    write_csv(out / "l2_client_selector_predictions.csv", [{k: v for k, v in row.items() if k != "feature_dict"} for row in client_candidate_rows])
    write_csv(out / "l2_unified_selector_summary.csv", summary_rows)
    write_report(out, summary_rows, profile_rows, client_profile_rows)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": args.data_root,
                "target_predictions": args.target_predictions,
                "target_clients": sorted(target_clients),
                "val_ratio": args.val_ratio,
                "outputs": {
                    "val_scores": str(out / "l2_unified_selector_val_scores.csv"),
                    "profile": str(out / "l2_unified_selector_profile.csv"),
                    "client_profile": str(out / "l2_client_selector_profile.csv"),
                    "predictions": str(out / "l2_unified_selector_predictions.csv"),
                    "client_predictions": str(out / "l2_client_selector_predictions.csv"),
                    "summary": str(out / "l2_unified_selector_summary.csv"),
                    "report": str(out / "l2_unified_selector_report.md"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote L2 unified selector results to {out}")


if __name__ == "__main__":
    main()
