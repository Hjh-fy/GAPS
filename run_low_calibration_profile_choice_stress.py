"""Low-calibration stress for choosing H2.3+ versus H8+C4 per client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from run_low_calibration_blend_stress import (
    aggregate_metric_rows,
    metric_rows_for_scopes,
    parse_clients,
    parse_int_grid,
    sample_rows_by_client,
)
from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import client_name, fnum, inum, metrics, read_csv, write_csv


DEFAULT_ROUTES = [
    {
        "route": "real-route",
        "h23_validation": "results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/fusion_profile_validation_predictions.csv",
        "h23_test": "results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/fusion_profile_predictions.csv",
        "h8_validation": "results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_c4_route_rescue_selector/formal_c4_route_rescue_validation_predictions.csv",
        "h8_test": "results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_c4_route_rescue_selector/formal_c4_route_rescue_predictions.csv",
    },
    {
        "route": "oracle-route",
        "h23_validation": "results/h2_3_plus_fusion_profile_20260630/r25_oracle_route_replay_gate/fusion_profile_validation_predictions.csv",
        "h23_test": "results/h2_3_plus_fusion_profile_20260630/r25_oracle_route_replay_gate/fusion_profile_predictions.csv",
        "h8_validation": "results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/formal_c4_route_rescue_selector/formal_c4_route_rescue_validation_predictions.csv",
        "h8_test": "results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/formal_c4_route_rescue_selector/formal_c4_route_rescue_predictions.csv",
    },
]


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        client_name(row.get("client") or row.get("client_id")),
        str(row.get("split", "")),
        inum(row.get("sample_index")),
    )


def matching_rows(rows: Sequence[dict[str, Any]], keys: set[tuple[str, str, int]]) -> list[dict[str, Any]]:
    by_key = {row_key(row): row for row in rows}
    missing = keys - set(by_key)
    if missing:
        preview = ", ".join(str(item) for item in sorted(missing)[:5])
        raise ValueError(f"Missing matched profile rows for {len(missing)} keys; first keys: {preview}")
    return [by_key[key] for key in sorted(keys)]


def rmse_for(rows: Sequence[dict[str, Any]], pred_key: str) -> float:
    if not rows:
        return float("inf")
    return fnum(metrics(rows, pred_key).get("RMSE"), float("inf"))


def choose_profiles_by_client(
    h23_val_rows: Sequence[dict[str, Any]],
    h8_val_rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    *,
    h23_key: str,
    h8_key: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for client in target_clients:
        h23_client = [row for row in h23_val_rows if client_name(row.get("client")) == client]
        h8_client = [row for row in h8_val_rows if client_name(row.get("client")) == client]
        h23_rmse = rmse_for(h23_client, h23_key)
        h8_rmse = rmse_for(h8_client, h8_key)
        selected = "H8+C4" if h8_rmse < h23_rmse else "H2.3+"
        out.append(
            {
                "client": client,
                "validation_N": len(h23_client),
                "h23_validation_RMSE": h23_rmse,
                "h8_validation_RMSE": h8_rmse,
                "selected_profile": selected,
            }
        )
    return out


def apply_profile_choices(
    h23_test_rows: Sequence[dict[str, Any]],
    h8_test_rows: Sequence[dict[str, Any]],
    choices: Sequence[dict[str, Any]],
    *,
    h23_key: str,
    h8_key: str,
    output_key: str,
) -> list[dict[str, Any]]:
    h8_by_key = {row_key(row): row for row in h8_test_rows}
    selected_by_client = {client_name(row["client"]): str(row["selected_profile"]) for row in choices}
    out: list[dict[str, Any]] = []
    for h23_row in h23_test_rows:
        client = client_name(h23_row.get("client"))
        key = row_key(h23_row)
        h8_row = h8_by_key.get(key)
        if h8_row is None:
            raise ValueError(f"Missing H8+C4 test row for key={key}")
        selected = selected_by_client.get(client, "H2.3+")
        source = h8_row if selected == "H8+C4" else h23_row
        pred_key = h8_key if selected == "H8+C4" else h23_key
        item = dict(source)
        item["selected_profile"] = selected
        item[output_key] = fnum(source.get(pred_key))
        out.append(item)
    return out


def profile_choice_repeat_rows(
    *,
    route: str,
    h23_val_rows: Sequence[dict[str, Any]],
    h8_val_rows: Sequence[dict[str, Any]],
    h23_test_rows: Sequence[dict[str, Any]],
    h8_test_rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    budget: int,
    repeat: int,
    seed: int,
    h23_key: str,
    h8_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampled_h23 = sample_rows_by_client(h23_val_rows, target_clients, budget=budget, seed=seed + repeat)
    sampled_keys = {row_key(row) for row in sampled_h23}
    sampled_h8 = matching_rows(h8_val_rows, sampled_keys)
    choices = choose_profiles_by_client(sampled_h23, sampled_h8, target_clients, h23_key=h23_key, h8_key=h8_key)
    selection_rows = [
        {
            "route": route,
            "budget_per_client": int(budget),
            "repeat": int(repeat),
            **choice,
        }
        for choice in choices
    ]
    hybrid_test = apply_profile_choices(
        h23_test_rows,
        h8_test_rows,
        choices,
        h23_key=h23_key,
        h8_key=h8_key,
        output_key="profile_choice_ppm",
    )
    metric_rows = metric_rows_for_scopes(
        route=route,
        budget=budget,
        repeat=repeat,
        rows=hybrid_test,
        pred_key="profile_choice_ppm",
        target_clients=target_clients,
    )
    return selection_rows, metric_rows


def aggregate_selection_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["route"]), inum(row["budget_per_client"]), client_name(row["client"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (route, budget, client), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        h23_count = sum(str(row["selected_profile"]) == "H2.3+" for row in values)
        h8_count = sum(str(row["selected_profile"]) == "H8+C4" for row in values)
        mode = "H8+C4" if h8_count > h23_count else "H2.3+"
        mode_count = max(h23_count, h8_count)
        out.append(
            {
                "route": route,
                "budget_per_client": budget,
                "client": client,
                "repeats": len(values),
                "H2_3_plus_rate": h23_count / len(values),
                "H8_C4_rate": h8_count / len(values),
                "profile_mode": mode,
                "profile_mode_rate": mode_count / len(values),
                "h23_validation_RMSE_mean": mean(fnum(row["h23_validation_RMSE"]) for row in values),
                "h8_validation_RMSE_mean": mean(fnum(row["h8_validation_RMSE"]) for row in values),
            }
        )
    return out


def write_report(out_dir: Path, metric_agg: Sequence[dict[str, Any]], selection_agg: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Low Calibration Profile Choice Stress",
        "",
        "Each repeat samples validation rows per client, chooses H2.3+ or H8+C4 by validation RMSE, and evaluates the selected per-client profile on test.",
        "",
        "## Test Metric Stability",
        "",
        "| route | budget/client | scope | RMSE mean +- std | RMSE range | NRMSE mean +- std | NRMSE range |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in metric_agg:
        lines.append(
            "| {route} | {budget} | {scope} | {rm} +- {rs} | {rmin}..{rmax} | {nm} +- {ns} | {nmin}..{nmax} |".format(
                route=row["route"],
                budget=row["budget_per_client"],
                scope=row["scope"],
                rm=format_float(row["RMSE_mean"], 3),
                rs=format_float(row["RMSE_std"], 3),
                rmin=format_float(row["RMSE_min"], 3),
                rmax=format_float(row["RMSE_max"], 3),
                nm=format_float(row["NRMSE_mean"], 4),
                ns=format_float(row["NRMSE_std"], 4),
                nmin=format_float(row["NRMSE_min"], 4),
                nmax=format_float(row["NRMSE_max"], 4),
            )
        )

    lines.extend(
        [
            "",
            "## Profile Selection Stability",
            "",
            "| route | budget/client | client | H2.3+ rate | H8+C4 rate | mode | mode rate | val H2.3+ RMSE | val H8+C4 RMSE |",
            "|---|---:|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in selection_agg:
        lines.append(
            "| {route} | {budget} | {client} | {h23}% | {h8}% | {mode} | {mode_rate}% | {h23_rmse} | {h8_rmse} |".format(
                route=row["route"],
                budget=row["budget_per_client"],
                client=row["client"],
                h23=format_float(100 * fnum(row["H2_3_plus_rate"]), 1),
                h8=format_float(100 * fnum(row["H8_C4_rate"]), 1),
                mode=row["profile_mode"],
                mode_rate=format_float(100 * fnum(row["profile_mode_rate"]), 1),
                h23_rmse=format_float(row["h23_validation_RMSE_mean"], 3),
                h8_rmse=format_float(row["h8_validation_RMSE_mean"], 3),
            )
        )
    (out_dir / "low_calibration_profile_choice_stress_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_routes(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.routes_json:
        return json.loads(Path(args.routes_json).read_text(encoding="utf-8"))
    return DEFAULT_ROUTES


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clients = parse_clients(args.target_clients)
    budgets = parse_int_grid(args.budgets)
    routes = load_routes(args)
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for route_spec in routes:
        route = str(route_spec["route"])
        h23_val = read_csv(route_spec["h23_validation"])
        h8_val = read_csv(route_spec["h8_validation"])
        h23_test = read_csv(route_spec["h23_test"])
        h8_test = read_csv(route_spec["h8_test"])
        for budget in budgets:
            for repeat in range(args.repeats):
                selected, metrics_out = profile_choice_repeat_rows(
                    route=route,
                    h23_val_rows=h23_val,
                    h8_val_rows=h8_val,
                    h23_test_rows=h23_test,
                    h8_test_rows=h8_test,
                    target_clients=target_clients,
                    budget=budget,
                    repeat=repeat,
                    seed=args.seed + budget * 1000,
                    h23_key=args.h23_key,
                    h8_key=args.h8_key,
                )
                selection_rows.extend(selected)
                metric_rows.extend(metrics_out)

    metric_agg = aggregate_metric_rows(metric_rows)
    selection_agg = aggregate_selection_rows(selection_rows)
    write_csv(out_dir / "low_calibration_profile_choice_selection_repeats.csv", selection_rows)
    write_csv(out_dir / "low_calibration_profile_choice_metric_repeats.csv", metric_rows)
    write_csv(out_dir / "low_calibration_profile_choice_metric_summary.csv", metric_agg)
    write_csv(out_dir / "low_calibration_profile_choice_selection_summary.csv", selection_agg)
    write_report(out_dir, metric_agg, selection_agg)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "routes": routes,
                "target_clients": target_clients,
                "budgets": budgets,
                "repeats": args.repeats,
                "seed": args.seed,
                "h23_key": args.h23_key,
                "h8_key": args.h8_key,
                "outputs": [
                    "low_calibration_profile_choice_selection_repeats.csv",
                    "low_calibration_profile_choice_metric_repeats.csv",
                    "low_calibration_profile_choice_metric_summary.csv",
                    "low_calibration_profile_choice_selection_summary.csv",
                    "low_calibration_profile_choice_stress_report.md",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "metric_rows": len(metric_rows), "selection_rows": len(selection_rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-json", default="")
    parser.add_argument("--output-dir", default="results/low_calibration_profile_choice_stress_20260630")
    parser.add_argument("--target-clients", default="C3,C4,C5")
    parser.add_argument("--budgets", default="12,24,48,96")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--h23-key", default="h2_3_plus_blend_ppm")
    parser.add_argument("--h8-key", default="formal_c4_route_rescue_ppm")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
