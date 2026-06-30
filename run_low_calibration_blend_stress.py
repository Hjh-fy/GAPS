"""Stress test H2.3+ blend-weight selection under low calibration budgets.

This does not retrain regression heads. It resamples already generated
calibration-validation predictions, reselects per-client blend weights, and
applies those weights to the fixed test predictions.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from run_h2_3_plus_fusion_profile import apply_client_blends, select_client_blend_weights
from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import client_name, fnum, inum, metrics, read_csv, write_csv


DEFAULT_ROUTES = [
    {
        "route": "real-route",
        "validation_predictions": "results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/fusion_profile_validation_predictions.csv",
        "test_predictions": "results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/fusion_profile_predictions.csv",
    },
    {
        "route": "oracle-route",
        "validation_predictions": "results/h2_3_plus_fusion_profile_20260630/r25_oracle_route_replay_gate/fusion_profile_validation_predictions.csv",
        "test_predictions": "results/h2_3_plus_fusion_profile_20260630/r25_oracle_route_replay_gate/fusion_profile_predictions.csv",
    },
]


def parse_clients(text: str) -> list[str]:
    return [client_name(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_grid(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def stable_row_key(row: dict[str, Any]) -> tuple[int, str]:
    return inum(row.get("sample_index")), str(row.get("filename", ""))


def sample_rows_by_client(
    rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    *,
    budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for client in target_clients:
        client_rows = sorted([row for row in rows if client_name(row.get("client")) == client], key=stable_row_key)
        if budget >= len(client_rows):
            out.extend(client_rows)
            continue
        indices = sorted(rng.sample(range(len(client_rows)), max(0, int(budget))))
        out.extend(client_rows[index] for index in indices)
    return out


def metric_rows_for_scopes(
    *,
    route: str,
    budget: int,
    repeat: int,
    rows: Sequence[dict[str, Any]],
    pred_key: str,
    target_clients: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in ["ALL", *target_clients]:
        selected = list(rows) if scope == "ALL" else [row for row in rows if client_name(row.get("client")) == scope]
        result = metrics(selected, pred_key)
        out.append(
            {
                "route": route,
                "budget_per_client": int(budget),
                "repeat": int(repeat),
                "scope": scope,
                "N": int(result.get("N") or 0),
                "RMSE": result.get("RMSE"),
                "NRMSE": result.get("NRMSE"),
                "MAE": result.get("MAE"),
                "P90AE": result.get("P90AE"),
            }
        )
    return out


def stress_repeat_rows(
    *,
    route: str,
    val_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    budget: int,
    repeat: int,
    seed: int,
    weight_grid: Sequence[float],
    anchor_key: str,
    candidate_key: str,
    max_nonco_delta: float,
    min_all_delta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sampled = sample_rows_by_client(val_rows, target_clients, budget=budget, seed=seed + repeat)
    weights, selection_audit = select_client_blend_weights(
        sampled,
        target_clients,
        weight_grid,
        anchor_key=anchor_key,
        candidate_key=candidate_key,
        max_nonco_delta=max_nonco_delta,
        min_all_delta=min_all_delta,
    )
    blended_test = apply_client_blends(
        test_rows,
        weights,
        anchor_key=anchor_key,
        candidate_key=candidate_key,
        output_key="stress_blend_ppm",
    )

    selection_rows: list[dict[str, Any]] = []
    for client in target_clients:
        selected_audit = next(
            row for row in selection_audit
            if client_name(row.get("client")) == client and inum(row.get("selected")) == 1
        )
        selection_rows.append(
            {
                "route": route,
                "budget_per_client": int(budget),
                "repeat": int(repeat),
                "client": client,
                "validation_N": len([row for row in sampled if client_name(row.get("client")) == client]),
                "selected_weight": float(weights[client]),
                "anchor_ALL_RMSE": selected_audit.get("anchor_ALL_RMSE"),
                "selected_blend_ALL_RMSE": selected_audit.get("blend_ALL_RMSE"),
                "anchor_nonCO_RMSE": selected_audit.get("anchor_nonCO_RMSE"),
                "selected_blend_nonCO_RMSE": selected_audit.get("blend_nonCO_RMSE"),
            }
        )

    return selection_rows, metric_rows_for_scopes(
        route=route,
        budget=budget,
        repeat=repeat,
        rows=blended_test,
        pred_key="stress_blend_ppm",
        target_clients=target_clients,
    )


def aggregate_metric_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["route"]), inum(row["budget_per_client"]), str(row["scope"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (route, budget, scope), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        rmses = [fnum(row.get("RMSE")) for row in values]
        nrmses = [fnum(row.get("NRMSE")) for row in values]
        out.append(
            {
                "route": route,
                "budget_per_client": budget,
                "scope": scope,
                "repeats": len(values),
                "RMSE_mean": mean(rmses),
                "RMSE_std": pstdev(rmses) if len(rmses) > 1 else 0.0,
                "RMSE_min": min(rmses),
                "RMSE_max": max(rmses),
                "NRMSE_mean": mean(nrmses),
                "NRMSE_std": pstdev(nrmses) if len(nrmses) > 1 else 0.0,
                "NRMSE_min": min(nrmses),
                "NRMSE_max": max(nrmses),
            }
        )
    return out


def aggregate_selection_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["route"]), inum(row["budget_per_client"]), str(row["client"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (route, budget, client), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        weights = [fnum(row.get("selected_weight")) for row in values]
        counts: dict[float, int] = {}
        for weight in weights:
            counts[weight] = counts.get(weight, 0) + 1
        mode_weight, mode_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        out.append(
            {
                "route": route,
                "budget_per_client": budget,
                "client": client,
                "repeats": len(values),
                "weight_mean": mean(weights),
                "weight_std": pstdev(weights) if len(weights) > 1 else 0.0,
                "weight_min": min(weights),
                "weight_max": max(weights),
                "weight_mode": mode_weight,
                "weight_mode_rate": mode_count / len(values),
                "weight_values": ";".join(str(weight) for weight in sorted(counts)),
            }
        )
    return out


def write_report(
    out_dir: Path,
    metric_agg: Sequence[dict[str, Any]],
    selection_agg: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Low Calibration Blend Stress",
        "",
        "Each repeat samples a fixed number of calibration-validation rows per client, reselects H2.3+ blend weights, and applies the weights to the fixed test predictions.",
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
            "## Selected Weight Stability",
            "",
            "| route | budget/client | client | weight mean +- std | range | mode | mode rate | values |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in selection_agg:
        lines.append(
            "| {route} | {budget} | {client} | {wm} +- {ws} | {wmin}..{wmax} | {mode} | {rate}% | {values} |".format(
                route=row["route"],
                budget=row["budget_per_client"],
                client=row["client"],
                wm=format_float(row["weight_mean"], 3),
                ws=format_float(row["weight_std"], 3),
                wmin=format_float(row["weight_min"], 2),
                wmax=format_float(row["weight_max"], 2),
                mode=format_float(row["weight_mode"], 2),
                rate=format_float(100 * fnum(row["weight_mode_rate"]), 1),
                values=row["weight_values"],
            )
        )

    (out_dir / "low_calibration_blend_stress_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_routes(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.routes_json:
        return json.loads(Path(args.routes_json).read_text(encoding="utf-8"))
    return DEFAULT_ROUTES


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clients = parse_clients(args.target_clients)
    budgets = parse_int_grid(args.budgets)
    weight_grid = parse_float_grid(args.blend_weights)
    routes = load_routes(args)

    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for route_spec in routes:
        route = str(route_spec["route"])
        val_rows = read_csv(route_spec["validation_predictions"])
        test_rows = read_csv(route_spec["test_predictions"])
        for budget in budgets:
            for repeat in range(args.repeats):
                selected, metrics_out = stress_repeat_rows(
                    route=route,
                    val_rows=val_rows,
                    test_rows=test_rows,
                    target_clients=target_clients,
                    budget=budget,
                    repeat=repeat,
                    seed=args.seed + budget * 1000,
                    weight_grid=weight_grid,
                    anchor_key=args.anchor_key,
                    candidate_key=args.candidate_key,
                    max_nonco_delta=args.max_nonco_delta,
                    min_all_delta=args.min_all_delta,
                )
                selection_rows.extend(selected)
                metric_rows.extend(metrics_out)

    metric_agg = aggregate_metric_rows(metric_rows)
    selection_agg = aggregate_selection_rows(selection_rows)
    write_csv(out_dir / "low_calibration_selection_repeats.csv", selection_rows)
    write_csv(out_dir / "low_calibration_metric_repeats.csv", metric_rows)
    write_csv(out_dir / "low_calibration_metric_summary.csv", metric_agg)
    write_csv(out_dir / "low_calibration_selection_summary.csv", selection_agg)
    write_report(out_dir, metric_agg, selection_agg)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "routes": routes,
                "target_clients": target_clients,
                "budgets": budgets,
                "repeats": args.repeats,
                "seed": args.seed,
                "blend_weights": weight_grid,
                "anchor_key": args.anchor_key,
                "candidate_key": args.candidate_key,
                "max_nonco_delta": args.max_nonco_delta,
                "min_all_delta": args.min_all_delta,
                "outputs": [
                    "low_calibration_selection_repeats.csv",
                    "low_calibration_metric_repeats.csv",
                    "low_calibration_metric_summary.csv",
                    "low_calibration_selection_summary.csv",
                    "low_calibration_blend_stress_report.md",
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
    parser.add_argument("--output-dir", default="results/low_calibration_blend_stress_20260630")
    parser.add_argument("--target-clients", default="C3,C4,C5")
    parser.add_argument("--budgets", default="12,24,48,96")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--blend-weights", default="0,0.1,0.25,0.5,0.75,1")
    parser.add_argument("--anchor-key", default="h2_3_current_ppm")
    parser.add_argument("--candidate-key", default="regfeat_ridge_plus_c4_rescue_ppm")
    parser.add_argument("--max-nonco-delta", type=float, default=0.0)
    parser.add_argument("--min-all-delta", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
