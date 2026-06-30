"""Guarded per-client profile selector for H2.3+ versus H8+C4.

The selector is intentionally conservative: H2.3+ is the default profile, and
H8+C4 is allowed only when validation evidence clears RMSE, NRMSE, nonCO, and
low-calibration stability guards.  This prevents small validation-set
fluctuations from over-switching clients such as C4.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from run_low_calibration_blend_stress import parse_clients
from run_low_calibration_profile_choice_stress import row_key
from run_profile_qc_coverage_audit import format_float
from run_regression_head_ablation import client_name, fnum, inum, metrics, read_csv, summarize, write_csv


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


def finite(value: Any) -> bool:
    return math.isfinite(fnum(value))


def pass_delta(candidate: Any, anchor: Any, max_delta: float, *, pass_if_missing: bool = True) -> int:
    candidate_f = fnum(candidate)
    anchor_f = fnum(anchor)
    if not math.isfinite(candidate_f) or not math.isfinite(anchor_f):
        return int(pass_if_missing)
    return int(candidate_f - anchor_f <= float(max_delta) + 1e-12)


def metric_bundle(rows: Sequence[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    result = metrics(rows, pred_key)
    nonco_rows = [row for row in rows if inum(row.get("true_class")) != 1]
    nonco = metrics(nonco_rows, pred_key) if nonco_rows else {"N": 0, "RMSE": None, "NRMSE": None}
    return {
        "N": int(result.get("N") or 0),
        "RMSE": result.get("RMSE"),
        "NRMSE": result.get("NRMSE"),
        "nonCO_N": int(nonco.get("N") or 0),
        "nonCO_RMSE": nonco.get("RMSE"),
        "nonCO_NRMSE": nonco.get("NRMSE"),
    }


def stability_rate(
    stability_rows: Sequence[dict[str, Any]],
    *,
    route: str,
    client: str,
    stability_budget: int,
) -> float:
    for row in stability_rows:
        if str(row.get("route")) != route:
            continue
        if inum(row.get("budget_per_client")) != int(stability_budget):
            continue
        if client_name(row.get("client")) != client_name(client):
            continue
        return fnum(row.get("H8_C4_rate"), 0.0)
    return 0.0


def guarded_profile_choices(
    h23_val_rows: Sequence[dict[str, Any]],
    h8_val_rows: Sequence[dict[str, Any]],
    target_clients: Sequence[str],
    *,
    route: str,
    stability_rows: Sequence[dict[str, Any]],
    stability_budget: int,
    h23_key: str,
    h8_key: str,
    min_rmse_margin: float,
    max_nrmse_delta: float,
    max_nonco_delta: float,
    min_h8_stability: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for client in [client_name(item) for item in target_clients]:
        h23_client = [row for row in h23_val_rows if client_name(row.get("client")) == client]
        h8_client = [row for row in h8_val_rows if client_name(row.get("client")) == client]
        h23 = metric_bundle(h23_client, h23_key)
        h8 = metric_bundle(h8_client, h8_key)
        h23_rmse = fnum(h23["RMSE"])
        h8_rmse = fnum(h8["RMSE"])
        rmse_gain = h23_rmse - h8_rmse
        h8_stability = stability_rate(
            stability_rows,
            route=route,
            client=client,
            stability_budget=stability_budget,
        )
        passes_rmse_margin = int(math.isfinite(rmse_gain) and rmse_gain >= float(min_rmse_margin) - 1e-12)
        passes_nrmse = pass_delta(h8["NRMSE"], h23["NRMSE"], max_nrmse_delta)
        passes_nonco = pass_delta(h8["nonCO_RMSE"], h23["nonCO_RMSE"], max_nonco_delta)
        passes_stability = int(h8_stability >= float(min_h8_stability) - 1e-12)
        allow_h8 = all([passes_rmse_margin, passes_nrmse, passes_nonco, passes_stability])
        out.append(
            {
                "route": route,
                "client": client,
                "selected_profile": "H8+C4" if allow_h8 else "H2.3+",
                "validation_N": h23["N"],
                "h23_validation_RMSE": h23["RMSE"],
                "h8_validation_RMSE": h8["RMSE"],
                "rmse_gain_h23_minus_h8": rmse_gain if math.isfinite(rmse_gain) else None,
                "min_rmse_margin": float(min_rmse_margin),
                "h23_validation_NRMSE": h23["NRMSE"],
                "h8_validation_NRMSE": h8["NRMSE"],
                "nrmse_delta_h8_minus_h23": fnum(h8["NRMSE"]) - fnum(h23["NRMSE"]) if finite(h8["NRMSE"]) and finite(h23["NRMSE"]) else None,
                "max_nrmse_delta": float(max_nrmse_delta),
                "h23_validation_nonCO_RMSE": h23["nonCO_RMSE"],
                "h8_validation_nonCO_RMSE": h8["nonCO_RMSE"],
                "nonco_delta_h8_minus_h23": fnum(h8["nonCO_RMSE"]) - fnum(h23["nonCO_RMSE"])
                if finite(h8["nonCO_RMSE"]) and finite(h23["nonCO_RMSE"])
                else None,
                "max_nonco_delta": float(max_nonco_delta),
                "h8_low_cal_stability_rate": h8_stability,
                "stability_budget": int(stability_budget),
                "min_h8_stability": float(min_h8_stability),
                "passes_rmse_margin": passes_rmse_margin,
                "passes_nrmse": passes_nrmse,
                "passes_nonco": passes_nonco,
                "passes_stability": passes_stability,
                "allow_h8": int(allow_h8),
            }
        )
    return out


def apply_guarded_profile_choices(
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
        h8_row = h8_by_key.get(row_key(h23_row))
        if h8_row is None:
            raise ValueError(f"Missing H8+C4 test row for key={row_key(h23_row)}")
        selected = selected_by_client.get(client, "H2.3+")
        source = h8_row if selected == "H8+C4" else h23_row
        pred_key = h8_key if selected == "H8+C4" else h23_key
        item = dict(source)
        item["selected_profile"] = selected
        item[output_key] = fnum(source.get(pred_key))
        out.append(item)
    return out


def write_report(
    out_dir: Path,
    selections: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    *,
    output_key: str,
) -> None:
    lines = [
        "# Guarded Profile Selector",
        "",
        "Default profile is H2.3+. H8+C4 is selected only when validation RMSE margin, NRMSE guard, nonCO guard, and low-calibration stability all pass.",
        "",
        "## Selection Decisions",
        "",
        "| route | client | selected | val H2.3+ RMSE/NRMSE | val H8+C4 RMSE/NRMSE | RMSE gain | nonCO delta | H8 stability | guards |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in selections:
        guards = "".join(
            [
                "M" if inum(row.get("passes_rmse_margin")) else "-",
                "N" if inum(row.get("passes_nrmse")) else "-",
                "C" if inum(row.get("passes_nonco")) else "-",
                "S" if inum(row.get("passes_stability")) else "-",
            ]
        )
        lines.append(
            "| {route} | {client} | {selected} | {h23_rmse} / {h23_nrmse} | {h8_rmse} / {h8_nrmse} | {gain} | {nonco_delta} | {stab}% | {guards} |".format(
                route=row["route"],
                client=row["client"],
                selected=row["selected_profile"],
                h23_rmse=format_float(row.get("h23_validation_RMSE"), 3),
                h23_nrmse=format_float(row.get("h23_validation_NRMSE"), 4),
                h8_rmse=format_float(row.get("h8_validation_RMSE"), 3),
                h8_nrmse=format_float(row.get("h8_validation_NRMSE"), 4),
                gain=format_float(row.get("rmse_gain_h23_minus_h8"), 3),
                nonco_delta=format_float(row.get("nonco_delta_h8_minus_h23"), 3),
                stab=format_float(100 * fnum(row.get("h8_low_cal_stability_rate")), 1),
                guards=guards,
            )
        )

    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            f"`{output_key}` is the guarded profile prediction column.",
            "",
            "| mode | scope | N | RMSE | NRMSE | MAE | P90AE |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        if row.get("scope") not in {"ALL", "C3", "C4", "C5", "nonCO_ALL"}:
            continue
        lines.append(
            "| {mode} | {scope} | {n} | {rmse} | {nrmse} | {mae} | {p90} |".format(
                mode=row["mode"],
                scope=row["scope"],
                n=row["N"],
                rmse=format_float(row.get("RMSE"), 3),
                nrmse=format_float(row.get("NRMSE"), 4),
                mae=format_float(row.get("MAE"), 3),
                p90=format_float(row.get("P90AE"), 3),
            )
        )
    (out_dir / "guarded_profile_selector_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_routes(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.routes_json:
        return json.loads(Path(args.routes_json).read_text(encoding="utf-8"))
    return DEFAULT_ROUTES


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clients = parse_clients(args.target_clients)
    routes = load_routes(args)
    stability_rows = read_csv(args.stability_summary) if args.stability_summary else []

    all_selections: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    prediction_outputs: list[str] = []

    for route_spec in routes:
        route = str(route_spec["route"])
        h23_val = read_csv(route_spec["h23_validation"])
        h8_val = read_csv(route_spec["h8_validation"])
        h23_test = read_csv(route_spec["h23_test"])
        h8_test = read_csv(route_spec["h8_test"])
        choices = guarded_profile_choices(
            h23_val,
            h8_val,
            target_clients,
            route=route,
            stability_rows=stability_rows,
            stability_budget=args.stability_budget,
            h23_key=args.h23_key,
            h8_key=args.h8_key,
            min_rmse_margin=args.min_rmse_margin,
            max_nrmse_delta=args.max_nrmse_delta,
            max_nonco_delta=args.max_nonco_delta,
            min_h8_stability=args.min_h8_stability,
        )
        guarded_rows = apply_guarded_profile_choices(
            h23_test,
            h8_test,
            choices,
            h23_key=args.h23_key,
            h8_key=args.h8_key,
            output_key=args.output_key,
        )
        safe_route = route.replace("-", "_")
        predictions_name = f"{safe_route}_guarded_profile_predictions.csv"
        write_csv(out_dir / predictions_name, guarded_rows)
        prediction_outputs.append(predictions_name)
        all_selections.extend(choices)
        all_metrics.extend(summarize(h23_test, args.h23_key, f"{route}_H2.3+", "test"))
        all_metrics.extend(summarize(h8_test, args.h8_key, f"{route}_H8+C4", "test"))
        all_metrics.extend(summarize(guarded_rows, args.output_key, f"{route}_guarded_profile", "test"))

    write_csv(out_dir / "guarded_profile_selection.csv", all_selections)
    write_csv(out_dir / "guarded_profile_metrics.csv", all_metrics)
    write_report(out_dir, all_selections, all_metrics, output_key=args.output_key)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "routes": routes,
                "target_clients": target_clients,
                "stability_summary": args.stability_summary,
                "stability_budget": args.stability_budget,
                "h23_key": args.h23_key,
                "h8_key": args.h8_key,
                "output_key": args.output_key,
                "guards": {
                    "min_rmse_margin": args.min_rmse_margin,
                    "max_nrmse_delta": args.max_nrmse_delta,
                    "max_nonco_delta": args.max_nonco_delta,
                    "min_h8_stability": args.min_h8_stability,
                },
                "outputs": [
                    "guarded_profile_selection.csv",
                    "guarded_profile_metrics.csv",
                    "guarded_profile_selector_report.md",
                    *prediction_outputs,
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "selection_rows": len(all_selections),
                "metric_rows": len(all_metrics),
                "prediction_outputs": prediction_outputs,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-json", default="")
    parser.add_argument("--output-dir", default="results/guarded_profile_selector_20260630")
    parser.add_argument("--target-clients", default="C3,C4,C5")
    parser.add_argument("--stability-summary", default="results/low_calibration_profile_choice_stress_20260630/low_calibration_profile_choice_selection_summary.csv")
    parser.add_argument("--stability-budget", type=int, default=96)
    parser.add_argument("--h23-key", default="h2_3_plus_blend_ppm")
    parser.add_argument("--h8-key", default="formal_c4_route_rescue_ppm")
    parser.add_argument("--output-key", default="guarded_profile_ppm")
    parser.add_argument("--min-rmse-margin", type=float, default=0.5)
    parser.add_argument("--max-nrmse-delta", type=float, default=0.0)
    parser.add_argument("--max-nonco-delta", type=float, default=0.0)
    parser.add_argument("--min-h8-stability", type=float, default=0.7)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
