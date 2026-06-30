"""Audit post-profile QC coverage for multiple regression profiles.

The QC decision/risk comes from the deployment QC records.  This script only
swaps the profile ppm output and recomputes RMSE/NRMSE under the same accepted,
review, reject, and risk-sorted coverage slices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from run_regression_head_ablation import client_name, fnum, inum, metrics, read_csv, write_csv


DEFAULT_PROFILES = [
    {
        "name": "H2.3 target direct-head",
        "predictions": "results/f6_fixed_da_strong_r25_profile_replay_20260630/h2_3_profile_replay/h2_3_profile_predictions.csv",
        "pred_key": "h2_3_ppm",
    },
    {
        "name": "H2.3+ reg-feat weak-blend",
        "predictions": "results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/fusion_profile_predictions.csv",
        "pred_key": "h2_3_plus_blend_ppm",
    },
    {
        "name": "H8 + formal C4 route rescue",
        "predictions": "results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_c4_route_rescue_selector/formal_c4_route_rescue_predictions.csv",
        "pred_key": "formal_c4_route_rescue_ppm",
    },
    {
        "name": "Client selector C34 H2.3+ / C5 H8+C4",
        "client_profiles": {
            "C3": {
                "predictions": "results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/fusion_profile_predictions.csv",
                "pred_key": "h2_3_plus_blend_ppm",
            },
            "C4": {
                "predictions": "results/h2_3_plus_fusion_profile_20260630/r25_balanced_replay_gate/fusion_profile_predictions.csv",
                "pred_key": "h2_3_plus_blend_ppm",
            },
            "C5": {
                "predictions": "results/f6_fixed_da_strong_r25_profile_replay_20260630/formal_c4_route_rescue_selector/formal_c4_route_rescue_predictions.csv",
                "pred_key": "formal_c4_route_rescue_ppm",
            },
        },
    },
]

DEFAULT_COVERAGES = [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        client_name(row.get("client") or row.get("client_id")),
        str(row.get("split", "test")),
        inum(row.get("sample_index")),
    )


def normalize_qc_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["client"] = client_name(item.get("client") or item.get("client_id"))
    item["split"] = str(item.get("split", "test"))
    item["sample_index"] = inum(item.get("sample_index"))
    item["true_class"] = inum(item.get("true_class"))
    item["true_ppm"] = fnum(item.get("true_ppm"))
    item["qc_decision"] = str(item.get("qc_decision", item.get("qc_status", ""))).lower()
    item["qc_risk_value"] = fnum(item.get("qc_risk_value", item.get("risk_score", 0.0)), 0.0)
    return item


def build_profile_rows(
    qc_rows: Sequence[dict[str, Any]],
    pred_rows: Sequence[dict[str, Any]],
    pred_key: str,
    output_key: str,
) -> list[dict[str, Any]]:
    pred_by_key = {row_key(row): row for row in pred_rows}
    out: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []
    for qc_row in qc_rows:
        item = normalize_qc_row(qc_row)
        pred = pred_by_key.get(row_key(item))
        if pred is None or pred_key not in pred:
            missing.append(row_key(item))
            continue
        item[output_key] = fnum(pred.get(pred_key))
        out.append(item)
    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise ValueError(f"Missing profile predictions for {len(missing)} rows; first keys: {preview}")
    return out


def build_client_hybrid_rows(
    qc_rows: Sequence[dict[str, Any]],
    predictions_by_client: dict[str, tuple[Sequence[dict[str, Any]], str]],
    output_key: str,
) -> list[dict[str, Any]]:
    pred_maps: dict[str, tuple[dict[tuple[str, str, int], dict[str, Any]], str]] = {
        client_name(client): ({row_key(row): row for row in rows}, pred_key)
        for client, (rows, pred_key) in predictions_by_client.items()
    }
    out: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []
    for qc_row in qc_rows:
        item = normalize_qc_row(qc_row)
        client = client_name(item.get("client"))
        if client not in pred_maps:
            missing.append(row_key(item))
            continue
        pred_by_key, pred_key = pred_maps[client]
        pred = pred_by_key.get(row_key(item))
        if pred is None or pred_key not in pred:
            missing.append(row_key(item))
            continue
        item[output_key] = fnum(pred.get(pred_key))
        out.append(item)
    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise ValueError(f"Missing client-hybrid profile predictions for {len(missing)} rows; first keys: {preview}")
    return out


def build_rows_for_profile(
    qc_rows: Sequence[dict[str, Any]],
    profile: dict[str, Any],
    output_key: str,
) -> list[dict[str, Any]]:
    if "client_profiles" in profile:
        predictions_by_client = {
            client_name(client): (read_csv(spec["predictions"]), str(spec["pred_key"]))
            for client, spec in profile["client_profiles"].items()
        }
        return build_client_hybrid_rows(qc_rows, predictions_by_client, output_key)
    return build_profile_rows(qc_rows, read_csv(profile["predictions"]), str(profile["pred_key"]), output_key)


def scoped_rows(rows: Sequence[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "ALL":
        return list(rows)
    return [row for row in rows if client_name(row.get("client")) == scope]


def metric_payload(rows: Sequence[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    result = metrics(rows, pred_key)
    return {
        "N": int(result.get("N") or 0),
        "RMSE": result.get("RMSE"),
        "NRMSE": result.get("NRMSE"),
        "MAE": result.get("MAE"),
        "P90AE": result.get("P90AE"),
    }


def post_qc_metric_rows(
    rows: Sequence[dict[str, Any]],
    profile_name: str,
    pred_key: str,
    clients: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scope in ["ALL", *clients]:
        subset = scoped_rows(rows, scope)
        accepted = [row for row in subset if str(row.get("qc_decision")) == "accept"]
        nonreject = [row for row in subset if str(row.get("qc_decision")) in {"accept", "review"}]
        full_metrics = metric_payload(subset, pred_key)
        accepted_metrics = metric_payload(accepted, pred_key)
        nonreject_metrics = metric_payload(nonreject, pred_key)
        total = len(subset)
        out.append(
            {
                "profile": profile_name,
                "scope": scope,
                "N": total,
                "pred_key": pred_key,
                "full_RMSE": full_metrics["RMSE"],
                "full_NRMSE": full_metrics["NRMSE"],
                "accepted_coverage": len(accepted) / total if total else 0.0,
                "accepted_N": len(accepted),
                "accepted_RMSE": accepted_metrics["RMSE"],
                "accepted_NRMSE": accepted_metrics["NRMSE"],
                "coverage_review": len(nonreject) / total if total else 0.0,
                "nonreject_N": len(nonreject),
                "coverage_review_RMSE": nonreject_metrics["RMSE"],
                "coverage_review_NRMSE": nonreject_metrics["NRMSE"],
                "review_rate": sum(str(row.get("qc_decision")) == "review" for row in subset) / total if total else 0.0,
                "reject_rate": sum(str(row.get("qc_decision")) == "reject" for row in subset) / total if total else 0.0,
                "qc_status_source": "qc_test_records.qc_decision",
            }
        )
    return out


def low_risk_subset(rows: Sequence[dict[str, Any]], coverage: float) -> tuple[list[dict[str, Any]], float]:
    ordered = sorted(rows, key=lambda row: (fnum(row.get("qc_risk_value")), inum(row.get("sample_index"))))
    if not ordered:
        return [], float("nan")
    if coverage >= 1.0:
        return ordered, float("inf")
    count = max(1, min(len(ordered), int(round(len(ordered) * float(coverage)))))
    selected = ordered[:count]
    return selected, fnum(selected[-1].get("qc_risk_value"))


def coverage_sweep_rows(
    rows: Sequence[dict[str, Any]],
    profile_name: str,
    pred_key: str,
    coverages: Sequence[float],
    *,
    by_client: bool,
) -> list[dict[str, Any]]:
    groups: list[tuple[str, list[dict[str, Any]]]]
    if by_client:
        clients = sorted({client_name(row.get("client")) for row in rows})
        groups = [(client, [row for row in rows if client_name(row.get("client")) == client]) for client in clients]
    else:
        groups = [("ALL", list(rows))]

    out: list[dict[str, Any]] = []
    for group_name, group_rows in groups:
        total = len(group_rows)
        for coverage in coverages:
            selected, threshold = low_risk_subset(group_rows, float(coverage))
            result = metric_payload(selected, pred_key)
            row = {
                "profile": profile_name,
                "target_coverage": f"{int(round(float(coverage) * 100))}%",
                "threshold": threshold,
                "coverage_review": len(selected) / total if total else 0.0,
                "N": len(selected),
                "RMSE": result["RMSE"],
                "NRMSE": result["NRMSE"],
                "MAE": result["MAE"],
                "P90AE": result["P90AE"],
            }
            if by_client:
                row["client"] = group_name
            else:
                row["scope"] = group_name
            out.append(row)
    return out


def format_float(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    value_f = fnum(value)
    if value_f == float("inf"):
        return "inf"
    return f"{value_f:.{digits}f}"


def best_profile_rows(rows: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(item) for item in group_keys)
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    def sort_key(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[Any, ...]:
        key, _values = item
        normalized: list[Any] = []
        for value in key:
            text = str(value)
            if text.endswith("%"):
                normalized.append(int(text.rstrip("%")))
            else:
                normalized.append(value)
        return tuple(normalized)

    for key, values in sorted(grouped.items(), key=sort_key):
        best_rmse = min(values, key=lambda row: fnum(row.get("RMSE"), float("inf")))
        best_nrmse = min(values, key=lambda row: fnum(row.get("NRMSE"), float("inf")))
        out.append(
            {
                **{name: value for name, value in zip(group_keys, key)},
                "best_RMSE_profile": best_rmse["profile"],
                "best_RMSE": best_rmse.get("RMSE"),
                "best_NRMSE_profile": best_nrmse["profile"],
                "best_NRMSE": best_nrmse.get("NRMSE"),
            }
        )
    return out


def write_report(
    out_dir: Path,
    post_rows: Sequence[dict[str, Any]],
    sweep_by_client: Sequence[dict[str, Any]],
    sweep_all: Sequence[dict[str, Any]],
    best_by_client: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Profile QC Coverage Audit",
        "",
        "QC decisions and risk values come from deployment `qc_test_records.csv`; only the profile ppm output changes.",
        "",
        "## Official QC Accepted+Review",
        "",
        "| profile | scope | full RMSE / NRMSE | accepted cov | accepted RMSE / NRMSE | accepted+review cov | accepted+review RMSE / NRMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in post_rows:
        lines.append(
            "| {profile} | {scope} | {full_rmse} / {full_nrmse} | {acov}% | {armse} / {anrmse} | {ncov}% | {nrmse} / {nnrmse} |".format(
                profile=row["profile"],
                scope=row["scope"],
                full_rmse=format_float(row.get("full_RMSE"), 2),
                full_nrmse=format_float(row.get("full_NRMSE"), 4),
                acov=format_float(100 * fnum(row.get("accepted_coverage")), 2),
                armse=format_float(row.get("accepted_RMSE"), 2),
                anrmse=format_float(row.get("accepted_NRMSE"), 4),
                ncov=format_float(100 * fnum(row.get("coverage_review")), 2),
                nrmse=format_float(row.get("coverage_review_RMSE"), 2),
                nnrmse=format_float(row.get("coverage_review_NRMSE"), 4),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Client Coverage Sweep",
            "",
            "| client | coverage | H2.3 RMSE/NRMSE | H2.3+ RMSE/NRMSE | H8+C4 RMSE/NRMSE | client selector RMSE/NRMSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    profiles = [
        "H2.3 target direct-head",
        "H2.3+ reg-feat weak-blend",
        "H8 + formal C4 route rescue",
        "Client selector C34 H2.3+ / C5 H8+C4",
    ]
    for client in sorted({row.get("client") for row in sweep_by_client}):
        coverages = sorted(
            {row.get("target_coverage") for row in sweep_by_client if row.get("client") == client},
            key=lambda text: int(str(text).rstrip("%")),
        )
        for coverage in coverages:
            by_profile = {
                row["profile"]: row
                for row in sweep_by_client
                if row.get("client") == client and row.get("target_coverage") == coverage
            }
            values = []
            for profile in profiles:
                row = by_profile.get(profile, {})
                values.append(f"{format_float(row.get('RMSE'), 3)} / {format_float(row.get('NRMSE'), 4)}")
            lines.append(f"| {client} | {coverage} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Best Profile By Client Coverage",
            "",
            "| client | coverage | best RMSE profile | RMSE | best NRMSE profile | NRMSE |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for row in best_by_client:
        lines.append(
            "| {client} | {coverage} | {brp} | {br} | {bnp} | {bn} |".format(
                client=row["client"],
                coverage=row["target_coverage"],
                brp=row["best_RMSE_profile"],
                br=format_float(row["best_RMSE"], 3),
                bnp=row["best_NRMSE_profile"],
                bn=format_float(row["best_NRMSE"], 4),
            )
        )

    lines.extend(
        [
            "",
            "## Global Coverage Sweep",
            "",
            "| profile | coverage | RMSE | NRMSE | N |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sweep_all:
        lines.append(
            f"| {row['profile']} | {row['target_coverage']} | {format_float(row.get('RMSE'), 3)} | {format_float(row.get('NRMSE'), 4)} | {row['N']} |"
        )

    (out_dir / "profile_qc_coverage_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_profiles(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.profiles_json:
        return json.loads(Path(args.profiles_json).read_text(encoding="utf-8"))
    return DEFAULT_PROFILES


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_rows = [normalize_qc_row(row) for row in read_csv(args.qc_records)]
    clients = [client_name(item.strip()) for item in args.clients.split(",") if item.strip()]
    coverages = [float(item.strip()) for item in args.coverages.split(",") if item.strip()]
    profiles = load_profiles(args)

    post_rows: list[dict[str, Any]] = []
    sweep_client_rows: list[dict[str, Any]] = []
    sweep_all_rows: list[dict[str, Any]] = []
    manifest_profiles: list[dict[str, Any]] = []

    for profile in profiles:
        name = str(profile["name"])
        profile_key = "profile_ppm"
        rows = build_rows_for_profile(qc_rows, profile, profile_key)
        post_rows.extend(post_qc_metric_rows(rows, name, profile_key, clients))
        sweep_client_rows.extend(coverage_sweep_rows(rows, name, profile_key, coverages, by_client=True))
        sweep_all_rows.extend(coverage_sweep_rows(rows, name, profile_key, coverages, by_client=False))
        manifest_profiles.append(profile)

    best_client_rows = best_profile_rows(sweep_client_rows, ["client", "target_coverage"])
    best_all_rows = best_profile_rows(sweep_all_rows, ["scope", "target_coverage"])

    write_csv(out_dir / "profile_post_qc_metrics.csv", post_rows)
    write_csv(out_dir / "profile_qc_threshold_sweep_by_client.csv", sweep_client_rows)
    write_csv(out_dir / "profile_qc_threshold_sweep.csv", sweep_all_rows)
    write_csv(out_dir / "profile_qc_best_by_client.csv", best_client_rows)
    write_csv(out_dir / "profile_qc_best_global.csv", best_all_rows)
    write_report(out_dir, post_rows, sweep_client_rows, sweep_all_rows, best_client_rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "qc_records": args.qc_records,
                "profiles": manifest_profiles,
                "clients": clients,
                "coverages": coverages,
                "outputs": [
                    "profile_post_qc_metrics.csv",
                    "profile_qc_threshold_sweep_by_client.csv",
                    "profile_qc_threshold_sweep.csv",
                    "profile_qc_best_by_client.csv",
                    "profile_qc_best_global.csv",
                    "profile_qc_coverage_audit_report.md",
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "profiles": [p["name"] for p in profiles]}, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-records", default="results/f6_fixed_da_strong_r25_profile_replay_20260630/inputs/qc_test_records.csv")
    parser.add_argument("--profiles-json", default="")
    parser.add_argument("--clients", default="3,4,5")
    parser.add_argument("--coverages", default="0.75,0.80,0.85,0.90,0.95,1.0")
    parser.add_argument("--output-dir", default="results/f6_fixed_da_strong_r25_profile_replay_20260630/profile_qc_coverage_audit")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
