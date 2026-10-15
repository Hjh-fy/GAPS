"""Independently recompute C5 factorial metrics using only csv + NumPy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric(rows: list[dict[str, str]], pred_key: str) -> dict[str, float | int]:
    true = np.asarray([float(r["true_ppm"]) for r in rows], dtype=np.float64)
    pred = np.asarray([float(r[pred_key]) for r in rows], dtype=np.float64)
    err = pred - true
    den = float(np.sum((true - np.mean(true)) ** 2))
    return {"N": len(rows), "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))),
            "Bias": float(np.mean(err)), "R2": float(1.0 - np.sum(err ** 2) / den) if den else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=ROOT / "results/iotj_c5_pipeline_audit_20260807"); args=ap.parse_args()
    out=args.output.resolve(); records=read(out/"p6_2x2_factorial_records.csv"); formal=read(out/"p6_2x2_factorial_summary.csv")
    formal_lookup={(r["experiment"],r["scope"]):r for r in formal}; result=[]; max_diff=0.0
    for exp in sorted(set(r["experiment"] for r in records)):
        exp_rows=[r for r in records if r["experiment"]==exp]
        scopes={"PIPELINE_ALL":(exp_rows,"pipeline_pred_ppm"),"S_CC":([r for r in exp_rows if int(r["route_correct"])],"pipeline_pred_ppm"),"ORACLE_ROUTE":(exp_rows,"oracle_pred_ppm")}
        for scope,(rows,key) in scopes.items():
            groupings=[("all","all",rows)]
            for gas in sorted(set(r["gas"] for r in rows)):
                gas_rows=[r for r in rows if r["gas"]==gas]; groupings.append(("gas",gas,gas_rows))
                for ppm in sorted(set(float(r["true_ppm"]) for r in gas_rows)):
                    groupings.append(("concentration",f"{gas}:{ppm:g}",[r for r in gas_rows if float(r["true_ppm"])==ppm]))
            for level,value,selected in groupings:
                m=metric(selected,key); diff=""
                if level=="all":
                    ref=formal_lookup[(exp,scope)]; diff=abs(m["RMSE"]-float(ref["RMSE"])); max_diff=max(max_diff,float(diff))
                result.append({"experiment":exp,"scope":scope,"group_level":level,"group_value":value,**m,"formal_RMSE_abs_difference":diff})
    with (out/"p8_independent_metric_check.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    status="PASS" if max_diff < 1e-8 else "FAIL"
    (out/"P8_METRIC_AUDIT.md").write_text(f"# P8 independent metric audit\n\n{status}. This script imports no project evaluator or RMSE helper. It reads persisted row records and directly computes `sqrt(mean((pred-true)**2))` in NumPy. Maximum absolute RMSE difference versus the P6 summary is `{max_diff:.17g}` (required < 1e-8). S_CC is independently filtered by `route_correct == 1`. Per-arm, per-gas and per-concentration N/RMSE/MAE/Bias/R2 are in `p8_independent_metric_check.csv`.\n",encoding="utf-8")
    if status != "PASS": raise RuntimeError("independent metric mismatch")
    print({"status":status,"max_abs_difference":max_diff})


if __name__ == "__main__": main()
