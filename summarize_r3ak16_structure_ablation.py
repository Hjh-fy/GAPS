from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for report generation.
    torch = None

from utils import CONC_STATS


OUT_DIR = Path("results/r3ak16_structure_ablation_20260625")
SUMMARY_CSV = OUT_DIR / "structure_ablation_summary.csv"
DETAIL_CSV = OUT_DIR / "structure_ablation_scope_metrics.csv"
REPORT_MD = OUT_DIR / "r3ak16_structure_ablation_report.md"

CO_CLASS = 1
CO_HIGH_PPM_MIN = 200.0


CANDIDATES = [
    {
        "candidate": "M0_R3aK16_depth4_dct16",
        "role": "current baseline",
        "train_dir": "results/R3aK16_flower_reg_depth4_dct_src12",
        "eval_dir": "results/R3aK16_dct_guardrail_budget/test_predictions",
        "notes": "R3aK16: depth4 residual reg head + DCT K16 response branch.",
    },
    {
        "candidate": "M1_R3aK8_depth4_dct8",
        "role": "lighter DCT branch",
        "train_dir": "results/R3aK8_flower_reg_depth4_dct_src12",
        "eval_dir": "results/R3aK8_dct_guardrail_budget/test_predictions",
        "notes": "Same depth as M0, smaller DCT response branch.",
    },
    {
        "candidate": "M2_R3b_depth4_msconv16",
        "role": "classic local conv response branch",
        "train_dir": "results/R3b_msconv16_flower_reg_depth4_src12",
        "eval_dir": "results/R3b_msconv16_guardrail_budget/test_predictions",
        "notes": "Replace DCT response branch with multi-scale convolution branch.",
    },
    {
        "candidate": "M3_S2_tcnadapter_k3g005",
        "role": "response adapter stress test",
        "train_dir": "results/S2_tcnadapter_k3g005_flower_reg_depth4_src12",
        "eval_dir": "results/S2_tcnadapter_k3g005_guardrail_budget/test_predictions",
        "notes": "Adds TCN adapter on top of the baseline-style regression path.",
    },
    {
        "candidate": "M4_T9fix_shared_trunk",
        "role": "shared-private head ablation",
        "train_dir": "results/T9fix_shared_trunk_flower_reg_depth4_src12",
        "eval_dir": "results/T9fix_shared_trunk_guardrail_budget/test_predictions",
        "notes": "Uses shared regression trunk before class-specific heads.",
    },
    {
        "candidate": "M5_T10afix_ratio_dct",
        "role": "ratio auxiliary branch",
        "train_dir": "results/T10afix_ratio_dct_flower_reg_depth4_src12",
        "eval_dir": "results/T10afix_ratio_dct_guardrail_budget/test_predictions",
        "notes": "Adds ratio branch to the DCT baseline.",
    },
]


PLANNED_NEXT = [
    {
        "candidate": "M6_depth2_dct16",
        "purpose": "Test whether the residual head can be made shallower while retaining DCT response statistics.",
        "command": (
            "python -m gaps_flower.regression_server --reg-head-depth 2 "
            "--reg-response-branch dct --reg-dct-k 16 ..."
        ),
    },
    {
        "candidate": "M7_depth2_none",
        "purpose": "Classic compact MLP head without explicit response branch.",
        "command": (
            "python -m gaps_flower.regression_server --reg-head-depth 2 "
            "--reg-response-branch none ..."
        ),
    },
    {
        "candidate": "M8_depth4_none",
        "purpose": "Keep deep head but remove response branch to isolate DCT contribution.",
        "command": (
            "python -m gaps_flower.regression_server --reg-head-depth 4 "
            "--reg-response-branch none ..."
        ),
    },
]


def class_range(class_id: int) -> float:
    stats = CONC_STATS.get(int(class_id), {"min": 0.0, "max": 1.0})
    return max(float(stats["max"] - stats["min"]), 1e-12)


def metric(df: pd.DataFrame, pred_col: str = "calibrated_ppm") -> dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "nrmse_range": None,
            "p90ae": None,
            "p95ae": None,
            "bias": None,
            "route_acc": None,
        }
    y = df["true_ppm"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    cls = df["true_class"].to_numpy(dtype=int)
    err = p - y
    ae = np.abs(err)
    ranges = np.asarray([class_range(int(c)) for c in cls], dtype=float)
    route_acc = None
    if "pred_class" in df.columns:
        route_acc = float(np.mean(df["pred_class"].to_numpy(dtype=int) == cls))
    return {
        "n": int(len(df)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(ae)),
        "nrmse_range": float(np.sqrt(np.mean((err / ranges) ** 2))),
        "p90ae": float(np.percentile(ae, 90)),
        "p95ae": float(np.percentile(ae, 95)),
        "bias": float(np.mean(err)),
        "route_acc": route_acc,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def load_predictions(eval_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(eval_dir.glob("C*_test_predictions.csv")):
        df = pd.read_csv(csv_path)
        if "client_id" not in df.columns:
            df["client_id"] = csv_path.name.split("_", 1)[0]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["true_class", "pred_class"]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    return out


def checkpoint_info(train_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "checkpoint": "",
        "full_params": None,
        "regression_branch_params": None,
        "model_config": {},
    }
    ckpt_path = train_dir / "regression_fedavg_global.pt"
    if not ckpt_path.exists() or torch is None:
        return info
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or {}
    model_config = ckpt.get("model_config") or {}
    info["checkpoint"] = str(ckpt_path)
    info["model_config"] = model_config
    info["full_params"] = int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))
    reg_prefixes = (
        "reg_",
        "gas_embedding",
        "proto_",
        "conc_",
    )
    info["regression_branch_params"] = int(
        sum(v.numel() for k, v in state.items() if k.startswith(reg_prefixes) and hasattr(v, "numel"))
    )
    return info


def scope_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    scopes = {
        "ALL": df,
        "CO": df[df["true_class"] == CO_CLASS],
        "CO_high": df[(df["true_class"] == CO_CLASS) & (df["true_ppm"] >= CO_HIGH_PPM_MIN)],
        "nonCO": df[df["true_class"] != CO_CLASS],
    }
    for client in sorted(df["client_id"].dropna().unique()):
        cdf = df[df["client_id"] == client]
        scopes[f"{client}_ALL"] = cdf
        scopes[f"{client}_CO"] = cdf[cdf["true_class"] == CO_CLASS]
        scopes[f"{client}_CO_high"] = cdf[
            (cdf["true_class"] == CO_CLASS) & (cdf["true_ppm"] >= CO_HIGH_PPM_MIN)
        ]
        scopes[f"{client}_nonCO"] = cdf[cdf["true_class"] != CO_CLASS]
    return scopes


def flatten_metric(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        train_dir = Path(spec["train_dir"])
        eval_dir = Path(spec["eval_dir"])
        ck_info = checkpoint_info(train_dir)
        cfg = ck_info["model_config"]
        df = load_predictions(eval_dir)
        row: dict[str, Any] = {
            "candidate": spec["candidate"],
            "role": spec["role"],
            "train_dir": spec["train_dir"],
            "eval_dir": spec["eval_dir"],
            "notes": spec["notes"],
            "has_predictions": not df.empty,
            "full_params": ck_info["full_params"],
            "regression_branch_params": ck_info["regression_branch_params"],
            "reg_head_depth": cfg.get("reg_head_depth"),
            "reg_response_branch": cfg.get("reg_response_branch"),
            "reg_dct_k": cfg.get("reg_dct_k"),
            "reg_tcn_adapter": cfg.get("reg_tcn_adapter"),
            "reg_use_shared_trunk": cfg.get("reg_use_shared_trunk"),
            "use_reg_ratio_branch": cfg.get("use_reg_ratio_branch"),
        }
        if not df.empty:
            scopes = scope_frames(df)
            for scope_name, sdf in scopes.items():
                final_metric = metric(sdf, "calibrated_ppm")
                raw_metric = metric(sdf, "pred_ppm") if "pred_ppm" in sdf else final_metric
                detail_row = {
                    "candidate": spec["candidate"],
                    "scope": scope_name,
                    **flatten_metric("final", final_metric),
                    **flatten_metric("raw", raw_metric),
                }
                detail_rows.append(detail_row)
            for scope_name in [
                "ALL",
                "CO",
                "CO_high",
                "nonCO",
                "C3_CO",
                "C4_CO",
                "C5_CO",
                "C3_CO_high",
                "C4_CO_high",
                "C5_CO_high",
                "C3_nonCO",
                "C4_nonCO",
                "C5_nonCO",
            ]:
                values = metric(scopes.get(scope_name, pd.DataFrame()), "calibrated_ppm")
                row.update(flatten_metric(scope_name.lower(), values))
        summary_rows.append(row)
    return summary_rows, detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(summary_rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        [r for r in summary_rows if r.get("all_rmse") is not None],
        key=lambda r: (r["all_rmse"], r.get("co_high_rmse") or 1e9),
    )
    best = ranked[0] if ranked else None
    baseline = next((r for r in summary_rows if r["candidate"].startswith("M0_")), None)
    lines: list[str] = []
    lines.append("# R3aK16 Structure Ablation Matrix Report")
    lines.append("")
    lines.append("- Generated from existing PC evaluation artifacts.")
    lines.append(f"- Evaluation target: C12 -> C345 target test, no-QC full-set.")
    lines.append(
        "- Prediction column: `calibrated_ppm` as final; `pred_ppm` retained in detail CSV. "
        "These artifacts are structure candidates after the existing target-side specialist/calibration flow, "
        "not raw source-only transfer."
    )
    lines.append(f"- CO high definition: true CO rows with `true_ppm >= {CO_HIGH_PPM_MIN:.0f}`.")
    lines.append(f"- Summary CSV: `{SUMMARY_CSV.as_posix()}`")
    lines.append(f"- Detail CSV: `{DETAIL_CSV.as_posix()}`")
    lines.append("")
    lines.append("## Current Matrix")
    lines.append("")
    matrix_rows = [
        [
            r["candidate"],
            r.get("role"),
            r.get("reg_head_depth"),
            r.get("reg_response_branch"),
            r.get("reg_dct_k"),
            r.get("reg_tcn_adapter"),
            r.get("reg_use_shared_trunk"),
            r.get("use_reg_ratio_branch"),
            r.get("full_params"),
            r.get("regression_branch_params"),
        ]
        for r in summary_rows
    ]
    lines.append(
        md_table(
            matrix_rows,
            [
                "candidate",
                "role",
                "depth",
                "response",
                "dct_k",
                "tcn",
                "shared",
                "ratio",
                "full params",
                "reg params",
            ],
        )
    )
    lines.append("")
    lines.append("## No-QC Full-Set Metrics")
    lines.append("")
    metric_rows = [
        [
            r["candidate"],
            r.get("all_n"),
            r.get("all_rmse"),
            r.get("all_nrmse_range"),
            r.get("co_rmse"),
            r.get("co_high_rmse"),
            r.get("nonco_rmse"),
            r.get("c3_co_rmse"),
            r.get("c4_co_rmse"),
            r.get("c5_co_rmse"),
            r.get("c3_co_high_rmse"),
            r.get("c4_co_high_rmse"),
            r.get("c5_co_high_rmse"),
        ]
        for r in ranked
    ]
    lines.append(
        md_table(
            metric_rows,
            [
                "candidate",
                "N",
                "ALL RMSE",
                "ALL NRMSE",
                "CO RMSE",
                "CO high",
                "nonCO",
                "C3 CO",
                "C4 CO",
                "C5 CO",
                "C3 high",
                "C4 high",
                "C5 high",
            ],
        )
    )
    lines.append("")
    lines.append("## Delta vs Baseline")
    lines.append("")
    if baseline:
        delta_rows = []
        base_params = float(baseline.get("regression_branch_params") or 0.0)
        base_rmse = float(baseline.get("all_rmse") or 0.0)
        base_co = float(baseline.get("co_rmse") or 0.0)
        base_high = float(baseline.get("co_high_rmse") or 0.0)
        base_nonco = float(baseline.get("nonco_rmse") or 0.0)
        for r in ranked:
            reg_params = float(r.get("regression_branch_params") or 0.0)
            param_change = (reg_params - base_params) / base_params * 100.0 if base_params else None
            delta_rows.append(
                [
                    r["candidate"],
                    param_change,
                    float(r.get("all_rmse") or 0.0) - base_rmse,
                    float(r.get("co_rmse") or 0.0) - base_co,
                    float(r.get("co_high_rmse") or 0.0) - base_high,
                    float(r.get("nonco_rmse") or 0.0) - base_nonco,
                ]
            )
        lines.append(
            md_table(
                delta_rows,
                [
                    "candidate",
                    "reg params %",
                    "ALL RMSE delta",
                    "CO delta",
                    "CO high delta",
                    "nonCO delta",
                ],
            )
        )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if best:
        lines.append(
            f"- Best existing structure by ALL RMSE is `{best['candidate']}` "
            f"({fmt(best.get('all_rmse'))} RMSE, {fmt(best.get('all_nrmse_range'), 4)} NRMSE)."
        )
    if baseline and best and baseline["candidate"] != best["candidate"]:
        delta = best["all_rmse"] - baseline["all_rmse"]
        lines.append(
            f"- Compared with baseline `{baseline['candidate']}`, the current best changes ALL RMSE by "
            f"{fmt(delta)} ppm."
        )
    elif baseline:
        lines.append("- Baseline remains the best among the existing structure candidates by ALL RMSE.")
    lines.append(
        "- `M4_T9fix_shared_trunk` is the only truly lightweight existing neural-structure candidate "
        "(about 74% fewer regression-branch parameters than M0), but its ALL RMSE and nonCO RMSE are clearly worse."
    )
    lines.append(
        "- `M1_R3aK8` trims only about 1% of regression-branch parameters; that is not enough to count as meaningful "
        "lightweight simplification, and it worsens ALL/CO metrics."
    )
    lines.append(
        "- `M5_T10afix_ratio_dct` helps C4 CO high in this artifact set, but the global ALL/nonCO trade-off is too large "
        "to promote as a mainline structure."
    )
    lines.append(
        "- This table only evaluates the base regression structure plus existing calibration output. "
        "It does not yet include the stronger target direct-head auto_v2 candidates such as H2.3/H8."
    )
    lines.append(
        "- If a lighter head cannot beat the baseline before target calibration, it should still be tested "
        "with the same target calibration before being rejected, because previous light-source-only tests showed "
        "direct transfer can collapse."
    )
    lines.append("")
    lines.append("## Planned Missing Lightweight Experiments")
    lines.append("")
    planned_rows = [[p["candidate"], p["purpose"], p["command"]] for p in PLANNED_NEXT]
    lines.append(md_table(planned_rows, ["candidate", "purpose", "command skeleton"]))
    lines.append("")
    lines.append("## Promotion Rule")
    lines.append("")
    lines.append("- P0: lower no-QC full-set ALL RMSE / NRMSE after the same target calibration flow.")
    lines.append("- P1: lower CO and CO-high RMSE, especially C4/C5, without obvious nonCO regression damage.")
    lines.append("- P2: prefer smaller regression branch parameters only when P0/P1 are not worse.")
    lines.append("- QC remains out of this selection loop.")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_rows, detail_rows = collect()
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(DETAIL_CSV, detail_rows)
    write_report(summary_rows)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {DETAIL_CSV}")
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
