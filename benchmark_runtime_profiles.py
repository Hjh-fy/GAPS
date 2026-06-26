"""Benchmark deployment runtime profiles for size and CPU latency.

This script is intentionally deployment-facing: it loads exported bundles from
``results/deployment_*`` and measures the actual ``FinalDeployRuntime`` path on
target test windows. Profiles without an exported runtime bundle are recorded as
pending instead of being inferred from analysis-only artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROFILES = [
    {
        "profile": "H2.3",
        "role": "balanced_mainline",
        "bundle": "results/deployment_h2_3_mlp_ridge_candidate_20260624",
        "expected_full_all_rmse": 18.62,
        "notes": "balanced no-QC full-set mainline",
    },
    {
        "profile": "H8",
        "role": "co_specialist",
        "bundle": "results/deployment_h8_source_aug_candidate_20260625",
        "expected_full_all_rmse": 18.47,
        "notes": "CO-specialist source-aug target ridge",
    },
    {
        "profile": "H8+C4",
        "role": "guarded_co_priority",
        "bundle": "results/deployment_h8_formal_c4_rescue_candidate_20260625",
        "expected_full_all_rmse": 18.30,
        "notes": "H8 plus calibration-selected formal C4 route-rescue",
    },
    {
        "profile": "L1",
        "role": "deployment_lite_candidate",
        "bundle": "results/deployment_l1_lightweight_candidate_20260626",
        "expected_full_all_rmse": 22.6,
        "notes": "pending exported runtime bundle; keep only if size/latency improves",
    },
    {
        "profile": "B0",
        "role": "baseline_reference",
        "bundle": "results/deployment_b0_r3ak16_candidate_20260626",
        "expected_full_all_rmse": 27.34,
        "notes": "pending exported runtime bundle for original auto_v2 baseline",
    },
]

MODEL_SUFFIXES = {".pt", ".pth", ".onnx", ".npz"}
CLASS_RANGES = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}


def parse_clients(text: str) -> list[str]:
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def client_num(client: str) -> int:
    return int(str(client).upper().replace("CLIENT_", "").replace("C", ""))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dir_size_mb(path: Path, suffixes: set[str] | None = None) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if suffixes is not None and item.suffix.lower() not in suffixes:
            continue
        total += item.stat().st_size
    return total / (1024.0 * 1024.0)


def load_metadata(path: Path, count: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(item) if isinstance(item, dict) else {} for item in payload[:count]]


def clear_runtime_modules() -> None:
    for name in list(sys.modules):
        if name == "gaps_deploy" or name.startswith("gaps_deploy."):
            del sys.modules[name]


def load_final_runtime(bundle: Path):
    runtime_src = bundle / "runtime_src"
    if not runtime_src.exists():
        raise FileNotFoundError(f"missing runtime_src: {runtime_src}")
    clear_runtime_modules()
    sys.path.insert(0, str(runtime_src))
    try:
        module = importlib.import_module("gaps_deploy.final_runtime")
        return module.FinalDeployRuntime
    finally:
        try:
            sys.path.remove(str(runtime_src))
        except ValueError:
            pass


def rmse_for_rows(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    pred = np.asarray([float(row["co_corrected_ppm"]) for row in rows], dtype=np.float64)
    true = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    cls = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    err = pred - true
    ranges = np.asarray([CLASS_RANGES[int(c)] for c in cls], dtype=np.float64)
    return float(np.sqrt(np.mean(err * err))), float(np.sqrt(np.mean((err / ranges) ** 2)))


def benchmark_profile(
    profile: dict[str, Any],
    data_root: Path,
    clients: list[str],
    limit: int,
    repeats: int,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = Path(profile["bundle"])
    artifact_size_mb = dir_size_mb(bundle)
    model_file_size_mb = dir_size_mb(bundle, MODEL_SUFFIXES)
    common = {
        "profile": profile["profile"],
        "role": profile["role"],
        "bundle": str(bundle),
        "artifact_size_mb": round(artifact_size_mb, 4),
        "model_file_size_mb": round(model_file_size_mb, 4),
        "expected_full_all_rmse": profile.get("expected_full_all_rmse", ""),
        "notes": profile.get("notes", ""),
    }
    if not bundle.exists() or not (bundle / "runtime_config.json").exists():
        return (
            [{**common, "client": "ALL", "status": "missing_bundle"}],
            [{**common, "status": "missing_bundle"}],
        )

    client_rows: list[dict[str, Any]] = []
    all_pred_rows: list[dict[str, Any]] = []
    try:
        Runtime = load_final_runtime(bundle)
    except Exception as exc:  # noqa: BLE001
        return (
            [{**common, "client": "ALL", "status": "runtime_import_failed", "error": repr(exc)}],
            [{**common, "status": "runtime_import_failed", "error": repr(exc)}],
        )

    for client in clients:
        cdir = data_root / f"client_{client_num(client)}"
        try:
            features = np.load(cdir / "test_features.npy").astype(np.float32)
            phases = np.load(cdir / "test_phase_labels.npy").astype(np.int64)
            cls = np.load(cdir / "test_classification_labels.npy").astype(np.int64)
            reg = np.load(cdir / "test_regression_labels.npy").astype(np.float32)
            count = min(int(limit), int(features.shape[0])) if limit > 0 else int(features.shape[0])
            features = features[:count]
            phases = phases[:count]
            metadata = load_metadata(cdir / "test_experiment_info.json", count)
            runtime = Runtime(bundle, client, device=device)
            warmup_count = min(8, count)
            if warmup_count > 0:
                runtime.predict_batch(features[:warmup_count], phase=phases[:warmup_count], metadata=metadata[:warmup_count])

            elapsed_per_window: list[float] = []
            runtime_total_per_window: list[float] = []
            last_rows: list[dict[str, Any]] = []
            for _ in range(int(repeats)):
                started = time.perf_counter()
                last_rows = runtime.predict_batch(features, phase=phases, metadata=metadata)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                elapsed_per_window.append(elapsed_ms / max(1, count))
                runtime_total_per_window.append(
                    float(runtime.last_timing_ms.get("runtime_total_ms", elapsed_ms)) / max(1, count)
                )

            pred_rows: list[dict[str, Any]] = []
            for idx, row in enumerate(last_rows):
                true_class = int(cls[idx])
                item = dict(row)
                item.update(
                    {
                        "profile": profile["profile"],
                        "client": client,
                        "sample_index": idx,
                        "true_class": true_class,
                        "true_ppm": float(reg[idx, true_class]),
                    }
                )
                pred_rows.append(item)
            all_pred_rows.extend(pred_rows)
            subset_rmse, subset_nrmse = rmse_for_rows(pred_rows)
            client_rows.append(
                {
                    **common,
                    "client": client,
                    "status": "ok",
                    "rows": count,
                    "repeats": repeats,
                    "load_ms": round(float(runtime.model_load_ms), 3),
                    "mean_latency_ms_per_window": round(float(np.mean(elapsed_per_window)), 5),
                    "p90_latency_ms_per_window": round(float(np.percentile(elapsed_per_window, 90)), 5),
                    "runtime_total_ms_per_window": round(float(np.mean(runtime_total_per_window)), 5),
                    "auto_output_field_present": bool(last_rows and "auto_output_ppm" in last_rows[0]),
                    "benchmark_subset_rmse": round(float(subset_rmse), 6) if subset_rmse is not None else "",
                    "benchmark_subset_nrmse": round(float(subset_nrmse), 6) if subset_nrmse is not None else "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            client_rows.append({**common, "client": client, "status": "benchmark_failed", "error": repr(exc)})
    summary_rmse, summary_nrmse = rmse_for_rows(all_pred_rows)
    summary = {
        **common,
        "status": "ok" if all(row.get("status") == "ok" for row in client_rows) else "partial",
        "clients": ",".join(clients),
        "rows": sum(int(row.get("rows", 0) or 0) for row in client_rows),
        "mean_latency_ms_per_window": round(
            float(np.mean([float(row["mean_latency_ms_per_window"]) for row in client_rows if row.get("status") == "ok"])),
            5,
        )
        if any(row.get("status") == "ok" for row in client_rows)
        else "",
        "p90_latency_ms_per_window": round(
            float(np.percentile([float(row["p90_latency_ms_per_window"]) for row in client_rows if row.get("status") == "ok"], 90)),
            5,
        )
        if any(row.get("status") == "ok" for row in client_rows)
        else "",
        "benchmark_subset_rmse": round(float(summary_rmse), 6) if summary_rmse is not None else "",
        "benchmark_subset_nrmse": round(float(summary_nrmse), 6) if summary_nrmse is not None else "",
        "auto_output_field_present": all(
            bool(row.get("auto_output_field_present")) for row in client_rows if row.get("status") == "ok"
        ),
    }
    return client_rows, [summary]


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "profile",
        "role",
        "status",
        "artifact_size_mb",
        "model_file_size_mb",
        "mean_latency_ms_per_window",
        "p90_latency_ms_per_window",
        "expected_full_all_rmse",
        "benchmark_subset_rmse",
        "auto_output_field_present",
    ]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(col, "")) for col in columns) + "|")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exported GAPS deployment runtime profiles.")
    parser.add_argument("--data-root", default="dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid")
    parser.add_argument("--clients", default="C3,C4,C5")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/runtime_profile_benchmark_20260626")
    parser.add_argument("--profiles-json", default="", help="Optional JSON list overriding the built-in profile matrix.")
    args = parser.parse_args()

    profiles = DEFAULT_PROFILES
    if args.profiles_json:
        profiles = json.loads(Path(args.profiles_json).read_text(encoding="utf-8"))
    clients = parse_clients(args.clients)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for profile in profiles:
        rows, summaries = benchmark_profile(
            profile,
            data_root=Path(args.data_root),
            clients=clients,
            limit=args.limit,
            repeats=args.repeats,
            device=args.device,
        )
        client_rows.extend(rows)
        summary_rows.extend(summaries)

    write_csv(out / "profile_client_latency.csv", client_rows)
    write_csv(out / "profile_summary.csv", summary_rows)
    manifest = {
        "data_root": str(Path(args.data_root)),
        "clients": clients,
        "limit": args.limit,
        "repeats": args.repeats,
        "device": args.device,
        "outputs": {
            "client_latency": str(out / "profile_client_latency.csv"),
            "summary": str(out / "profile_summary.csv"),
            "report": str(out / "runtime_profile_benchmark_report.md"),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Runtime Profile Benchmark",
            "",
            f"- data_root: `{args.data_root}`",
            f"- clients: `{','.join(clients)}`",
            f"- limit per client: `{args.limit}`",
            f"- repeats: `{args.repeats}`",
            f"- device: `{args.device}`",
            "",
            markdown_table(summary_rows),
            "",
            "Notes:",
            "- `benchmark_subset_rmse` is measured on the benchmark subset only; use `expected_full_all_rmse` for the fixed no-QC full-set model comparison.",
            "- Profiles with `missing_bundle` are not deployment-validated yet and should not be claimed as runtime-ready.",
            "- `auto_output_field_present` checks whether the public runtime row exposes the deployment-only accepted-output field.",
            "",
        ]
    )
    (out / "runtime_profile_benchmark_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote runtime profile benchmark to {out}")


if __name__ == "__main__":
    main()
