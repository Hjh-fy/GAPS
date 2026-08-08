"""Benchmark the exact canonical-v1 deployment runtime on Raspberry Pi 5."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.canonical_v1_runtime import CanonicalV1Runtime


COMPONENTS = (
    "preprocessing_ms",
    "classifier_ms",
    "r84_ms",
    "qc_ms",
    "total_pipeline_ms",
)


def latency_summary(component: str, values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("latency values must be finite and non-empty")
    return {
        "component": component,
        "N": len(array),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "P50_ms": float(np.quantile(array, 0.50)),
        "P90_ms": float(np.quantile(array, 0.90)),
        "P95_ms": float(np.quantile(array, 0.95)),
        "P99_ms": float(np.quantile(array, 0.99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _proc_memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, number, _unit = line.split()
            values[name.rstrip(":")] = int(number) * 1024
    return values.get("VmRSS", 0), values.get("VmHWM", 0)


def _governor() -> str:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unknown"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def benchmark(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"benchmark output exists: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    payload = np.load(args.inputs, allow_pickle=False)
    windows = payload["windows"]
    phases = payload["phases"]
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if len(windows) != len(phases) or len(windows) != len(metadata):
        raise RuntimeError("benchmark input alignment differs")
    runtime = CanonicalV1Runtime(args.package, args.target, device="cpu")
    for index in range(args.warmup):
        source = index % len(windows)
        runtime.infer_one(windows[source], metadata[source], int(phases[source]))
    steady_rss, _ = _proc_memory()
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    process_start = time.process_time()
    for index in range(args.windows):
        source = index % len(windows)
        result, timing = runtime.infer_one_timed(
            windows[source], metadata[source], int(phases[source])
        )
        rss, peak = _proc_memory()
        rows.append(
            {
                "iteration": index,
                "source_index": source,
                "target": args.target,
                "runtime_status": result["runtime_status"],
                **timing,
                "rss_bytes": rss,
                "peak_rss_bytes": peak,
            }
        )
    wall_seconds = time.perf_counter() - wall_start
    process_seconds = time.process_time() - process_start
    _write_csv(output / "pi5_latency_raw.csv", rows)
    summary = [latency_summary(component, [float(row[component]) for row in rows]) for component in COMPONENTS]
    peak_rss = max(int(row["peak_rss_bytes"]) for row in rows)
    final_rss = int(rows[-1]["rss_bytes"])
    for row in summary:
        row.update(
            {
                "runtime_status": "FINAL_DEPLOYED_RUNTIME",
                "target": args.target,
                "warmup_windows": args.warmup,
                "measured_windows": args.windows,
                "batch_size": 1,
                "thread_count": args.threads,
                "throughput_windows_per_second": args.windows / wall_seconds,
                "steady_rss_bytes": steady_rss,
                "final_rss_bytes": final_rss,
                "peak_rss_bytes": peak_rss,
                "process_cpu_utilization_percent_of_one_core": 100.0 * process_seconds / wall_seconds,
                "deployment_package_sha256": args.package_sha256,
                "runtime_commit": args.runtime_commit,
            }
        )
    _write_csv(output / "pi5_benchmark_summary.csv", summary)
    environment = {
        "runtime_status": "FINAL_DEPLOYED_RUNTIME",
        "target": args.target,
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "thread_count": args.threads,
        "batch_size": 1,
        "governor": _governor(),
        "warmup_windows": args.warmup,
        "measured_windows": args.windows,
        "deployment_package_sha256": args.package_sha256,
        "runtime_commit": args.runtime_commit,
    }
    (output / "benchmark_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    total = next(row for row in summary if row["component"] == "total_pipeline_ms")
    (output / "PI5_BENCHMARK.md").write_text(
        "# Raspberry Pi 5 formal benchmark\n\n"
        f"Status: **FINAL_DEPLOYED_RUNTIME**. Target profile: {args.target}. "
        f"Measured {args.windows} batch-1 windows after {args.warmup} warm-up windows. "
        f"Total pipeline P50/P95/P99 = {total['P50_ms']:.3f}/{total['P95_ms']:.3f}/{total['P99_ms']:.3f} ms; "
        f"throughput = {args.windows / wall_seconds:.3f} windows/s; peak RSS = {peak_rss / 2**20:.3f} MiB. "
        "No model, preprocessing, Ridge, or QC setting was changed after this measurement.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="C5")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--windows", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--runtime-commit", required=True)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
