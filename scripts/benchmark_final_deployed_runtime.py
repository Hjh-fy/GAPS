"""Benchmark FINAL_DEPLOYED_RUNTIME with the frozen batch-one protocol."""

from __future__ import annotations

import argparse, json, math, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import psutil
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR if (SCRIPT_DIR / "gaps_deploy").is_dir() else SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.final_a4_runtime import FinalA4Runtime


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def thermal() -> float | None:
    try: return float(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    except (OSError, ValueError): return None


def throttled() -> str | None:
    try: return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError): return None


def main(args: argparse.Namespace) -> None:
    if args.runs < 5000 or args.runs > 10000 or args.warmup != 100 or args.threads != 1:
        raise ValueError("frozen final protocol requires 5000-10000 runs, warmup=100, threads=1")
    if args.output.exists(): raise FileExistsError(args.output)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    windows = np.load(args.data_root / "test_features.npy", mmap_mode="r")
    phases = np.load(args.data_root / "test_phase_labels.npy", mmap_mode="r")
    metadata = json.loads((args.data_root / "test_experiment_info.json").read_text(encoding="utf-8"))
    if windows.shape != (1360, 100, 8) or len(metadata) != 1360: raise ValueError("C5 test universe drift")
    process = psutil.Process(os.getpid()); baseline = process.memory_info().rss
    runtime = FinalA4Runtime(args.package_root)
    first = runtime.infer_one(windows[0], metadata[0], int(phases[0]))
    if first["runtime_status"] != "FINAL_DEPLOYED_RUNTIME": raise ValueError("runtime status drift")
    for index in range(args.warmup):
        item = index % len(windows); runtime.infer_one(windows[item], metadata[item], int(phases[item]))
    values: list[float] = []; peak = process.memory_info().rss; temp_start = thermal(); temp_peak = temp_start
    throttle_start = throttled(); start = time.perf_counter_ns()
    for index in range(args.runs):
        item = index % len(windows); tick = time.perf_counter_ns()
        result = runtime.infer_one(windows[item], metadata[item], int(phases[item]))
        if result["runtime_status"] != "FINAL_DEPLOYED_RUNTIME": raise ValueError("runtime status drift")
        values.append((time.perf_counter_ns() - tick) / 1e6)
        peak = max(peak, process.memory_info().rss)
        current = thermal(); temp_peak = current if temp_peak is None else (max(temp_peak, current) if current is not None else temp_peak)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    output = {
        "schema_version": "iotj.final_deployed_runtime_benchmark.v1", "status": "PASS", "runtime": "FINAL_DEPLOYED_RUNTIME", "platform": "Raspberry Pi 5",
        "protocol": {"runs": args.runs, "warmup": args.warmup, "batch_size": 1, "threads": 1, "device": "cpu", "input_order": "C5 test canonical order modulo 1360", "test_used_for_selection": False, "disk_io_in_steady_state": False},
        "latency": {"p50_ms": percentile(values,50), "p95_ms": percentile(values,95), "p99_ms": percentile(values,99), "mean_ms": float(np.mean(values)), "min_ms": min(values), "max_ms": max(values), "throughput_windows_per_s": args.runs/elapsed},
        "resources": {"rss_baseline_bytes": baseline, "peak_rss_bytes": peak, "peak_rss_mib": peak/1048576, "temperature_start_c": temp_start, "temperature_peak_c": temp_peak, "temperature_end_c": thermal(), "throttled_before": throttle_start, "throttled_after": throttled()},
        "environment": {"hostname": platform.node(), "machine": platform.machine(), "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "row_universe": {"source_windows": 1360, "measured_windows": args.runs, "unique_windows": 1360},
    }
    if not all(math.isfinite(x) for x in values): raise ValueError("non-finite latency")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--package-root",type=Path,required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--runs",type=int,default=5000); p.add_argument("--warmup",type=int,default=100); p.add_argument("--threads",type=int,default=1); p.add_argument("--output",type=Path,required=True); main(p.parse_args())
