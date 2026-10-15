"""Measure child-process launch, runtime readiness, and first inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def child(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from scripts.benchmark_iotj_final_runtime import _load_inputs, _runtime_factory

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    windows, metadata, phases, _keys = _load_inputs(args.data_root)
    tick = time.perf_counter_ns()
    _runtime, infer = _runtime_factory(args.runtime, args.contract, "cpu")
    bundle_load_ms = (time.perf_counter_ns() - tick) / 1e6
    print(json.dumps({"event": "runtime_ready", "bundle_load_ms": bundle_load_ms}), flush=True)
    tick = time.perf_counter_ns()
    rows = infer(windows[0:1], metadata[0:1], phases[0:1])
    if len(rows) != 1:
        raise ValueError("cold-start first inference row count differs")
    print(json.dumps({"event": "first_inference_complete", "first_inference_ms": (time.perf_counter_ns() - tick) / 1e6}), flush=True)


def parent(args: argparse.Namespace) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--runtime", args.runtime, "--contract", str(args.contract), "--data-root", str(args.data_root), "--output", str(args.output)]
    started = time.perf_counter_ns()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    ready = json.loads(process.stdout.readline())
    ready_elapsed = (time.perf_counter_ns() - started) / 1e6
    first = json.loads(process.stdout.readline())
    first_elapsed = (time.perf_counter_ns() - started) / 1e6
    stderr = process.stderr.read() if process.stderr is not None else ""
    returncode = process.wait()
    if returncode != 0 or ready.get("event") != "runtime_ready" or first.get("event") != "first_inference_complete":
        raise RuntimeError(f"cold-start child failed: returncode={returncode}; stderr={stderr}")
    payload = {
        "schema_version": "iotj.runtime_cold_start.v1", "runtime": args.runtime,
        "python_launch_to_runtime_ready_ms": ready_elapsed,
        "python_launch_to_first_inference_complete_ms": first_elapsed,
        "bundle_load_ms": ready["bundle_load_ms"], "first_inference_ms": first["first_inference_ms"],
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--runtime", required=True, choices=["RUNTIME_V4_FULL", "RUNTIME_V5_REGRESSION_CORE", "RUNTIME_V5_QC2_CANDIDATE"])
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    child(parsed) if parsed.child else parent(parsed)
