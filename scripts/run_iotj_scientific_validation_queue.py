"""Continue the frozen scientific-validation queue after the active A0T run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
A0T_ROOT = ROOT / "results/iotj_canonical_v1_final_20260808/a0t_equal_label/classification"
QUEUE_ROOT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/queue"


def required_a0t_markers() -> list[Path]:
    return [A0T_ROOT / f"CANONICAL-V1-A0T-C{client}/fixed_endpoint_complete.json" for client in (3, 4, 5)]


def process_exists(pid: int) -> bool:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"],
        cwd=ROOT,
    )
    return completed.returncode == 0


def write_state(status: str, step: str, **extra: object) -> None:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "step": step, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra}
    (QUEUE_ROOT / "queue_state.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_step(name: str, command: list[str], timeout_hours: float = 24.0) -> None:
    write_state("RUNNING", name, command=command)
    subprocess.run(command, cwd=ROOT, check=True, timeout=timeout_hours * 3600)


def run_queue(a0t_pid: int, poll_seconds: int) -> None:
    write_state("WAITING", "A0T", a0t_pid=a0t_pid)
    while process_exists(a0t_pid):
        time.sleep(poll_seconds)
    missing = [str(path) for path in required_a0t_markers() if not path.is_file()]
    if missing:
        raise RuntimeError(f"FAIL_CLOSED A0T process ended without all endpoints: {missing}")
    python = sys.executable
    run_step("CANONICAL_COMPARATORS", [python, "scripts/run_iotj_canonical_v1_comparators.py", "--timeout-hours", "12"])
    run_step("DEPLOY_STRICT_DATASET", [python, "scripts/deploy_iotj_strict_dataset.py"], timeout_hours=2)
    run_step("STRICT_A4_R84", [python, "scripts/run_iotj_canonical_v1_strict_nonoverlap.py", "--timeout-hours", "12"], timeout_hours=36)
    run_step("SEALED_COMPARATOR_EVALUATION", [python, "scripts/evaluate_iotj_canonical_v1_comparators.py", "--device", "cpu"], timeout_hours=3)
    run_step("STRICT_ANALYSIS", [python, "scripts/analyze_iotj_canonical_v1_strict_nonoverlap.py"], timeout_hours=3)
    write_state("COMPLETE", "PENDING_FINAL_AUDIT_AND_REPORT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0t-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    try:
        run_queue(args.a0t_pid, args.poll_seconds)
    except Exception as exc:
        write_state("FAILED", "QUEUE", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
