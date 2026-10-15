"""Run one preserved-evidence B5 formal OFF repeat with a fresh attempt ID."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.diagnose_b5_fixed_state_order_replay import recursive_manifest
from scripts import run_iotj_observer_equivalence_gate as gate
from scripts.run_iotj_confirmation_observability import (
    DEFAULT_PC_RUNTIME_ROOT,
    Attempt,
    FormalSmokeConfig,
    ProductionRuntime,
    _execute_hook_lifecycle,
    build_production_hooks,
    canonical_sha256,
)


_RUN_ID_RE = re.compile(r"^c12_to_c5__(b2|b5)__s(42|43|44|45|46)$")
REPO_ROOT = Path(__file__).resolve().parents[1]


def noncanonical_attempt_id(run_id: str, attempt_number: int) -> str:
    """Reserve 900--997 for explicit probes; 998/999 remain formal OFF/ON."""

    if _RUN_ID_RE.fullmatch(str(run_id)) is None:
        raise ValueError("invalid confirmation run_id")
    if type(attempt_number) is not int or not 900 <= attempt_number <= 997:
        raise ValueError("probe attempt number must be an integer in 900..997")
    return f"{run_id}__a{attempt_number:03d}"


def _run_numbered_off_attempt(
    output: Path,
    *,
    run_id: str,
    attempt_number: int,
    provenance: Any,
    hooks: Any,
) -> Attempt:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"probe output already exists: {output}")
    output.mkdir(parents=True)
    attempt_id = noncanonical_attempt_id(run_id, attempt_number)
    attempt = Attempt(run_id=run_id, attempt_id=attempt_id, path=output)
    marker = {
        "schema_version": 1,
        "namespace": "noncanonical_smoke",
        "noncanonical_smoke": True,
        "mode": "off",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "confirmation_commit": provenance.confirmation_commit,
        "source_archive_sha256": provenance.source_archive_sha256,
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "algorithm_config_sha256": provenance.algorithm_config_sha256,
        "diagnostic_purpose": "independent_formal_off_repeatability",
    }
    marker["binding_sha256"] = canonical_sha256(marker)
    (output / "noncanonical_smoke.json").write_text(
        json.dumps(marker, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    try:
        _execute_hook_lifecycle(attempt, hooks)
    except BaseException as exc:
        failure = {
            **marker,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (output / "noncanonical_smoke_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        raise
    success = {**marker, "status": "evidence_recovered"}
    (output / "noncanonical_smoke_complete.json").write_text(
        json.dumps(success, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return attempt


def run_probe(
    protocol_manifest: Path,
    reference_root: Path,
    output_root: Path,
    *,
    attempt_number: int,
) -> dict[str, Any]:
    reference = Path(reference_root).resolve(strict=True)
    reference_before = recursive_manifest(reference)
    binding = gate._load_formal_frozen_binding(Path(protocol_manifest), "B5")
    frozen = binding["_frozen"]
    frozen_run = binding["_frozen_run"]
    root = gate._prepare_output_root(Path(output_root))
    initial = gate._create_frozen_initial_checkpoint(
        root / "frozen_initial_checkpoint.pth", "B5"
    )
    runtime = ProductionRuntime(
        frozen=frozen,
        frozen_run=frozen_run,
        deployments={},
        ecs_host=os.environ.get("IOTJ_ECS_HOST", "root@121.40.139.213"),
        pi_host=os.environ.get("IOTJ_PI_HOST", "gaps@192.168.31.184"),
        validator=REPO_ROOT / "scripts" / "validate_iotj_confirmation_attempt.py",
        poll_seconds=1.0,
        timeout_seconds=1800.0,
        pc_runtime_root=Path(
            os.environ.get("IOTJ_PC_RUNTIME_ROOT", str(DEFAULT_PC_RUNTIME_ROOT))
        ),
    )
    smoke = FormalSmokeConfig(
        observer_enabled=False,
        mode="off",
        trace_output="{ecs_raw}/common_trace.jsonl",
        initial_checkpoint="{ecs_root}/frozen_initial_checkpoint.pth",
        initial_checkpoint_source=Path(initial["path"]),
        initial_checkpoint_sha256=initial["raw_sha256"],
    )
    repeat_root = root / "off_repeat"
    repeat = _run_numbered_off_attempt(
        repeat_root,
        run_id=frozen_run.run_id,
        attempt_number=attempt_number,
        provenance=frozen_run.provenance,
        hooks=build_production_hooks(runtime, smoke=smoke),
    )
    old_artifacts = gate._capture_formal_artifacts(reference / "off")
    repeat_artifacts = gate._capture_formal_artifacts(repeat_root)
    comparison = gate.compare_fingerprints(
        old_artifacts, old_artifacts, repeat_artifacts
    )
    reference_after = recursive_manifest(reference)
    report = {
        "schema_version": "iotj.b5_formal_off_repeat_probe.v1",
        "diagnostic_only": True,
        "status": comparison["status"],
        "off_pair_equal": comparison["off_pair_equal"],
        "max_abs_delta": comparison["max_abs_delta"],
        "mismatches": comparison["mismatches"],
        "artifact_hashes": comparison["artifact_hashes"],
        "content_set_sha256": comparison["content_set_sha256"],
        "attempt_id": repeat.attempt_id,
        "binding": {
            key: value for key, value in binding.items() if not key.startswith("_")
        },
        "frozen_initial_checkpoint": initial,
        "reference_manifest": {
            "before_sha256": reference_before["manifest_sha256"],
            "after_sha256": reference_after["manifest_sha256"],
            "unchanged": reference_before == reference_after,
        },
        "freeze_record_created": False,
        "formal_25_round_runs_started": False,
    }
    gate._atomic_write_json(root / "b5_formal_off_repeat_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=root
        / "results/iotj_main_confirmation_observability_20260715_summary/confirmation_protocol_manifest.json",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=root
        / "results/iotj_main_confirmation_observability_20260715/smoke/b5",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-number", type=int, default=997)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_probe(
        args.protocol_manifest,
        args.reference_root,
        args.output_root,
        attempt_number=args.attempt_number,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
