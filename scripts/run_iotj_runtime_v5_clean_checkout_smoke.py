"""Run the Runtime-v5 portable archive smoke in a fresh fixed-commit checkout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


RELEASE_ID = "gaps_runtime_v5_core_20260726"
EXPECTED_ARCHIVE_SHA256 = (
    "740e8237384041523e51969b88795c27e43e88650c73e1a5209092880cf547de"
)
OUTPUT_SCHEMA_VERSION = "gaps.runtime_v5.inference_output.v1"
REQUIRED_ROW_FIELDS = {
    "sample_index",
    "pred_class",
    "source_h1_ppm",
    "prediction_ppm",
    "max_probability",
    "qc_status",
    "auto_output_ppm",
}
FORBIDDEN_FORMAL_PATHS = (
    "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5/test_features.npy",
    "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5/test_experiment_info.json",
    "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5/test_phase_labels.npy",
    "results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_hc95_records.csv",
    "results/iotj_b5_c5_deployment_p1_20260722/high_coverage_qc/test_hc90_records.csv",
    "results/iotj_b5_c5_runtime_v5_candidate_20260724/target_ridge/offline_reference_1360.csv",
)


class CleanCheckoutSmokeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    record = {
        "command": list(command),
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    if completed.returncode != 0:
        raise CleanCheckoutSmokeError(
            f"command failed: {json.dumps(record, ensure_ascii=False)}"
        )
    return record


def _extract_safely(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as error:
                raise CleanCheckoutSmokeError(
                    f"archive member escapes restore root: {member.filename}"
                ) from error
        archive.extractall(root)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_smoke(
    source_repository: Path,
    code_commit: str,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    source_repository = source_repository.resolve()
    output_dir = output_dir.resolve()
    receipt_path = receipt_path.resolve()
    if output_dir.exists() or receipt_path.exists():
        raise CleanCheckoutSmokeError(
            f"REFUSE_TO_OVERWRITE: output={output_dir}, receipt={receipt_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkout = output_dir / "checkout"
    restore_root = output_dir / "restored"
    smoke_output = output_dir / "synthetic_output.json"
    clone = _run(
        [
            "git",
            "clone",
            "--shared",
            "--no-checkout",
            str(source_repository),
            str(checkout),
        ],
        cwd=output_dir,
    )
    checkout_record = _run(
        ["git", "checkout", "--detach", code_commit],
        cwd=checkout,
    )
    actual_commit = _run(["git", "rev-parse", "HEAD"], cwd=checkout)["stdout"]
    if actual_commit != code_commit:
        raise CleanCheckoutSmokeError(
            f"fresh checkout commit differs: {actual_commit} != {code_commit}"
        )
    archive = checkout / "release" / f"{RELEASE_ID}.zip"
    sidecar = checkout / "release" / f"{RELEASE_ID}.zip.sha256"
    if not archive.is_file() or not sidecar.is_file():
        raise CleanCheckoutSmokeError("release archive/sidecar is missing in checkout")
    archive_sha256 = sha256_file(archive)
    sidecar_sha256 = sidecar.read_text(encoding="ascii").split()[0]
    if (
        archive_sha256 != EXPECTED_ARCHIVE_SHA256
        or sidecar_sha256 != EXPECTED_ARCHIVE_SHA256
    ):
        raise CleanCheckoutSmokeError("release archive SHA256 differs")
    _extract_safely(archive, restore_root)
    release = restore_root / RELEASE_ID
    if not release.is_dir():
        raise CleanCheckoutSmokeError("restored release root differs")
    forbidden_presence = [
        relative
        for relative in FORBIDDEN_FORMAL_PATHS
        if (checkout / relative).exists() or (release / relative).exists()
    ]
    if forbidden_presence:
        raise CleanCheckoutSmokeError(
            f"formal test material is present: {forbidden_presence}"
        )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(checkout)
    import_check = _run(
        [
            sys.executable,
            "-c",
            (
                "from gaps_deploy.runtime_v5_portable import "
                "load_runtime_v5_from_portable_binding; "
                "from gaps_deploy.runtime_v5_cli import main"
            ),
        ],
        cwd=checkout,
        environment=environment,
    )
    archive_verify = _run(
        [
            sys.executable,
            "scripts/build_iotj_runtime_v5_portable_release.py",
            "--verify-only",
            "--output-dir",
            str(release),
        ],
        cwd=checkout,
        environment=environment,
    )
    binding = release / "portable_binding.json"
    binding_verify = _run(
        [
            sys.executable,
            "-m",
            "gaps_deploy.runtime_v5_cli",
            "--contract",
            str(binding),
            "--verify-only",
        ],
        cwd=checkout,
        environment=environment,
    )
    inference = _run(
        [
            sys.executable,
            "-m",
            "gaps_deploy.runtime_v5_cli",
            "--contract",
            str(binding),
            "--input",
            str(release / "synthetic/input.npy"),
            "--metadata",
            str(release / "synthetic/metadata.json"),
            "--phase-file",
            str(release / "synthetic/phase.npy"),
            "--output",
            str(smoke_output),
            "--device",
            "cpu",
        ],
        cwd=checkout,
        environment=environment,
    )
    try:
        output = json.loads(smoke_output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanCheckoutSmokeError("synthetic output is invalid") from error
    rows = output.get("rows")
    if (
        output.get("schema_version") != OUTPUT_SCHEMA_VERSION
        or output.get("row_count") != 1
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], dict)
        or set(rows[0]) != REQUIRED_ROW_FIELDS
        or rows[0].get("sample_index") != 0
        or rows[0].get("pred_class") not in (0, 1, 2, 3)
        or rows[0].get("qc_status") != "disabled_pending_dependency_audit"
        or rows[0].get("auto_output_ppm") is not None
    ):
        raise CleanCheckoutSmokeError("synthetic output schema differs")
    checkout_status = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=checkout
    )["stdout"]
    if checkout_status:
        raise CleanCheckoutSmokeError(
            f"fresh checkout was modified by smoke: {checkout_status}"
        )
    binding_payload = json.loads(binding.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": "gaps.runtime_v5.clean_checkout_receipt.v1",
        "status": "CLEAN_CHECKOUT_RUNTIME_V5_CORE_READY",
        "verified_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "code": {
            "commit": code_commit,
            "checkout_head": actual_commit,
            "checkout_clean_after_smoke": True,
            "clone_mode": "fresh local shared-object clone; independent working tree",
        },
        "archive": {
            "release_id": RELEASE_ID,
            "path_in_checkout": f"release/{archive.name}",
            "bytes": archive.stat().st_size,
            "sha256": archive_sha256,
            "sidecar_matched": True,
            "restored_to_new_directory": True,
        },
        "portable_binding": {
            "schema_version": binding_payload["schema_version"],
            "sha256": sha256_file(binding),
            "source_frozen": binding_payload["source_frozen"],
            "assets": {
                name: {
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
                for name, record in binding_payload["assets"].items()
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "numpy_version": _package_version("numpy"),
            "torch_version": _package_version("torch"),
        },
        "checks": {
            "fresh_checkout": "PASS",
            "archive_sha256": "PASS",
            "archive_restore": "PASS",
            "sha256sums_and_manifest": "PASS",
            "runtime_import": "PASS",
            "portable_binding_verify": "PASS",
            "runtime_load": "PASS",
            "synthetic_inference": "PASS",
            "output_schema": "PASS",
        },
        "synthetic_output": {
            "schema_version": output["schema_version"],
            "row_count": output["row_count"],
            "row_fields": sorted(rows[0]),
            "pred_class": rows[0]["pred_class"],
            "qc_status": rows[0]["qc_status"],
            "auto_output_ppm": rows[0]["auto_output_ppm"],
        },
        "evidence_boundary": {
            "formal_test_accessed": False,
            "formal_test_presence_check": "PASS",
            "formal_test_presence_method": (
                "fresh checkout contains no ignored dataset/results assets; restored "
                "archive membership and explicit forbidden paths were checked"
            ),
            "training_run": False,
            "evaluation_run": False,
            "benchmark_run": False,
            "frozen_results_modified": False,
            "runtime_v4_portability_claimed": False,
            "runtime_v5_qc_reproduction_claimed": False,
            "full_system_ready_claimed": False,
        },
        "commands": {
            "clone": clone,
            "checkout": checkout_record,
            "import": import_check,
            "archive_verify": archive_verify,
            "binding_verify": binding_verify,
            "synthetic_inference": inference,
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh-checkout synthetic smoke for portable Runtime-v5 core."
    )
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = run_smoke(
            args.source_repository,
            args.code_commit,
            args.output_dir,
            args.receipt,
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "code_commit": receipt["code"]["commit"],
                    "archive_sha256": receipt["archive"]["sha256"],
                    "checks": receipt["checks"],
                    "evidence_boundary": receipt["evidence_boundary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (OSError, CleanCheckoutSmokeError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
