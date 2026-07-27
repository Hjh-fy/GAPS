#!/usr/bin/env python3
"""Wrap the frozen portable Runtime-v5 release for guarded UI replay.

The builder never trains, converts, or edits model assets. It copies an already
verified portable release, writes the UI streaming contract, and refuses to
overwrite an existing destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from edge_ai_runtime import EdgeAIPackage, EdgeAIRuntime


DEFAULT_SENSOR_FIELDS = [
    "adc_ch0_pa0",
    "adc_ch1_pa1",
    "adc_ch2_pa2",
    "adc_ch3_pa3",
    "adc_ch4_pa4",
    "adc_ch5_pa5",
    "adc_ch6_pa6",
    "adc_ch7_pa7",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a schema-v3 upper-computer package around the frozen "
            "GAPS Runtime-v5 portable release."
        )
    )
    parser.add_argument("--portable-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--runtime-python-root",
        required=True,
        help=(
            "Repository/project root containing model.py and gaps_deploy. "
            "The required Python files are copied and hash-locked into the package."
        ),
    )
    parser.add_argument(
        "--package-name", default="gaps_runtime_v5_public_c5_10hz_replay"
    )
    parser.add_argument(
        "--dataset-profile",
        default="public_c5_timeaware_100hz_to_10hz_calibration_replay",
    )
    parser.add_argument(
        "--device-profile",
        default="public_dataset_c5_precomputed_replay_only",
    )
    parser.add_argument("--fixed-phase-id", type=int, default=2)
    parser.add_argument("--phase-label", default="late")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.portable_release).expanduser().resolve()
    binding_source = source / "portable_binding.json"
    if not source.is_dir() or not binding_source.is_file():
        raise FileNotFoundError(
            "portable release must contain portable_binding.json"
        )
    binding = json.loads(binding_source.read_text(encoding="utf-8"))
    if binding.get("schema_version") != "gaps.runtime_v5.portable_binding.v1":
        raise ValueError("portable binding schema differs")
    if binding.get("status") != "READY":
        raise ValueError("portable binding is not READY")
    if binding.get("dependency_contract", {}).get("formal_test_material") is not False:
        raise ValueError("portable release does not declare formal-test isolation")
    if args.fixed_phase_id not in (0, 1, 2):
        raise ValueError("--fixed-phase-id must be 0, 1, or 2")
    expected_phase_label = {0: "early", 1: "middle", 2: "late"}[
        args.fixed_phase_id
    ]
    if str(args.phase_label).strip().lower() != expected_phase_label:
        raise ValueError(
            "--phase-label must match --fixed-phase-id "
            f"({args.fixed_phase_id} -> {expected_phase_label})"
        )

    runtime_source_root = Path(args.runtime_python_root).expanduser().resolve()
    if not (
        (runtime_source_root / "model.py").is_file()
        and (runtime_source_root / "gaps_deploy/runtime_v5_portable.py").is_file()
    ):
        raise FileNotFoundError(
            "--runtime-python-root must contain model.py and "
            "gaps_deploy/runtime_v5_portable.py"
        )
    source_commit = (
        subprocess.check_output(
            ["git", "-C", str(runtime_source_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        .strip()
        .lower()
    )
    if len(source_commit) != 40:
        raise ValueError("runtime source commit is invalid")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {output}")
    output.mkdir(parents=True, exist_ok=False)
    release_destination = output / "runtime_v5_core"
    shutil.copytree(source, release_destination)
    code_destination = output / "runtime_code"
    (code_destination / "gaps_deploy").mkdir(parents=True, exist_ok=False)
    shutil.copy2(runtime_source_root / "model.py", code_destination / "model.py")
    for source_path in sorted(
        (runtime_source_root / "gaps_deploy").rglob("*.py")
    ):
        relative = source_path.relative_to(runtime_source_root)
        destination = code_destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    code_files = {}
    for path in sorted(code_destination.rglob("*.py")):
        relative = path.relative_to(code_destination).as_posix()
        code_files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    code_manifest_path = code_destination / "code_manifest.json"
    code_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "gaps.runtime_v5.code_bundle.v1",
                "source_commit": source_commit,
                "files": code_files,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    binding_destination = release_destination / "portable_binding.json"
    manifest = {
        "schema_version": 3,
        "package_name": args.package_name,
        "dataset_profile": args.dataset_profile,
        "device_profile": args.device_profile,
        "model_backend": "gaps_runtime_v5",
        "runtime_v5": {
            "binding_file": "runtime_v5_core/portable_binding.json",
            "release_id": binding["release_id"],
            "fixed_phase_id": args.fixed_phase_id,
            "metadata": {"phase_label": expected_phase_label},
            "code_root": "runtime_code",
            "code_manifest_file": "runtime_code/code_manifest.json",
            "code_source_commit": source_commit,
        },
        "input": {
            "sensor_fields": DEFAULT_SENSOR_FIELDS,
            "feature_mode": "precomputed",
            "raw_sample_hz": 10.0,
            "target_sample_hz": 10.0,
            "unstable_duration_s": 0.0,
            "baseline_duration_s": 0.0,
            "window_duration_s": 10.0,
            "stride_duration_s": 10.0,
            "min_rate_ratio": 0.95,
            "max_rate_ratio": 1.05,
            "allow_rate_mismatch": False,
            "max_gap_s": 0.3,
            "reject_implausible_frames": True,
        },
        "normalization": {"enabled": False},
        "phase_control": {"mode": "automatic", "inference_phases": ["automatic"]},
        "output": {
            "gas_names": ["Ethanol", "CO", "Ethylene", "Methane"],
            "qc_status": "disabled_pending_dependency_audit",
            "auto_output": False,
        },
        "integrity": {
            "runtime_v5_binding_sha256": sha256(binding_destination),
            "runtime_v5_code_manifest_sha256": sha256(code_manifest_path),
        },
        "usage_guard": {
            "purpose": "public_dataset_precomputed_stream_replay_only",
            "live_raw_stm32_allowed": False,
            "formal_test_material": False,
            "labels_required": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    package = EdgeAIPackage(output)
    EdgeAIRuntime(output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "package_name": package.package_name,
                "package_fingerprint": package.package_fingerprint,
                "release_id": package.runtime_v5_release_id,
                "runtime_code_source_commit": source_commit,
                "verification": "full_runtime_load",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
