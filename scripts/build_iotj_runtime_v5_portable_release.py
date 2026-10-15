"""Build the minimal, test-free portable Runtime-v5 core release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_deploy.c5_federated_source_ridge_bundle import sha256_file
from gaps_deploy.runtime_v5_portable import (
    CLASSIFIER_MODEL,
    DEPENDENCY_CONTRACT,
    EXPECTED_ASSET_SHA256,
    FROZEN_BUNDLE_MANIFEST_SHA256,
    FROZEN_RUNTIME_CONTRACT_SHA256,
    RUNTIME_DESCRIPTOR,
    SCHEMA_VERSION,
    STATUS,
    verify_portable_binding,
)


RELEASE_ID = "gaps_runtime_v5_core_20260726"
BASE_COMMIT = "4d0e6b84341142a80ffd265e2e95dcda06fd1c72"
DEFAULT_OUTPUT = ROOT / "release" / RELEASE_ID
DEFAULT_ZIP = ROOT / "release" / f"{RELEASE_ID}.zip"
SOURCE_ASSETS = {
    "classifier": ROOT
    / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/classifier.pth",
    "federated_h1": ROOT
    / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/federated_h1.json",
    "target_ridge": ROOT
    / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/assets/target_ridge_105d.json",
    "calibration_lock": ROOT
    / "results/iotj_b5_c5_runtime_v5_candidate_20260724/target_ridge/calibration_lock.json",
}
SOURCE_FROZEN = {
    "runtime_contract": ROOT
    / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json",
    "bundle_manifest": ROOT
    / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/bundle_manifest.json",
}
ARCHIVE_PATHS = {
    "classifier": "assets/classifier.pth",
    "federated_h1": "assets/federated_h1.json",
    "target_ridge": "assets/target_ridge_105d.json",
    "calibration_lock": "lineage/calibration_lock.json",
}
FORBIDDEN_RELEASE_TOKENS = (
    "test_features",
    "test_labels",
    "hc95",
    "hc90",
    "offline_reference",
    "offline_prediction",
    "test_records",
)


class PortableReleaseError(RuntimeError):
    pass


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _descriptor(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _canonical_tree_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(
            f"{record['path']}\0{record['bytes']}\0{record['sha256']}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _payload_records(root: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _verify_frozen_sources() -> None:
    expected_frozen = {
        "runtime_contract": FROZEN_RUNTIME_CONTRACT_SHA256,
        "bundle_manifest": FROZEN_BUNDLE_MANIFEST_SHA256,
    }
    for name, path in SOURCE_FROZEN.items():
        if not path.is_file() or sha256_file(path) != expected_frozen[name]:
            raise PortableReleaseError(f"frozen source identity differs: {name}")
    for name, path in SOURCE_ASSETS.items():
        if not path.is_file() or sha256_file(path) != EXPECTED_ASSET_SHA256[name]:
            raise PortableReleaseError(f"frozen asset identity differs: {name}")


def _build_synthetic_examples(root: Path) -> None:
    synthetic = root / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0.1, 1.0, 800, dtype=np.float32).reshape(1, 100, 8)
    np.save(synthetic / "input.npy", values, allow_pickle=False)
    np.save(synthetic / "phase.npy", np.asarray([0], dtype=np.int64), allow_pickle=False)
    metadata = [
        {
            "synthetic": True,
            "window_start_s": 0.0,
            "window_end_s": 99.0,
            "window_center_s": 49.5,
            "t_onset": 10.0,
            "t_min": 20.0,
            "interpolated_ratio": 0.0,
            "max_gap_inside_window": 0.0,
            "response_phase": "main_response",
            "phase_label": "early",
        }
    ]
    _write_new_json(synthetic / "metadata.json", metadata)
    _write_new_json(
        synthetic / "expected_output_schema.json",
        {
            "schema_version": "gaps.runtime_v5.inference_output.v1",
            "row_count": 1,
            "required_row_fields": RUNTIME_DESCRIPTOR["output_fields"],
            "formal_test_material": False,
        },
    )


def _binding_payload(copied: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "release_id": RELEASE_ID,
        "source_frozen": {
            "runtime_contract_sha256": FROZEN_RUNTIME_CONTRACT_SHA256,
            "bundle_manifest_sha256": FROZEN_BUNDLE_MANIFEST_SHA256,
        },
        "classifier_model": CLASSIFIER_MODEL,
        "runtime": RUNTIME_DESCRIPTOR,
        "assets": {
            name: _descriptor(copied[name], ARCHIVE_PATHS[name])
            for name in sorted(copied)
        },
        "dependency_contract": DEPENDENCY_CONTRACT,
    }


def _readme() -> str:
    return """# GAPS Runtime-v5 Core Portable Release

Status: `CANDIDATE_FOR_CLEAN_CHECKOUT_SMOKE`

This archive contains only the final B5 classifier, sufficient-statistics
Federated H1, C5 105D target Ridge, calibration lineage lock, a strict
relative-path portable binding, provenance records, and synthetic inputs.

It contains no C5 formal test windows, labels, HC95/HC90 records, offline
formal predictions, Runtime-v4 assets, or Runtime-v5 QC policy.

Verify assets:

```powershell
python -m gaps_deploy.runtime_v5_cli --contract portable_binding.json --verify-only
```

Describe the binding:

```powershell
python -m gaps_deploy.runtime_v5_cli --contract portable_binding.json --describe-contract
```

Run the synthetic example:

```powershell
python -m gaps_deploy.runtime_v5_cli `
  --contract portable_binding.json `
  --input synthetic/input.npy `
  --metadata synthetic/metadata.json `
  --phase-file synthetic/phase.npy `
  --output synthetic/output.json `
  --device cpu
```

The CLI refuses missing or mismatched assets, invalid shapes, NaN/Inf, malformed
metadata/phases, and an existing output path.
"""


def build_release(output: Path, zip_path: Path) -> dict[str, Any]:
    output = output.resolve()
    zip_path = zip_path.resolve()
    sidecar = Path(f"{zip_path}.sha256")
    existing = [str(path) for path in (output, zip_path, sidecar) if path.exists()]
    if existing:
        raise PortableReleaseError(f"REFUSE_TO_OVERWRITE: {existing}")
    _verify_frozen_sources()
    output.mkdir(parents=True, exist_ok=False)
    copied: dict[str, Path] = {}
    for name, source in SOURCE_ASSETS.items():
        destination = output / ARCHIVE_PATHS[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != EXPECTED_ASSET_SHA256[name]:
            raise PortableReleaseError(f"copied asset identity differs: {name}")
        copied[name] = destination
    binding = output / "portable_binding.json"
    _write_new_json(binding, _binding_payload(copied))
    verify_portable_binding(binding)
    _write_new_json(
        output / "provenance_mapping.json",
        {
            "schema_version": "gaps.runtime_v5.provenance_mapping.v1",
            "status": "LOCKED",
            "release_id": RELEASE_ID,
            "frozen_base_commit": BASE_COMMIT,
            "source_frozen": {
                name: {
                    "source_path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for name, path in SOURCE_FROZEN.items()
            },
            "assets": {
                name: {
                    "source_path": SOURCE_ASSETS[name].relative_to(ROOT).as_posix(),
                    "archive_path": ARCHIVE_PATHS[name],
                    "bytes": copied[name].stat().st_size,
                    "sha256": sha256_file(copied[name]),
                }
                for name in sorted(copied)
            },
            "formal_test_material": False,
            "frozen_source_rewritten": False,
        },
    )
    _build_synthetic_examples(output)
    readme = output / "README.md"
    with readme.open("x", encoding="utf-8") as handle:
        handle.write(_readme())
    payload_records = _payload_records(
        output, {"archive_manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "schema_version": "gaps.runtime_v5.portable_archive.v1",
        "status": "READY_FOR_CLEAN_CHECKOUT_SMOKE",
        "release_id": RELEASE_ID,
        "frozen_base_commit": BASE_COMMIT,
        "portable_binding": _descriptor(binding, "portable_binding.json"),
        "payload_file_count": len(payload_records),
        "payload_bytes": sum(record["bytes"] for record in payload_records),
        "payload_tree_sha256": _canonical_tree_sha256(payload_records),
        "payload_files": payload_records,
        "formal_test_material": False,
        "frozen_results_modified": False,
    }
    manifest_path = output / "archive_manifest.json"
    _write_new_json(manifest_path, manifest)
    sums_records = _payload_records(output, {"SHA256SUMS"})
    sums = output / "SHA256SUMS"
    with sums.open("x", encoding="utf-8", newline="\n") as handle:
        for record in sorted(sums_records, key=lambda item: str(item["path"])):
            handle.write(f"{record['sha256']}  {record['path']}\n")
    verify_release_directory(output)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.name
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_STORED) as archive:
        for source in sorted(item for item in output.rglob("*") if item.is_file()):
            relative = source.relative_to(output).as_posix()
            info = zipfile.ZipInfo(f"{prefix}/{relative}", (2026, 7, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
    archive_sha256 = sha256_file(zip_path)
    with sidecar.open("x", encoding="ascii", newline="\n") as handle:
        handle.write(f"{archive_sha256}  {zip_path.name}\n")
    return {
        "status": "BUILT",
        "release_directory": str(output),
        "archive": str(zip_path),
        "archive_bytes": zip_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "sidecar": str(sidecar),
        "formal_test_material": False,
    }


def verify_release_directory(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    sums = root / "SHA256SUMS"
    manifest_path = root / "archive_manifest.json"
    binding_path = root / "portable_binding.json"
    if not sums.is_file() or not manifest_path.is_file() or not binding_path.is_file():
        raise PortableReleaseError("release control files are missing")
    verify_portable_binding(binding_path)
    expected: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in expected:
            raise PortableReleaseError("SHA256SUMS schema differs")
        expected[parts[1]] = parts[0]
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums
    }
    if set(expected) != actual_files:
        raise PortableReleaseError("SHA256SUMS file membership differs")
    for relative, expected_sha in expected.items():
        path = root / relative
        if sha256_file(path) != expected_sha:
            raise PortableReleaseError(f"release SHA256 differs: {relative}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PortableReleaseError("archive manifest is invalid") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "gaps.runtime_v5.portable_archive.v1"
        or manifest.get("status") != "READY_FOR_CLEAN_CHECKOUT_SMOKE"
        or manifest.get("formal_test_material") is not False
        or manifest.get("frozen_results_modified") is not False
    ):
        raise PortableReleaseError("archive manifest schema/status differs")
    payload = _payload_records(root, {"archive_manifest.json", "SHA256SUMS"})
    if (
        manifest.get("payload_file_count") != len(payload)
        or manifest.get("payload_bytes")
        != sum(record["bytes"] for record in payload)
        or manifest.get("payload_tree_sha256") != _canonical_tree_sha256(payload)
        or manifest.get("payload_files") != payload
    ):
        raise PortableReleaseError("archive payload identity differs")
    forbidden = [
        relative
        for relative in actual_files
        if any(token in relative.lower() for token in FORBIDDEN_RELEASE_TOKENS)
    ]
    if forbidden:
        raise PortableReleaseError(f"formal test/QC material found: {forbidden}")
    return {
        "status": "PASS",
        "release_id": manifest.get("release_id"),
        "verified_file_count": len(actual_files),
        "payload_tree_sha256": manifest["payload_tree_sha256"],
        "formal_test_material": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the minimal portable Runtime-v5 core release."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = (
            verify_release_directory(args.output_dir)
            if args.verify_only
            else build_release(args.output_dir, args.zip_path)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, PortableReleaseError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
