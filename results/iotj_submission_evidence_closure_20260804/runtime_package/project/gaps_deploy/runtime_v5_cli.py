"""Thin fail-closed CLI for the portable Runtime-v5 regression core."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .c5_federated_source_ridge_bundle import sha256_file
from .c5_federated_source_ridge_runtime import OUTPUT_FIELDS
from .runtime_v5_portable import (
    RuntimeV5PortableBindingError,
    describe_portable_binding,
    load_runtime_v5_from_portable_binding,
    verify_portable_binding,
)


OUTPUT_SCHEMA_VERSION = "gaps.runtime_v5.inference_output.v1"


class RuntimeV5CLIError(RuntimeError):
    pass


def _load_inputs(
    input_path: Path, metadata_path: Path, phase_path: Path
) -> tuple[np.ndarray, list[Mapping[str, Any]], np.ndarray]:
    try:
        windows = np.load(input_path, allow_pickle=False)
        phases = np.load(phase_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeV5CLIError("input package cannot be loaded") from error
    if (
        not isinstance(windows, np.ndarray)
        or windows.ndim != 3
        or windows.shape[1:] != (100, 8)
        or not np.issubdtype(windows.dtype, np.number)
        or not np.isfinite(windows).all()
    ):
        raise RuntimeV5CLIError("input must be a finite numeric N×100×8 NPY array")
    if (
        not isinstance(metadata, list)
        or len(metadata) != len(windows)
        or any(not isinstance(row, Mapping) for row in metadata)
    ):
        raise RuntimeV5CLIError("metadata must be a JSON list aligned with input rows")
    if (
        not isinstance(phases, np.ndarray)
        or phases.shape != (len(windows),)
        or not np.issubdtype(phases.dtype, np.integer)
        or not np.isin(phases, (0, 1, 2)).all()
    ):
        raise RuntimeV5CLIError(
            "phase file must be an aligned integer NPY array with values 0..2"
        )
    return (
        np.asarray(windows, dtype=np.float32),
        metadata,
        np.asarray(phases, dtype=np.int64),
    )


def _validate_rows(rows: object, row_count: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != row_count:
        raise RuntimeV5CLIError("runtime output row count differs")
    validated: list[dict[str, Any]] = []
    required = set(OUTPUT_FIELDS)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != required:
            raise RuntimeV5CLIError(f"runtime output schema differs at row {index}")
        if row.get("sample_index") != index:
            raise RuntimeV5CLIError(f"runtime sample index differs at row {index}")
        route = row.get("pred_class")
        if not isinstance(route, int) or isinstance(route, bool) or route not in (0, 1, 2, 3):
            raise RuntimeV5CLIError(f"runtime route differs at row {index}")
        for field in ("source_h1_ppm", "prediction_ppm", "max_probability"):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise RuntimeV5CLIError(
                    f"runtime output contains NaN/Inf at row {index}: {field}"
                )
        if not 0.0 <= float(row["max_probability"]) <= 1.0:
            raise RuntimeV5CLIError(
                f"runtime max_probability is out of range at row {index}"
            )
        if row.get("qc_status") != "disabled_pending_dependency_audit":
            raise RuntimeV5CLIError(f"runtime QC status differs at row {index}")
        if row.get("auto_output_ppm") is not None:
            raise RuntimeV5CLIError(
                f"runtime core must not emit auto_output_ppm at row {index}"
            )
        validated.append(row)
    return validated


def run_inference(
    contract: Path,
    input_path: Path,
    metadata_path: Path,
    phase_path: Path,
    output_path: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    destination = Path(output_path)
    if destination.exists():
        raise RuntimeV5CLIError(f"REFUSE_TO_OVERWRITE: {destination}")
    windows, metadata, phases = _load_inputs(
        Path(input_path), Path(metadata_path), Path(phase_path)
    )
    runtime = load_runtime_v5_from_portable_binding(Path(contract), device=device)
    rows = _validate_rows(runtime.infer(windows, metadata, phases), len(windows))
    binding = verify_portable_binding(Path(contract))
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "release_id": binding.payload["release_id"],
        "portable_binding_sha256": sha256_file(binding.path),
        "device": device,
        "row_count": len(rows),
        "output_fields": OUTPUT_FIELDS,
        "rows": rows,
        "qc_status": "disabled_pending_dependency_audit",
        "formal_test_material_declared": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return {
        "status": "PASS",
        "output": str(destination),
        "row_count": len(rows),
        "schema_version": OUTPUT_SCHEMA_VERSION,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portable Runtime-v5 core inference; no QC or formal test dependency."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--describe-contract", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_only:
            binding = verify_portable_binding(args.contract)
            result = {
                "status": "PASS",
                "schema_version": binding.payload["schema_version"],
                "release_id": binding.payload["release_id"],
                "verified_assets": sorted(binding.asset_paths),
                "formal_test_material": False,
            }
        elif args.describe_contract:
            result = describe_portable_binding(args.contract)
        else:
            missing = [
                name
                for name, value in (
                    ("--input", args.input),
                    ("--metadata", args.metadata),
                    ("--phase-file", args.phase_file),
                    ("--output", args.output),
                )
                if value is None
            ]
            if missing:
                raise RuntimeV5CLIError(
                    f"inference mode requires: {', '.join(missing)}"
                )
            result = run_inference(
                args.contract,
                args.input,
                args.metadata,
                args.phase_file,
                args.output,
                device=args.device,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        RuntimeV5PortableBindingError,
        RuntimeV5CLIError,
    ) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
