"""Fail-closed comparison of B5/C5 offline and deployment runtime streams."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "sample_index",
    "pred_class",
    "selected_profile",
    "qc_decision",
    "final_ppm",
)
EXPECTED_ROWS = 1360
PPM_TOLERANCE = 1e-6


def _read_indexed(path: Path) -> dict[int, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing parity fields in {path}: {missing}")
        indexed: dict[int, dict[str, str]] = {}
        for row in reader:
            try:
                key = int(str(row["sample_index"]))
            except ValueError as error:
                raise ValueError(f"invalid sample_index in {path}: {row.get('sample_index')!r}") from error
            if key in indexed:
                raise ValueError(f"duplicate sample_index in {path}: {key}")
            indexed[key] = row
    if len(indexed) != EXPECTED_ROWS:
        raise ValueError(f"expected exactly {EXPECTED_ROWS} parity rows in {path}; got {len(indexed)}")
    return indexed


def _as_float(value: str, field: str, key: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric {field} for sample_index {key}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} for sample_index {key}")
    return result


def validate_parity(reference_path: Path, runtime_path: Path) -> dict[str, Any]:
    """Compare exactly 1,360 C5 rows, without permitting partial equivalence."""
    reference = _read_indexed(Path(reference_path))
    runtime = _read_indexed(Path(runtime_path))
    reference_keys, runtime_keys = set(reference), set(runtime)
    missing_runtime = sorted(reference_keys - runtime_keys)
    unexpected_runtime = sorted(runtime_keys - reference_keys)
    class_mismatches = 0
    profile_mismatches = 0
    qc_mismatches = 0
    max_abs_ppm_delta = 0.0
    examples: list[dict[str, Any]] = []
    for key in sorted(reference_keys & runtime_keys):
        offline = reference[key]
        deployed = runtime[key]
        class_bad = str(offline["pred_class"]) != str(deployed["pred_class"])
        profile_bad = str(offline["selected_profile"]) != str(deployed["selected_profile"])
        qc_bad = str(offline["qc_decision"]) != str(deployed["qc_decision"])
        ppm_delta = abs(
            _as_float(offline["final_ppm"], "offline final_ppm", key)
            - _as_float(deployed["final_ppm"], "runtime final_ppm", key)
        )
        class_mismatches += int(class_bad)
        profile_mismatches += int(profile_bad)
        qc_mismatches += int(qc_bad)
        max_abs_ppm_delta = max(max_abs_ppm_delta, ppm_delta)
        if (class_bad or profile_bad or qc_bad or ppm_delta > PPM_TOLERANCE) and len(examples) < 20:
            examples.append(
                {
                    "sample_index": key,
                    "class_match": not class_bad,
                    "profile_match": not profile_bad,
                    "qc_match": not qc_bad,
                    "abs_ppm_delta": ppm_delta,
                }
            )
    equivalent = (
        not missing_runtime
        and not unexpected_runtime
        and class_mismatches == 0
        and profile_mismatches == 0
        and qc_mismatches == 0
        and max_abs_ppm_delta <= PPM_TOLERANCE
    )
    return {
        "schema_version": "iotj.b5_c5_runtime_parity.v1",
        "status": "equivalent" if equivalent else "failed",
        "reference_rows": len(reference),
        "runtime_rows": len(runtime),
        "missing_runtime_rows": missing_runtime,
        "unexpected_runtime_rows": unexpected_runtime,
        "class_mismatches": class_mismatches,
        "selected_profile_mismatches": profile_mismatches,
        "qc_decision_mismatches": qc_mismatches,
        "max_abs_ppm_delta": max_abs_ppm_delta,
        "ppm_tolerance": PPM_TOLERANCE,
        "first_mismatches": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_parity(args.reference, args.runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "max_abs_ppm_delta": report["max_abs_ppm_delta"]}))


if __name__ == "__main__":
    main()
