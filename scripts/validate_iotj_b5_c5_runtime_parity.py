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
C5_H8_PPM_TOLERANCE = 2e-3
C5_H8_RISK_TOLERANCE = 1e-12
SUPPORTED_WORKPOINTS = frozenset({"HC95", "HC90"})


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


def _read_c5_h8_rows(path: Path, *, reference: bool, workpoint: str) -> dict[int, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"sample_index", "pred_class", "deployment_risk_full", "qc_decision"}
        ppm_field = "target_ridge_plus_source_preds_ppm" if reference else "h8_ppm"
        required.add(ppm_field)
        if not reference:
            required.add("auto_output_ppm")
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"missing C5/H8 parity fields in {path}: {missing}")
        rows: dict[int, dict[str, Any]] = {}
        for raw in reader:
            try:
                key = int(str(raw["sample_index"]))
                pred_class = int(str(raw["pred_class"]))
            except ValueError as error:
                raise ValueError(f"invalid C5/H8 row key or class in {path}") from error
            if key in rows:
                raise ValueError(f"duplicate sample_index in {path}: {key}")
            if pred_class not in (0, 1, 2, 3):
                raise ValueError(f"invalid pred_class in {path}: {pred_class}")
            if raw.get("qc_workpoint") and raw["qc_workpoint"] != workpoint:
                raise ValueError(f"C5/H8 reference/runtime workpoint differs in {path}: {raw['qc_workpoint']}")
            ppm = _as_float(raw[ppm_field], ppm_field, key)
            decision = str(raw["qc_decision"])
            if decision not in {"accept", "review", "reject"}:
                raise ValueError(f"invalid qc_decision for sample_index {key}")
            expected_auto: float | str = ppm if decision == "accept" else ""
            auto: float | str = expected_auto if reference else raw["auto_output_ppm"]
            if auto != "":
                auto = _as_float(str(auto), "auto_output_ppm", key)
            rows[key] = {
                "pred_class": pred_class,
                "h8_ppm": ppm,
                "deployment_risk_full": _as_float(raw["deployment_risk_full"], "deployment_risk_full", key),
                "qc_decision": decision,
                "auto_output_ppm": auto,
            }
    expected = set(range(EXPECTED_ROWS))
    if set(rows) != expected:
        raise ValueError(f"C5/H8 parity keys in {path} must be exactly 0..{EXPECTED_ROWS - 1}")
    return rows


def validate_c5_h8_parity(reference_path: Path, runtime_path: Path, workpoint: str) -> dict[str, Any]:
    """Compare the formal six-field C5/H8 deployment contract fail-closed."""
    if workpoint not in SUPPORTED_WORKPOINTS:
        raise ValueError(f"unsupported C5/H8 parity workpoint: {workpoint}")
    reference = _read_c5_h8_rows(Path(reference_path), reference=True, workpoint=workpoint)
    runtime = _read_c5_h8_rows(Path(runtime_path), reference=False, workpoint=workpoint)
    counters = {"class_mismatches": 0, "risk_mismatches": 0, "qc_decision_mismatches": 0, "auto_output_mismatches": 0}
    max_ppm = 0.0
    max_risk = 0.0
    examples: list[dict[str, Any]] = []
    for key in range(EXPECTED_ROWS):
        offline, deployed = reference[key], runtime[key]
        class_bad = offline["pred_class"] != deployed["pred_class"]
        ppm_delta = abs(offline["h8_ppm"] - deployed["h8_ppm"])
        risk_delta = abs(offline["deployment_risk_full"] - deployed["deployment_risk_full"])
        qc_bad = offline["qc_decision"] != deployed["qc_decision"]
        offline_auto, deployed_auto = offline["auto_output_ppm"], deployed["auto_output_ppm"]
        auto_bad = (offline_auto == "") != (deployed_auto == "") or (
            offline_auto != "" and abs(float(offline_auto) - float(deployed_auto)) > C5_H8_PPM_TOLERANCE
        )
        counters["class_mismatches"] += int(class_bad)
        counters["risk_mismatches"] += int(risk_delta > C5_H8_RISK_TOLERANCE)
        counters["qc_decision_mismatches"] += int(qc_bad)
        counters["auto_output_mismatches"] += int(auto_bad)
        max_ppm, max_risk = max(max_ppm, ppm_delta), max(max_risk, risk_delta)
        if (class_bad or ppm_delta > C5_H8_PPM_TOLERANCE or risk_delta > C5_H8_RISK_TOLERANCE or qc_bad or auto_bad) and len(examples) < 20:
            examples.append({"sample_index": key, "abs_h8_ppm_delta": ppm_delta, "abs_risk_delta": risk_delta, "class_match": not class_bad, "qc_match": not qc_bad, "auto_output_match": not auto_bad})
    equivalent = not examples and all(value == 0 for value in counters.values()) and max_ppm <= C5_H8_PPM_TOLERANCE
    return {
        "schema_version": "iotj.c5_h8_runtime_parity.v1",
        "status": "equivalent" if equivalent else "failed",
        "workpoint": workpoint,
        "reference_rows": len(reference),
        "runtime_rows": len(runtime),
        **counters,
        "max_abs_h8_ppm_delta": max_ppm,
        "max_abs_risk_delta": max_risk,
        "ppm_tolerance": C5_H8_PPM_TOLERANCE,
        "risk_tolerance": C5_H8_RISK_TOLERANCE,
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
