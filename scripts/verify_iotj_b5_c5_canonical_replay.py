"""Fail-closed contract verifier for one B5 C5 canonical deployment replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 1360
FORBIDDEN_RUNTIME_DEPENDENCIES = ["C3", "C4", "R3aK16", "H8+C4", "P4"]
R4_FIELD = "target_ridge_plus_source_preds_ppm"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _verify_runtime_policy(path: Path, label: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("forbidden_runtime_dependencies") != FORBIDDEN_RUNTIME_DEPENDENCIES:
        raise ValueError(f"{label} forbidden runtime dependency contract differs")
    return payload


def _verify_r4_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_index", "pred_class", R4_FIELD}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"R4 stream missing fields: {missing}")
        rows = list(reader)
    keys = [int(str(row["sample_index"])) for row in rows]
    if len(rows) != EXPECTED_ROWS or len(set(keys)) != EXPECTED_ROWS:
        raise ValueError(f"R4 stream must contain exactly {EXPECTED_ROWS} unique rows")
    return len(rows)


def verify_canonical_replay(root: Path) -> dict[str, Any]:
    """Validate that a replay contains the runtime assets needed for parity."""
    root = Path(root)
    r4_policy = _require_file(root / "h8_no_rescue" / "r4_policy.json", "r4_policy")
    h23_reference = _require_file(root / "h23_plus" / "h23_reference.json", "h23_reference")
    qc_dir = root / "high_coverage_qc"
    qc_manifest = _require_file(qc_dir / "manifest.json", "QC manifest")
    for name in ("risk_policy.json", "component_calibrator.json", "feature_reference.json", "risk_selection.json"):
        _require_file(qc_dir / name, name)
    r4_stream = _require_file(
        root / "h8_no_rescue" / "target_predictions_plus_source_preds.csv", "R4 test stream"
    )
    _verify_runtime_policy(r4_policy, "r4_policy")
    _verify_runtime_policy(h23_reference, "h23_reference")
    manifest = _read_json(qc_manifest)
    if manifest.get("pred_key") != R4_FIELD or manifest.get("secondary_workpoint") != "HC90":
        raise ValueError("QC manifest is not bound to the R4 HC90 runtime contract")
    return {
        "status": "ready",
        "root": str(root),
        "r4_policy": (Path("h8_no_rescue") / "r4_policy.json").as_posix(),
        "h23_reference": (Path("h23_plus") / "h23_reference.json").as_posix(),
        "runtime_rows": _verify_r4_rows(r4_stream),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_canonical_replay(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
