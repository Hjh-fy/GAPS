"""Bind explicit B5/C5 deployment inputs without inferring legacy artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


REQUIRED_KEYS = (
    "classifier",
    "r4_policy",
    "h23_reference",
    "qc_risk_policy",
    "qc_component_calibrator",
    "qc_feature_reference",
    "qc_risk_selection",
    "feature_schema",
    "class_map",
    "normalization",
    "offline_reference_1360",
)
FORBIDDEN_TOKENS = ("c3", "c4", "r3ak16", "h8+c4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_input_paths(paths: Mapping[str, Path]) -> dict[str, object]:
    """Return a fail-closed audit of explicit bundle input paths."""
    reasons: list[str] = []
    assets: dict[str, object] = {}
    for key, value in paths.items():
        path = Path(value)
        rendered = path.as_posix().lower()
        if any(token in key.lower() or token in rendered for token in FORBIDDEN_TOKENS):
            reasons.append(f"legacy_forbidden:{key}")
            continue
        if not path.is_file():
            reasons.append(f"missing_required:{key}")
            continue
        assets[key] = {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    for key in REQUIRED_KEYS:
        if key not in paths:
            reasons.append(f"missing_required:{key}")
    reasons = sorted(set(reasons))
    return {"status": "ready" if not reasons else "blocked", "reasons": reasons, "assets": assets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit explicit B5/C5 deployment inputs")
    parser.add_argument("--input-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input_map.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(value, str) for value in payload.values()):
        raise ValueError("input map must be a JSON object of string paths")
    result = audit_input_paths({key: Path(value) for key, value in payload.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "reason_count": len(result["reasons"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
