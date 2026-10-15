"""Fail-closed packager for explicitly audited B5/C5 deployment assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from .iotj_b5_c5_bundle_contract import (
        FORBIDDEN_TOKENS,
        PARITY_REFERENCE_KEY,
        REQUIRED_KEYS,
    )
except ImportError:  # pragma: no cover - supports direct ``python scripts/...`` invocation.
    from iotj_b5_c5_bundle_contract import (
        FORBIDDEN_TOKENS,
        PARITY_REFERENCE_KEY,
        REQUIRED_KEYS,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_reason(audit: dict[str, Any]) -> bool:
    reasons = [str(item).lower() for item in audit.get("reasons", [])]
    if any("legacy_forbidden" in item for item in reasons):
        return True
    for key, item in audit.get("assets", {}).items():
        if any(token in str(key).lower() for token in FORBIDDEN_TOKENS):
            return True
        rendered = str(item.get("path", "")).replace("\\", "/").lower()
        if any(token in rendered for token in FORBIDDEN_TOKENS):
            return True
    return False


def _validate_asset_contract(assets: object) -> dict[str, Any]:
    if not isinstance(assets, dict):
        raise ValueError("input audit has no bound assets")
    observed = set(assets)
    missing = sorted(set(REQUIRED_KEYS) - observed)
    if missing:
        raise ValueError(f"missing required audited asset: {missing[0]}")
    unexpected = sorted(observed - set(REQUIRED_KEYS))
    if unexpected:
        raise ValueError(f"unexpected audited asset: {unexpected[0]}")
    return assets


def build_bundle(input_audit: Path, output_dir: Path) -> dict[str, Any]:
    """Copy audited assets and emit a fresh, content-addressed manifest.

    The function intentionally performs no model conversion or fitting. Its only
    authority is to copy exact, already-audited B5/C5 artifacts into a new
    immutable-layout candidate.
    """
    audit_path = Path(input_audit)
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes.decode("utf-8"))
    if _legacy_reason(audit):
        raise ValueError("forbidden legacy input in audit")
    if audit.get("status") != "ready":
        raise ValueError("input audit must be ready before building bundle")
    assets = _validate_asset_contract(audit.get("assets"))

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing bundle: {output_dir}")
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    packaged: dict[str, Any] = {}
    parity_reference: dict[str, Any] | None = None
    for key in sorted(assets):
        descriptor = assets[key]
        if not isinstance(descriptor, dict):
            raise ValueError(f"invalid asset descriptor: {key}")
        source = Path(str(descriptor.get("path", "")))
        if not source.is_file():
            raise FileNotFoundError(f"audited asset disappeared: {key}: {source}")
        source_sha = _sha256(source)
        if source_sha != str(descriptor.get("sha256", "")):
            raise ValueError(f"audited asset hash changed: {key}")
        if key == PARITY_REFERENCE_KEY:
            parity_reference = {
                "source_path": source.as_posix(),
                "bytes": source.stat().st_size,
                "sha256": source_sha,
            }
            continue
        destination = asset_dir / f"{key}{source.suffix}"
        shutil.copy2(source, destination)
        packaged[key] = {
            "source_path": source.as_posix(),
            "bundle_path": destination.relative_to(output_dir).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
    if parity_reference is None:
        raise ValueError("input audit is missing parity reference")
    manifest = {
        "schema_version": "iotj.b5_c5_deployment_bundle.v1",
        "status": "ready",
        "input_audit": audit_path.as_posix(),
        "input_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "assets": packaged,
        "parity_reference": parity_reference,
        "forbidden": ["C3", "C4", "R3aK16", "H8+C4", "P4"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_bundle(args.input_audit, args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
