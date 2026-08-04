"""Strict bundle loader for the independent Runtime v5 QC extension."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtime_v5_qc import descriptor, sha256_file


class RuntimeV5QCBundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeV5QCBundle:
    root: Path
    manifest: Mapping[str, Any]
    base_runtime_contract: Path
    qc_policy: Path


def load_runtime_v5_qc_bundle(path: Path) -> RuntimeV5QCBundle:
    requested = Path(path)
    manifest_path = requested if requested.is_file() else requested / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeV5QCBundleError("Runtime v5 QC bundle manifest is invalid") from error
    required = {"schema_version", "status", "build_commit", "assets", "dependency_contract"}
    if set(manifest) != required or manifest.get("schema_version") != "iotj.runtime_v5_qc_bundle.v1" or manifest.get("status") != "locked":
        raise RuntimeV5QCBundleError("Runtime v5 QC bundle schema/status differs")
    commit = manifest.get("build_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeV5QCBundleError("Runtime v5 QC bundle build commit differs")
    dependencies = manifest.get("dependency_contract")
    expected_dependencies = {
        "allowed": ["base_runtime_contract", "qc_policy"],
        "forbidden": ["H2", "H3", "H2.3", "all_prior", "legacy_rescue", "runtime_v4_risk"],
        "legacy_fallback": False,
    }
    if dependencies != expected_dependencies:
        raise RuntimeV5QCBundleError("Runtime v5 QC dependency contract differs")
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != {"base_runtime_contract", "qc_policy"}:
        raise RuntimeV5QCBundleError("Runtime v5 QC assets differ")
    root = manifest_path.parent.resolve()
    loaded: dict[str, Path] = {}
    for name, record in assets.items():
        if not isinstance(record, Mapping) or set(record) != {"bundle_path", "bytes", "sha256"}:
            raise RuntimeV5QCBundleError(f"Runtime v5 QC asset descriptor differs: {name}")
        asset = (root / str(record["bundle_path"])).resolve()
        try:
            asset.relative_to(root)
        except ValueError as error:
            raise RuntimeV5QCBundleError(f"Runtime v5 QC asset escapes bundle: {name}") from error
        if not asset.is_file() or asset.stat().st_size != record["bytes"] or sha256_file(asset) != record["sha256"]:
            raise RuntimeV5QCBundleError(f"Runtime v5 QC asset identity differs: {name}")
        loaded[name] = asset
    return RuntimeV5QCBundle(root, manifest, loaded["base_runtime_contract"], loaded["qc_policy"])


def bundle_asset_record(root: Path, asset: Path) -> dict[str, Any]:
    record = descriptor(asset)
    return {"bundle_path": str(Path(asset).resolve().relative_to(Path(root).resolve())).replace("\\", "/"), "bytes": record["bytes"], "sha256": record["sha256"]}
