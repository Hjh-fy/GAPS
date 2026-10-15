"""Fail-closed asset loader for the B5 + Federated-H1 Runtime v5 candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class FederatedSourceRidgeBundleError(RuntimeError):
    pass


ALLOWED_ASSETS = {"classifier", "federated_h1", "target_ridge"}
FORBIDDEN_ASSET_TOKENS = ("h2", "h3", "h8", "c4", "r3ak16", "p4", "qc")
BUNDLE_KEYS = {
    "schema_version", "status", "method", "build_commit", "assets",
    "feature_schema", "route_schema", "output_schema", "calibration_lineage",
    "dependency_contract",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FederatedSourceRidgeBundle:
    root: Path
    manifest: Mapping[str, Any]
    asset_paths: Mapping[str, Path]


def load_federated_source_ridge_bundle(root: Path) -> FederatedSourceRidgeBundle:
    requested = Path(root)
    manifest_path = requested if requested.is_file() else requested / "manifest.json"
    directory = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FederatedSourceRidgeBundleError("bundle manifest is invalid") from error
    if set(manifest) != BUNDLE_KEYS:
        raise FederatedSourceRidgeBundleError("bundle top-level schema differs")
    if manifest.get("schema_version") != "iotj.c5_federated_source_ridge_bundle.v1" or manifest.get("status") != "ready":
        raise FederatedSourceRidgeBundleError("bundle manifest is not ready")
    build_commit = manifest.get("build_commit")
    if not isinstance(build_commit, str) or len(build_commit) != 40 or any(character not in "0123456789abcdef" for character in build_commit):
        raise FederatedSourceRidgeBundleError("bundle build commit differs")
    feature_schema = manifest.get("feature_schema")
    if not isinstance(feature_schema, Mapping) or set(feature_schema) != {"rich_dimension", "federated_h1_input_dimension", "target_ridge_input_dimension", "target_added_feature", "rich_feature_schema_sha256"} or feature_schema.get("rich_dimension") != 104 or feature_schema.get("federated_h1_input_dimension") != 104 or feature_schema.get("target_ridge_input_dimension") != 105 or feature_schema.get("target_added_feature") != "srcpred_H1_source_ridge_ppm":
        raise FederatedSourceRidgeBundleError("bundle feature schema differs")
    route_schema = manifest.get("route_schema")
    if not isinstance(route_schema, Mapping) or set(route_schema) != {"semantics", "valid_class_ids", "class_names"} or route_schema.get("valid_class_ids") != [0, 1, 2, 3]:
        raise FederatedSourceRidgeBundleError("bundle route schema differs")
    output_schema = manifest.get("output_schema")
    if not isinstance(output_schema, Mapping) or set(output_schema) != {"required", "qc_status"} or output_schema.get("qc_status") != "disabled_pending_dependency_audit":
        raise FederatedSourceRidgeBundleError("bundle output schema differs")
    lineage = manifest.get("calibration_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != {"path", "bytes", "sha256"}:
        raise FederatedSourceRidgeBundleError("bundle calibration lineage descriptor differs")
    lineage_path = Path(str(lineage["path"]))
    if not lineage_path.is_file() or lineage_path.stat().st_size != lineage["bytes"] or sha256_file(lineage_path) != lineage["sha256"]:
        raise FederatedSourceRidgeBundleError("bundle calibration lineage identity differs")
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != ALLOWED_ASSETS:
        raise FederatedSourceRidgeBundleError("bundle assets must be exactly classifier, federated_h1, target_ridge")
    paths: dict[str, Path] = {}
    dependencies = manifest.get("dependency_contract")
    if not isinstance(dependencies, Mapping) or set(dependencies) != {"allowed", "source_heads", "forbidden", "qc", "legacy_fallback"}:
        raise FederatedSourceRidgeBundleError("bundle dependency contract schema differs")
    if dependencies.get("allowed") != sorted(ALLOWED_ASSETS) or dependencies.get("source_heads") != ["H1"] or dependencies.get("qc") != "disabled_pending_dependency_audit" or dependencies.get("legacy_fallback") is not False:
        raise FederatedSourceRidgeBundleError("bundle dependency contract differs")
    for name, descriptor in assets.items():
        if any(token in name.lower() for token in FORBIDDEN_ASSET_TOKENS):
            raise FederatedSourceRidgeBundleError(f"forbidden dependency asset: {name}")
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"bundle_path", "bytes", "sha256"} or not isinstance(descriptor.get("bundle_path"), str):
            raise FederatedSourceRidgeBundleError(f"asset descriptor is invalid: {name}")
        path = (directory / descriptor["bundle_path"]).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError as error:
            raise FederatedSourceRidgeBundleError(f"asset escapes bundle root: {name}") from error
        if not path.is_file() or path.stat().st_size != descriptor.get("bytes") or sha256_file(path) != descriptor.get("sha256"):
            raise FederatedSourceRidgeBundleError(f"asset identity differs: {name}")
        paths[name] = path
    return FederatedSourceRidgeBundle(directory.resolve(), manifest, paths)
