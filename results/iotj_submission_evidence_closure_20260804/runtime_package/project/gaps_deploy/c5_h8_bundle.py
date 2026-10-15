"""Immutable contract loader for the formal C1/C2-to-C5 fixed-H8 runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .package_contract import DeploymentPackageError

try:
    from scripts.iotj_b5_c5_bundle_contract import RUNTIME_ASSET_KEYS
except ImportError:  # pragma: no cover - direct module execution fallback.
    from iotj_b5_c5_bundle_contract import RUNTIME_ASSET_KEYS


BUNDLE_SCHEMA = "iotj.b5_c5_deployment_bundle.v1"
R4_POLICY_SCHEMA = "iotj.b5_c5_r4_policy.v1"
DEFAULT_WORKPOINT = "HC95"
SUPPORTED_WORKPOINTS = frozenset({"HC95", "HC90"})
CANONICAL_FORBIDDEN = ("C3", "C4", "R3aK16", "H8+C4", "P4")


class C5H8BundleError(DeploymentPackageError):
    """Raised when an immutable C5/H8 deployment bundle violates its contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise C5H8BundleError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise C5H8BundleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise C5H8BundleError(f"{label} must be a JSON object: {path}")
    return value


def _require_hash(path: Path, expected: object, label: str) -> Path:
    if not isinstance(expected, str) or len(expected) != 64:
        raise C5H8BundleError(f"{label} has no valid SHA-256")
    if not path.is_file():
        raise C5H8BundleError(f"missing {label}: {path}")
    if _sha256(path) != expected:
        raise C5H8BundleError(f"{label} SHA-256 differs from manifest")
    return path


def _verify_forbidden(value: object, label: str) -> None:
    expected = list(CANONICAL_FORBIDDEN)
    if value != expected:
        raise C5H8BundleError(f"{label} forbidden dependency contract differs")


def _verify_assets(root: Path, manifest: Mapping[str, Any]) -> dict[str, Path]:
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, dict):
        raise C5H8BundleError("manifest assets must be an object")
    if set(raw_assets) != set(RUNTIME_ASSET_KEYS):
        missing = sorted(set(RUNTIME_ASSET_KEYS) - set(raw_assets))
        extra = sorted(set(raw_assets) - set(RUNTIME_ASSET_KEYS))
        raise C5H8BundleError(f"manifest asset roles differ: missing={missing}, extra={extra}")
    paths: dict[str, Path] = {}
    for key in RUNTIME_ASSET_KEYS:
        descriptor = raw_assets[key]
        if not isinstance(descriptor, dict):
            raise C5H8BundleError(f"manifest asset descriptor is invalid: {key}")
        relative = descriptor.get("bundle_path")
        if not isinstance(relative, str) or not relative:
            raise C5H8BundleError(f"manifest asset has no bundle path: {key}")
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise C5H8BundleError(f"manifest asset escapes bundle: {key}")
        paths[key] = _require_hash(candidate, descriptor.get("sha256"), f"asset {key}")
    return paths


def _verify_reference(manifest: Mapping[str, Any]) -> Path:
    descriptor = manifest.get("parity_reference")
    if not isinstance(descriptor, dict):
        raise C5H8BundleError("manifest has no parity reference")
    source = descriptor.get("source_path")
    if not isinstance(source, str) or not source:
        raise C5H8BundleError("parity reference has no source path")
    return _require_hash(Path(source), descriptor.get("sha256"), "parity reference")


def _verify_r4_policy(path: Path) -> None:
    policy = _read_json(path, "R4 policy")
    if policy.get("schema_version") != R4_POLICY_SCHEMA:
        raise C5H8BundleError("R4 policy schema differs")
    if policy.get("direction") != "C1_C2_to_C5":
        raise C5H8BundleError("R4 policy direction is not C1/C2-to-C5")
    _verify_forbidden(policy.get("forbidden_runtime_dependencies"), "R4 policy")
    route = policy.get("source_aug_target_ridge_policy")
    if not isinstance(route, dict):
        raise C5H8BundleError("R4 policy has no source-augmented target route")
    switch = route.get("switch_rule")
    if not isinstance(switch, dict) or switch.get("class_ids") != [0, 1, 2, 3]:
        raise C5H8BundleError("R4 policy must route all four predicted classes")
    if switch.get("enabled_clients") != ["C5"]:
        raise C5H8BundleError("R4 policy must route C5 only")


def _validate_workpoint(name: str, value: object) -> None:
    if not isinstance(value, dict):
        raise C5H8BundleError(f"{name} workpoint is invalid")
    try:
        accept = float(value["accept_threshold"])
        reject = float(value["reject_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise C5H8BundleError(f"{name} workpoint thresholds are invalid") from error
    if not math.isfinite(accept) or not math.isfinite(reject) or not 0.0 <= accept < reject:
        raise C5H8BundleError(f"{name} workpoint thresholds are invalid")


def _verify_risk_policy(path: Path) -> dict[str, Any]:
    policy = _read_json(path, "QC risk policy")
    workpoints = policy.get("workpoints")
    if not isinstance(workpoints, dict) or DEFAULT_WORKPOINT not in workpoints:
        raise C5H8BundleError("QC risk policy must contain HC95")
    _validate_workpoint(DEFAULT_WORKPOINT, workpoints[DEFAULT_WORKPOINT])
    if "HC90" in workpoints:
        _validate_workpoint("HC90", workpoints["HC90"])
    return policy


@dataclass(frozen=True)
class C5H8Bundle:
    """Verified immutable assets for one fixed-H8 C5 deployment runtime."""

    root: Path
    manifest: Mapping[str, Any]
    asset_paths: Mapping[str, Path]
    parity_reference: Path
    risk_policy: Mapping[str, Any]
    default_workpoint: str = DEFAULT_WORKPOINT

    def select_workpoint(self, requested: str | None = None) -> str:
        selected = requested or self.default_workpoint
        if selected not in SUPPORTED_WORKPOINTS:
            raise C5H8BundleError(f"unsupported C5/H8 workpoint: {selected}")
        if selected not in self.risk_policy["workpoints"]:
            raise C5H8BundleError(f"workpoint is not frozen in bundle policy: {selected}")
        return selected


def load_c5_h8_bundle(bundle_dir: Path) -> C5H8Bundle:
    """Load the one formal C1/C2-to-C5 B5/R4 asset contract fail-closed."""
    root = Path(bundle_dir).resolve()
    manifest = _read_json(root / "manifest.json", "bundle manifest")
    if manifest.get("schema_version") != BUNDLE_SCHEMA or manifest.get("status") != "ready":
        raise C5H8BundleError("bundle manifest is not a ready C5/H8 bundle")
    _verify_forbidden(manifest.get("forbidden"), "bundle manifest")
    paths = _verify_assets(root, manifest)
    _verify_r4_policy(paths["r4_policy"])
    risk_policy = _verify_risk_policy(paths["qc_risk_policy"])
    return C5H8Bundle(root, manifest, paths, _verify_reference(manifest), risk_policy)
