"""Strict relative-path binding for the final Runtime-v5 regression core."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .c5_federated_source_ridge_bundle import sha256_file
from .c5_federated_source_ridge_runtime import (
    C5FederatedSourceRidgeRuntime,
    C5FederatedSourceRidgeRuntimeError,
    OUTPUT_FIELDS,
)


SCHEMA_VERSION = "gaps.runtime_v5.portable_binding.v1"
STATUS = "READY"
FROZEN_RUNTIME_CONTRACT_SHA256 = (
    "bca1471198f0505d4536fba71100e87279156a0c69fdd54d300ffad991b36482"
)
FROZEN_BUNDLE_MANIFEST_SHA256 = (
    "f239c3b4929d1748574ec4d2fe4f61e09030087c2104b8c726046b5a39bffb1e"
)
EXPECTED_ASSET_SHA256 = {
    "classifier": "9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c",
    "federated_h1": "1ca10939f09e744fdddc0dce6f5fd959ccf769e9b78142030aa7e50aa6b2f3d4",
    "target_ridge": "2039d049776e7dfe0e8c4e6405dff2ae56a6e09b63f60ff2627ac0975aa075de",
    "calibration_lock": "4edf75222e41d8bf43097625a076964e9338493478edc76c9a04a08794d5affe",
}
ASSET_KEYS = set(EXPECTED_ASSET_SHA256)
RUNTIME_MODEL_ASSETS = {"classifier", "federated_h1", "target_ridge"}
TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "release_id",
    "source_frozen",
    "classifier_model",
    "runtime",
    "assets",
    "dependency_contract",
}
DESCRIPTOR_KEYS = {"path", "bytes", "sha256"}
CLASSIFIER_MODEL = {
    "architecture": "FedGasBaseModel",
    "num_sensors": 8,
    "num_classes": 4,
    "feat_dim": 64,
    "encoder_type": "tcn",
    "tcn_norm": "instance",
    "use_cls_proj": True,
}
RUNTIME_DESCRIPTOR = {
    "implementation": (
        "gaps_deploy.c5_federated_source_ridge_runtime."
        "C5FederatedSourceRidgeRuntime"
    ),
    "rich_feature_dimension": 104,
    "target_ridge_dimension": 105,
    "output_fields": OUTPUT_FIELDS,
    "qc_status": "disabled_pending_dependency_audit",
}
DEPENDENCY_CONTRACT = {
    "allowed_assets": [
        "calibration_lock",
        "classifier",
        "federated_h1",
        "target_ridge",
    ],
    "formal_test_material": False,
    "qc": False,
    "legacy_fallback": False,
    "forbidden": [
        "C5_test_features",
        "test_labels",
        "HC95_test_records",
        "HC90_test_records",
        "offline_formal_predictions",
        "H2",
        "H3",
        "C4",
        "legacy_rescue",
    ],
}
FORBIDDEN_PATH_TOKENS = (
    "test",
    "hc95",
    "hc90",
    "offline",
    "prediction",
    "qc",
)


class RuntimeV5PortableBindingError(RuntimeError):
    """Raised when the portable Runtime-v5 binding fails closed."""


@dataclass(frozen=True)
class RuntimeV5PortableBinding:
    path: Path
    root: Path
    payload: Mapping[str, Any]
    asset_paths: Mapping[str, Path]


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value.lower() == value
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeV5PortableBindingError(
            f"{label} must be a non-empty POSIX relative path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ":" in path.parts[0]
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise RuntimeV5PortableBindingError(f"{label} is not portable")
    if any(token in value.lower() for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeV5PortableBindingError(
            f"{label} contains a forbidden test/QC token"
        )
    return value


def verify_portable_binding(path: Path) -> RuntimeV5PortableBinding:
    binding_path = Path(path).resolve()
    try:
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeV5PortableBindingError(
            f"portable binding is invalid: {binding_path}"
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != TOP_LEVEL_KEYS:
        raise RuntimeV5PortableBindingError("portable binding top-level schema differs")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != STATUS:
        raise RuntimeV5PortableBindingError("portable binding schema/status differs")
    release_id = payload.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise RuntimeV5PortableBindingError("portable binding release_id differs")
    source = payload.get("source_frozen")
    expected_source = {
        "runtime_contract_sha256": FROZEN_RUNTIME_CONTRACT_SHA256,
        "bundle_manifest_sha256": FROZEN_BUNDLE_MANIFEST_SHA256,
    }
    if source != expected_source:
        raise RuntimeV5PortableBindingError("frozen source identity differs")
    if payload.get("classifier_model") != CLASSIFIER_MODEL:
        raise RuntimeV5PortableBindingError("classifier model contract differs")
    if payload.get("runtime") != RUNTIME_DESCRIPTOR:
        raise RuntimeV5PortableBindingError("runtime descriptor differs")
    if payload.get("dependency_contract") != DEPENDENCY_CONTRACT:
        raise RuntimeV5PortableBindingError("portable dependency contract differs")
    assets = payload.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != ASSET_KEYS:
        raise RuntimeV5PortableBindingError("portable assets differ")
    root = binding_path.parent.resolve()
    loaded: dict[str, Path] = {}
    for name, record in assets.items():
        if not isinstance(record, Mapping) or set(record) != DESCRIPTOR_KEYS:
            raise RuntimeV5PortableBindingError(
                f"portable asset descriptor differs: {name}"
            )
        relative = _safe_relative_path(record.get("path"), f"assets.{name}.path")
        size = record.get("bytes")
        expected_sha = record.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _valid_sha256(expected_sha)
        ):
            raise RuntimeV5PortableBindingError(
                f"portable asset identity is invalid: {name}"
            )
        if expected_sha != EXPECTED_ASSET_SHA256[name]:
            raise RuntimeV5PortableBindingError(
                f"portable asset is not the frozen final asset: {name}"
            )
        asset = (root / PurePosixPath(relative)).resolve()
        try:
            asset.relative_to(root)
        except ValueError as error:
            raise RuntimeV5PortableBindingError(
                f"portable asset escapes release root: {name}"
            ) from error
        if (
            not asset.is_file()
            or asset.stat().st_size != size
            or sha256_file(asset) != expected_sha
        ):
            raise RuntimeV5PortableBindingError(
                f"portable asset bytes/SHA256 differ: {name}"
            )
        loaded[name] = asset
    return RuntimeV5PortableBinding(binding_path, root, payload, loaded)


def load_runtime_v5_from_portable_binding(
    path: Path, *, device: str = "cpu"
) -> C5FederatedSourceRidgeRuntime:
    binding = verify_portable_binding(path)
    try:
        return C5FederatedSourceRidgeRuntime._from_verified_assets(
            {
                name: binding.asset_paths[name]
                for name in sorted(RUNTIME_MODEL_ASSETS)
            },
            binding.payload["classifier_model"],
            device=device,
            contract=binding.payload,
        )
    except C5FederatedSourceRidgeRuntimeError as error:
        raise RuntimeV5PortableBindingError(str(error)) from error


def describe_portable_binding(path: Path) -> dict[str, Any]:
    binding = verify_portable_binding(path)
    return {
        "schema_version": binding.payload["schema_version"],
        "status": binding.payload["status"],
        "release_id": binding.payload["release_id"],
        "source_frozen": binding.payload["source_frozen"],
        "runtime": binding.payload["runtime"],
        "assets": {
            name: {
                "path": binding.payload["assets"][name]["path"],
                "bytes": binding.payload["assets"][name]["bytes"],
                "sha256": binding.payload["assets"][name]["sha256"],
            }
            for name in sorted(binding.asset_paths)
        },
        "formal_test_material": False,
    }
