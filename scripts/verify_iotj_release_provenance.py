"""Hash-only verifier for the GAPS IoT-J release provenance contract.

This utility deliberately does not import runtime modules, deserialize model
checkpoints, run inference, or rewrite frozen artifacts.  It verifies only the
release manifest schema, path safety, byte sizes, and SHA-256 identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "iotj.release_provenance.v1"
STATUS = "PROVENANCE_LOCKED"
TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "release_id",
    "created_date",
    "code",
    "source_roots",
    "profiles",
    "assets",
    "loader_dependency_audit",
    "constraints",
}
ASSET_KEYS = {
    "asset_id",
    "component",
    "role",
    "source_root",
    "source_path",
    "restore_path",
    "bytes",
    "sha256",
    "evidence_class",
    "contains_formal_test_material",
}
PROFILE_KEYS = {"description", "asset_ids", "runtime_loadable_after_raw_restore"}
ALLOWED_EVIDENCE_CLASSES = {
    "RUNTIME_ASSET",
    "LOADER_COUPLED_AUDIT_ASSET",
    "PROVENANCE_AUDIT_ASSET",
    "FORMAL_TEST_MATERIAL",
}


class ProvenanceError(RuntimeError):
    """Raised when the release provenance contract fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"manifest is not valid UTF-8 JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ProvenanceError("manifest top level must be an object")
    return payload


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProvenanceError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ":" in path.parts[0]
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ProvenanceError(f"{label} is not a safe relative path: {value!r}")
    return value


def _validate_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProvenanceError(f"{label} must be a lowercase SHA-256")
    return value


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if set(payload) != TOP_LEVEL_KEYS:
        raise ProvenanceError("manifest top-level schema differs")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ProvenanceError("manifest schema version differs")
    if payload.get("status") != STATUS:
        raise ProvenanceError("manifest is not provenance-locked")
    if not isinstance(payload.get("release_id"), str) or not payload["release_id"]:
        raise ProvenanceError("release_id is invalid")
    if not isinstance(payload.get("created_date"), str) or not payload["created_date"]:
        raise ProvenanceError("created_date is invalid")

    code = payload.get("code")
    if not isinstance(code, Mapping) or set(code) != {
        "repository",
        "branch",
        "provenance_base_commit",
        "frozen_evidence_commit",
    }:
        raise ProvenanceError("code provenance schema differs")
    for key in ("provenance_base_commit", "frozen_evidence_commit"):
        value = code.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ProvenanceError(f"code.{key} is not a full commit SHA")

    roots = payload.get("source_roots")
    if not isinstance(roots, Mapping) or set(roots) != {"repository", "workspace"}:
        raise ProvenanceError("source_roots must define repository and workspace")
    if roots != {
        "repository": "Git worktree root containing this manifest",
        "workspace": "Parent workspace root containing dataset/ and the worktree collection",
    }:
        raise ProvenanceError("source root semantics differ")

    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ProvenanceError("assets must be a non-empty list")
    asset_ids: set[str] = set()
    restore_paths: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping) or set(asset) != ASSET_KEYS:
            raise ProvenanceError(f"asset schema differs at index {index}")
        asset_id = asset.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id or asset_id in asset_ids:
            raise ProvenanceError(f"asset_id is invalid or duplicated: {asset_id!r}")
        asset_ids.add(asset_id)
        if asset.get("source_root") not in roots:
            raise ProvenanceError(f"asset source_root is invalid: {asset_id}")
        _safe_relative_path(asset.get("source_path"), f"{asset_id}.source_path")
        restore = _safe_relative_path(asset.get("restore_path"), f"{asset_id}.restore_path")
        if restore in restore_paths:
            raise ProvenanceError(f"restore_path is duplicated: {restore}")
        restore_paths.add(restore)
        size = asset.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProvenanceError(f"asset bytes is invalid: {asset_id}")
        _validate_sha256(asset.get("sha256"), f"{asset_id}.sha256")
        if asset.get("evidence_class") not in ALLOWED_EVIDENCE_CLASSES:
            raise ProvenanceError(f"asset evidence_class is invalid: {asset_id}")
        if not isinstance(asset.get("contains_formal_test_material"), bool):
            raise ProvenanceError(
                f"asset contains_formal_test_material is invalid: {asset_id}"
            )
        if (
            asset["evidence_class"] == "FORMAL_TEST_MATERIAL"
            and not asset["contains_formal_test_material"]
        ):
            raise ProvenanceError(f"formal test asset is not marked sensitive: {asset_id}")

    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ProvenanceError("profiles must be a non-empty object")
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ProvenanceError("profile id is invalid")
        if not isinstance(profile, Mapping) or set(profile) != PROFILE_KEYS:
            raise ProvenanceError(f"profile schema differs: {profile_id}")
        members = profile.get("asset_ids")
        if (
            not isinstance(members, list)
            or not members
            or len(members) != len(set(members))
            or any(member not in asset_ids for member in members)
        ):
            raise ProvenanceError(f"profile asset membership differs: {profile_id}")
        if not isinstance(profile.get("runtime_loadable_after_raw_restore"), bool):
            raise ProvenanceError(f"profile loadability flag differs: {profile_id}")

    loader_audit = payload.get("loader_dependency_audit")
    if not isinstance(loader_audit, Mapping) or set(loader_audit) != {
        "runtime_v4",
        "runtime_v5_core",
        "runtime_v5_qc2_candidate",
    }:
        raise ProvenanceError("loader dependency audit schema differs")
    constraints = payload.get("constraints")
    if not isinstance(constraints, Mapping) or set(constraints) != {
        "hash_only_verification",
        "no_checkpoint_deserialization",
        "no_formal_test_material_in_default_archive",
        "no_frozen_asset_rewrite",
        "absolute_path_contracts_are_not_portable",
    }:
        raise ProvenanceError("release constraints schema differs")
    if not all(value is True for value in constraints.values()):
        raise ProvenanceError("all release constraints must remain enabled")


def resolve_roots(repository_root: Path, workspace_root: Path | None) -> dict[str, Path]:
    repository = repository_root.resolve()
    workspace = (
        workspace_root.resolve()
        if workspace_root is not None
        else repository.parent.parent.resolve()
    )
    return {"repository": repository, "workspace": workspace}


def _selected_assets(
    payload: Mapping[str, Any], profile_ids: Sequence[str] | None
) -> list[Mapping[str, Any]]:
    assets = {asset["asset_id"]: asset for asset in payload["assets"]}
    if not profile_ids:
        return list(payload["assets"])
    selected: list[str] = []
    for profile_id in profile_ids:
        try:
            members = payload["profiles"][profile_id]["asset_ids"]
        except KeyError as error:
            raise ProvenanceError(f"unknown profile: {profile_id}") from error
        for asset_id in members:
            if asset_id not in selected:
                selected.append(asset_id)
    return [assets[asset_id] for asset_id in selected]


def verify_sources(
    payload: Mapping[str, Any],
    roots: Mapping[str, Path],
    profile_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = _selected_assets(payload, profile_ids)
    failures: list[dict[str, str]] = []
    verified: list[dict[str, Any]] = []
    for asset in selected:
        path = roots[asset["source_root"]] / PurePosixPath(asset["source_path"])
        if not path.is_file():
            failures.append(
                {"asset_id": asset["asset_id"], "reason": "missing", "path": str(path)}
            )
            continue
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_size != asset["bytes"]:
            failures.append(
                {
                    "asset_id": asset["asset_id"],
                    "reason": f"bytes:{actual_size}!={asset['bytes']}",
                    "path": str(path),
                }
            )
            continue
        if actual_sha256 != asset["sha256"]:
            failures.append(
                {
                    "asset_id": asset["asset_id"],
                    "reason": f"sha256:{actual_sha256}!={asset['sha256']}",
                    "path": str(path),
                }
            )
            continue
        verified.append(
            {
                "asset_id": asset["asset_id"],
                "bytes": actual_size,
                "sha256": actual_sha256,
            }
        )
    return {
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "profiles": list(profile_ids or payload["profiles"].keys()),
        "verified_asset_count": len(verified),
        "verified_bytes": sum(item["bytes"] for item in verified),
        "failures": failures,
        "verified_assets": verified,
    }


def _git(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def build_receipt(
    manifest_path: Path,
    payload: Mapping[str, Any],
    verification: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "iotj.release_provenance_verification_receipt.v1",
        "status": verification["status"],
        "release_id": payload["release_id"],
        "verified_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "manifest": {
            "path": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "code": {
            "head": _git(repository_root, "rev-parse", "HEAD"),
            "branch": _git(repository_root, "branch", "--show-current"),
        },
        "verification": verification,
        "verification_boundary": {
            "checkpoint_deserialized": False,
            "runtime_imported": False,
            "inference_or_evaluation_run": False,
            "frozen_asset_rewritten": False,
            "clean_checkout_deployment_proven": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify GAPS release provenance using bytes and SHA-256 only."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Defaults to the parent of the .worktrees directory.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        help="Verify one named profile; repeat to verify a union of profiles.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Write a verification receipt. The destination must not already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    try:
        payload = _read_json(manifest_path)
        validate_manifest(payload)
        roots = resolve_roots(args.repository_root, args.workspace_root)
        verification = verify_sources(payload, roots, args.profiles)
        receipt = build_receipt(
            manifest_path, payload, verification, roots["repository"]
        )
        if args.receipt is not None:
            destination = args.receipt.resolve()
            if destination.exists():
                raise ProvenanceError(f"receipt destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 0 if verification["status"] == "PASS" else 2
    except (OSError, ProvenanceError, subprocess.CalledProcessError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
