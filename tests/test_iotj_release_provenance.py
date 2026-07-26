from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_iotj_release_provenance.py"
SPEC = importlib.util.spec_from_file_location("verify_iotj_release_provenance", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(repository_payload: bytes, workspace_payload: bytes) -> dict:
    assets = [
        {
            "asset_id": "repo_asset",
            "component": "runtime_v5_core",
            "role": "classifier",
            "source_root": "repository",
            "source_path": "assets/model.bin",
            "restore_path": "repo/assets/model.bin",
            "bytes": len(repository_payload),
            "sha256": _sha256(repository_payload),
            "evidence_class": "RUNTIME_ASSET",
            "contains_formal_test_material": False,
        },
        {
            "asset_id": "workspace_asset",
            "component": "formal_test_evidence",
            "role": "test_input",
            "source_root": "workspace",
            "source_path": "dataset/test.bin",
            "restore_path": "workspace/dataset/test.bin",
            "bytes": len(workspace_payload),
            "sha256": _sha256(workspace_payload),
            "evidence_class": "FORMAL_TEST_MATERIAL",
            "contains_formal_test_material": True,
        },
    ]
    return {
        "schema_version": MODULE.SCHEMA_VERSION,
        "status": MODULE.STATUS,
        "release_id": "synthetic-release",
        "created_date": "2026-07-26",
        "code": {
            "repository": "GAPS",
            "branch": "test",
            "provenance_base_commit": "a" * 40,
            "frozen_evidence_commit": "b" * 40,
        },
        "source_roots": {
            "repository": "Git worktree root containing this manifest",
            "workspace": "Parent workspace root containing dataset/ and the worktree collection",
        },
        "profiles": {
            "runtime": {
                "description": "synthetic runtime",
                "asset_ids": ["repo_asset"],
                "runtime_loadable_after_raw_restore": False,
            },
            "audit": {
                "description": "synthetic audit",
                "asset_ids": ["repo_asset", "workspace_asset"],
                "runtime_loadable_after_raw_restore": False,
            },
        },
        "assets": assets,
        "loader_dependency_audit": {
            "runtime_v4": {},
            "runtime_v5_core": {},
            "runtime_v5_qc2_candidate": {},
        },
        "constraints": {
            "hash_only_verification": True,
            "no_checkpoint_deserialization": True,
            "no_formal_test_material_in_default_archive": True,
            "no_frozen_asset_rewrite": True,
            "absolute_path_contracts_are_not_portable": True,
        },
    }


def _write_source_tree(tmp_path: Path) -> tuple[Path, Path, dict]:
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    repository_payload = b"model"
    workspace_payload = b"test"
    (repository / "assets").mkdir(parents=True)
    (workspace / "dataset").mkdir(parents=True)
    (repository / "assets/model.bin").write_bytes(repository_payload)
    (workspace / "dataset/test.bin").write_bytes(workspace_payload)
    return repository, workspace, _manifest(repository_payload, workspace_payload)


def test_manifest_and_profile_verify_hash_only(tmp_path: Path) -> None:
    repository, workspace, manifest = _write_source_tree(tmp_path)
    MODULE.validate_manifest(manifest)
    result = MODULE.verify_sources(
        manifest,
        {"repository": repository, "workspace": workspace},
        ["runtime"],
    )
    assert result["status"] == "PASS"
    assert result["verified_asset_count"] == 1
    assert result["verified_assets"][0]["asset_id"] == "repo_asset"


def test_verification_fails_closed_on_sha_drift(tmp_path: Path) -> None:
    repository, workspace, manifest = _write_source_tree(tmp_path)
    (repository / "assets/model.bin").write_bytes(b"changed")
    result = MODULE.verify_sources(
        manifest,
        {"repository": repository, "workspace": workspace},
        ["runtime"],
    )
    assert result["status"] == "FAIL_CLOSED"
    assert result["failures"][0]["asset_id"] == "repo_asset"
    assert result["failures"][0]["reason"].startswith("bytes:")


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_path", "D:/absolute/model.bin"),
        ("source_path", "../escape.bin"),
        ("restore_path", "repo/../escape.bin"),
        ("restore_path", r"repo\assets\model.bin"),
    ],
)
def test_manifest_rejects_nonportable_or_escaping_paths(
    tmp_path: Path, field: str, value: str
) -> None:
    _, _, manifest = _write_source_tree(tmp_path)
    manifest["assets"][0][field] = value
    with pytest.raises(MODULE.ProvenanceError):
        MODULE.validate_manifest(manifest)


def test_manifest_rejects_unmarked_formal_test_asset(tmp_path: Path) -> None:
    _, _, manifest = _write_source_tree(tmp_path)
    manifest["assets"][1]["contains_formal_test_material"] = False
    with pytest.raises(MODULE.ProvenanceError):
        MODULE.validate_manifest(manifest)


def test_checked_in_manifest_schema_is_valid() -> None:
    payload = json.loads(
        (ROOT / "docs/system/iotj_release_provenance_manifest_20260726.json").read_text(
            encoding="utf-8"
        )
    )
    MODULE.validate_manifest(payload)
