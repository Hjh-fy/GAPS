import hashlib
import json
from pathlib import Path

import pytest


def _bound_b5_assets(tmp_path: Path) -> dict[str, Path]:
    from scripts.inspect_b5_c5_deployment_inputs import REQUIRED_KEYS

    paths: dict[str, Path] = {}
    for key in REQUIRED_KEYS:
        suffix = ".csv" if key == "offline_reference_1360" else ".json"
        path = tmp_path / f"{key}{suffix}"
        path.write_bytes(key.encode("utf-8"))
        paths[key] = path
    return paths


def test_audit_requires_the_b5_policy_asset_roles(tmp_path: Path) -> None:
    from scripts.inspect_b5_c5_deployment_inputs import audit_input_paths

    result = audit_input_paths(_bound_b5_assets(tmp_path))

    assert result["status"] == "ready"
    assert set(result["assets"]) == {
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
    }


def test_bundle_keeps_parity_reference_outside_runtime_assets(tmp_path: Path) -> None:
    from scripts.build_iotj_b5_c5_deployment_bundle import build_bundle
    from scripts.inspect_b5_c5_deployment_inputs import audit_input_paths

    paths = _bound_b5_assets(tmp_path)
    audit = audit_input_paths(paths)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    manifest = build_bundle(audit_path, tmp_path / "bundle")

    assert "offline_reference_1360" not in manifest["assets"]
    assert manifest["parity_reference"]["sha256"] == hashlib.sha256(
        paths["offline_reference_1360"].read_bytes()
    ).hexdigest()
    assert not (tmp_path / "bundle" / "assets" / "offline_reference_1360.csv").exists()


def test_audit_rejects_obsolete_and_p4_asset_keys(tmp_path: Path) -> None:
    from scripts.inspect_b5_c5_deployment_inputs import audit_input_paths

    paths = _bound_b5_assets(tmp_path)
    paths["source_ridge"] = tmp_path / "source_ridge.json"
    paths["source_ridge"].write_text("obsolete", encoding="utf-8")
    paths["p4"] = tmp_path / "neutral.json"
    paths["p4"].write_text("forbidden", encoding="utf-8")

    result = audit_input_paths(paths)

    assert result["status"] == "blocked"
    assert "unknown_asset_key:source_ridge" in result["reasons"]
    assert "legacy_forbidden:p4" in result["reasons"]


def test_bundle_rejects_incomplete_ready_audit(tmp_path: Path) -> None:
    from scripts.build_iotj_b5_c5_deployment_bundle import build_bundle

    classifier = tmp_path / "classifier.pth"
    reference = tmp_path / "offline_reference_1360.csv"
    classifier.write_bytes(b"classifier")
    reference.write_bytes(b"reference")
    audit = {
        "status": "ready",
        "reasons": [],
        "assets": {
            "classifier": {
                "path": str(classifier),
                "sha256": hashlib.sha256(classifier.read_bytes()).hexdigest(),
            },
            "offline_reference_1360": {
                "path": str(reference),
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            },
        },
    }
    audit_path = tmp_path / "incomplete_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required audited asset"):
        build_bundle(audit_path, tmp_path / "bundle")


def test_bundle_manifest_records_input_audit_sha256(tmp_path: Path) -> None:
    from scripts.build_iotj_b5_c5_deployment_bundle import build_bundle
    from scripts.inspect_b5_c5_deployment_inputs import audit_input_paths

    audit = audit_input_paths(_bound_b5_assets(tmp_path))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")

    manifest = build_bundle(audit_path, tmp_path / "bundle")

    assert manifest["input_audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()
