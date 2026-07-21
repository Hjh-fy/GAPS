import hashlib
import json
from pathlib import Path


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
