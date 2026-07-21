import json
import hashlib
from pathlib import Path

import pytest


def _ready_audit(tmp_path: Path) -> Path:
    from scripts.inspect_b5_c5_deployment_inputs import REQUIRED_KEYS

    assets = {}
    for key in REQUIRED_KEYS:
        suffix = ".pth" if key == "classifier" else ".csv" if key == "offline_reference_1360" else ".json"
        asset = tmp_path / f"{key}{suffix}"
        asset.write_bytes(b"b5" if key == "classifier" else key.encode("utf-8"))
        assets[key] = asset
    audit = {
        "status": "ready",
        "reasons": [],
        "assets": {
            key: {
                "path": str(asset),
                "bytes": asset.stat().st_size,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }
            for key, asset in assets.items()
        },
    }
    path = tmp_path / "input_audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def test_bundle_rejects_legacy_input_audit(tmp_path: Path) -> None:
    from scripts.build_iotj_b5_c5_deployment_bundle import build_bundle

    audit = _ready_audit(tmp_path)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["reasons"] = ["legacy_forbidden:r3ak16_reference"]
    payload["status"] = "blocked"
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden legacy"):
        build_bundle(audit, tmp_path / "bundle")


def test_bundle_copies_bound_assets_and_hashes_them(tmp_path: Path) -> None:
    from scripts.build_iotj_b5_c5_deployment_bundle import build_bundle

    manifest = build_bundle(_ready_audit(tmp_path), tmp_path / "bundle")

    assert manifest["status"] == "ready"
    copied = tmp_path / "bundle" / "assets" / "classifier.pth"
    assert copied.read_bytes() == b"b5"
    assert manifest["assets"]["classifier"]["sha256"]
