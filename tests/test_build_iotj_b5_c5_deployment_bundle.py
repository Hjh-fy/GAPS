import json
import hashlib
from pathlib import Path

import pytest


def _ready_audit(tmp_path: Path) -> Path:
    asset = tmp_path / "classifier.pth"
    asset.write_bytes(b"b5")
    reference = tmp_path / "offline_reference_1360.csv"
    reference.write_text("sample_index,final_ppm\n0,25.0\n", encoding="utf-8")
    audit = {
        "status": "ready",
        "reasons": [],
        "assets": {
            "classifier": {
                "path": str(asset),
                "bytes": asset.stat().st_size,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            },
            "offline_reference_1360": {
                "path": str(reference),
                "bytes": reference.stat().st_size,
                "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            },
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
