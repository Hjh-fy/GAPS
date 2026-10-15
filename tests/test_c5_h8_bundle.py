from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


FORBIDDEN = ["C3", "C4", "R3aK16", "H8+C4", "P4"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _valid_r4_policy() -> dict[str, object]:
    return {
        "schema_version": "iotj.b5_c5_r4_policy.v1",
        "direction": "C1_C2_to_C5",
        "classifier_sha256": "a" * 64,
        "forbidden_runtime_dependencies": FORBIDDEN,
        "source_aug_target_ridge_policy": {
            "switch_rule": {"class_ids": [0, 1, 2, 3], "enabled_clients": ["C5"]}
        },
    }


def _valid_risk_policy(workpoints: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "score_key": "deployment_risk_full",
        "workpoints": workpoints
        or {
            "HC95": {"accept_threshold": 0.8, "reject_threshold": 0.9},
            "HC90": {"accept_threshold": 0.7, "reject_threshold": 0.85},
        },
    }


def _write_bundle(tmp_path: Path) -> Path:
    from scripts.iotj_b5_c5_bundle_contract import RUNTIME_ASSET_KEYS

    bundle = tmp_path / "bundle"
    assets_dir = bundle / "assets"
    assets_dir.mkdir(parents=True)
    assets: dict[str, dict[str, object]] = {}
    for key in RUNTIME_ASSET_KEYS:
        suffix = ".pth" if key == "classifier" else ".json"
        path = assets_dir / f"{key}{suffix}"
        if key == "r4_policy":
            _write_json(path, _valid_r4_policy())
        elif key == "qc_risk_policy":
            _write_json(path, _valid_risk_policy())
        else:
            _write_json(path, {"asset": key})
        assets[key] = {"bundle_path": path.relative_to(bundle).as_posix(), "sha256": _sha256(path)}
    reference = bundle / "offline_reference_1360.csv"
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index"])
        writer.writeheader()
    _write_json(
        bundle / "manifest.json",
        {
            "schema_version": "iotj.b5_c5_deployment_bundle.v1",
            "status": "ready",
            "forbidden": FORBIDDEN,
            "assets": assets,
            "parity_reference": {
                "source_path": reference.as_posix(),
                "sha256": _sha256(reference),
            },
        },
    )
    return bundle


def test_ready_hashed_bundle_loads_with_hc95(tmp_path: Path) -> None:
    from gaps_deploy.c5_h8_bundle import load_c5_h8_bundle
    from scripts.iotj_b5_c5_bundle_contract import RUNTIME_ASSET_KEYS

    loaded = load_c5_h8_bundle(_write_bundle(tmp_path))

    assert loaded.default_workpoint == "HC95"
    assert set(loaded.asset_paths) == set(RUNTIME_ASSET_KEYS)


@pytest.mark.parametrize("mutation", ["schema", "hash", "forbidden", "missing_role"])
def test_bad_bundle_fails_closed(tmp_path: Path, mutation: str) -> None:
    from gaps_deploy.c5_h8_bundle import C5H8BundleError, load_c5_h8_bundle

    bundle = _write_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        manifest["schema_version"] = "unknown"
    elif mutation == "hash":
        manifest["assets"]["classifier"]["sha256"] = "0" * 64
    elif mutation == "forbidden":
        manifest["forbidden"] = ["C3"]
    else:
        del manifest["assets"]["normalization"]
    _write_json(manifest_path, manifest)

    with pytest.raises(C5H8BundleError):
        load_c5_h8_bundle(bundle)


def test_bundle_rejects_missing_hc95_or_invalid_hc90_thresholds(tmp_path: Path) -> None:
    from gaps_deploy.c5_h8_bundle import C5H8BundleError, load_c5_h8_bundle

    bundle = _write_bundle(tmp_path)
    risk_path = bundle / "assets" / "qc_risk_policy.json"
    _write_json(risk_path, _valid_risk_policy({"HC90": {"accept_threshold": 0.9, "reject_threshold": 0.8}}))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"]["qc_risk_policy"]["sha256"] = _sha256(risk_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(C5H8BundleError, match="HC95"):
        load_c5_h8_bundle(bundle)
