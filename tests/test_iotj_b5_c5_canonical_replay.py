from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_ready_replay(root: Path) -> None:
    _write_json(root / "h8_no_rescue" / "r4_policy.json", {
        "forbidden_runtime_dependencies": ["C3", "C4", "R3aK16", "H8+C4", "P4"]
    })
    _write_json(root / "h23_plus" / "h23_reference.json", {
        "forbidden_runtime_dependencies": ["C3", "C4", "R3aK16", "H8+C4", "P4"]
    })
    _write_json(root / "high_coverage_qc" / "manifest.json", {
        "pred_key": "target_ridge_plus_source_preds_ppm", "secondary_workpoint": "HC90"
    })
    for name in ("risk_policy.json", "component_calibrator.json", "feature_reference.json", "risk_selection.json"):
        _write_json(root / "high_coverage_qc" / name, {})
    rows = [
        {"sample_index": index, "pred_class": 0, "target_ridge_plus_source_preds_ppm": 10.0}
        for index in range(1360)
    ]
    target = root / "h8_no_rescue" / "target_predictions_plus_source_preds.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_canonical_replay_requires_r4_policy(tmp_path: Path) -> None:
    from scripts.verify_iotj_b5_c5_canonical_replay import verify_canonical_replay

    with pytest.raises(FileNotFoundError, match="r4_policy"):
        verify_canonical_replay(tmp_path)


def test_canonical_replay_accepts_complete_c5_only_contract(tmp_path: Path) -> None:
    from scripts.verify_iotj_b5_c5_canonical_replay import verify_canonical_replay

    _make_ready_replay(tmp_path)

    report = verify_canonical_replay(tmp_path)

    assert report["status"] == "ready"
    assert report["runtime_rows"] == 1360
    assert report["r4_policy"].endswith("h8_no_rescue/r4_policy.json")


def test_regression_suite_declares_runtime_assets_as_required_outputs(tmp_path: Path) -> None:
    from scripts.run_iotj_c5_regression_suite import expected_outputs

    outputs = {path.as_posix() for path in expected_outputs(tmp_path / "suite")}

    assert (tmp_path / "suite" / "h23_plus" / "h23_reference.json").as_posix() in outputs
    assert (tmp_path / "suite" / "h8_no_rescue" / "r4_policy.json").as_posix() in outputs
