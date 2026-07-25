from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.benchmark_iotj_final_runtime import latency_statistics, require_same_row_universe
from scripts.build_iotj_final_system_evidence import (
    FrozenAssetError,
    summarize_selective_rows,
    verify_frozen_assets,
)
from scripts.prepare_iotj_final_benchmark_package import prepare
from pathlib import PurePosixPath


def test_latency_statistics_reports_requested_distribution() -> None:
    result = latency_statistics([1.0, 2.0, 3.0, 4.0])
    assert result["n"] == 4
    assert result["mean_ms"] == pytest.approx(2.5)
    assert result["sample_std_ms"] == pytest.approx(math.sqrt(5.0 / 3.0))
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["p90_ms"] == pytest.approx(3.7)
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["p99_ms"] == pytest.approx(3.97)
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0


def test_require_same_row_universe_fails_closed() -> None:
    require_same_row_universe(["C5:test:0", "C5:test:1"], ["C5:test:0", "C5:test:1"])
    with pytest.raises(ValueError, match="row universe"):
        require_same_row_universe(["C5:test:0"], ["C5:test:1"])


def test_verify_frozen_assets_checks_hash_and_size(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"frozen")
    record = {
        "asset": {
            "path": str(asset),
            "bytes": 6,
            "sha256": hashlib.sha256(b"frozen").hexdigest(),
        }
    }
    verified = verify_frozen_assets(record)
    assert verified["asset"]["status"] == "PASS"
    asset.write_bytes(b"drift")
    with pytest.raises(FrozenAssetError, match="asset"):
        verify_frozen_assets(record)


def test_summarize_selective_rows_preserves_quality_coverage_scope() -> None:
    rows = [
        {"row_key": "r0", "true_class": 1, "pred_class": 1, "true_ppm": 200.0, "prediction_ppm": 190.0, "qc_decision": "accept"},
        {"row_key": "r1", "true_class": 1, "pred_class": 0, "true_ppm": 250.0, "prediction_ppm": 220.0, "qc_decision": "review"},
        {"row_key": "r2", "true_class": 0, "pred_class": 0, "true_ppm": 100.0, "prediction_ppm": 140.0, "qc_decision": "reject"},
    ]
    result = summarize_selective_rows(
        rows,
        runtime="V5",
        regression_structure="B5 + Federated H1 + C5 Ridge",
        workpoint="HC95",
        deployment_status="VALID_CANDIDATE_NOT_PROMOTED",
    )
    assert result["total_N"] == 3
    assert (result["accept_N"], result["review_N"], result["reject_N"]) == (1, 1, 1)
    assert result["accepted_yield"] == pytest.approx(1 / 3)
    assert result["accepted_plus_review_yield"] == pytest.approx(2 / 3)
    assert result["accepted_RMSE"] == pytest.approx(10.0)
    assert result["accepted_plus_review_RMSE"] == pytest.approx(math.sqrt(500.0))
    assert result["misclassified_accept_N"] == 0
    assert result["misclassified_review_N"] == 1
    assert result["CO_high_N"] == 2
    assert result["CO_high_accepted_yield"] == pytest.approx(0.5)


def test_portable_package_closes_runtime_import_and_reference_dependencies(tmp_path: Path) -> None:
    output = tmp_path / "portable"
    prepare(output, PurePosixPath("/opt/iotj-benchmark"))
    assert (output / "project/scripts/iotj_b5_c5_bundle_contract.py").is_file()
    assert (output / "project/scripts/benchmark_iotj_final_runtime.py").is_file()
    assert (output / "project/probe_iotj_runtime_cold_start.py").is_file()
    manifest = __import__("json").loads((output / "v4_bundle/manifest.json").read_text(encoding="utf-8"))
    assert manifest["parity_reference"]["source_path"] == "/opt/iotj-benchmark/v4_bundle/offline_reference_1360.csv"
    assert (output / "v4_bundle/offline_reference_1360.csv").is_file()
    v5_manifest = __import__("json").loads((output / "v5_bundle/bundle_manifest.json").read_text(encoding="utf-8"))
    assert v5_manifest["calibration_lineage"]["path"] == "/opt/iotj-benchmark/v5_bundle/calibration_lock.json"
    assert (output / "v5_bundle/calibration_lock.json").is_file()


def test_cold_start_child_direct_entry_emits_two_events() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, str(root / "scripts/probe_iotj_runtime_cold_start.py"), "--child",
        "--runtime", "RUNTIME_V5_REGRESSION_CORE",
        "--contract", str(root / "results/iotj_b5_c5_runtime_v5_candidate_20260724/runtime_v5/runtime_contract_v5.json"),
        "--data-root", str(root / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5"),
        "--output", str(root / ".unused_cold_start_test.json"),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    events = [__import__("json").loads(line)["event"] for line in result.stdout.splitlines()]
    assert events == ["runtime_ready", "first_inference_complete"]
