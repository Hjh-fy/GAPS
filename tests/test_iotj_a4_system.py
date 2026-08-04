from __future__ import annotations

from scripts.finalize_iotj_a4_system import build_system_evidence


def test_build_system_evidence_keeps_measured_and_theoretical_bytes_distinct() -> None:
    efficiency = [
        {
            "runtime": "RUNTIME_V5_REGRESSION_CORE",
            "Pi_p50_ms": "3.7",
            "Pi_p95_ms": "3.8",
            "Pi_peak_RSS_MiB": "234.2",
            "Pi_throughput_windows_per_s": "254.2",
            "deployment_status": "FINAL_SIMPLIFIED_REGRESSION",
        }
    ]
    fl = [{"measured_application_total_25round_bytes": "17572650", "rounds": "25"}]
    h1 = [
        {
            "direction": "TOTAL",
            "kind": "one_shot_sufficient_statistics_exchange",
            "theoretical_serialized_exchange_bytes": "7710128",
        }
    ]
    locked = {
        "client_c1": ["python", "--device", "cpu"],
        "client_c2": ["python", "--device", "cpu"],
        "server": ["python", "--rounds", "25"],
        "protocol": {"seed": 42, "target": "C5"},
    }
    completed = {"fixed_endpoint": {"round": 25}, "experiment_id": "FCL-E4-A4"}
    run = {"wall_seconds": 4641.0, "target_test_opened": False}

    system, physical = build_system_evidence(efficiency, fl, h1, locked, completed, run)

    communication = [row for row in system if row["record_type"] == "communication"]
    assert communication[0]["evidence_type"] == "measured_application_bytes"
    assert communication[1]["evidence_type"] == "theoretical_serialized_exchange_bytes"
    assert physical[0]["status"] == "PASS"
    assert physical[0]["completed_rounds"] == 25
    assert physical[0]["target_test_opened_during_training"] is False
