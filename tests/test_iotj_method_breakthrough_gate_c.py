from __future__ import annotations

import numpy as np
import pytest


def test_incremental_cost_matrix_clips_only_primary_off_diagonal() -> None:
    from scripts.run_iotj_method_breakthrough_gate_c import cost_matrix_rows

    rows = [
        {"true_class": 0, "forced_route": 0, "true_ppm": 10.0, "pred_ppm": 12.0},
        {"true_class": 0, "forced_route": 1, "true_ppm": 10.0, "pred_ppm": 11.0},
        {"true_class": 0, "forced_route": 0, "true_ppm": 20.0, "pred_ppm": 24.0},
        {"true_class": 0, "forced_route": 1, "true_ppm": 20.0, "pred_ppm": 30.0},
    ]
    result = cost_matrix_rows(rows, class_ranges={0: 100.0})
    diagonal = next(row for row in result if row["forced_route"] == 0)
    offdiag = next(row for row in result if row["forced_route"] == 1)
    assert diagonal["primary_cost_squared_ppm"] == 0.0
    assert offdiag["mean_incremental_squared_ppm"] == pytest.approx((1 - 4 + 100 - 16) / 2)
    assert offdiag["primary_cost_squared_ppm"] == pytest.approx(40.5)
    assert offdiag["mean_range_normalized_incremental_squared"] == pytest.approx(0.00405)


def test_gate_c_decision_requires_heterogeneity_and_distributed_files() -> None:
    from scripts.run_iotj_method_breakthrough_gate_c import decide_gate_c

    motivated = decide_gate_c(
        positive_offdiagonal_costs=[1.0, 2.0, 20.0],
        positive_contribution_files=3,
        top_file_share=0.50,
        actual_misroute_windows=8,
    )
    assert motivated["decision"] == "COST_AWARE_ROUTING_MOTIVATED"

    uniform = decide_gate_c(
        positive_offdiagonal_costs=[10.0, 11.0, 12.0],
        positive_contribution_files=3,
        top_file_share=0.50,
        actual_misroute_windows=8,
    )
    assert uniform["decision"] == "COST_AWARE_ROUTING_NOT_SUPPORTED"

    single_file = decide_gate_c(
        positive_offdiagonal_costs=[1.0, 2.0, 20.0],
        positive_contribution_files=1,
        top_file_share=1.0,
        actual_misroute_windows=8,
    )
    assert single_file["decision"] == "COST_AWARE_ROUTING_NOT_SUPPORTED"


def test_grouped_bootstrap_resamples_whole_files_reproducibly() -> None:
    from scripts.run_iotj_method_breakthrough_gate_c import grouped_bootstrap

    rows = []
    for filename, offset in (("a.txt", 0.0), ("b.txt", 10.0), ("c.txt", 20.0)):
        for index in range(2):
            rows.append(
                {
                    "filename": filename,
                    "true_ppm": 10.0 + index,
                    "a0t_pred_ppm": 20.0 + offset,
                    "a4_pred_ppm": 12.0 + offset / 10.0,
                    "a0t_excess_se": 5.0 + offset,
                    "a4_excess_se": 1.0 + offset / 10.0,
                }
            )
    first = grouped_bootstrap(rows, replicates=20, seed=42)
    second = grouped_bootstrap(rows, replicates=20, seed=42)
    assert first == second
    assert len(first) == 20
    assert all(row["sampled_file_count"] == 3 for row in first)


def test_gate_c_protocol_is_calibration_only_and_fixed() -> None:
    from scripts.run_iotj_method_breakthrough_gate_c import gate_c_protocol

    protocol = gate_c_protocol()
    assert protocol["cost_matrix_source"] == "C5_canonical_calibration_only"
    assert protocol["primary_cost"] == "max_0_mean_incremental_squared_ppm"
    assert protocol["bootstrap_replicates"] == 2000
    assert protocol["bootstrap_seed"] == 42
    assert protocol["target_test_used_for_cost_matrix"] is False
    assert protocol["hyperparameter_search"] is False


def test_gate_c_rejects_test_analysis_before_matrix_lock(tmp_path) -> None:
    from scripts.run_iotj_method_breakthrough_gate_c import require_cost_matrix_lock

    with pytest.raises(RuntimeError, match="cost matrix lock"):
        require_cost_matrix_lock(tmp_path)
