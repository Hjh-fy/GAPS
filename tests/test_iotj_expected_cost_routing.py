"""Protocol tests for Phase-4 expected downstream-cost routing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_expected_cost_router_uses_probability_weighted_cost_without_threshold() -> None:
    from scripts.run_iotj_expected_cost_routing import expected_cost_routes

    probabilities = np.asarray([[0.55, 0.45], [0.10, 0.90]], dtype=np.float64)
    # Rows are latent true classes c; columns are candidate routes j.
    costs = np.asarray([[0.0, 1.0], [100.0, 0.0]], dtype=np.float64)

    routes, risks = expected_cost_routes(probabilities, costs)

    assert routes.tolist() == [1, 1]
    assert risks[0].tolist() == pytest.approx([45.0, 0.55])


def test_cost_router_decision_uses_frozen_thresholds() -> None:
    from scripts.run_iotj_expected_cost_routing import decide_cost_router

    supported = decide_cost_router(
        argmax_rmse=20.0,
        cost_rmse=18.0,
        argmax_macro_f1=0.98,
        cost_macro_f1=0.977,
        bootstrap_probability_negative=0.8,
    )
    assert supported["decision"] == "COST_AWARE_ROUTING_SUPPORTED"

    modest = decide_cost_router(
        argmax_rmse=20.0,
        cost_rmse=19.4,
        argmax_macro_f1=0.98,
        cost_macro_f1=0.978,
        bootstrap_probability_negative=0.4,
    )
    assert modest["decision"] == "COST_AWARE_ROUTING_MODEST"

    classification_cost = decide_cost_router(
        argmax_rmse=20.0,
        cost_rmse=19.0,
        argmax_macro_f1=0.98,
        cost_macro_f1=0.96,
        bootstrap_probability_negative=0.9,
    )
    assert classification_cost["decision"] == "QUANTITATIVE_GAIN_WITH_CLASSIFICATION_COST"


def test_cost_matrix_lock_rejects_mutation(tmp_path: Path) -> None:
    from scripts.run_iotj_expected_cost_routing import require_cost_matrix_lock, sha256_file

    matrix = tmp_path / "EXPECTED_COST_MATRIX.csv"
    matrix.write_text("true_class,route,cost\n0,0,0\n", encoding="utf-8")
    lock = tmp_path / "CALIBRATION_COST_MATRIX_LOCK.json"
    lock.write_text(
        json.dumps(
            {
                "status": "LOCKED_BEFORE_TARGET_TEST",
                "target_test_opened": False,
                "cost_matrix_sha256": sha256_file(matrix),
            }
        ),
        encoding="utf-8",
    )
    assert require_cost_matrix_lock(tmp_path)["status"] == "PASS"
    matrix.write_text("true_class,route,cost\n0,0,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cost matrix lock"):
        require_cost_matrix_lock(tmp_path)


def test_grouped_bootstrap_is_raw_filename_seed42_reproducible() -> None:
    from scripts.run_iotj_expected_cost_routing import grouped_bootstrap

    rows = []
    for filename, offset in (("a.txt", 0.0), ("b.txt", 2.0), ("c.txt", 4.0)):
        for truth in (10.0, 20.0):
            rows.append(
                {
                    "filename": filename,
                    "true_ppm": truth,
                    "argmax_pred_ppm": truth + 3.0 + offset,
                    "cost_pred_ppm": truth + 1.0 + offset,
                }
            )
    first = grouped_bootstrap(rows, replicates=20, seed=42)
    second = grouped_bootstrap(rows, replicates=20, seed=42)
    assert first == second
    assert len(first) == 20
    assert all(row["sampled_raw_file_count"] == 3 for row in first)
