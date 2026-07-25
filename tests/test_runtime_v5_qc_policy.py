from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gaps_deploy.runtime_v5_qc import (
    QCCandidate,
    RuntimeV5QCPolicy,
    assign_group_folds,
    empirical_percentile,
    fit_feature_reference,
    fit_regression_consistency_scales,
    make_selection_lock,
    require_selection_lock,
)
from scripts.evaluate_iotj_runtime_v5_qc import v4_baseline


def _metadata() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gas in range(4):
        for concentration in range(10):
            for repeat in range(2):
                filename = f"G{gas}_C{concentration}_R{repeat}.txt"
                rows.extend(
                    {
                        "filename": filename,
                        "classification_label": gas,
                        "gas_code": f"G{gas}",
                        "concentration_code": f"{concentration:03d}",
                        "repeat_id": repeat,
                    }
                    for _ in range(4)
                )
    return rows


def test_group_folds_are_deterministic_isolated_and_balanced() -> None:
    metadata = _metadata()
    folds, audit = assign_group_folds(metadata, n_splits=5, seed=20260725)
    assert folds == assign_group_folds(metadata, n_splits=5, seed=20260725)[0]
    assert len(folds) == 320
    assert sorted(set(folds)) == [0, 1, 2, 3, 4]
    assert audit["group_count"] == 80
    assert audit["group_cross_fold_count"] == 0
    assert [folds.count(i) for i in range(5)] == [64] * 5
    for name in {str(row["filename"]) for row in metadata}:
        assert len({folds[i] for i, row in enumerate(metadata) if row["filename"] == name}) == 1


def test_variable_size_filename_groups_remain_whole_and_row_balanced() -> None:
    sizes = [1] * 2 + [2] * 6 + [3] * 15 + [4] * 34 + [5] * 15 + [6] * 6 + [7] * 2
    assert len(sizes) == 80 and sum(sizes) == 320
    metadata: list[dict[str, object]] = []
    for group, size in enumerate(sizes):
        gas = group % 4
        concentration = (group // 4) % 10
        filename = f"G{gas}_C{concentration:02d}_R{group // 40}.txt"
        metadata.extend({"filename": filename, "classification_label": gas, "gas_code": f"G{gas}", "concentration_code": f"{concentration:03d}", "repeat_id": group // 40} for _ in range(size))
    folds, audit = assign_group_folds(metadata, n_splits=5, seed=20260725)
    assert audit["group_size_min"] == 1
    assert audit["group_size_max"] == 7
    assert audit["group_size_distribution"] == {"1": 2, "2": 6, "3": 15, "4": 34, "5": 15, "6": 6, "7": 2}
    assert max(audit["fold_row_counts"]) - min(audit["fold_row_counts"]) <= 7
    for filename in {str(row["filename"]) for row in metadata}:
        assert len({folds[index] for index, row in enumerate(metadata) if row["filename"] == filename}) == 1


@pytest.mark.parametrize("mutation", ["missing", "wrong_size", "mixed_label"])
def test_group_folds_fail_closed_on_invalid_filename_groups(mutation: str) -> None:
    metadata = _metadata()
    if mutation == "missing":
        metadata[0].pop("filename")
    elif mutation == "wrong_size":
        metadata.pop()
    else:
        metadata[0]["classification_label"] = 3
    with pytest.raises(ValueError, match="filename|group"):
        assign_group_folds(metadata, n_splits=5, seed=20260725)


def test_feature_reference_and_regression_scales_are_training_rows_only() -> None:
    rows = []
    for index in range(8):
        gas = index % 2
        rows.append(
            {
                "true_class": gas,
                "pred_class": gas,
                "representation": [float(index), float(index + 1)],
                "source_h1_ppm": float(index),
                "prediction_ppm": float(index + (1 if gas == 0 else 2)),
            }
        )
    reference = fit_feature_reference(rows, epsilon=1e-8)
    scales = fit_regression_consistency_scales(rows, epsilon=1e-8)
    assert reference["feature_dimension"] == 2
    assert set(reference["classes"]) == {"0", "1"}
    assert set(scales["per_predicted_gas"]) == {"0", "1"}
    assert all(value["scale"] >= 1e-8 for value in scales["per_predicted_gas"].values())
    with pytest.raises(ValueError, match="NaN/Inf"):
        fit_feature_reference([{**rows[0], "representation": [np.nan, 0.0]}], epsilon=1e-8)


def test_candidate_group_means_and_risk_direction() -> None:
    distributions = {
        key: [0.0, 1.0, 2.0, 3.0]
        for key in (
            "entropy",
            "inverse_margin",
            "prototype_distance",
            "support_distance",
            "normalized_regression_disagreement",
        )
    }
    low = {key: 0.0 for key in distributions}
    high = {key: 3.0 for key in distributions}
    policy = RuntimeV5QCPolicy.from_payload(
        {
            "schema_version": "iotj.runtime_v5_qc_policy.v1",
            "status": "locked",
            "selected_candidate": "QC3",
            "epsilon": 1e-8,
            "feature_reference": {
                "feature_dimension": 2,
                "classes": {
                    "0": {"mean": [0.0, 0.0], "scale": [1.0, 1.0], "support": [[0.0, 0.0]], "n": 1}
                },
            },
            "regression_consistency_scale": {"per_predicted_gas": {"0": {"scale": 1.0}}},
            "component_distributions": distributions,
            "workpoints": {
                "HC95": {"accept_threshold": 0.8, "reject_threshold": 0.9},
                "HC90": {"accept_threshold": 0.7, "reject_threshold": 0.85},
            },
            "decision_semantics": {"auto_output_only_for_accept": True},
        }
    )
    low_score = policy.aggregate_percentiles({key: empirical_percentile(value, distributions[key]) for key, value in low.items()})
    high_score = policy.aggregate_percentiles({key: empirical_percentile(value, distributions[key]) for key, value in high.items()})
    assert low_score["deployment_risk"] < high_score["deployment_risk"]
    assert low_score["confidence_group"] == np.mean([0.25, 0.25])
    assert high_score["deployment_risk"] == 1.0
    assert QCCandidate.QC3.value == "QC3"


def test_selection_lock_is_immutable_and_hash_binds_assets(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    policy.write_text("{}", encoding="utf-8")
    lock_path = tmp_path / "qc_selection_lock.json"
    payload = make_selection_lock(
        selected_candidate="QC2",
        selection_reason="simplest passing",
        policy_path=policy,
        bound_assets={"qc_policy": policy},
        build_commit="a" * 40,
    )
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = require_selection_lock(lock_path, {"qc_policy": policy})
    assert loaded["test_opened_after_lock"] is False
    assert loaded["selected_candidate"] == "QC2"

    policy.write_text("{\"changed\": true}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash differs"):
        require_selection_lock(lock_path, {"qc_policy": policy})


def test_policy_decisions_emit_auto_output_only_for_accept() -> None:
    assert RuntimeV5QCPolicy.decision(0.2, 10.0, {"accept_threshold": 0.5, "reject_threshold": 0.8}) == ("accept", 10.0)
    assert RuntimeV5QCPolicy.decision(0.6, 10.0, {"accept_threshold": 0.5, "reject_threshold": 0.8}) == ("review", None)
    assert RuntimeV5QCPolicy.decision(0.9, 10.0, {"accept_threshold": 0.5, "reject_threshold": 0.8}) == ("reject", None)
    with pytest.raises(ValueError, match="NaN/Inf"):
        RuntimeV5QCPolicy.decision(np.nan, 10.0, {"accept_threshold": 0.5, "reject_threshold": 0.8})


def test_v4_baseline_loader_carries_route_for_automatic_guard() -> None:
    summary, gas_rows = v4_baseline("HC95")
    assert summary["N"] == 1360
    assert summary["misclassified_N"] >= 0
    assert len(gas_rows) == 4
