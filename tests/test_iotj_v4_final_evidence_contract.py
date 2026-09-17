import numpy as np
import pytest

from gaps_flower.iotj_v4_final_evidence import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BUDGET_COUNTS,
    RIDGE_ALPHAS,
    classify_provenance,
    match_budget_identities,
    paired_grouped_interval,
    pooled_classification_metrics,
)


def test_frozen_protocol_constants():
    assert BUDGET_COUNTS == {20: 320, 15: 240, 10: 160, 5: 80}
    assert RIDGE_ALPHAS == (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    assert (BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED) == (5000, 42)


def test_pooled_metrics_merge_predictions_before_macro_f1():
    rows = [
        {"target": "C3", "true_class": 0, "predicted_class": 0},
        {"target": "C3", "true_class": 1, "predicted_class": 0},
        {"target": "C4", "true_class": 1, "predicted_class": 1},
        {"target": "C5", "true_class": 1, "predicted_class": 1},
    ]
    result = pooled_classification_metrics(rows)
    assert result["aggregation"] == "merged_predictions_global_metric"
    assert result["N"] == 4
    assert result["accuracy"] == 0.75
    assert result["macro_f1"] == pytest.approx(0.7333333333333334)


def _canonical_rows():
    return [
        {"physical_identity": "a", "true_class": 0, "value": 1.0},
        {"physical_identity": "b", "true_class": 1, "value": 2.0},
    ]


def _budget_rows():
    return [
        {"physical_identity": "b", "true_class": 1, "value": 20.0, "budget_pct": 20, "method": "R84_CONCAT"},
        {"physical_identity": "a", "true_class": 0, "value": 10.0, "budget_pct": 20, "method": "R83_TARGET_ONLY"},
    ]


def test_identity_matching_returns_stable_canonical_order():
    matched = match_budget_identities(_budget_rows(), _canonical_rows())
    assert [row["physical_identity"] for row in matched] == ["a", "b"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda rows: rows + [dict(rows[0])], "duplicate"),
        (lambda rows: rows[:1], "missing"),
        (lambda rows: [{**rows[0], "true_class": 0}, rows[1]], "label"),
        (lambda rows: [{**rows[0], "value": np.nan}, rows[1]], "non-finite"),
        (lambda rows: [{**rows[0], "budget_pct": 19}, rows[1]], "budget"),
        (lambda rows: [{**rows[0], "method": "OTHER"}, rows[1]], "method"),
    ],
)
def test_identity_matching_fails_closed(mutator, message):
    with pytest.raises((ValueError, RuntimeError), match=message):
        match_budget_identities(mutator(_budget_rows()), _canonical_rows())


def test_grouped_bootstrap_fails_when_filename_is_missing():
    with pytest.raises(RuntimeError, match="grouping key"):
        paired_grouped_interval(
            truth=np.array([0.0, 1.0]),
            pred_r83=np.array([0.0, 1.0]),
            pred_r84=np.array([0.0, 1.0]),
            gas_range=np.ones(2),
            groups=np.array(["", ""]),
        )


def test_grouped_bootstrap_reports_c5_point_and_group_count(monkeypatch):
    seen = {}

    def fake_bootstrap(truth, pred83, pred84, groups, ranges, *, replicates, seed):
        seen.update(replicates=replicates, seed=seed)
        return {name: [0.0, 0.0] for name in ("rmse_delta", "mae_delta", "nrmse_range_delta")}

    monkeypatch.setattr("gaps_flower.iotj_v4_final_evidence.bootstrap_paired_group_deltas", fake_bootstrap)
    result = paired_grouped_interval(
        truth=np.array([0.0, 1.0]),
        pred_r83=np.array([0.0, 0.0]),
        pred_r84=np.array([0.0, 1.0]),
        gas_range=np.ones(2),
        groups=np.array(["C5|file-a", "C5|file-b"]),
    )
    assert seen == {"replicates": 5000, "seed": 42}
    assert {row["scope"] for row in result} == {"C5"}
    assert {row["n_groups"] for row in result} == {2}
    assert {row["n_replicates"] for row in result} == {5000}


def test_provenance_classification_is_machine_readable_and_fail_closed():
    assert classify_provenance({"checkpoint": True, "dataset": True}) == {
        "classification": "VERIFIED",
        "passed": True,
        "failed_checks": [],
    }
    result = classify_provenance({"checkpoint": True, "dataset": False})
    assert result == {
        "classification": "FAIL_CLOSED",
        "passed": False,
        "failed_checks": ["dataset"],
    }
    with pytest.raises(ValueError, match="checks"):
        classify_provenance({})
    with pytest.raises(ValueError, match="boolean"):
        classify_provenance({"checkpoint": "yes"})
