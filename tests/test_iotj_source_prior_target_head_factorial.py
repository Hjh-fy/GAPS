from pathlib import Path

import pytest

from scripts.evaluate_iotj_source_prior_target_head_factorial import (
    FORMAL_OUTPUT_FILES,
    MLP_ALPHAS,
    MLP_HIDDEN_GRID,
    SOURCE_KEYS,
    VARIANTS,
    effect_row,
    freeze_decision_gate,
    require_new_empty_output,
)


def _summaries(ridge: float, mlp: float):
    values = {
        "E1_RIDGE_RICH": 20.0,
        "E2_RIDGE_H1": 19.0,
        "E2_RIDGE_H2": 18.0,
        "E2_RIDGE_H3": 17.0,
        "E1_RIDGE_PRIOR": ridge,
        "E1_MLP_RICH": 16.0,
        "E1_MLP_PRIOR": mlp,
    }
    return {
        key: {"calibration_validation_RMSE": value}
        for key, value in values.items()
    }


def test_factorial_schema_is_exact() -> None:
    assert VARIANTS["E1_RIDGE_RICH"]["source_keys"] == ()
    assert VARIANTS["E1_RIDGE_PRIOR"]["source_keys"] == SOURCE_KEYS
    assert VARIANTS["E1_MLP_RICH"]["head"] == "mlp"
    assert VARIANTS["E1_MLP_PRIOR"]["source_keys"] == SOURCE_KEYS
    assert MLP_HIDDEN_GRID == ((16,), (32,), (64,), (32, 16))
    assert MLP_ALPHAS == (0.001, 0.01, 0.1, 1.0)


def test_output_contract_contains_gate_and_requested_metrics() -> None:
    assert {
        "calibration_validation_summary.csv",
        "test_summary.csv",
        "per_gas_summary.csv",
        "factorial_effects.csv",
        "decision_gate.json",
        "target_head_manifest.json",
    }.issubset(FORMAL_OUTPUT_FILES)


def test_gate_requires_strictly_more_than_five_percent() -> None:
    at_threshold = freeze_decision_gate(
        _summaries(100.0, 95.0), "a" * 40
    )
    assert at_threshold["selection_status"] == "KEEP_RUNTIME_V4"
    assert at_threshold["runtime_action"] == "none"
    promoted = freeze_decision_gate(
        _summaries(100.0, 94.999), "a" * 40
    )
    assert (
        promoted["selection_status"]
        == "NEW_CANDIDATE_PENDING_CONFIRMATION"
    )
    assert promoted["runtime_action"] == "none"
    assert promoted["test_opened_after_selection"] is False
    assert promoted["test_metrics_used_for_selection"] is False


def test_effect_sign_is_positive_when_intervention_is_better() -> None:
    row = effect_row(
        "calibration_validation",
        "source_prior_gain_for_Ridge",
        "baseline",
        "intervention",
        {"baseline": 10.0, "intervention": 9.0},
    )
    assert row["absolute_RMSE_reduction"] == pytest.approx(1.0)
    assert row["relative_improvement_percent"] == pytest.approx(10.0)


def test_output_directory_is_non_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    require_new_empty_output(output)
    (output / "evidence.txt").write_text("frozen", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        require_new_empty_output(output)
