from scripts.evaluate_iotj_calibration_protocol_harmonization import (
    classify_protocol_sensitivity,
    historical_nested_subsets,
)


def _metadata():
    rows = []
    for index in range(320):
        gas = index % 4
        rows.append({
            "filename": f"file_{index // 4:03d}",
            "classification_label": gas,
            "concentration": float((index // 4) % 10) * 25.0,
        })
    return rows


def test_historical_subsets_are_exact_nested_and_pool_preserving():
    metadata = _metadata()
    fit = list(range(240))
    validation = list(range(240, 320))
    result = historical_nested_subsets(metadata, fit, validation, 2026072500)
    assert {budget: len(value["fit"]) for budget, value in result.items()} == {160: 120, 80: 60, 40: 30}
    assert {budget: len(value["validation"]) for budget, value in result.items()} == {160: 40, 80: 20, 40: 10}
    assert set(result[40]["fit"]) <= set(result[80]["fit"]) <= set(result[160]["fit"]) <= set(fit)
    assert set(result[40]["validation"]) <= set(result[80]["validation"]) <= set(result[160]["validation"]) <= set(validation)
    assert set(result[160]["fit"]).isdisjoint(result[160]["validation"])


def test_protocol_decision_A_when_both_degrade_with_similar_magnitude():
    decision, audit = classify_protocol_sensitivity(
        {160: .20, 80: .35, 40: .50}, {160: .18, 80: .30, 40: .42}
    )
    assert decision == "BUDGET_SENSITIVITY_ROBUST_ACROSS_PROTOCOLS"
    assert audit["max_absolute_relative_degradation_gap"] < .20


def test_protocol_decision_B_when_both_degrade_but_magnitude_differs():
    decision, _ = classify_protocol_sensitivity(
        {160: .40, 80: .70, 40: 1.0}, {160: .12, 80: .20, 40: .30}
    )
    assert decision == "SENSITIVITY_PARTLY_PROTOCOL_DEPENDENT"


def test_protocol_decision_C_when_mainly_groupaware_degrades():
    decision, _ = classify_protocol_sensitivity(
        {160: .20, 80: .40, 40: .60}, {160: .02, 80: .05, 40: .08}
    )
    assert decision == "SENSITIVITY_STRONGLY_PROTOCOL_DEPENDENT"
