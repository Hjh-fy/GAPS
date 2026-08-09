from scripts.analyze_iotj_c5_label_budget import (
    build_comparison,
    scientific_decision,
)


def metric(method: str, budget: int, macro_f1: float, source_f1: float) -> dict:
    return {
        "method": method,
        "budget_pct": budget,
        "macro_f1": macro_f1,
        "accuracy": macro_f1,
        "nll": 0.1,
        "ece": 0.01,
        "source_macro_f1": source_f1,
    }


def test_build_comparison_reuses_20pct_and_calculates_ordered_budget_deltas() -> None:
    existing = [
        metric("A0T", 20, 0.99, 0.98),
        metric("A4", 20, 0.99, 0.97),
    ]
    new = [
        metric("A0T", 15, 0.97, 0.98), metric("A4", 15, 0.98, 0.97),
        metric("A0T", 10, 0.94, 0.96), metric("A4", 10, 0.97, 0.97),
        metric("A0T", 5, 0.80, 0.90), metric("A4", 5, 0.96, 0.95),
    ]

    rows = build_comparison(existing, new)

    assert [row["budget_pct"] for row in rows] == [20, 15, 10, 5]
    assert rows[0]["a4_minus_a0t_macro_f1"] == 0.0
    assert rows[2]["a0t_delta_vs_20pct"] == -0.05
    assert rows[2]["a4_delta_vs_20pct"] == -0.02
    assert rows[3]["a4_minus_a0t_macro_f1"] == 0.16
    assert rows[3]["a0t_source_macro_f1"] == 0.90
    assert rows[3]["a4_source_macro_f1"] == 0.95


def test_scientific_decision_requires_practical_low_budget_gap_for_support() -> None:
    negligible = [
        {"budget_pct": 20, "a4_minus_a0t_macro_f1": 0.0},
        {"budget_pct": 15, "a4_minus_a0t_macro_f1": 0.003},
        {"budget_pct": 10, "a4_minus_a0t_macro_f1": 0.006},
        {"budget_pct": 5, "a4_minus_a0t_macro_f1": 0.009},
    ]
    supported = [*negligible[:-1], {"budget_pct": 5, "a4_minus_a0t_macro_f1": 0.01}]
    assert scientific_decision(negligible) == {
        "conclusion": "LABEL_EFFICIENCY_NOT_SUPPORTED",
        "multi_seed_recommendation": "NO",
    }
    assert scientific_decision(supported) == {
        "conclusion": "LABEL_EFFICIENCY_SUPPORTED",
        "multi_seed_recommendation": "YES_PROPOSAL_ONLY",
    }
