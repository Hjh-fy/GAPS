from __future__ import annotations

import pytest

from scripts.lab_three_gas_3class.summarize_a4_crossboard_results import (
    build_code_identity,
    summarize_experiment,
)


def _metric(correct: int, total: int) -> dict:
    incorrect = total - correct
    return {
        "accuracy": correct / total,
        "macro_f1": correct / total,
        "confusion_matrix": [[correct, incorrect, 0], [0, 0, 0], [0, 0, 0]],
        "n_samples": total,
    }


def _scope_payload() -> dict:
    return {
        "checkpoint": "/runs/server_round_025_adapted.pth",
        "target_client": 3,
        "scopes": {
            "stable360": {
                "global": {
                    "window": _metric(3, 4),
                    "exposure": {
                        **_metric(1, 1),
                        "n_exposures": 1,
                    },
                }
            },
            "early60": {
                "global": {
                    "window": _metric(1, 2),
                    "exposure": {
                        **_metric(0, 1),
                        "n_exposures": 1,
                    },
                }
            },
            "full420": {
                "global": {
                    "window": _metric(4, 6),
                    "exposure": {
                        **_metric(1, 1),
                        "n_exposures": 1,
                    },
                }
            },
        },
    }


def _audit_payload() -> dict:
    return {
        "status": "valid",
        "direction": "P1_to_P3",
        "source_clients": [1],
        "target_client": 3,
        "selected_round": 25,
        "rounds": 25,
        "local_epochs": 3,
        "model_profile": "proto_replay",
        "domain_adaptation_mode": "corrected_b2",
        "target_ce_weight": 0.0,
        "selection_policy": "last_round",
        "metrics": {
            "unadapted": {"target_test_window_accuracy": 0.5},
            "adapted": {"target_test_window_accuracy": 0.75},
        },
    }


def test_summary_preserves_scope_counts_and_fixed_checkpoint() -> None:
    summary = summarize_experiment(
        experiment_id="A4-XB-E2-P1P3-S42",
        scope_payload=_scope_payload(),
        audit_payload=_audit_payload(),
        scope_source="scope.json",
        audit_source="audit.json",
    )

    assert summary["selected_round"] == 25
    assert summary["scopes"]["stable360"]["window_correct"] == 3
    assert summary["scopes"]["early60"]["window_correct"] == 1
    assert summary["scopes"]["full420"]["window_total"] == 6
    assert summary["formal_stable"]["adapted_accuracy"] == 0.75


def test_summary_rejects_nonvalid_postflight_audit() -> None:
    audit = _audit_payload()
    audit["status"] = "invalid"

    with pytest.raises(ValueError, match="postflight audit is not valid"):
        summarize_experiment(
            experiment_id="A4-XB-E2-P1P3-S42",
            scope_payload=_scope_payload(),
            audit_payload=audit,
            scope_source="scope.json",
            audit_source="audit.json",
        )


def test_code_identity_includes_posthoc_evaluators(tmp_path) -> None:
    scope_evaluator = tmp_path / "scope.py"
    checkpoint_evaluator = tmp_path / "checkpoint.py"
    scope_evaluator.write_bytes(b"scope\n")
    checkpoint_evaluator.write_bytes(b"checkpoint\n")

    identity = build_code_identity(
        {"source_archive_sha256": "archive-sha"},
        scope_evaluator,
        checkpoint_evaluator,
    )

    assert identity == {
        "training_source_archive_sha256": "archive-sha",
        "posthoc_scope_evaluator_sha256": (
            "d8233a294e3028f552d2219dafbd9f417ccc039567d14e45ceecce60262d87ff"
        ),
        "checkpoint_evaluator_sha256": (
            "74c24dae6de5def220b3b9c31540dbb31934b9e5b6dbd37427ee1f21abe7e7e6"
        ),
    }
