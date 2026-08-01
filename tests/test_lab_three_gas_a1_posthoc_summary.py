from __future__ import annotations

import pytest

from scripts.lab_three_gas_3class.summarize_a1_posthoc_results import (
    build_summary,
    summarize_experiment,
    write_artifacts,
)


def _metric(correct: int, total: int) -> dict:
    return {
        "accuracy": correct / total,
        "macro_f1": correct / total,
        "confusion_matrix": [
            [correct, total - correct, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        "n_samples": total,
    }


def _scope_payload() -> dict:
    return {
        "checkpoint": "/runs/server_round_025_adapted.pth",
        "target_client": 3,
        "scopes": {
            "early60": {
                "global": {
                    "window": _metric(1, 2),
                    "exposure": {**_metric(1, 1), "n_exposures": 1},
                }
            },
            "stable360": {
                "global": {
                    "window": _metric(3, 4),
                    "exposure": {**_metric(1, 1), "n_exposures": 1},
                }
            },
            "full420": {
                "global": {
                    "window": _metric(4, 6),
                    "exposure": {**_metric(1, 1), "n_exposures": 1},
                }
            },
        },
    }


def _audit(primary_accuracy: float) -> dict:
    return {
        "status": "valid",
        "direction": "P2_to_P3",
        "source_clients": [2],
        "target_client": 3,
        "rounds": 25,
        "local_epochs": 1,
        "da_steps_per_round": 100,
        "model_profile": "proto_replay",
        "domain_adaptation_mode": "corrected_b2",
        "target_ce_weight": 0.0,
        "selection_policy": "last_round",
        "selected_round": 25,
        "metrics": {
            "unadapted": {"target_test_window_accuracy": 0.5},
            "adapted": {"target_test_window_accuracy": primary_accuracy},
        },
    }


def test_a1_uses_full420_as_formal_primary_scope() -> None:
    result = summarize_experiment(
        experiment_id="A1-FULL-E1-P2P3-S42",
        protocol="A1",
        primary_scope="full420",
        scope_payload=_scope_payload(),
        audit_payload=_audit(4 / 6),
        scope_source="scope.json",
        audit_source="audit.json",
    )

    assert result["primary_scope"] == "full420"
    assert result["scopes"]["early60"]["window_correct"] == 1
    assert result["scopes"]["stable360"]["window_correct"] == 3
    assert result["scopes"]["full420"]["window_correct"] == 4


def test_a4_uses_stable360_as_formal_primary_scope() -> None:
    result = summarize_experiment(
        experiment_id="A4-CTRL-E2-P2P3-LE1-S42",
        protocol="A4",
        primary_scope="stable360",
        scope_payload=_scope_payload(),
        audit_payload=_audit(3 / 4),
        scope_source="scope.json",
        audit_source="audit.json",
    )

    assert result["primary_scope"] == "stable360"
    assert result["formal_primary"]["adapted_accuracy"] == 3 / 4


def test_summary_rejects_audit_mismatch_for_declared_primary_scope() -> None:
    with pytest.raises(ValueError, match="primary metric/audit mismatch"):
        summarize_experiment(
            experiment_id="A1-FULL-E1-P2P3-S42",
            protocol="A1",
            primary_scope="full420",
            scope_payload=_scope_payload(),
            audit_payload=_audit(3 / 4),
            scope_source="scope.json",
            audit_source="audit.json",
        )


def test_build_summary_and_write_artifacts_preserve_single_seed_limits(
    tmp_path,
) -> None:
    posthoc_root = tmp_path / "posthoc"
    controller_root = tmp_path / "controller"
    run_dir = "run_a1"
    (posthoc_root / run_dir / "evaluation").mkdir(parents=True)
    (controller_root / run_dir).mkdir(parents=True)
    scope_payload = _scope_payload()
    scope_payload["checkpoint"] = (
        f"/results/{run_dir}/server_round_025_adapted.pth"
    )
    (posthoc_root / run_dir / "evaluation" / "summary.json").write_text(
        __import__("json").dumps(scope_payload), encoding="utf-8"
    )
    (controller_root / run_dir / "postflight_attempt_audit.json").write_text(
        __import__("json").dumps(_audit(4 / 6)), encoding="utf-8"
    )
    manifest = {
        "schema_version": "gaps.lab_three_gas.a1_posthoc_manifest.v1",
        "source_archive_sha256": "a" * 64,
        "seed_set": [42],
        "experiments": [
            {
                "experiment_id": "A1-FULL-E1-P2P3-S42",
                "protocol": "A1",
                "primary_scope": "full420",
                "run_dir": run_dir,
                "dataset": "dataset_v1",
                "split_protocol": "a1_full_crossboard_p2_to_p3_v1",
                "config_path": "dataset_v1/fold_1/fold_config.json",
                "dataset_path": "dataset_v1/fold_1",
            }
        ],
    }

    summary, rows, registry = build_summary(
        manifest=manifest,
        posthoc_root=posthoc_root,
        controller_root=controller_root,
    )

    assert summary["limitations"] == [
        "single_seed_descriptive_only",
        "overlapping_windows_within_exposure",
        "nominal_gas_boundaries",
        "all_retained_concentrations_in_target_calibration",
        "post_hoc_time_scope_diagnostics",
    ]
    assert len(rows) == 3
    assert {row["calculation_status"] for row in rows} == {"recomputed"}
    assert registry[0]["status"] == "audited"

    output_dir = tmp_path / "formal"
    write_artifacts(
        output_dir=output_dir,
        summary=summary,
        metric_rows=rows,
        registry_rows=registry,
    )
    assert (output_dir / "combined_summary.json").is_file()
    assert (output_dir / "combined_metrics.csv").is_file()
    assert (output_dir / "experiment_registry.csv").is_file()
    assert (output_dir / "RESULT_ANALYSIS.md").is_file()
    assert (output_dir / "EXPERIMENT_AUDIT.md").is_file()
    assert (output_dir / "FORMAL_REPORT.zh.md").is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_artifacts(
            output_dir=output_dir,
            summary=summary,
            metric_rows=rows,
            registry_rows=registry,
        )
