"""Combine audited A4 cross-board stable, early, and full-scope results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCOPE_ORDER = ("stable360", "early60", "full420")
EXPERIMENT_ARGS = (
    ("A4-XB-E0-FULL420", "e0"),
    ("A4-XB-E1-P2P1-S42", "p2p1"),
    ("A4-XB-E2-P1P3-S42", "p1p3"),
    ("A4-XB-E3-P12P3-S42", "p12p3"),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_code_identity(
    source_manifest: dict[str, Any],
    scope_evaluator: Path,
    checkpoint_evaluator: Path,
) -> dict[str, str]:
    return {
        "training_source_archive_sha256": source_manifest[
            "source_archive_sha256"
        ],
        "posthoc_scope_evaluator_sha256": sha256_file(scope_evaluator),
        "checkpoint_evaluator_sha256": sha256_file(checkpoint_evaluator),
    }


def correct_count(confusion_matrix: list[list[int]]) -> int:
    return sum(
        int(confusion_matrix[index][index])
        for index in range(len(confusion_matrix))
    )


def summarize_experiment(
    *,
    experiment_id: str,
    scope_payload: dict[str, Any],
    audit_payload: dict[str, Any],
    scope_source: str,
    audit_source: str,
) -> dict[str, Any]:
    if audit_payload.get("status") != "valid":
        raise ValueError(f"{experiment_id}: postflight audit is not valid")
    if int(audit_payload.get("selected_round", -1)) != 25:
        raise ValueError(f"{experiment_id}: selected round must be 25")
    if int(scope_payload.get("target_client", -1)) != int(
        audit_payload["target_client"]
    ):
        raise ValueError(f"{experiment_id}: target client mismatch")

    scopes: dict[str, Any] = {}
    for scope_name in SCOPE_ORDER:
        result = scope_payload["scopes"][scope_name]["global"]
        window = result["window"]
        exposure = result["exposure"]
        scopes[scope_name] = {
            "window_correct": correct_count(window["confusion_matrix"]),
            "window_total": int(window["n_samples"]),
            "window_accuracy": float(window["accuracy"]),
            "window_macro_f1": float(window["macro_f1"]),
            "window_confusion_matrix": window["confusion_matrix"],
            "exposure_correct": correct_count(exposure["confusion_matrix"]),
            "exposure_total": int(exposure["n_exposures"]),
            "exposure_accuracy": float(exposure["accuracy"]),
            "exposure_macro_f1": float(exposure["macro_f1"]),
            "exposure_confusion_matrix": exposure["confusion_matrix"],
            "calculation_status": "reported",
        }

    adapted_accuracy = float(
        audit_payload["metrics"]["adapted"]["target_test_window_accuracy"]
    )
    if abs(scopes["stable360"]["window_accuracy"] - adapted_accuracy) > 1e-12:
        raise ValueError(f"{experiment_id}: stable metric/audit mismatch")

    return {
        "experiment_id": experiment_id,
        "direction": audit_payload["direction"],
        "source_clients": audit_payload["source_clients"],
        "target_client": int(audit_payload["target_client"]),
        "seed": 42,
        "rounds": int(audit_payload["rounds"]),
        "local_epochs": int(audit_payload["local_epochs"]),
        "model_profile": audit_payload["model_profile"],
        "domain_adaptation_mode": audit_payload["domain_adaptation_mode"],
        "target_ce_weight": float(audit_payload["target_ce_weight"]),
        "selection_policy": audit_payload["selection_policy"],
        "selected_round": int(audit_payload["selected_round"]),
        "checkpoint": scope_payload["checkpoint"],
        "formal_stable": {
            "unadapted_accuracy": float(
                audit_payload["metrics"]["unadapted"][
                    "target_test_window_accuracy"
                ]
            ),
            "adapted_accuracy": adapted_accuracy,
        },
        "scopes": scopes,
        "provenance": {
            "scope_summary": scope_source,
            "postflight_audit": audit_source,
        },
        "status": "audited",
    }


def metric_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for experiment in experiments:
        for scope_name in SCOPE_ORDER:
            scope = experiment["scopes"][scope_name]
            rows.append(
                {
                    "experiment_id": experiment["experiment_id"],
                    "direction": experiment["direction"],
                    "source_clients": ";".join(
                        str(item) for item in experiment["source_clients"]
                    ),
                    "target_client": experiment["target_client"],
                    "seed": experiment["seed"],
                    "checkpoint_round": experiment["selected_round"],
                    "scope": scope_name,
                    "window_correct": scope["window_correct"],
                    "window_total": scope["window_total"],
                    "window_accuracy": scope["window_accuracy"],
                    "window_macro_f1": scope["window_macro_f1"],
                    "exposure_correct": scope["exposure_correct"],
                    "exposure_total": scope["exposure_total"],
                    "exposure_accuracy": scope["exposure_accuracy"],
                    "exposure_macro_f1": scope["exposure_macro_f1"],
                    "calculation_status": scope["calculation_status"],
                    "source_path": experiment["provenance"]["scope_summary"],
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for _experiment_id, prefix in EXPERIMENT_ARGS:
        parser.add_argument(f"--{prefix}-scope", type=Path, required=True)
        parser.add_argument(f"--{prefix}-audit", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--scope-evaluator", type=Path, required=True)
    parser.add_argument("--checkpoint-evaluator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")

    experiments = []
    for experiment_id, prefix in EXPERIMENT_ARGS:
        scope_path: Path = getattr(args, f"{prefix}_scope")
        audit_path: Path = getattr(args, f"{prefix}_audit")
        experiments.append(
            summarize_experiment(
                experiment_id=experiment_id,
                scope_payload=load_json(scope_path),
                audit_payload=load_json(audit_path),
                scope_source=scope_path.as_posix(),
                audit_source=audit_path.as_posix(),
            )
        )

    source_manifest = load_json(args.source_manifest)
    code_identity = build_code_identity(
        source_manifest,
        args.scope_evaluator,
        args.checkpoint_evaluator,
    )
    payload = {
        "schema_version": "gaps.lab_three_gas.a4_crossboard_summary.v1",
        "protocol": {
            "task": "three_gas_classification",
            "input_shape": [100, 6],
            "selected_channels": [1, 2, 4, 6, 8, 9],
            "seed_set": [42],
            "rounds": 25,
            "local_epochs": 3,
            "checkpoint_policy": "fixed_round_25",
            "code_identity": code_identity,
        },
        "experiments": experiments,
        "limitations": [
            "single_seed_descriptive_only",
            "overlapping_windows_within_exposure",
            "nominal_gas_boundaries",
            "all_retained_concentrations_in_target_calibration",
            "early_and_full_scopes_are_post_hoc_diagnostics",
            "p12_has_no_matched_budget_control",
        ],
    }
    rows = metric_rows(experiments)

    output_dir.mkdir(parents=True)
    (output_dir / "combined_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "combined_metrics.csv").open(
        "x", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
