from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


UNKNOWN = "unknown"
SCHEMA_VERSION = "gaps.iotj.legacy_evidence_inventory.v1"

COMMON_COLUMNS = [
    "experiment_id",
    "source_clients",
    "target_clients",
    "seed",
    "data_root",
    "split_protocol",
    "calibration_rows",
    "test_rows",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_role",
    "code_commit",
    "adaptation_config",
    "routing_assumption",
    "base_accuracy",
    "adapted_accuracy",
    "accuracy",
    "macro_f1",
    "nll",
    "ece",
    "nrmse_cc",
    "nrmse_all",
    "metric_calculation_status",
    "current_c12_to_c5_compatibility",
    "single_seed_status",
    "historical_semantics",
    "target_test_tuning",
    "classification_regression_routing_consistency",
    "evaluation_replay_status",
    "provenance_status",
    "evidence_tier",
    "source_artifacts",
    "notes",
]


LEGACY_DIRECTIONS = {
    "F1": ("C1", "C5", "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid", 320, 1360),
    "F2": ("C1;C2", "C5", "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid", 320, 1360),
    "F3": ("C1;C2;C3", "C5", "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid", 320, 1360),
    "F4": ("C1;C2;C3;C4", "C5", "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid", 320, 1360),
    "F5": ("C1", "C2;C3;C4;C5", "dataset/client_data_c1src_c2345tgt_2080_timeaware_60_170_window_fullgrid", 2000, 8080),
    "R1": ("C5", "C1", "dataset/client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid", 680, 2680),
    "R2": ("C4;C5", "C1", "dataset/client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid", 680, 2680),
    "R3": ("C3;C4;C5", "C1", "dataset/client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid", 680, 2680),
    "R4": ("C2;C3;C4;C5", "C1", "dataset/client_data_c2345src_c1tgt_2080_timeaware_60_170_window_fullgrid", 680, 2680),
}

CLEAN_RUNS = {
    "F4": "F4_C1234_to_C5_fixed_da_strong_r25",
    "F5": "F5_C1_to_C2345_fixed_da_strong_r25",
    "R1": "R1_C5_to_C1_fixed_da_strong_r25",
    "R2": "R2_C45_to_C1_fixed_da_strong_r25",
    "R3": "R3_C345_to_C1_fixed_da_strong_r25",
    "R4": "R4_C2345_to_C1_fixed_da_strong_r25",
}

OLD_RUNS = {
    "F1": "F1_C1_to_C5_fixed_da_strong_r25",
    "F2": "F2_C12_to_C5_fixed_da_strong_r25",
    "F3": "F3_C123_to_C5_fixed_da_strong_r25",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMMON_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, UNKNOWN) for key in COMMON_COLUMNS})


def sum_confusion_matrices(matrices: Iterable[list[list[int]]]) -> list[list[int]]:
    matrices = list(matrices)
    if not matrices:
        raise ValueError("at least one confusion matrix is required")
    size = len(matrices[0])
    total = [[0 for _ in range(size)] for _ in range(size)]
    for matrix in matrices:
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("confusion matrices must be square and shape-compatible")
        for i in range(size):
            for j in range(size):
                total[i][j] += int(matrix[i][j])
    return total


def macro_f1_from_confusion(matrix: list[list[int]]) -> float:
    size = len(matrix)
    scores: list[float] = []
    for cls in range(size):
        tp = float(matrix[cls][cls])
        fp = float(sum(matrix[row][cls] for row in range(size)) - matrix[cls][cls])
        fn = float(sum(matrix[cls]) - matrix[cls][cls])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def relative_history_path(path: Path, history_root: Path) -> str:
    try:
        return path.relative_to(history_root).as_posix()
    except ValueError:
        return str(path)


def checkpoint_identity(path: Path, history_root: Path) -> tuple[str, str]:
    if not path.is_file():
        return UNKNOWN, UNKNOWN
    return relative_history_path(path, history_root), sha256_file(path)


def adaptation_subset(run_config: dict[str, Any] | None) -> str:
    if not run_config:
        return json_text({"reported_profile": "fixed_da_strong", "detail": UNKNOWN})
    args = run_config.get("args", {})
    keys = [
        "profile",
        "da_preset",
        "domain_adapt_steps",
        "da_server_opt_lr",
        "da_use_coral",
        "da_use_mmd",
        "da_use_adversarial",
        "da_coral_class_conditional",
        "da_lambda_coral",
        "da_lambda_global_mmd",
        "da_lambda_class_mmd",
        "da_lambda_stage_mmd",
        "da_lambda_adv",
        "use_selective_agg",
        "use_proto_mmd",
    ]
    return json_text({key: args.get(key, UNKNOWN) for key in keys})


def classification_base_row(exp_id: str) -> dict[str, Any]:
    source, target, data_root, calibration_rows, test_rows = LEGACY_DIRECTIONS[exp_id]
    same_direction = exp_id == "F2"
    tier = "supplement_only" if exp_id in {"F1", "F2", "F3", "F4"} else "historical_diagnostic_only"
    return {
        "experiment_id": exp_id,
        "source_clients": source,
        "target_clients": target,
        "seed": UNKNOWN,
        "data_root": data_root,
        "split_protocol": "historical time-aware calibrated-target held-out-window 20:80 split; original-file independence not established",
        "calibration_rows": calibration_rows,
        "test_rows": test_rows,
        "checkpoint_role": "adapted_classifier",
        "code_commit": UNKNOWN,
        "routing_assumption": "classification_only; no regression routing metric is implied",
        "nrmse_cc": UNKNOWN,
        "nrmse_all": UNKNOWN,
        "current_c12_to_c5_compatibility": (
            "partial_same_clients_and_named_split_but_legacy_semantics_topology_and_commit"
            if same_direction
            else "incompatible_source_or_target_roles_with_current_C1_C2_to_C5_protocol"
        ),
        "single_seed_status": "unknown_seed; cannot label seed42",
        "historical_semantics": True,
        "target_test_tuning": "not_proven_absent; historical test-visible development context",
        "classification_regression_routing_consistency": "not_auditable_without_direction-bound_regression_predictions",
        "evaluation_replay_status": "classification_metrics_reported_or_confusion_replay_only",
        "evidence_tier": tier,
    }


def build_old_classification_rows(history_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    source_csv = history_root / "results/source_target_classification_matrix_20260630_summary_f1_f4/final_target_metrics.csv"
    records = read_csv(source_csv)
    rows: list[dict[str, Any]] = []
    sources = [source_csv]
    for exp_id, run_name in OLD_RUNS.items():
        matching = [row for row in records if row["run_id"] == run_name]
        by_variant = {row["variant"]: row for row in matching}
        if set(by_variant) != {"base", "adapted"}:
            raise ValueError(f"{run_name}: expected exactly base and adapted rows")
        base = by_variant["base"]
        adapted = by_variant["adapted"]
        row = classification_base_row(exp_id)
        checkpoint = history_root / adapted["checkpoint"]
        checkpoint_path, checkpoint_sha = checkpoint_identity(checkpoint, history_root)
        config_path = history_root / f"results/source_target_classification_matrix_20260630/{run_name}/run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
        if config_path.is_file():
            sources.append(config_path)
        if checkpoint.is_file():
            sources.append(checkpoint)
        row.update(
            {
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
                "adaptation_config": adaptation_subset(config),
                "base_accuracy": base["accuracy"],
                "adapted_accuracy": adapted["accuracy"],
                "accuracy": adapted["accuracy"],
                "macro_f1": adapted["macro_f1"],
                "nll": adapted["nll"],
                "ece": adapted["ece"],
                "metric_calculation_status": "reported",
                "provenance_status": "partial" if checkpoint_sha == UNKNOWN else "checkpoint_hash_bound_but_seed_and_commit_unknown",
                "source_artifacts": ";".join(
                    [relative_history_path(source_csv, history_root)]
                    + ([relative_history_path(config_path, history_root)] if config_path.is_file() else [])
                ),
                "notes": "F1/F3 checkpoints are absent locally when shown as unknown; no value was inferred from the path.",
            }
        )
        rows.append(row)
    return rows, sources


def build_clean_classification_rows(history_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    clean_root = history_root / "results/source_target_classification_matrix_20260708_clean"
    clean_summary = history_root / "results/source_target_classification_matrix_20260708_clean_summary/clean_matrix_final_target_metrics.csv"
    rows: list[dict[str, Any]] = []
    sources: list[Path] = [clean_summary]
    for exp_id, run_name in CLEAN_RUNS.items():
        run_dir = clean_root / run_name
        base_path = run_dir / "target_summary/target_test_base.json"
        adapted_path = run_dir / "target_summary/target_test_adapted.json"
        config_path = run_dir / "run_config.json"
        checkpoint = run_dir / "server_latest_adapted.pth"
        for required in (base_path, adapted_path, config_path, checkpoint):
            if not required.is_file():
                raise FileNotFoundError(required)
        base = json.loads(base_path.read_text(encoding="utf-8"))
        adapted = json.loads(adapted_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        macro_f1 = macro_f1_from_confusion(
            sum_confusion_matrices(client["confusion_matrix"] for client in adapted["clients"])
        )
        checkpoint_path, checkpoint_sha = checkpoint_identity(checkpoint, history_root)
        row = classification_base_row(exp_id)
        provenance = "checkpoint_hash_bound_but_seed_and_commit_unknown"
        notes = "Macro-F1 recomputed from the persisted confusion matrix; Accuracy/NLL/ECE copied from target summary."
        if exp_id == "F4":
            provenance = "conflict_resolved_by_documented_canonical_recovery"
            notes += " Earlier 2026-06-30 F4 metrics differ; this row uses the notebook-designated 2026-07-08 canonical recovery and preserves the conflict in the audit."
        row.update(
            {
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
                "adaptation_config": adaptation_subset(config),
                "base_accuracy": base["weighted_accuracy"],
                "adapted_accuracy": adapted["weighted_accuracy"],
                "accuracy": adapted["weighted_accuracy"],
                "macro_f1": macro_f1,
                "nll": adapted["weighted_nll"],
                "ece": adapted["weighted_ece"],
                "metric_calculation_status": "reported_accuracy_nll_ece; recomputed_macro_f1_from_persisted_confusion_matrix",
                "provenance_status": provenance,
                "source_artifacts": ";".join(relative_history_path(p, history_root) for p in (base_path, adapted_path, config_path)),
                "notes": notes,
            }
        )
        rows.append(row)
        sources.extend([base_path, adapted_path, config_path, checkpoint])
    return rows, sources


def build_component_rows(repo_root: Path, history_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    inventory_path = repo_root / "results/iotj_minimal_gap_audit_20260726/component_ablation_inventory.csv"
    metrics_path = repo_root / "results/iotj_classification_ablation_20260712_v3_summary/classification_per_run.csv"
    inventory = {row["group_id"]: row for row in read_csv(inventory_path)}
    metrics = {row["group_id"]: row for row in read_csv(metrics_path)}
    configs = {
        "B1": {"semantic_core": True, "CORAL": 0.5, "MMD2": False, "stage_alignment": False, "adversarial_wasserstein_min": False},
        "B2": {"semantic_core": True, "CORAL": False, "global_MMD2": 0.5, "class_MMD2": 0.5, "stage_alignment": False, "adversarial_wasserstein_min": False},
        "B3": {"semantic_core": True, "CORAL": False, "MMD2": False, "stage_MMD2": 0.2, "stage_semantics": "cross_domain_same_class_phase", "adversarial_wasserstein_min": False},
        "B4": {"semantic_core": True, "CORAL": False, "MMD2": False, "stage_alignment": False, "adversarial_wasserstein_min": 0.5},
        "B5": {"semantic_core": True, "CORAL": 0.5, "global_MMD2": 0.5, "class_MMD2": 0.5, "stage_MMD2": 0.2, "stage_semantics": "cross_domain_same_class_phase", "adversarial_wasserstein_min": 0.5},
    }
    rows: list[dict[str, Any]] = []
    sources: list[Path] = [inventory_path, metrics_path]
    for group in ("B1", "B2", "B3", "B4", "B5"):
        inv = inventory[group]
        metric = metrics[group]
        checkpoint_path = history_root / inv["checkpoint_path"]
        manifest_path = history_root / inv["manifest_path"]
        for path, expected_sha in (
            (checkpoint_path, inv["checkpoint_sha256"]),
            (manifest_path, inv["manifest_sha256"]),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                raise ValueError(f"SHA256 mismatch for {path}: expected {expected_sha}, got {actual_sha}")
            sources.append(path)
        rows.append(
            {
                "experiment_id": group,
                "source_clients": inv["source_clients"].replace(",", ";"),
                "target_clients": inv["target_client"],
                "seed": inv["seed"],
                "data_root": inv["dataset_path"],
                "split_protocol": "calibrated-target held-out-window evaluation; historical screening",
                "calibration_rows": 320,
                "test_rows": inv["N"],
                "checkpoint": inv["checkpoint_path"],
                "checkpoint_sha256": inv["checkpoint_sha256"],
                "checkpoint_role": "adapted_classifier",
                "code_commit": inv["code_commit"],
                "adaptation_config": json_text(configs[group]),
                "routing_assumption": "classification_only; no regression routing metric is implied",
                "base_accuracy": UNKNOWN,
                "adapted_accuracy": metric["accuracy"],
                "accuracy": metric["accuracy"],
                "macro_f1": metric["macro_f1"],
                "nll": metric["nll"],
                "ece": metric["ece"],
                "nrmse_cc": UNKNOWN,
                "nrmse_all": UNKNOWN,
                "metric_calculation_status": "reported",
                "current_c12_to_c5_compatibility": "partial_same_named_data_and_window_split_but_legacy_PC_C2_topology_old_commit_and_test_visible_screening",
                "single_seed_status": "seed42_single_seed",
                "historical_semantics": True,
                "target_test_tuning": "test_visible_screening; no test-fit documented, but results cannot support untouched-test selection claims",
                "classification_regression_routing_consistency": "not_applicable_classification_component_screen",
                "evaluation_replay_status": "not_requested_for_classification_component_metrics",
                "provenance_status": "hash_bound_legacy_screening",
                "evidence_tier": "supplement_only",
                "source_artifacts": f"{inv['manifest_path']};{inv['result_path']}",
                "notes": inv["compatibility_reason"],
            }
        )
    return rows, sources


def build_regression_rows(classification_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classifier_by_id = {row["experiment_id"]: row for row in classification_rows}
    rows: list[dict[str, Any]] = []
    for exp_id in ("F4", "F5", "R1", "R2", "R3", "R4"):
        cls = classifier_by_id[exp_id]
        rows.append(
            {
                "experiment_id": f"{exp_id}-REG",
                "source_clients": cls["source_clients"],
                "target_clients": cls["target_clients"],
                "seed": UNKNOWN,
                "data_root": cls["data_root"],
                "split_protocol": cls["split_protocol"],
                "calibration_rows": cls["calibration_rows"],
                "test_rows": cls["test_rows"],
                "checkpoint": cls["checkpoint"],
                "checkpoint_sha256": cls["checkpoint_sha256"],
                "checkpoint_role": "classification_route_only; regression_checkpoint_unknown",
                "code_commit": UNKNOWN,
                "adaptation_config": "regression model/config not bound by a direction-specific manifest",
                "routing_assumption": "unknown; no direction-bound persisted regression prediction stream was located",
                "base_accuracy": UNKNOWN,
                "adapted_accuracy": cls["adapted_accuracy"],
                "accuracy": cls["accuracy"],
                "macro_f1": cls["macro_f1"],
                "nll": cls["nll"],
                "ece": cls["ece"],
                "nrmse_cc": UNKNOWN,
                "nrmse_all": UNKNOWN,
                "metric_calculation_status": "classification_context_only; regression_metrics_unavailable",
                "current_c12_to_c5_compatibility": cls["current_c12_to_c5_compatibility"],
                "single_seed_status": "unknown_seed; cannot label seed42",
                "historical_semantics": True,
                "target_test_tuning": "unknown_for_regression; historical development context",
                "classification_regression_routing_consistency": "not_auditable",
                "evaluation_replay_status": "NOT_RUN_NO_FROZEN_PREDICTION_ASSET",
                "provenance_status": "missing_direction_bound_regression_metric_provenance",
                "evidence_tier": "historical_diagnostic_only",
                "source_artifacts": cls["source_artifacts"],
                "notes": "No NRMSE value was copied from another source/target direction. Existing C1/C2→C5 formal oracle-route evidence and broad C4/C5→C1/C2/C3 legacy regression evidence are not substitutes for this row.",
            }
        )
    return rows


def markdown_table_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        tier: sum(1 for row in rows if row["evidence_tier"] == tier)
        for tier in ("main_paper_ready", "supplement_only", "historical_diagnostic_only")
    }


def build_inventory_md(classification: list[dict[str, Any]], components: list[dict[str, Any]], regression: list[dict[str, Any]]) -> str:
    all_rows = classification + components + regression
    counts = markdown_table_counts(all_rows)
    return f"""# IoT-J Legacy Evidence Inventory

- Schema: `{SCHEMA_VERSION}`
- Audit mode: read-only; no training, checkpoint inference, model fitting, or benchmark.
- Scope: F1–F5/R1–R4 historical classification matrix, B1–B5 corrected server-adaptation screen, and requested F4/F5/R1–R4 regression provenance.
- Evidence tiers: `main_paper_ready={counts['main_paper_ready']}`, `supplement_only={counts['supplement_only']}`, `historical_diagnostic_only={counts['historical_diagnostic_only']}`.

## Inventory boundary

The complete legacy matrix assets are local historical files outside the current worktree's tracked result set. The CSV manifests preserve that storage boundary. An existing path is not treated as a portable or paper-ready artifact unless seed, commit, checkpoint, split, and metric provenance are bound.

## Classification matrix

- F1–F3 use the 2026-06-30 final target summary.
- F4/F5/R1–R4 use the 2026-07-08 canonical full-name recovery recorded in the experiment notebook.
- F4 has two traceable result sources with different values. The inventory uses the notebook-designated recovery and records the earlier value as a conflict; it does not average or silently replace either source.
- Clean-matrix Macro-F1 is recomputed only from persisted confusion matrices. Accuracy, NLL, and ECE remain copied reported values.

## B1–B5 server adaptation

B1 is CORAL, B2 is conventional global/class MMD², B3 is cross-domain same-class/same-phase stage MMD², B4 is corrected Wasserstein-min adversarial alignment, and B5 is their predeclared combination on the shared semantic core. All five are seed-42 historical screens using the older Windows-PC C2 topology and test-visible development context. They are supplementary mechanism evidence, not final-B5 five-seed causal evidence.

## Cross-direction regression

No direction-bound frozen row-level regression prediction stream was found for F4/F5/R1–R4 in the canonical matrix roots or the tracked evidence indexes. Therefore `NRMSE_CC` and `NRMSE_ALL` remain `unknown`, and evaluation replay is explicitly `NOT_RUN_NO_FROZEN_PREDICTION_ASSET`. C1/C2→C5 formal R4/oracle metrics and the broad C4/C5→C1/C2/C3 legacy pipeline were not substituted because they use different source/target and routing identities.

## Machine-readable files

- `classification_cross_direction_summary.csv`
- `server_adaptation_component_summary.csv`
- `regression_cross_direction_summary.csv`
- `EXPERIMENT_AUDIT.md`
- `sha256_index.json`
"""


def build_audit_md(classification: list[dict[str, Any]], components: list[dict[str, Any]], regression: list[dict[str, Any]]) -> str:
    return f"""# IoT-J Legacy Experiment Audit

## Verdict

No requested historical row qualifies as `main_paper_ready`. `{sum(r['evidence_tier'] == 'supplement_only' for r in classification + components)}` classification/component rows are `supplement_only`; all `{len(regression)}` requested cross-direction regression rows are `historical_diagnostic_only` because their metric-producing assets are not direction-bound.

## Blocking findings

1. **Cross-direction regression replay unavailable.** F4/F5/R1–R4 have classification checkpoints and summaries, but no verified row-level regression truth/prediction pair with an explicit route schema. Recomputing NRMSE would require guessing an unrelated pipeline identity, so replay was not run.
2. **Seed and commit are unbound for the legacy F1–F5/R1–R4 matrix.** These rows cannot be called seed42 and cannot support across-seed stability.

## Major findings

1. **Current protocol compatibility is limited.** Only historical F2 shares the C1/C2→C5 role labels and named 320/1360 window split. It still differs from the final canonical ECS-C2 topology, code identity, and corrected/frozen method evidence. F1/F3/F4 change source sets; F5 and R1–R4 change target scope or direction.
2. **Historical semantics.** The old fixed-da-strong matrix is not final B5. B1–B5 are corrected mechanism screens, but are single-seed, older-topology and test-visible; the historical B5 name must not be equated with the final five-seed B5 evidence.
3. **F4 provenance conflict.** The 2026-06-30 F4 summary and 2026-07-08 canonical recovery report different target metrics. The inventory follows the documented recovery for the active row and preserves the conflict in `notes`/`provenance_status`.
4. **Target-test boundary.** Historical matrix and B1–B5 results were visible during development. No per-run fitting to test is documented, but absence of test-driven method screening is not established. They cannot support untouched-test or prospective-confirmatory wording.

## Routing audit

- Classification rows have no regression route assumption.
- Requested cross-direction regression rows lack persisted prediction columns and route identity; classification-vs-regression routing consistency is therefore `not_auditable`.
- Existing formal C1/C2→C5 oracle-route records are a different experiment family and were intentionally excluded.

## Evidence-tier decision

- `main_paper_ready`: none.
- `supplement_only`: F1–F4 classification context and B1–B5 historical component screening.
- `historical_diagnostic_only`: F5/R1–R4 classification context and all requested cross-direction regression placeholders with unknown metrics.

## Integrity boundary

The audit reads historical summaries, run configs, confusion matrices and checkpoint bytes only for identity hashing. It does not load checkpoints, execute inference, train models, open formal C5 test arrays, or modify frozen assets.
"""


def generate(repo_root: Path, history_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    old_rows, old_sources = build_old_classification_rows(history_root)
    clean_rows, clean_sources = build_clean_classification_rows(history_root)
    classification = old_rows + clean_rows
    components, component_sources = build_component_rows(repo_root, history_root)
    regression = build_regression_rows(classification)

    classification_path = output_dir / "classification_cross_direction_summary.csv"
    component_path = output_dir / "server_adaptation_component_summary.csv"
    regression_path = output_dir / "regression_cross_direction_summary.csv"
    inventory_path = output_dir / "LEGACY_EVIDENCE_INVENTORY.md"
    audit_path = output_dir / "EXPERIMENT_AUDIT.md"
    index_path = output_dir / "sha256_index.json"

    write_csv(classification_path, classification)
    write_csv(component_path, components)
    write_csv(regression_path, regression)
    inventory_path.write_text(build_inventory_md(classification, components, regression), encoding="utf-8")
    audit_path.write_text(build_audit_md(classification, components, regression), encoding="utf-8")

    generated = [inventory_path, classification_path, component_path, regression_path, audit_path]
    source_paths = sorted({p.resolve() for p in old_sources + clean_sources + component_sources}, key=str)
    index = {
        "schema_version": "gaps.iotj.legacy_evidence.sha256_index.v1",
        "inventory_schema": SCHEMA_VERSION,
        "audit_mode": "read_only_no_training_no_checkpoint_inference",
        "history_root": str(history_root.resolve()),
        "repo_root": str(repo_root.resolve()),
        "generated_artifacts": [
            {"path": path.relative_to(repo_root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in generated
        ],
        "source_artifacts": [
            {"path": relative_history_path(path, history_root), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "counts": {
            "classification_rows": len(classification),
            "component_rows": len(components),
            "regression_rows": len(regression),
            "recomputed_regression_rows": 0,
            "unknown_regression_nrmse_rows": len(regression),
        },
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only IoT-J legacy evidence inventory")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index = generate(args.repo_root.resolve(), args.history_root.resolve(), args.output_dir.resolve())
    print(json.dumps(index["counts"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
