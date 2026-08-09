"""Audit canonical-v1 claim coverage and freeze the minimum missing runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATASET_HASH = "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6"
STUDY = ROOT / "results/iotj_canonical_v1_final_20260808"
OUTPUT = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809"
DOCS = ROOT / "docs/experiments/iotj_canonical_v1_final"
LEGACY_DATASET = "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def audit_existing_evidence(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    state = _read_json(root / "results/iotj_canonical_v1_final_20260808/FINAL_EXPERIMENT_STATE.json")
    if state["dataset_hash"] != CANONICAL_DATASET_HASH:
        raise RuntimeError("canonical final state dataset hash differs")
    legacy_audit = _read_json(
        root / "results/iotj_final_classification_le1_20260804/post_run_audit.json"
    )
    dataset_finding = next(
        item for item in legacy_audit["findings"] if item["check_id"] == "DATASET-C1-C5-COMPLETE"
    )
    legacy_root = Path(dataset_finding["details"]["data_root"]).name
    if legacy_root != LEGACY_DATASET:
        raise RuntimeError("legacy comparator dataset provenance changed")
    a0t = _read_json(
        root / "results/iotj_canonical_v1_final_20260808/a0t_equal_label/A0T_PRE_RUN_FREEZE.json"
    )
    if a0t["status"] != "FROZEN":
        raise RuntimeError("A0T preregistration is unavailable")
    canonical = {
        "status": "CANONICAL_COMPLETE",
        "dataset_hash": CANONICAL_DATASET_HASH,
        "observed_dataset": "iotj_canonical_v1",
    }
    legacy_missing = {
        "status": "CANONICAL_COMPARATOR_MISSING",
        "dataset_hash": "unknown",
        "observed_dataset": legacy_root,
    }
    return {
        "FedAvg": dict(legacy_missing),
        "FedProx": dict(legacy_missing),
        "SCAFFOLD": dict(legacy_missing),
        "MMD": dict(legacy_missing),
        "A0T": {
            "status": "BLOCKED_NOT_RUN",
            "dataset_hash": CANONICAL_DATASET_HASH,
            "observed_dataset": "iotj_canonical_v1_preregistered_only",
        },
        "GAPS/A4": dict(canonical),
        "routing": dict(canonical),
        "FedRidge_83D_84D": dict(canonical),
        "QC": dict(canonical),
        "Pi5": dict(canonical),
    }


def validate_missing_run_matrix(root: Path = ROOT) -> list[dict[str, Any]]:
    audit = audit_existing_evidence(root)
    if audit["GAPS/A4"]["status"] != "CANONICAL_COMPLETE":
        raise RuntimeError("formal A4 prerequisite is incomplete")
    common = {
        "dataset_hash": CANONICAL_DATASET_HASH,
        "source_clients": "C1;C2",
        "rounds": 25,
        "local_epochs": 1,
        "batch_size": 32,
        "seed": 42,
        "target_test_selection": False,
        "hyperparameter_search": False,
    }
    rows: list[dict[str, Any]] = []
    for method in ("FedAvg", "FedProx", "SCAFFOLD"):
        rows.append({
            **common,
            "experiment_id": f"CAN-V1-CMP-{method.upper()}",
            "method": method,
            "targets": "C3;C4;C5",
            "target_fields": "none",
            "execution": "one_source_FL_then_three_fixed_evaluations",
        })
    for target in ("C3", "C4", "C5"):
        rows.extend([
            {
                **common,
                "experiment_id": f"CAN-V1-CMP-MMD-{target}",
                "method": "MMD",
                "targets": target,
                "target_fields": "calibration_x_only",
                "execution": "fixed_100_step_posthoc_from_canonical_FedAvg_round25",
            },
            {
                **common,
                "experiment_id": f"CANONICAL-V1-A0T-{target}",
                "method": "A0T",
                "targets": target,
                "target_fields": "calibration_x_class",
                "execution": "target_CE_only_100_steps_per_round",
            },
            {
                **common,
                "experiment_id": f"CAN-V1-STRICT-A4-R84-{target}",
                "method": "STRICT_A4_R84",
                "targets": target,
                "target_fields": "strict_calibration_x_class_phase",
                "execution": "strict_split_A4_then_R84_fixed_protocol",
            },
        ])
    ids = [row["experiment_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("missing-run experiment IDs are not unique")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"scientific-validation output already exists: {OUTPUT}")
    for path in (
        DOCS / "FINAL_CLAIM_EVIDENCE_AUDIT.md",
        DOCS / "A0T_REQUIRED_RUN_PLAN.md",
        DOCS / "SCIENTIFIC_VALIDATION_PRE_RUN_FREEZE.json",
        DOCS / "SCIENTIFIC_VALIDATION_EXPERIMENT_MATRIX.csv",
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    audit = audit_existing_evidence(ROOT)
    rows = validate_missing_run_matrix(ROOT)
    OUTPUT.mkdir(parents=True)
    report = f"""# Final claim-evidence audit

## Verdict before new execution

| Claim | Existing canonical evidence | Finding | Severity | Required action |
|---|---|---|---|---|
| Standard FL is insufficient under target shift | A4 only; FedAvg/FedProx/SCAFFOLD are legacy preprocessing | `CANONICAL_COMPARATOR_MISSING` | blocking | Run minimal canonical comparators |
| Unlabeled alignment and labeled commissioning are distinct regimes | No canonical MMD/A0T | `CANONICAL_COMPARATOR_MISSING` | blocking | Run MMD and A0T with explicit information table |
| GAPS benefit is not only target-label access | A0T preregistered, no endpoint | `SUBMISSION_BLOCKER_P0` | blocking | Run frozen equal-label A0T |
| Routing errors propagate to regression | Canonical prediction and S_ALL/S_CC/oracle artifacts exist | reusable | informational | Read-only routing analysis |
| Federated H1 contributes to regression | Matched canonical 83D/84D predictions exist | reusable with uncertainty gap | major | Raw-file-grouped paired bootstrap |
| QC identifies risk beyond reduced coverage | HC90/HC95 and same-budget random exist | reusable with capture/AURC gap | major | Read-only risk/capture analysis |
| Strict non-overlap conclusion holds | Exact identity overlap 0; raw-time overlap about 29% | `SUBMISSION_BLOCKER_P0` | blocking | Separate strict grouped robustness run |
| Edge claims match the deployed package | Package/Pi/model-size hashes exist | reusable | informational | Hash and communication audit |

The historical comparator root uses `{LEGACY_DATASET}` and cannot populate the canonical table. Existing canonical assets remain read-only. Exactly {len(rows)} missing executable configurations are frozen; no other algorithm is authorized.
"""
    (OUTPUT / "FINAL_CLAIM_EVIDENCE_AUDIT.md").write_text(report, encoding="utf-8")
    (DOCS / "FINAL_CLAIM_EVIDENCE_AUDIT.md").write_text(report, encoding="utf-8")
    a0t_plan = """# A0T required run plan

Status: **SUBMISSION_BLOCKER_P0 / REQUIRED**.

Run C3, C4, and C5 independently on canonical-v1 with the same source roles, fresh seed42 initialization, backbone, 25 rounds, local_epochs=1, batch size 32, Adam lr=5e-4, calibration identities, and target-label budget as A4. The only target adaptation loss is supervised target CE for 100 fixed steps per round. MMD, CORAL, DANN, prototype, semantic, stage, consistency, and other A4-specific losses are unavailable/disabled. Checkpoint selection is fixed round25 and target test opens only after all three endpoints complete. No result may trigger A4 tuning.
"""
    (DOCS / "A0T_REQUIRED_RUN_PLAN.md").write_text(a0t_plan, encoding="utf-8")
    _write_csv(DOCS / "SCIENTIFIC_VALIDATION_EXPERIMENT_MATRIX.csv", rows)
    freeze = {
        "schema_version": "iotj.canonical_v1.scientific_validation.freeze.v1",
        "status": "FROZEN_BEFORE_MISSING_RUNS",
        "parent_commit": _head(),
        "freeze_commit": "SELF_GIT_COMMIT_CONTAINING_THIS_FREEZE",
        "canonical_dataset_sha256": CANONICAL_DATASET_HASH,
        "immutable_protocol": {
            "preprocessing": "HZ5_MEAN_W10S", "source": ["C1", "C2"],
            "targets": ["C3", "C4", "C5"], "rounds": 25,
            "local_epochs": 1, "batch_size": 32, "seed": 42,
            "classifier": "A4", "regression": "R84_FED_H1",
            "qc": "frozen_equal_mean",
        },
        "existing_evidence_audit": audit,
        "missing_run_ids": [row["experiment_id"] for row in rows],
        "matrix_sha256": _sha256(DOCS / "SCIENTIFIC_VALIDATION_EXPERIMENT_MATRIX.csv"),
        "forbidden": [
            "target_test_selection", "hyperparameter_search", "new_algorithms",
            "preprocessing_change", "A4_change", "R84_change", "QC_change",
            "outlier_deletion",
        ],
    }
    (DOCS / "SCIENTIFIC_VALIDATION_PRE_RUN_FREEZE.json").write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate()
