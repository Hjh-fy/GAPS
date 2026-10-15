"""Fail-closed pre-run audit for the frozen IoT-J v4 final evidence scope.

Dry-run is the only mode used during Task 2.  It reads immutable evidence,
prints one machine-readable verdict, and performs no writes, test release, or
training.  The explicit ``--write`` mode exists for the later authorized
formal audit and creates a new result root exclusively.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/iotj_v4_final_evidence_20260917"
FROZEN_SCOPE_COMMIT = "5f554cdfa05af272a4d496c7063f317174bc0d81"

POOLED_RULE = (
    "concatenate C3/C4/C5 predictions, then compute one Accuracy and one global Macro-F1"
)
INFORMATION_NOTE = (
    "Cross-regime numbers are reported for contextual comparison and must not "
    "be interpreted as an equal-supervision ranking."
)
GROUPING_KEY = "C5|filename"

EXPECTED_COUNTS = {20: 320, 15: 240, 10: 160, 5: 80}
EXPECTED_POPULATION = {"C3": 2677, "C4": 1360, "C5": 1360}
EXPECTED_83D_SCHEMA_SHA256 = (
    "4cb9e621b39cf726b18d0102d2ec395ba11b109b6ffcabb381c729dd44f26248"
)
EXPECTED_104D_SCHEMA_SHA256 = (
    "df696d3cfbe43eff40b515f6f1a7bb51c9cd11900dba93e231a3ded0755c3259"
)
EXPECTED_SCHEMA_BUNDLE_SHA256 = (
    "059a6c87449a55aff503dee5986a4d68356eb65606a7dfb7cb71b47e3de53753"
)
EXPECTED_R0_INDEX_SHA256 = (
    "04e62b0a2b363fcb79cbeb60ad6c2c6c9a3dedef380b2b0f7c2aa808732d4d82"
)
EXPECTED_R0_MODEL_LOCK_SHA256 = (
    "40c06848f19d211920b200946328e6e95cae9656a17ae0d18036951b7c5d67f6"
)
EXPECTED_C5_R1_TEST_IDENTITY_SHA256 = (
    "424f361de4dbafd440af6cbf66ad2eddc67c2362be4f6d99fe3eadf96ff6d16f"
)
EXPECTED_ALPHA_RULE = {
    "grid": [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    "folds": 5,
    "grouping": "raw_filename",
    "objective": "pooled RMSE",
    "tie_break": "first-in-grid",
    "selection_data": "calibration-only",
}

CLASSIFICATION_TABLE = Path(
    "docs/experiments/iotj_canonical_v1_final/canonical_classification_comparison.csv"
)
CLASSIFICATION_MANIFEST = Path(
    "docs/experiments/iotj_canonical_v1_final/canonical_comparator_evaluation_manifest.json"
)
COMPARATOR_ROOT = Path("results/iotj_final_classification_le1_20260804")
BUDGET_ROOT = Path("results/iotj_canonical_v1_c5_budget_20260810")
DATASET_C5 = Path("dataset/iotj_canonical_v1/client_5")
R0_ROOT = Path(
    "results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812"
)
R0_PROTOCOL = Path(
    "docs/experiments/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/protocol_manifest.json"
)
R1_ROOT = Path(
    "results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812"
)
R1_PROTOCOL = Path(
    "docs/experiments/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812/protocol_manifest.json"
)
Q1_ROOT = Path(
    "results/iotj_canonical_v1_final/canonical_q1_conformal_qc_v2_20260812"
)
CURRENT_QC_ROOT = Path(
    "results/iotj_canonical_v1_final/current_authoritative_summary_20260812"
)
LEGACY_QC = Path("docs/experiments/iotj_canonical_v1_final/05_qc_final.csv")
LEGACY_RANDOM = Path(
    "results/iotj_v4_evidence_completion_20260912/qc_random_retention_summary.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_pass(value: Any) -> bool:
    if type(value) is bool:
        return value
    if isinstance(value, Mapping):
        return value.get("status") == "PASS"
    return value == "PASS"


def final_verdict(checks: Mapping[str, Any]) -> dict[str, str]:
    """Reduce critical checks without spreading one component's uncertainty."""
    if not isinstance(checks, Mapping) or not checks:
        raise ValueError("checks must be a non-empty mapping")
    component_for_field = {
        "budget_grouping_key": "regression_budget",
        "budget_manifest": "regression_budget",
        "schema_hashes": "regression_budget",
        "r0_model_lock": "regression_budget",
        "alpha_rule": "regression_budget",
        "r1_test_identity": "regression_budget",
        "qc_authority": "qc",
        "qc_provenance": "qc",
        "classification_population": "classification",
    }
    components: dict[str, str] = {}
    for field, value in checks.items():
        if field in {"classification", "regression_budget", "qc"}:
            component = field
        else:
            component = component_for_field.get(field)
        if component is None:
            raise ValueError(f"unknown critical field: {field}")
        components.setdefault(component, "PASS")
        if not _is_pass(value):
            components[component] = "BLOCKED"
    return components


def audit_comparator_population(
    name: str,
    observed: Mapping[str, Any],
    canonical: Mapping[str, Any] = EXPECTED_POPULATION,
) -> dict[str, Any]:
    expected = {key: int(value) for key, value in canonical.items()}
    try:
        actual = {key: int(value) for key, value in observed.items()}
    except (TypeError, ValueError):
        actual = {}
    matches = set(actual) == set(expected) and actual == expected
    return {
        "comparator": name,
        "status": "PASS" if matches else "EXCLUDED_NONCANONICAL",
        "observed_population": actual,
        "canonical_population": expected,
        "retrain": False,
        "replacement": False,
        "reason": "canonical population identity" if matches else "test population mismatch",
    }


def audit_budget_manifest(
    budget_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    test_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed: list[str] = []
    normalized = {int(key): list(value) for key, value in budget_rows.items()}
    counts = {pct: len(normalized.get(pct, [])) for pct in EXPECTED_COUNTS}
    if set(normalized) != set(EXPECTED_COUNTS) or counts != EXPECTED_COUNTS:
        failed.append("budget_counts")

    identities: dict[int, set[str]] = {}
    for pct, rows in normalized.items():
        values = [str(row.get("canonical_identity", "")).strip() for row in rows]
        identities[pct] = {value for value in values if value}
        if len(identities[pct]) != len(rows):
            failed.append(f"budget_{pct}_identities")
    nested = all(
        identities.get(smaller, set()).issubset(identities.get(larger, set()))
        for smaller, larger in ((5, 10), (10, 15), (15, 20))
    )
    if not nested:
        failed.append("nested_identities")

    test_identity_values = [
        str(row.get("physical_identity", "")).strip() for row in test_rows
    ]
    if not test_identity_values or any(not value for value in test_identity_values):
        failed.append("test_identities")
    test_identities = {value for value in test_identity_values if value}
    overlap = len(identities.get(20, set()) & test_identities)
    if overlap:
        failed.append("calibration_test_exact_identity_overlap")

    group_to_gas: dict[str, str] = {}
    invalid_groups = False
    for row in test_rows:
        filename = str(row.get("filename", "")).strip()
        gas = str(row.get("gas_code", "")).strip()
        if not filename or not gas:
            invalid_groups = True
            continue
        previous = group_to_gas.setdefault(filename, gas)
        if previous != gas:
            invalid_groups = True
    group_counts = dict(sorted(Counter(group_to_gas.values()).items()))
    if invalid_groups or len(group_to_gas) != 80:
        failed.append("test_filename_groups")
    if group_counts != {"CO": 20, "Ea": 20, "Ey": 20, "Me": 20}:
        failed.append("test_filename_groups_per_gas")

    return {
        "status": "BLOCKED" if failed else "PASS",
        "failed_checks": sorted(set(failed)),
        "counts": counts,
        "nested": nested,
        "calibration_test_exact_identity_overlap": overlap,
        "test_filename_groups": len(group_to_gas),
        "test_filename_groups_per_gas": group_counts,
        "grouping_key": GROUPING_KEY,
    }


def audit_schema_hashes(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    required = ("83D", "104D")
    failed = [
        key
        for key in required
        if not isinstance(observed.get(key), str)
        or observed.get(key) != expected.get(key)
    ]
    if set(observed) != set(required) or set(expected) != set(required):
        failed.extend(sorted((set(observed) | set(expected)) - set(required)))
        for key in required:
            if key not in observed or key not in expected:
                failed.append(key)
    return {
        "status": "BLOCKED" if failed else "PASS",
        "failed_checks": sorted(set(failed)),
        "observed": dict(observed),
        "expected": dict(expected),
    }


def audit_exact_hash(observed: Any, expected: Any, label: str) -> dict[str, Any]:
    def valid(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    passed = valid(observed) and valid(expected) and observed == expected
    return {
        "status": "PASS" if passed else "BLOCKED",
        "failed_checks": [] if passed else [label],
        "observed": observed,
        "expected": expected,
    }


def audit_alpha_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(rule)
    try:
        observed["grid"] = [float(value) for value in rule.get("grid", [])]
        observed["folds"] = int(rule.get("folds"))
    except (TypeError, ValueError):
        pass
    failed = [
        key for key, expected in EXPECTED_ALPHA_RULE.items() if observed.get(key) != expected
    ]
    if set(observed) != set(EXPECTED_ALPHA_RULE):
        failed.append("fields")
    return {
        "status": "BLOCKED" if failed else "PASS",
        "failed_checks": sorted(set(failed)),
        "observed": observed,
        "expected": dict(EXPECTED_ALPHA_RULE),
    }


def audit_qc_provenance(
    *,
    q1: Mapping[str, Any],
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    fields = ("backend", "population", "risk_definition", "policy", "authority")
    required = {
        "backend": "R84_CONCAT",
        "population": "C3+C4+C5 canonical test",
        "risk_definition": "classification confidence",
        "policy": "CONFIDENCE_QC_FINAL",
        "authority": "AUTHORITATIVE",
    }
    failed: list[str] = []
    for field in fields:
        values = (q1.get(field), metrics.get(field), thresholds.get(field))
        if any(value != required[field] for value in values) or len(set(values)) != 1:
            failed.append(field)
    passed = not failed
    return {
        "status": "PASS" if passed else "BLOCKED",
        "failed_checks": failed,
        "policy_scope": "confidence-only",
        "authorized_overlays": (
            ["FINAL_CONFIDENCE_QC_METRICS.csv", "FINAL_CONFIDENCE_QC_THRESHOLDS.csv"]
            if passed
            else []
        ),
        "legacy": {
            "05_qc_final.csv": "HISTORICAL_SUPERSEDED",
            "legacy_random_retention": "HISTORICAL_SUPERSEDED",
        },
        "compatibility": {field: required[field] for field in fields},
    }


def build_candidate_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in (15, 10, 5):
        for method in ("R83_TARGET_ONLY", "R84_CONCAT"):
            rows.append(
                {
                    "experiment_id": f"IOTJ-V4-C5-B{budget:02d}-{method}",
                    "source_clients": "C1;C2",
                    "target_clients": "C5",
                    "split_protocol": "canonical-v1 calibration/test roles",
                    "method": method,
                    "model": method,
                    "checkpoint": (
                        "R0-v2 immutable source prior" if method == "R84_CONCAT" else "none"
                    ),
                    "DA": "none",
                    "calibration": "deterministic nested calibration-only grouped CV",
                    "QC": "none",
                    "seed": 42,
                    "budget_pct": budget,
                    "calibration_n": EXPECTED_COUNTS[budget],
                    "result_path": "NOT_CREATED",
                    "status": "REGISTERED_NOT_RUN",
                    "notes": "20% pair is reused and is intentionally not a candidate run",
                }
            )
    return rows


def _scope_check(repo_root: Path, scope_commit: str) -> dict[str, Any]:
    if scope_commit != FROZEN_SCOPE_COMMIT:
        return {"status": "BLOCKED", "reason": "scope commit differs from frozen scope"}
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{scope_commit}^{{commit}}"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "BLOCKED",
        "scope_commit": scope_commit,
        "reason": "frozen scope commit exists" if completed.returncode == 0 else "missing commit",
    }


def _classification_audit(repo_root: Path) -> tuple[dict[str, Any], list[Path]]:
    table_path = repo_root / CLASSIFICATION_TABLE
    manifest_path = repo_root / CLASSIFICATION_MANIFEST
    rows = _read_csv(table_path)
    manifest = _read_json(manifest_path)
    failed: list[str] = []
    required_methods = {"FedAvg", "FedProx", "SCAFFOLD", "MMD", "A0T", "GAPS/A4"}
    by_method: dict[str, dict[str, int]] = {}
    for row in rows:
        method = row.get("method", "")
        target = row.get("target", "")
        if method in required_methods and target in {*EXPECTED_POPULATION, "ALL"}:
            try:
                by_method.setdefault(method, {})[target] = int(row["N"])
            except (KeyError, TypeError, ValueError):
                failed.append(f"{method}_population")
    expected_with_pool = {**EXPECTED_POPULATION, "ALL": sum(EXPECTED_POPULATION.values())}
    for method in sorted(required_methods):
        if by_method.get(method) != expected_with_pool:
            failed.append(f"{method}_population")
    if manifest.get("status") != "PASS" or manifest.get("selection_performed") is not False:
        failed.append("classification_manifest")

    inputs = [table_path, manifest_path]
    comparators: dict[str, Any] = {}
    for method in ("CORAL", "DANN"):
        observed: dict[str, int] = {}
        for target in ("C3", "C4", "C5"):
            path = (
                repo_root
                / COMPARATOR_ROOT
                / f"FCL-E2-{method}-{target}"
                / f"final_evaluation_{target}.json"
            )
            inputs.append(path)
            payload = _read_json(path)
            predictions = payload.get("predictions", [])
            target_n = payload.get("target_metrics", {}).get("num_examples")
            if not isinstance(predictions, list) or len(predictions) != target_n:
                failed.append(f"{method}_{target}_prediction_count")
            observed[target] = int(target_n)
        comparators[method] = audit_comparator_population(
            method, observed, EXPECTED_POPULATION
        )
        if comparators[method]["status"] != "EXCLUDED_NONCANONICAL":
            failed.append(f"{method}_must_be_excluded")
    return (
        {
            "status": "BLOCKED" if failed else "PASS",
            "failed_checks": sorted(set(failed)),
            "pooled_rule": POOLED_RULE,
            "information_note": INFORMATION_NOTE,
            "canonical_population": EXPECTED_POPULATION,
            "comparators": comparators,
        },
        inputs,
    )


def _regression_audit(repo_root: Path) -> tuple[dict[str, Any], list[Path]]:
    inputs: list[Path] = []
    budget_rows: dict[int, list[dict[str, str]]] = {}
    for pct in (20, 15, 10, 5):
        path = repo_root / BUDGET_ROOT / f"c5_calibration_budget_{pct:02d}pct.csv"
        inputs.append(path)
        budget_rows[pct] = _read_csv(path)
    test_info_path = repo_root / DATASET_C5 / "test_experiment_info.json"
    calibration_info_path = repo_root / DATASET_C5 / "calibration_experiment_info.json"
    inputs.extend((test_info_path, calibration_info_path))
    test_rows = _read_json(test_info_path)
    budget = audit_budget_manifest(budget_rows, test_rows)

    calibration_ids = {
        str(row.get("physical_identity", "")).strip()
        for row in _read_json(calibration_info_path)
    }
    if calibration_ids != {
        str(row.get("canonical_identity", "")).strip() for row in budget_rows[20]
    }:
        budget["status"] = "BLOCKED"
        budget["failed_checks"] = sorted(
            set(budget["failed_checks"] + ["20pct_calibration_identity"])
        )

    r0_protocol_path = repo_root / R0_PROTOCOL
    r1_cache_path = (
        repo_root / R1_ROOT / "canonical_feature_caches/C5/test/cache_manifest.json"
    )
    inputs.extend((r0_protocol_path, r1_cache_path))
    feature_protocol = _read_json(r0_protocol_path).get("feature_protocol", {})
    cache_manifest = _read_json(r1_cache_path)
    observed_schema = {
        "83D": cache_manifest.get("ordered_sensor_feature_names_sha256"),
        "104D": cache_manifest.get("ordered_h1_feature_names_sha256"),
    }
    schema = audit_schema_hashes(
        observed_schema,
        {"83D": EXPECTED_83D_SCHEMA_SHA256, "104D": EXPECTED_104D_SCHEMA_SHA256},
    )
    if (
        feature_protocol.get("sensor_dimensions") != 83
        or feature_protocol.get("h1_dimensions") != 104
        or feature_protocol.get("ordered_sensor_feature_names_sha256")
        != EXPECTED_83D_SCHEMA_SHA256
        or feature_protocol.get("ordered_h1_feature_names_sha256")
        != EXPECTED_104D_SCHEMA_SHA256
        or cache_manifest.get("schema_bundle_sha256") != EXPECTED_SCHEMA_BUNDLE_SHA256
    ):
        schema["status"] = "BLOCKED"
        schema["failed_checks"] = sorted(
            set(schema["failed_checks"] + ["dimensions_or_schema_bundle"])
        )

    r0_index_path = repo_root / R0_ROOT / "sha256_index.json"
    r0_lock_path = repo_root / R0_ROOT / "model_lock.json"
    inputs.extend((r0_index_path, r0_lock_path))
    r0_index = _read_json(r0_index_path)
    r0_lock_hash = sha256_file(r0_lock_path)
    r0_index_hash = sha256_file(r0_index_path)
    r0_lock_pass = (
        audit_exact_hash(r0_index_hash, EXPECTED_R0_INDEX_SHA256, "R0-v2 index")["status"]
        == "PASS"
        and audit_exact_hash(
            r0_lock_hash, EXPECTED_R0_MODEL_LOCK_SHA256, "R0-v2 model lock"
        )["status"]
        == "PASS"
        and r0_index.get("model_lock.json") == EXPECTED_R0_MODEL_LOCK_SHA256
    )
    r0_lock = {
        "status": "PASS" if r0_lock_pass else "BLOCKED",
        "model_lock_sha256": r0_lock_hash,
        "index_sha256": r0_index_hash,
    }

    r1_protocol_path = repo_root / R1_PROTOCOL
    inputs.append(r1_protocol_path)
    r1_protocol = _read_json(r1_protocol_path)
    selection_text = r1_protocol.get("alpha_selection", "")
    alpha = audit_alpha_rule(
        {
            "grid": r1_protocol.get("alpha_grid", []),
            "folds": 5 if "five-fold" in selection_text else None,
            "grouping": "raw_filename" if "raw-filename-grouped" in selection_text else None,
            "objective": "pooled RMSE" if "pooled RMSE" in selection_text else None,
            "tie_break": "first-in-grid" if "first tie" in selection_text else None,
            "selection_data": "calibration-only" if "calibration CV" in selection_text else None,
        }
    )

    predictions_path = repo_root / R1_ROOT / "predictions.csv"
    inputs.append(predictions_path)
    prediction_rows = _read_csv(predictions_path)
    dataset_test_ids = {str(row["physical_identity"]) for row in test_rows}
    by_method = {
        method: {
            row["physical_identity"]
            for row in prediction_rows
            if row.get("target") == "C5" and row.get("method") == method
        }
        for method in ("TARGET_ONLY_83D_RIDGE", "R84_CONCAT")
    }
    identity_pass = (
        cache_manifest.get("row_count") == 1360
        and cache_manifest.get("row_identities_sha256")
        == EXPECTED_C5_R1_TEST_IDENTITY_SHA256
        and all(values == dataset_test_ids for values in by_method.values())
    )
    identity = {
        "status": "PASS" if identity_pass else "BLOCKED",
        "row_identities_sha256": cache_manifest.get("row_identities_sha256"),
        "expected_sha256": EXPECTED_C5_R1_TEST_IDENTITY_SHA256,
        "method_identity_counts": {key: len(value) for key, value in by_method.items()},
    }

    registry = build_candidate_registry()
    component = final_verdict(
        {
            "budget_manifest": budget,
            "schema_hashes": schema,
            "r0_model_lock": r0_lock,
            "alpha_rule": alpha,
            "r1_test_identity": identity,
        }
    )["regression_budget"]
    return (
        {
            "status": component,
            "budget_manifest": budget,
            "schema_hashes": schema,
            "r0_model_lock": r0_lock,
            "alpha_rule": alpha,
            "r1_test_identity": identity,
            "full_budget_20pct": "REUSE_ONLY_NOT_REGISTERED",
            "candidate_registry_count": len(registry),
            "candidate_runs_executed": 0,
        },
        inputs,
    )


def _qc_audit(repo_root: Path) -> tuple[dict[str, Any], list[Path]]:
    decision_path = repo_root / Q1_ROOT / "Q1_DECISION.json"
    lock_path = repo_root / Q1_ROOT / "conformal_policy_lock.json"
    curves_path = repo_root / Q1_ROOT / "q1_risk_coverage_curves.csv"
    provenance_path = repo_root / CURRENT_QC_ROOT / "RESULT_PROVENANCE_INDEX.json"
    index_path = repo_root / CURRENT_QC_ROOT / "sha256_index.json"
    metrics_path = repo_root / CURRENT_QC_ROOT / "FINAL_CONFIDENCE_QC_METRICS.csv"
    thresholds_path = repo_root / CURRENT_QC_ROOT / "FINAL_CONFIDENCE_QC_THRESHOLDS.csv"
    legacy_path = repo_root / LEGACY_QC
    random_path = repo_root / LEGACY_RANDOM
    inputs = [
        decision_path,
        lock_path,
        curves_path,
        provenance_path,
        index_path,
        metrics_path,
        thresholds_path,
        legacy_path,
    ]
    if random_path.is_file():
        inputs.append(random_path)

    decision = _read_json(decision_path)
    lock = _read_json(lock_path)
    provenance = _read_json(provenance_path)
    current_index = _read_json(index_path)
    metrics_rows = _read_csv(metrics_path)
    thresholds_rows = _read_csv(thresholds_path)
    curve_rows = _read_csv(curves_path)
    authoritative = {
        (row.get("study"), row.get("source_path")): row
        for row in provenance.get("source_artifacts", [])
        if row.get("status") == "AUTHORITATIVE"
    }
    q1_sources = [key for key in authoritative if key[0] == "Q1"]
    policies = {row.get("policy") for row in metrics_rows + thresholds_rows}
    targets = {row.get("target") for row in metrics_rows}
    threshold_targets = {row.get("target") for row in thresholds_rows}
    curve_policies = {row.get("policy") for row in curve_rows}
    hash_pass = (
        current_index.get("FINAL_CONFIDENCE_QC_METRICS.csv") == sha256_file(metrics_path)
        and current_index.get("FINAL_CONFIDENCE_QC_THRESHOLDS.csv")
        == sha256_file(thresholds_path)
    )
    authority_pass = (
        decision.get("backend") == "R84_CONCAT"
        and decision.get("decision") == "CONFIDENCE_QC_FINAL"
        and provenance.get("final_regression") == "R84_CONCAT"
        and provenance.get("final_qc") == "CONFIDENCE_QC_FINAL"
        and "CLASSIFICATION_CONFIDENCE_ONLY" in lock.get("risk_policies", [])
        and policies == {"CONFIDENCE_QC_FINAL"}
        and targets == {"C3", "C4", "C5", "POOLED_C3_C4_C5"}
        and threshold_targets == {"C3", "C4", "C5"}
        and "CLASSIFICATION_CONFIDENCE_ONLY" in curve_policies
        and len(q1_sources) >= 2
        and hash_pass
    )
    common = {
        "backend": "R84_CONCAT" if authority_pass else "unknown",
        "population": "C3+C4+C5 canonical test" if authority_pass else "unknown",
        "risk_definition": "classification confidence" if authority_pass else "unknown",
        "policy": "CONFIDENCE_QC_FINAL" if authority_pass else "unknown",
        "authority": "AUTHORITATIVE" if authority_pass else "unknown",
    }
    audit = audit_qc_provenance(q1=common, metrics=common, thresholds=common)
    audit.update(
        {
            "q1_study_id": decision.get("study_id"),
            "q1_sources_authoritative": len(q1_sources),
            "current_table_hashes_verified": hash_pass,
            "legacy_05_qc_path": LEGACY_QC.as_posix(),
            "legacy_random_retention_path": LEGACY_RANDOM.as_posix(),
        }
    )
    return audit, inputs


def build_audit(repo_root: Path, scope_commit: str) -> tuple[dict[str, Any], dict[str, str]]:
    repo_root = repo_root.resolve()
    scope = _scope_check(repo_root, scope_commit)
    classification, classification_inputs = _classification_audit(repo_root)
    regression, regression_inputs = _regression_audit(repo_root)
    qc, qc_inputs = _qc_audit(repo_root)
    components = {
        "classification": classification["status"],
        "regression_budget": regression["status"],
        "qc": qc["status"],
    }
    if scope["status"] != "PASS":
        components = {component: "BLOCKED" for component in components}
    status = "PASS" if all(value == "PASS" for value in components.values()) else "BLOCKED"
    paths = classification_inputs + regression_inputs + qc_inputs
    hashes = {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(set(paths))
    }
    audit = {
        "schema_version": "iotj.v4.final_pre_run_audit.v1",
        "status": status,
        "scope": scope,
        "components": components,
        "classification": classification,
        "regression_budget": regression,
        "qc_provenance": qc,
        "candidate_registry": {
            "status": "PASS" if len(build_candidate_registry()) == 6 else "BLOCKED",
            "count": len(build_candidate_registry()),
            "budgets": [15, 10, 5],
            "methods": ["R83_TARGET_ONLY", "R84_CONCAT"],
        },
        "training_performed": False,
        "test_release_performed": False,
        "test_tensor_or_label_parsed_for_new_runs": False,
    }
    return audit, hashes


def _markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# IoT-J v4 final pre-run audit",
        "",
        f"Verdict: **{audit['status']}**",
        "",
        "## Components",
        "",
        "| Component | Verdict |",
        "|---|---|",
    ]
    lines.extend(
        f"| {component} | {status} |"
        for component, status in audit["components"].items()
    )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            f"- Pooled rule: {POOLED_RULE}",
            f"- Information note: {INFORMATION_NOTE}",
            "- CORAL/DANN mismatched endpoints remain EXCLUDED_NONCANONICAL; no retraining or replacement.",
            "- 20% R83/R84 is reuse-only; the registry contains only six 15/10/5% candidates.",
            "- Legacy QC and legacy random-retention evidence are HISTORICAL_SUPERSEDED.",
            "- This audit performs no training and releases no target test for a new run.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_root: Path,
    audit: Mapping[str, Any],
    registry: Sequence[Mapping[str, Any]],
    input_hashes: Mapping[str, str],
) -> list[Path]:
    """Create a brand-new formal root and its five contracted pre-run files."""
    output_root = Path(output_root)
    if not registry:
        raise RuntimeError("refusing empty experiment registry")
    if output_root.exists():
        raise FileExistsError(f"exclusive formal output root exists; refusing overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    pre_run = output_root / "pre_run"
    pre_run.mkdir(exist_ok=False)
    paths = {
        "markdown": pre_run / "FINAL_PRE_RUN_AUDIT.md",
        "audit": pre_run / "final_pre_run_audit.json",
        "qc": pre_run / "qc_provenance_audit.json",
        "registry": pre_run / "experiment_registry.csv",
        "hashes": pre_run / "input_sha256.json",
    }
    paths["markdown"].write_text(_markdown(audit), encoding="utf-8")
    paths["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["qc"].write_text(
        json.dumps(audit.get("qc_provenance", {}), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = list(registry[0])
    with paths["registry"].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(registry)
    paths["hashes"].write_text(
        json.dumps(dict(input_hashes), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return list(paths.values())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--scope-commit", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit, hashes = build_audit(REPO_ROOT, args.scope_commit)
    if args.dry_run:
        payload = {
            "schema_version": "iotj.v4.final_pre_run_dry_run.v1",
            "mode": "dry-run",
            "status": audit["status"],
            "components": audit["components"],
            "candidate_registry_count": audit["candidate_registry"]["count"],
            "comparator_statuses": {
                name: item["status"]
                for name, item in audit["classification"]["comparators"].items()
            },
            "writes_performed": False,
            "test_release_performed": False,
            "training_performed": False,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    created = write_outputs(args.output_root, audit, build_candidate_registry(), hashes)
    print(
        json.dumps(
            {
                "mode": "write",
                "status": audit["status"],
                "created": [str(path) for path in created],
                "test_release_performed": False,
                "training_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
