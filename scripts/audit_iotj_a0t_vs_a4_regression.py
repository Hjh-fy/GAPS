"""Strictly audit the canonical-v1 A0T versus A4 regression evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_iotj_a0t_vs_a4_regression import regression_decision
from scripts.run_iotj_a0t_vs_a4_regression import (
    DATA_ROOT,
    DEFAULT_OUTPUT,
    EXPECTED_DATASET_SHA256,
    audit_checkpoint,
    endpoint_specs,
    expected_counts,
    frozen_alphas,
)
from tools.verify_iotj_canonical_v1_hashes import verify as verify_dataset


SCOPES = {
    "S_ALL": "test_s_all.csv",
    "S_CC": "test_s_cc.csv",
    "Oracle_ALL": "test_oracle_all.csv",
    "Oracle_CC": "test_oracle_cc.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_hash_index(base: Path, index: Mapping[str, str]) -> list[str]:
    return [relative for relative, expected in index.items() if not (base / relative).is_file() or sha256(base / relative) != expected]


def verify_decision(c5_delta: float, pooled_delta: float, recorded: str) -> bool:
    expected = regression_decision(c5_delta, pooled_delta)
    if expected != recorded:
        raise RuntimeError(f"FAIL_CLOSED dual-gate decision differs: {recorded} != {expected}")
    return True


def _prediction_indices(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for spec in endpoint_specs():
        endpoint = root / "endpoints" / spec.experiment_id
        for filename in SCOPES.values():
            relative = (endpoint / filename).relative_to(root).as_posix()
            output[relative] = sha256(root / relative)
        qc = endpoint / "test_qc_frozen.csv"
        output[qc.relative_to(root).as_posix()] = sha256(qc)
    return output


def audit_study(root: Path = DEFAULT_OUTPUT, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    failures: list[str] = []
    dataset = verify_dataset(DATA_ROOT)
    if dataset["status"] != "PASS" or dataset["aggregate_sha256"] != EXPECTED_DATASET_SHA256:
        failures.append("dataset_hash")

    protocol = read_json(root / "protocol_manifest.json")
    required_protocol = {
        "status": "FIXED_ENDPOINTS_COMPLETE",
        "endpoint_count": 6,
        "seed": 42,
        "target_test_opened_after_all_calibration_locks": True,
        "target_test_used_for_selection": False,
        "classifier_training_performed": False,
        "alpha_selection_performed": False,
    }
    for key, value in required_protocol.items():
        if protocol.get(key) != value:
            failures.append(f"protocol:{key}")

    checkpoint_index: dict[str, Any] = {}
    test_index: dict[str, Any] = {}
    prediction_index = _prediction_indices(root)
    for spec in endpoint_specs():
        endpoint = root / "endpoints" / spec.experiment_id
        manifest = read_json(endpoint / "endpoint_manifest.json")
        lock = read_json(endpoint / "calibration_lock.json")
        if manifest.get("status") != "COMPLETE" or manifest.get("experiment_id") != spec.experiment_id:
            failures.append(f"endpoint:{spec.experiment_id}")
        if lock.get("status") != "SEALED_BEFORE_TARGET_TEST" or lock.get("target_test_opened") is not False:
            failures.append(f"calibration_lock:{spec.experiment_id}")
        expected_alpha = {str(key): value for key, value in frozen_alphas()[spec.target].items()}
        if lock.get("fixed_alphas") != expected_alpha or lock.get("alpha_selection_performed") is not False:
            failures.append(f"alpha_lock:{spec.experiment_id}")
        models = read_json(endpoint / "r84_models.json")
        if {key: float(value["alpha"]) for key, value in models.items()} != expected_alpha:
            failures.append(f"model_alpha:{spec.experiment_id}")

        provenance = audit_checkpoint(spec)
        checkpoint_index[spec.experiment_id] = {
            "path": str(spec.checkpoint.resolve()),
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "ordered_state_content_fingerprint": provenance["ordered_state_content_fingerprint"],
        }
        if provenance["checkpoint_sha256"] != manifest.get("checkpoint_sha256"):
            failures.append(f"checkpoint:{spec.experiment_id}")

        target_test = DATA_ROOT / f"client_{spec.target[1:]}"
        test_index.setdefault(spec.target, {})
        for filename, expected in manifest["test_manifest_sha256"].items():
            observed = sha256(target_test / filename)
            test_index[spec.target][filename] = observed
            if observed != expected:
                failures.append(f"test_manifest:{spec.target}:{filename}")

        rows = {scope: read_csv(endpoint / filename) for scope, filename in SCOPES.items()}
        expected_n = expected_counts(spec.target)["test"]
        if len(rows["S_ALL"]) != expected_n or len(rows["Oracle_ALL"]) != expected_n:
            failures.append(f"row_count:{spec.experiment_id}")
        scc = [row["sample_index"] for row in rows["S_CC"]]
        oracle_cc = [row["sample_index"] for row in rows["Oracle_CC"]]
        if scc != oracle_cc:
            failures.append(f"paired_scope:{spec.experiment_id}")
        if spec.target == "C5":
            special = read_csv(endpoint / "special_slices.csv")
            if not special or special[0]["slice"] != "methane_225ppm_repeat1" or int(special[0]["N"]) <= 0:
                failures.append(f"special_slice:{spec.experiment_id}")

    if verify_hash_index(root, prediction_index):
        failures.append("prediction_hash_index")
    qc = read_json(root / "qc_protocol_manifest.json")
    if qc.get("status") != "FROZEN_QC_COMPLETE" or qc.get("threshold_refit") is not False or qc.get("qc_threshold_changed") is not False:
        failures.append("frozen_qc")

    comparison = read_csv(root / "regression_comparison.csv")
    lookup = {(row["method"], row["target"], row["scope"]): float(row["RMSE"]) for row in comparison}
    c5_delta = lookup[("A4", "C5", "S_ALL")] - lookup[("A0T", "C5", "S_ALL")]
    pooled_delta = lookup[("A4", "POOLED_C3_C4_C5", "S_ALL")] - lookup[("A0T", "POOLED_C3_C4_C5", "S_ALL")]
    recorded = "REGRESSION_ADVANTAGE_SUPPORTED" if "`REGRESSION_ADVANTAGE_SUPPORTED`" in (root / "A0T_VS_GAPS_FINAL_CONCLUSION.md").read_text(encoding="utf-8") else "REGRESSION_ADVANTAGE_NOT_SUPPORTED"
    try:
        verify_decision(c5_delta, pooled_delta, recorded)
    except RuntimeError:
        failures.append("dual_gate_decision")

    source = (ROOT / "scripts" / "run_iotj_a0t_vs_a4_regression.py").read_text(encoding="utf-8")
    if "RIDGE_ALPHAS" in source or "best_alpha" in source:
        failures.append("alpha_search_api")

    result = {
        "schema_version": "iotj.canonical_v1.a0t_vs_a4_regression.strict_audit.v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "endpoint_count": 6,
        "dataset_aggregate_sha256": dataset["aggregate_sha256"],
        "checkpoint_count": len(checkpoint_index),
        "prediction_file_count": len(prediction_index),
        "test_target_count": len(test_index),
        "dual_gate": {"c5_A4_minus_A0T_RMSE": c5_delta, "pooled_A4_minus_A0T_RMSE": pooled_delta, "decision": recorded},
        "stop_rule": "STOP_NO_FURTHER_EXPERIMENTS",
    }
    if write:
        outputs = ("checkpoint_sha256.json", "prediction_sha256.json", "test_manifest_sha256.json", "STRICT_AUDIT.json")
        existing = [name for name in outputs if (root / name).exists()]
        if existing:
            raise FileExistsError(f"FAIL_CLOSED audit outputs already exist: {existing}")
        write_json(root / "checkpoint_sha256.json", checkpoint_index)
        write_json(root / "prediction_sha256.json", prediction_index)
        write_json(root / "test_manifest_sha256.json", test_index)
        write_json(root / "STRICT_AUDIT.json", result)
    if failures:
        raise RuntimeError(f"FAIL_CLOSED strict audit failures: {failures}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = audit_study(args.root, write=not args.check_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
