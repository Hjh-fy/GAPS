"""Strict fail-closed audit for P0-U zero-label commissioning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.run_iotj_p0_zero_label_commissioning import (
    EXPECTED_CHECKPOINT_SHA256,
    PSEUDO_THRESHOLD,
    STEPS,
    static_label_access_audit,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(f"FAIL_CLOSED: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--result-root", required=True, type=Path); parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(); root = args.result_root.resolve()
    report = root / "EXPERIMENT_AUDIT.md"; index_path = root / "sha256_index.json"
    if report.exists() or index_path.exists(): raise FileExistsError("REFUSE_TO_OVERWRITE audit output")
    manifest = json.loads((root / "protocol_manifest.json").read_text(encoding="utf-8"))
    static = static_label_access_audit(); require(static["status"] == "PASS", "static label-access audit")
    require(manifest["source_checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256, "frozen source checkpoint")
    require(manifest["seed"] == 42 and manifest["steps"] == 100 and manifest["model_lr"] == 5e-4, "seed/steps/LR")
    require(manifest["threshold"] == PSEUDO_THRESHOLD == 0.90 and not manifest["hyperparameter_search"], "fixed threshold/no search")
    require(manifest["checkpoint_selection"] == "fixed_source_round25" and not manifest["target_test_used_for_selection"], "checkpoint/test selection")
    require(manifest["u1"]["target_api"] == "x_only" and manifest["u2"]["target_api"] == "x_only", "x-only APIs")
    expected_forbidden = {"target_ce", "class_conditional_coral", "class_mmd", "same_class_phase_stage_mmd", "target_proto_anchor", "target_label_semantic_matching", "pseudo_labels"}
    require(set(manifest["u1"]["forbidden_target_losses"]) == expected_forbidden, "U1 forbidden loss set")
    require(manifest["u2"]["teacher"] == "frozen_source_round25" and manifest["u2"]["pseudo_label_origin"] == "teacher_argmax", "U2 pseudo origin")

    u1 = rows(root / "unsupervised_alignment_diagnostics.csv"); require(len(u1) == STEPS, "100 U1 rows")
    for row in u1:
        require(row["target_label_object_present"] == "False", "U1 target label object absent")
        require(row["target_ce_status"] == "UNAVAILABLE" and row["class_conditional_coral_status"] == "DISABLED", "U1 target CE/CORAL")
        require(row["class_mmd_status"] == "DISABLED" and row["stage_mmd_status"] == "DISABLED", "U1 conditional MMD")
        require(row["target_proto_anchor_status"] == "UNAVAILABLE" and row["pseudo_label_status"] == "DISABLED", "U1 anchor/pseudo")
    u2 = rows(root / "pseudo_label_diagnostics.csv")
    training = [row for row in u2 if row["record_type"] == "training_step"]; posthoc = [row for row in u2 if row["record_type"] == "posthoc_truth_audit"]
    require(len(training) == STEPS and len(posthoc) == 1, "U2 100 steps plus one posthoc audit")
    require(all(float(row["threshold"]) == 0.90 for row in training), "U2 threshold fixed")
    require(all(row["pseudo_label_origin"] == "teacher_argmax_only" and row["target_label_object_present"] == "False" for row in training), "U2 origin/no truth")
    require(posthoc[0]["target_label_object_present"] == "posthoc_truth_only", "truth access posthoc only")

    ledger = manifest["label_access_ledger"]; names = [event["event"] for event in ledger]
    require(names.index("u1_training_completed") < names.index("u2_training_completed") < names.index("calibration_truth_opened_posthoc") < names.index("c5_test_opened"), "runtime label/test access order")
    require(all(event.get("target_labels_loaded") is False for event in ledger if event["event"] in {"u1_target_x_loader_created", "u1_training_completed", "u2_target_x_loader_created", "u2_training_completed"}), "runtime target labels absent")

    comparison = rows(root / "zero_label_commissioning_comparison.csv")
    expected_methods = {"source_only", "simple_target_ce", "unsupervised_global_alignment", "pseudo_label_self_training"}
    require(len(comparison) == 4 and {row["method"] for row in comparison} == expected_methods, "unified four-row comparison")
    require(all(int(row["num_examples"]) == 1360 and int(row["seed"]) == 42 for row in comparison), "formal sample/seed scope")
    for method in ("unsupervised_global_alignment", "pseudo_label_self_training"):
        row = next(item for item in comparison if item["method"] == method)
        require(row["source_checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256 and row["target_label_access"].startswith("none_x_only"), f"{method} origin/access")
    require(next(row for row in comparison if row["method"] == "simple_target_ce")["target_label_access"] == "C5_calibration_true_labels", "supervised reference declared")

    main_files = [root / name for name in ("zero_label_commissioning_comparison.csv", "unsupervised_alignment_diagnostics.csv", "pseudo_label_diagnostics.csv", "LABEL_ACCESS_AUDIT.md", "protocol_manifest.json", "label_access_static_preflight.json")]
    require(all(path.is_file() for path in main_files), "required output coverage")
    hashes = [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in main_files]
    index_path.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    report.write_text(
        "# P0-U Experiment Audit\n\n## Verdict: approved for seed42 descriptive evidence\n\n"
        "Static and runtime label-access audits passed. U1/U2 target training APIs received x-only tensors; U1 label-conditioned losses were unavailable/disabled; U2 pseudo labels came only from a frozen source teacher at the fixed 0.90 threshold. Calibration truth opened once after both training branches solely for pseudo-label precision, and C5 test opened afterward for final evaluation. Both branches identify the same hash-pinned source checkpoint. No hyperparameter, threshold, early-stopping, or checkpoint search occurred.\n\n"
        "## Limitations\n\nSeed42 only. Existing Source-only and supervised Target-CE rows are read-only references. Results do not support uncertainty, significance, or automatic follow-up optimization.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "result_root": str(root)}))


if __name__ == "__main__": main()
