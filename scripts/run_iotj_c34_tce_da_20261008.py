"""Six fixed C3/C4 TCE diagnostics reusing the unchanged C5 adaptation code."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts import run_iotj_c5_labeled_conditional_da as old

RESULT = ROOT / "results/c34_tce_da_extension_20261008"
METHODS = ("coral_tce", "mmd_tce", "dann_tce")
TARGETS = {"C3": (3, 320, 680), "C4": (4, 160, 320)}
DISPLAY = dict(zip(METHODS, ("CORAL+TCE", "MMD+TCE", "DANN+TCE")))


def matrix():
    return [
        {"experiment_id": f"{target}LCD-{method.upper().replace('_', '-')}-S42-20261008",
         "target": target, "method_key": method, "method": DISPLAY[method],
         "calibration_examples": counts[1], "test_examples": counts[2],
         **old.registered_method_specs()[method]}
        for target, counts in TARGETS.items() for method in METHODS
    ]


def check_inputs():
    if old.sha256_file(old.SOURCE_CHECKPOINT) != old.SOURCE_WHOLE_FILE_SHA256:
        raise RuntimeError("Source checkpoint SHA256 mismatch")
    hashes = {}
    paths = [old.SOURCE_CHECKPOINT, Path(__file__),
             ROOT / "gaps_flower/labeled_conditional_da.py",
             ROOT / "scripts/run_iotj_c5_labeled_conditional_da.py",
             old.DATA_ROOT / "norm_stats.npz"]
    for client in (1, 2, 3, 4):
        directory = old.DATA_ROOT / f"client_{client}"
        for split in ("calibration", "test"):
            for field in ("features", "classification_labels", "regression_labels", "phase_labels"):
                paths.append(directory / f"{split}_{field}.npy")
    paths.append(old.RESULT_ROOT / "c5_labeled_conditional_da_comparison.csv")
    for path in paths:
        hashes[path.relative_to(ROOT).as_posix()] = old.sha256_file(path)
    for target, (client, count, _) in TARGETS.items():
        _, info = old.load_target_calibration_xy(
            old.DATA_ROOT / f"client_{client}", batch_size=32, seed=42)
        if info["num_examples"] != count or info["feature_shape"][1:] != [100, 8]:
            raise RuntimeError(f"Unexpected calibration data for {target}: {info}")
    return hashes


def execute():
    if RESULT.exists():
        raise FileExistsError(f"Preserve existing output: {RESULT}")
    hashes = check_inputs()
    device = torch.device("cpu")
    source, _ = old.checkpoint_model(old.SOURCE_CHECKPOINT, device)
    fingerprint = old.ordered_state_content_fingerprint(source.state_dict())
    if fingerprint != old.SOURCE_ORDERED_FINGERPRINT:
        raise RuntimeError("Source state-content fingerprint mismatch")
    RESULT.mkdir()
    specs = matrix()
    manifest = {
        "study_id": "C34-TCE-DA-EXTENSION-20261008", "status": "PRE_RUN_FROZEN",
        "source_clients": ["C1", "C2"], "dataset": str(old.DATA_ROOT.relative_to(ROOT)),
        "source_checkpoint": str(old.SOURCE_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_role": "pure_FedAvg_round25_LE1_seed42",
        "source_checkpoint_sha256": old.SOURCE_WHOLE_FILE_SHA256,
        "source_checkpoint_ordered_fingerprint": fingerprint, "matrix": specs,
        "optimizer": "Adam", "optimizer_lr": 5e-4, "batch_size": 32,
        "seed": 42, "steps": 100, "alignment_weight": 0.5, "target_ce_weight": 1.0,
        "target_fields": ["x", "class"], "source_batch": "unchanged_C1_C2_calibration_loader",
        "device": str(device), "torch_threads": torch.get_num_threads(),
        "independent_source_reload": True, "new_source_FL": False,
        "test_release": "after_all_six_step100_markers", "hyperparameter_search": False,
        "early_stopping": False, "input_hashes": hashes,
        "test_hashing_note": "byte provenance only before endpoints; no test arrays loaded for adaptation",
        "C5": "reported_read_only; no retraining",
        "scope": "legacy supplementary diagnostic; existing C3/C4 train splits are unused",
    }
    old.write_json(RESULT / "protocol_manifest.json", manifest)
    old.write_csv(RESULT / "EXPERIMENT_MATRIX.csv", specs)
    for run_index, spec in enumerate(specs):
        target, method = spec["target"], spec["method_key"]
        old.write_json(RESULT / "RUN_PROGRESS.json", {
            "status": "ADAPTING", "completed": run_index, "total": 6,
            "current": spec["experiment_id"]})
        client = TARGETS[target][0]
        run = RESULT / target / method
        run.mkdir(parents=True)
        loader, info = old.load_target_calibration_xy(
            old.DATA_ROOT / f"client_{client}", batch_size=32, seed=42)
        info["split"] = f"{target}_calibration"
        source_loader = old.create_merged_calibration_loader(
            [old.DATA_ROOT / "client_1", old.DATA_ROOT / "client_2"],
            batch_size=32, num_workers=0)
        model, payload = old.checkpoint_model(old.SOURCE_CHECKPOINT, device)
        adapted, diagnostics, seconds = old.run_labeled_conditional_da(
            method, model, source_loader, loader, device,
            expected_source_fingerprint=fingerprint, formal=True)
        if len(diagnostics) != 100 or not all(math.isfinite(row["total_loss"]) for row in diagnostics):
            raise RuntimeError("Incomplete/nonfinite adaptation")
        checkpoint = run / "adapted_step_100.pth"
        torch.save({**payload, "model_state": adapted.state_dict(), "step": 100,
                    "experiment_id": spec["experiment_id"], "seed": 42}, checkpoint)
        old.write_csv(run / "adaptation_diagnostics.csv", diagnostics)
        old.write_json(run / "fixed_endpoint_complete.json", {
            **spec, "checkpoint_sha256": old.sha256_file(checkpoint),
            "source_fingerprint": fingerprint, "target_test_opened": False,
            "calibration": info, "commissioning_seconds": seconds})
        print(f"ENDPOINT {target} {DISPLAY[method]} step100 complete ({seconds:.2f}s)", flush=True)
    for spec in specs:
        if not (RESULT / spec["target"] / spec["method_key"] / "fixed_endpoint_complete.json").is_file():
            raise RuntimeError("All six endpoints required before test release")
    old.write_json(RESULT / "test_release.json", {"all_six_complete": True, "selection": False})
    old.write_json(RESULT / "RUN_PROGRESS.json", {"status": "EVALUATING", "completed": 6, "total": 6})
    source_test = old.create_merged_test_loader(
        [old.DATA_ROOT / "client_1", old.DATA_ROOT / "client_2"], batch_size=32, num_workers=0)
    rows, predictions = [], []
    for target, (client, _, ntest) in TARGETS.items():
        target_test = old.create_client_test_only_loader(
            old.DATA_ROOT / f"client_{client}", batch_size=32, num_workers=0)
        for method in METHODS:
            run = RESULT / target / method
            checkpoint = run / "adapted_step_100.pth"
            model, _ = old.checkpoint_model(checkpoint, device)
            probs, truth = old.predict_probabilities(model, target_test, device)
            sp, sy = old.predict_probabilities(model, source_test, device)
            if len(truth) != ntest:
                raise RuntimeError(f"Unexpected test count for {target}")
            tm, sm = old.classification_metrics(probs, truth), old.classification_metrics(sp, sy)
            endpoint = json.loads((run / "fixed_endpoint_complete.json").read_text())
            row = old.comparison_row(method=method, target_metrics=tm, source_metrics=sm,
                                     seconds=endpoint["commissioning_seconds"], checkpoint=checkpoint)
            row.update(target=target, experiment_id=endpoint["experiment_id"],
                       calibration_examples=TARGETS[target][1],
                       checkpoint=str(checkpoint.relative_to(ROOT)), calculation_status="recomputed")
            rows.append(row)
            old.write_json(run / f"final_evaluation_{target}.json", {
                "comparison_row": row, "target_metrics": tm, "source_metrics": sm,
                "selection_role": "none_fixed_step100"})
            for i, (y, p) in enumerate(zip(truth, probs)):
                predictions.append({"target": target, "method": DISPLAY[method], "window_index": i,
                                    "true_class": int(y), "predicted_class": int(np.argmax(p)),
                                    **{f"prob_class_{k}": float(p[k]) for k in range(4)}})
            print(f"RESULT {target} {DISPLAY[method]} Accuracy={100*tm['accuracy']:.3f}% F1={100*tm['macro_f1']:.3f}%", flush=True)
    old.write_csv(RESULT / "c34_tce_comparison.csv", rows)
    old.write_csv(RESULT / "per_window_predictions.csv", predictions)
    with (old.RESULT_ROOT / "c5_labeled_conditional_da_comparison.csv").open(newline="", encoding="utf-8") as f:
        c5 = [dict(row, calculation_status="reported_read_only", calibration_examples=320)
              for row in csv.DictReader(f) if row["method_key"] in METHODS]
    if len(c5) != 3:
        raise RuntimeError("Three existing C5 TCE rows required")
    all_rows = rows + c5
    old.write_csv(RESULT / "c345_tce_comparison.csv", all_rows)
    with (old.BASE_RESULT_ROOT / "classification_main_comparison.csv").open(newline="", encoding="utf-8") as f:
        refs = [row for row in csv.DictReader(f) if row["method"] in ("CORAL", "MMD", "DANN")]
    deltas = []
    for row in all_rows:
        baseline = next(r for r in refs if r["method"] == row["method"].split("+")[0]
                        and r["target_id"] == row["target"])
        if int(baseline["num_examples"]) != int(row["num_examples"]):
            raise RuntimeError("Reference population mismatch")
        deltas.append({"target": row["target"], "method": row["method"],
                       "accuracy_delta_pp": 100*(float(row["accuracy"])-float(baseline["accuracy"])),
                       "macro_f1_delta_pp": 100*(float(row["macro_f1"])-float(baseline["macro_f1"]))})
    old.write_csv(RESULT / "tce_vs_xonly_deltas.csv", deltas)
    for rel, expected in hashes.items():
        if old.sha256_file(ROOT / rel) != expected:
            raise RuntimeError(f"Input changed: {rel}")
    old.write_json(RESULT / "EXPERIMENT_AUDIT.json", {
        "status": "PASS", "new_endpoints": 6, "steps_each": 100,
        "new_prediction_rows": len(predictions), "comparison_rows_with_reported_C5": len(all_rows),
        "independent_source_reload": True, "inputs_unchanged": True,
        "target_adaptation_fields": ["x", "class"], "target_test_used_for_selection": False,
        "target_test_opened_after_all_endpoints": True, "scope": "legacy seed42 descriptive diagnostic"})
    old.write_json(RESULT / "RUN_PROGRESS.json", {"status": "COMPLETE", "completed": 6, "total": 6})
    index = {p.relative_to(RESULT).as_posix(): old.sha256_file(p)
             for p in sorted(RESULT.rglob("*")) if p.is_file()}
    old.write_json(RESULT / "sha256_index.json", index)
    old.write_json(RESULT / "COMPLETE.json", {"status": "PASS", "new_runs": 6,
                                               "C5_reused": 3, "output": str(RESULT)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "run"))
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps({"output_exists": RESULT.exists(), "matrix": matrix(),
                          "input_hashes": check_inputs()}, indent=2))
    else:
        execute()
