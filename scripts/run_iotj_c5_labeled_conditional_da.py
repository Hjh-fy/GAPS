"""Run the frozen C5 labeled/conditional DA mechanism diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from federated_dataset import (
    create_client_test_only_loader,
    create_merged_calibration_loader,
    create_merged_test_loader,
)
from gaps_flower.evaluate_checkpoint import load_checkpoint_model
from gaps_flower.labeled_conditional_da import (
    registered_method_specs,
    run_labeled_conditional_da,
)
from gaps_flower.state_fingerprint import ordered_state_content_fingerprint
from scripts.evaluate_iotj_final_classification_le1 import classification_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RESULT_ROOT = REPO_ROOT / "results/iotj_final_classification_le1_20260804"
RESULT_ROOT = BASE_RESULT_ROOT / "c5_labeled_conditional_da_probe_20260824"
DOC_ROOT = REPO_ROOT / "docs/experiments/iotj_c5_labeled_conditional_da_20260824"
DATA_ROOT = (
    REPO_ROOT
    / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
)
SOURCE_CHECKPOINT = BASE_RESULT_ROOT / "inputs/P0A_round25_source.pth"
SOURCE_WHOLE_FILE_SHA256 = (
    "4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c"
)
SOURCE_ORDERED_FINGERPRINT = (
    "c0379777c63c7a5b0910e5d2ab8a1c37869692db14fd940e0a86778c9f5a1769"
)
BATCH_SIZE = 32
SEED = 42


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise RuntimeError(f"FAIL_CLOSED refusing empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in values:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def build_protocol_manifest(
    *,
    source_checkpoint: str | Path,
    source_fingerprint: str,
    data_root: str | Path,
) -> dict[str, Any]:
    return {
        "schema_version": "iotj.c5_labeled_conditional_da.protocol.v1",
        "study_id": "C5-LABELED-CONDITIONAL-DA-PROBE-20260824",
        "purpose": "mechanism_diagnostic_not_table_ii_replacement",
        "source_clients": ["C1", "C2"],
        "target": "C5",
        "dataset_path": str(Path(data_root)),
        "split_protocol": "existing C5 calibration 320 / sealed test 1360",
        "calibration_examples": 320,
        "test_examples": 1360,
        "source_checkpoint": str(Path(source_checkpoint)),
        "source_checkpoint_role": "P0A_round25_pure_FedAvg_LE1_seed42",
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_checkpoint_ordered_fingerprint": str(source_fingerprint),
        "methods": registered_method_specs(),
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "target_test_selection": False,
        "checkpoint_selection": "fixed_step_100_only",
        "early_stopping": False,
        "hyperparameter_search": False,
        "table_ii_mutated": False,
    }


def load_target_calibration_xy(
    client_dir: str | Path, *, batch_size: int, seed: int
) -> tuple[DataLoader, dict[str, Any]]:
    directory = Path(client_dir)
    feature_path = directory / "calibration_features.npy"
    label_path = directory / "calibration_classification_labels.npy"
    features = np.load(feature_path, allow_pickle=False)
    labels = np.load(label_path, allow_pickle=False)
    if (
        features.ndim != 3
        or tuple(features.shape[1:]) not in {(50, 8), (100, 8)}
        or not np.issubdtype(features.dtype, np.number)
        or not np.all(np.isfinite(features))
    ):
        raise RuntimeError(f"FAIL_CLOSED invalid target calibration features: {features.shape}")
    labels = np.asarray(labels).reshape(-1)
    if len(features) != len(labels) or np.any(labels < 0) or np.any(labels >= 4):
        raise RuntimeError("FAIL_CLOSED invalid target calibration class labels")
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.from_numpy(labels.astype(np.int64, copy=False)),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    return loader, {
        "split": "C5_calibration",
        "fields": ["x", "class"],
        "num_examples": int(len(labels)),
        "feature_shape": list(features.shape),
        "feature_sha256": sha256_file(feature_path),
        "class_label_sha256": sha256_file(label_path),
        "phase_access": False,
        "concentration_access": False,
        "target_test_access": False,
    }


def require_all_endpoints(root: str | Path, methods: Iterable[str]) -> list[str]:
    ordered = list(methods)
    missing = [
        method
        for method in ordered
        if not (Path(root) / method / "fixed_endpoint_complete.json").is_file()
    ]
    if missing:
        raise RuntimeError(
            "FAIL_CLOSED all six fixed endpoints required before target test: "
            + ",".join(missing)
        )
    return ordered


def predict_probabilities(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].long()
            logits, _, _ = model(x)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
    if not probabilities:
        raise RuntimeError("FAIL_CLOSED empty evaluation loader")
    return np.concatenate(probabilities), np.concatenate(labels)


def checkpoint_model(
    checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model, _config, payload = load_checkpoint_model(
        str(checkpoint), device, BATCH_SIZE
    )
    return model, payload


def comparison_row(
    *,
    method: str,
    target_metrics: dict[str, Any],
    source_metrics: dict[str, Any],
    seconds: float,
    checkpoint: Path,
) -> dict[str, Any]:
    specs = registered_method_specs()
    spec = specs[method]
    display = {
        "coral_tce": "CORAL+TCE",
        "mmd_tce": "MMD+TCE",
        "dann_tce": "DANN+TCE",
        "cc_coral": "CC-CORAL",
        "class_mmd": "Class-MMD",
        "cc_dann": "CC-DANN",
    }[method]
    return {
        "experiment_id": f"C5LCD-{method.upper().replace('_', '-')}-S42",
        "method_key": method,
        "method": display,
        "target": "C5",
        "group": (
            "supervised_global_alignment"
            if spec["target_ce_weight"] > 0.0
            else "true_class_conditional_no_target_ce"
        ),
        "accuracy": target_metrics["accuracy"],
        "macro_f1": target_metrics["macro_f1"],
        "nll": target_metrics["nll"],
        "ece": target_metrics["ece"],
        "num_examples": target_metrics["num_examples"],
        "per_class": json.dumps(target_metrics["per_class"], separators=(",", ":")),
        "confusion_matrix": json.dumps(
            target_metrics["confusion_matrix"], separators=(",", ":")
        ),
        "source_accuracy": source_metrics["accuracy"],
        "source_macro_f1": source_metrics["macro_f1"],
        "source_target_f1_gap": source_metrics["macro_f1"]
        - target_metrics["macro_f1"],
        "commissioning_seconds": float(seconds),
        "optimizer": "Adam",
        "optimizer_lr": 5e-4,
        "steps": 100,
        "alignment_weight": 0.5,
        "target_ce_weight": spec["target_ce_weight"],
        "alignment_scope": spec["alignment_scope"],
        "target_label_role": spec["target_label_role"],
        "seed": 42,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_checkpoint_sha256": SOURCE_WHOLE_FILE_SHA256,
        "checkpoint_selection": "fixed_step_100",
        "target_test_selection": False,
        "calculation_status": "recomputed",
    }


def reference_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    table_path = BASE_RESULT_ROOT / "classification_main_comparison.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    for method in ("CORAL", "MMD", "DANN", "GAPS"):
        selected = [
            row
            for row in table
            if row.get("method") == method and row.get("target_id") == "C5"
        ]
        if len(selected) != 1:
            raise RuntimeError(f"FAIL_CLOSED missing unique C5 reference: {method}")
        row = selected[0]
        output.append(
            {
                "method": method,
                "reference_role": (
                    "existing_x_only_E2" if method != "GAPS" else "existing_GAPS_E3"
                ),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "nll": float(row["nll"]),
                "ece": float(row["ece"]),
                "num_examples": int(row["num_examples"]),
                "source_path": str(table_path),
                "calculation_status": "reported_read_only",
            }
        )
    target_ce_path = (
        REPO_ROOT
        / "docs/experiments/iotj_p0_zero_label_commissioning_20260803"
        / "zero_label_commissioning_comparison.csv"
    )
    with target_ce_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["method"] == "simple_target_ce"]
    if len(selected) != 1:
        raise RuntimeError("FAIL_CLOSED missing unique Simple Target-CE reference")
    row = selected[0]
    output.append(
        {
            "method": "Simple Target-CE",
            "reference_role": "existing_target_ce_only",
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "nll": float(row["nll"]),
            "ece": float(row["ece"]),
            "num_examples": int(row["num_examples"]),
            "source_path": str(target_ce_path),
            "calculation_status": "reported_read_only",
        }
    )
    return output


def render_analysis(new_rows: list[dict[str, Any]], refs: list[dict[str, Any]]) -> str:
    by_new = {row["method_key"]: row for row in new_rows}
    by_ref = {row["method"]: row for row in refs}
    lines = [
        "# C5 Labeled and Class-Conditional DA Diagnostic",
        "",
        "## Scope",
        "",
        "All six branches independently reload the same P0A round25 source checkpoint. "
        "This is a seed42 mechanism diagnostic and does not replace the x-only TABLE II rows.",
        "",
        "## Fixed-endpoint results",
        "",
        "| Method | Accuracy (%) | Macro-F1 (%) | NLL | ECE | Source F1 (%) | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in new_rows:
        lines.append(
            f"| {row['method']} | {100*row['accuracy']:.3f} | "
            f"{100*row['macro_f1']:.3f} | {row['nll']:.4f} | {row['ece']:.4f} | "
            f"{100*row['source_macro_f1']:.3f} | {row['commissioning_seconds']:.2f} |"
        )
    lines.extend(["", "## Descriptive mechanism deltas", ""])
    pairs = [
        ("coral_tce", "CORAL", "global alignment + target CE"),
        ("mmd_tce", "MMD", "global alignment + target CE"),
        ("dann_tce", "DANN", "global alignment + target CE"),
        ("cc_coral", "CORAL", "true-class conditioning without target CE"),
        ("class_mmd", "MMD", "true-class conditioning without target CE"),
        ("cc_dann", "DANN", "true-class conditioning without target CE"),
    ]
    for key, baseline, explanation in pairs:
        new = by_new[key]
        old = by_ref[baseline]
        lines.append(
            f"- **{new['method']} vs {baseline}:** "
            f"Accuracy {100*(new['accuracy']-old['accuracy']):+.3f} pp; "
            f"Macro-F1 {100*(new['macro_f1']-old['macro_f1']):+.3f} pp "
            f"({explanation})."
        )
    target_ce = by_ref["Simple Target-CE"]
    lines.extend(["", "## Target-CE control", ""])
    for key in ("coral_tce", "mmd_tce", "dann_tce"):
        row = by_new[key]
        lines.append(
            f"- {row['method']} minus Simple Target-CE: "
            f"{100*(row['accuracy']-target_ce['accuracy']):+.3f} pp Accuracy, "
            f"{100*(row['macro_f1']-target_ce['macro_f1']):+.3f} pp Macro-F1."
        )
    best = max(new_rows, key=lambda row: row["macro_f1"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The highest new single-seed C5 Macro-F1 is {best['method']} "
            f"({100*best['macro_f1']:.3f}%). These values are descriptive; "
            "one seed cannot establish stability or statistical significance. "
            "Because target labels are used by all six diagnostics, they are not "
            "unsupervised baselines and must remain outside the canonical E2 rows.",
            "",
        ]
    )
    return "\n".join(lines)


def execute() -> Path:
    if RESULT_ROOT.exists():
        raise FileExistsError(f"FAIL_CLOSED result destination exists: {RESULT_ROOT}")
    if sha256_file(SOURCE_CHECKPOINT) != SOURCE_WHOLE_FILE_SHA256:
        raise RuntimeError("FAIL_CLOSED source checkpoint whole-file SHA mismatch")
    RESULT_ROOT.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_model, _source_payload = checkpoint_model(SOURCE_CHECKPOINT, device)
    source_fingerprint = ordered_state_content_fingerprint(source_model.state_dict())
    if source_fingerprint != SOURCE_ORDERED_FINGERPRINT:
        raise RuntimeError("FAIL_CLOSED source checkpoint ordered fingerprint mismatch")
    protocol = build_protocol_manifest(
        source_checkpoint=SOURCE_CHECKPOINT,
        source_fingerprint=source_fingerprint,
        data_root=DATA_ROOT,
    )
    write_json(RESULT_ROOT / "protocol_manifest.json", protocol)

    access_events: list[dict[str, Any]] = []
    methods = list(registered_method_specs())
    trained_models: dict[str, torch.nn.Module] = {}
    seconds_by_method: dict[str, float] = {}
    for method in methods:
        run_dir = RESULT_ROOT / method
        run_dir.mkdir()
        target_loader, target_summary = load_target_calibration_xy(
            DATA_ROOT / "client_5", batch_size=BATCH_SIZE, seed=SEED
        )
        if target_summary["num_examples"] != 320:
            raise RuntimeError("FAIL_CLOSED C5 calibration count is not 320")
        access_events.append(
            {
                "sequence": len(access_events) + 1,
                "method": method,
                "stage": "adaptation",
                **target_summary,
            }
        )
        source_loader = create_merged_calibration_loader(
            [DATA_ROOT / "client_1", DATA_ROOT / "client_2"],
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
        model, _payload = checkpoint_model(SOURCE_CHECKPOINT, device)
        if ordered_state_content_fingerprint(model.state_dict()) != source_fingerprint:
            raise RuntimeError("FAIL_CLOSED method did not reload original source checkpoint")
        adapted, diagnostics, seconds = run_labeled_conditional_da(
            method,
            model,
            source_loader,
            target_loader,
            device,
            num_steps=100,
            model_lr=5e-4,
            alignment_weight=0.5,
            expected_source_fingerprint=source_fingerprint,
            seed=SEED,
            formal=True,
        )
        checkpoint = run_dir / "adapted_step_100.pth"
        torch.save(
            {
                "model_state": adapted.state_dict(),
                "source_checkpoint_ordered_fingerprint": source_fingerprint,
                "source_checkpoint_sha256": SOURCE_WHOLE_FILE_SHA256,
                "experiment_id": f"C5LCD-{method.upper().replace('_', '-')}-S42",
                "method": method,
                "step": 100,
                "seed": SEED,
            },
            checkpoint,
        )
        write_csv(run_dir / "adaptation_diagnostics.csv", diagnostics)
        write_json(
            run_dir / "run_manifest.json",
            {
                "method": method,
                "spec": registered_method_specs()[method],
                "source_checkpoint_sha256": SOURCE_WHOLE_FILE_SHA256,
                "source_checkpoint_ordered_fingerprint": source_fingerprint,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "adaptation_seconds": seconds,
                "target_calibration": target_summary,
                "target_test_opened": False,
            },
        )
        write_json(
            run_dir / "fixed_endpoint_complete.json",
            {
                "method": method,
                "fixed_endpoint": 100,
                "checkpoint_sha256": sha256_file(checkpoint),
                "target_test_opened": False,
            },
        )
        trained_models[method] = adapted
        seconds_by_method[method] = seconds

    require_all_endpoints(RESULT_ROOT, methods)
    access_events.append(
        {
            "sequence": len(access_events) + 1,
            "stage": "final_evaluation_release",
            "all_six_endpoints_complete": True,
            "split": "C5_test",
            "fields": ["x", "class"],
            "purpose": "fixed_endpoint_metrics_only",
            "target_test_selection": False,
        }
    )
    target_test_loader = create_client_test_only_loader(
        DATA_ROOT / "client_5", batch_size=BATCH_SIZE, num_workers=0
    )
    source_test_loader = create_merged_test_loader(
        [DATA_ROOT / "client_1", DATA_ROOT / "client_2"],
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method in methods:
        model = trained_models[method]
        target_probs, target_labels = predict_probabilities(
            model, target_test_loader, device
        )
        source_probs, source_labels = predict_probabilities(
            model, source_test_loader, device
        )
        if len(target_labels) != 1360:
            raise RuntimeError("FAIL_CLOSED C5 test count is not 1360")
        target_metrics = classification_metrics(target_probs, target_labels)
        source_metrics = classification_metrics(source_probs, source_labels)
        checkpoint = RESULT_ROOT / method / "adapted_step_100.pth"
        row = comparison_row(
            method=method,
            target_metrics=target_metrics,
            source_metrics=source_metrics,
            seconds=seconds_by_method[method],
            checkpoint=checkpoint,
        )
        rows.append(row)
        for index, (truth, probs) in enumerate(zip(target_labels, target_probs)):
            prediction_rows.append(
                {
                    "method": row["method"],
                    "method_key": method,
                    "target": "C5",
                    "window_index": index,
                    "true_class": int(truth),
                    "predicted_class": int(np.argmax(probs)),
                    "confidence": float(np.max(probs)),
                    **{
                        f"prob_class_{class_id}": float(probs[class_id])
                        for class_id in range(4)
                    },
                }
            )
        write_json(
            RESULT_ROOT / method / "final_evaluation_C5.json",
            {
                "comparison_row": row,
                "source_metrics": source_metrics,
                "target_metrics": target_metrics,
                "selection_role": "none_fixed_endpoint_only",
            },
        )

    write_csv(RESULT_ROOT / "c5_labeled_conditional_da_comparison.csv", rows)
    write_csv(RESULT_ROOT / "per_window_predictions.csv", prediction_rows)
    refs = reference_rows()
    write_csv(RESULT_ROOT / "read_only_reference_comparison.csv", refs)
    write_csv(
        RESULT_ROOT / "c5_full_diagnostic_comparison.csv",
        [
            {
                "method": row["method"],
                "role": row["group"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "nll": row["nll"],
                "ece": row["ece"],
                "num_examples": row["num_examples"],
                "calculation_status": "recomputed",
            }
            for row in rows
        ]
        + [
            {
                "method": row["method"],
                "role": row["reference_role"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "nll": row["nll"],
                "ece": row["ece"],
                "num_examples": row["num_examples"],
                "calculation_status": row["calculation_status"],
            }
            for row in refs
        ],
    )
    write_json(
        RESULT_ROOT / "target_access_audit.json",
        {
            "status": "PASS",
            "events": access_events,
            "calibration_fields": ["x", "class"],
            "test_opened_only_after_all_six_fixed_endpoints": True,
            "target_test_used_for_selection": False,
            "phase_or_concentration_loaded_for_adaptation": False,
        },
    )
    (RESULT_ROOT / "RESULT_ANALYSIS.md").write_text(
        render_analysis(rows, refs), encoding="utf-8"
    )
    (RESULT_ROOT / "EXPERIMENT_AUDIT.md").write_text(
        """# Experiment Audit

## Verdict: PASS for a single-seed supplementary mechanism diagnostic

- All six branches independently reloaded the exact P0A round25 checkpoint.
- C1/C2 source calibration, C5 calibration identities, 100 steps, Adam 5e-4,
  alignment weight 0.5, seed42, and fixed step100 were held constant.
- C5 class labels were intentionally available during adaptation. TCE branches
  used them in target CE; conditional branches used them only for grouping.
- Target phase and concentration were not loaded by the target adaptation API.
- The sealed C5 test was opened only after all six fixed endpoints existed and
  was not used for tuning, stopping, thresholding, or checkpoint selection.
- Existing x-only E2, Simple Target-CE, and GAPS rows were copied read-only.
- Limitation: seed42 only; results are descriptive and do not establish
  statistical significance or replace the canonical unsupervised TABLE II rows.
""",
        encoding="utf-8",
    )
    (RESULT_ROOT / "LABEL_ACCESS_AUDIT.md").write_text(
        """# Label Access Audit

- Target calibration API receives exactly two tensors: C5 calibration x and class.
- CORAL+TCE, MMD+TCE, and DANN+TCE activate target CE with fixed weight 1.0.
- CC-CORAL, Class-MMD, and CC-DANN set target CE structurally inactive; class
  labels only construct true-class masks for the alignment objective.
- CC-DANN uses one shared discriminator and per-shared-class GRL/binary-BCE.
- Target phase, concentration, pseudo labels, and test labels are unavailable to
  adaptation. Test labels are opened only for final fixed-endpoint evaluation.
""",
        encoding="utf-8",
    )
    index = {}
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path.name != "sha256_index.json":
            index[path.relative_to(RESULT_ROOT).as_posix()] = sha256_file(path)
    write_json(RESULT_ROOT / "sha256_index.json", index)
    write_json(
        RESULT_ROOT / "COMPLETE.json",
        {
            "status": "PASS",
            "study_id": protocol["study_id"],
            "method_count": len(rows),
            "target_test_examples": 1360,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256_index_sha256": sha256_file(RESULT_ROOT / "sha256_index.json"),
        },
    )
    return RESULT_ROOT


def inspect() -> dict[str, Any]:
    return {
        "source_checkpoint_exists": SOURCE_CHECKPOINT.is_file(),
        "source_checkpoint_sha256": (
            sha256_file(SOURCE_CHECKPOINT) if SOURCE_CHECKPOINT.is_file() else None
        ),
        "data_root_exists": DATA_ROOT.is_dir(),
        "output_exists": RESULT_ROOT.exists(),
        "methods": list(registered_method_specs()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inspect", "run"))
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect(), indent=2, sort_keys=True))
        return
    started = time.perf_counter()
    output = execute()
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "wall_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

