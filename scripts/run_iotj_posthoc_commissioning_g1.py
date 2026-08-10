"""Run Gate 1 true post-hoc C5 commissioning from one source-only endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated_dataset import GasSensorWindowDataset  # noqa: E402
from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays  # noqa: E402
from gaps_flower.evaluate_checkpoint import load_checkpoint_model  # noqa: E402
from gaps_flower.posthoc_commissioning import (  # noqa: E402
    BATCH_SIZE,
    LR,
    SEED,
    STEPS,
    a4_posthoc_adapt,
    load_calibration_identity_manifest,
    ordered_state_fingerprint,
    sha256_file,
    supervised_ce_adapt,
)
from scripts.summarize_iotj_classification_ablation import (  # noqa: E402
    classification_metrics,
    evaluate_checkpoint_stream,
)


DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
SOURCE_RUN = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/source_fl/CAN-V1-CMP-FEDAVG"
SOURCE_CHECKPOINT = SOURCE_RUN / "remote_server/server_latest.pth"
HISTORICAL = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/classification_comparison/canonical_classification_comparison.csv"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate1_posthoc"
METHODS = ("a0t_full", "a4", "target_head")
DISPLAY = {
    "source_only": "Source-only",
    "a0t_full": "Posthoc A0T-full",
    "a4": "Posthoc A4",
    "target_head": "Posthoc Target-head",
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class _RSSMonitor:
    def __init__(self) -> None:
        try:
            import psutil

            self.process = psutil.Process(os.getpid())
        except ImportError:
            self.process = None
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        if self.process is not None:
            self.peak = int(self.process.memory_info().rss)

            def sample() -> None:
                while not self.stop_event.wait(0.01):
                    self.peak = max(self.peak, int(self.process.memory_info().rss))

            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.process is not None:
            self.peak = max(self.peak, int(self.process.memory_info().rss))


def _loader(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    limit: int | None,
) -> DataLoader:
    features, classes, phases = arrays
    dataset = GasSensorWindowDataset(
        features,
        np.zeros((len(features), 4), dtype=np.float32),
        classes,
        phases,
        normalize=False,
        mean_std=None,
    )
    if limit is not None and len(dataset) > limit:
        indices = np.random.RandomState(SEED).choice(len(dataset), size=limit, replace=False)
        dataset = Subset(dataset, indices)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    method: str,
    source_sha: str,
    source_state_sha: str,
) -> None:
    torch.save(
        {
            "step": STEPS,
            "model_state": model.state_dict(),
            "method": method,
            "source_checkpoint_sha256": source_sha,
            "source_state_fingerprint": source_state_sha,
            "seed": SEED,
        },
        path,
    )


def verify_adaptation_gate(output: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_dir = output / method
        marker = method_dir / "fixed_endpoint_complete.json"
        manifest = method_dir / "run_manifest.json"
        if not marker.is_file() or not manifest.is_file():
            raise RuntimeError(f"missing endpoint for {method}")
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        checkpoint = Path(manifest_payload["checkpoint"])
        if int(marker_payload.get("step", -1)) != STEPS:
            raise RuntimeError(f"invalid endpoint step for {method}")
        if marker_payload.get("target_test_opened") is not False:
            raise RuntimeError(f"target test opened before endpoint lock: {method}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != manifest_payload["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {method}")
        if manifest_payload.get("source_state_fingerprint") != marker_payload.get("source_state_fingerprint"):
            raise RuntimeError(f"source state fingerprint mismatch for {method}")
        result[method] = manifest_payload
    fingerprints = {item["source_state_fingerprint"] for item in result.values()}
    if len(fingerprints) != 1:
        raise RuntimeError("post-hoc methods did not reload the same source state")
    return result


def decide_gate1(
    *,
    source_f1: float,
    a0t_f1: float,
    a4_f1: float,
    head_f1: float,
    a0t_retention: float,
    a4_retention: float,
    head_retention: float,
    a0t_seconds: float,
    a4_seconds: float,
    head_seconds: float,
    a0t_trainable: int,
    head_trainable: int,
    historical_a0t: float,
    historical_a4: float,
) -> dict[str, Any]:
    best = max(a0t_f1, a4_f1, head_f1)
    if best >= 0.95 and best - source_f1 >= 0.05:
        lifecycle = "POSTHOC_LIFECYCLE_SUPPORTED"
    elif best - source_f1 >= 0.05 or best >= max(historical_a0t, historical_a4) - 0.05:
        lifecycle = "POSTHOC_LIFECYCLE_WEAK"
    else:
        lifecycle = "POSTHOC_LIFECYCLE_FAILED"
    a4_keep = (a4_f1 - a0t_f1 >= 0.005) or (
        abs(a4_f1 - a0t_f1) <= 0.005 and a4_retention - a0t_retention >= 0.01
    )
    head_promising = (
        a0t_f1 - head_f1 <= 0.005
        and head_trainable < a0t_trainable
        and head_retention >= a0t_retention
        and head_seconds < a0t_seconds
    )
    risk = (
        historical_a0t - a0t_f1 > 0.05
        and historical_a4 - a4_f1 > 0.05
        and max(historical_a0t, historical_a4) - head_f1 > 0.05
    )
    return {
        "lifecycle": lifecycle,
        "a4": "KEEP" if a4_keep else "RETIRE_AS_CORE",
        "target_head": "PROMISING" if head_promising else "NOT_COMPETITIVE",
        "interleaved_dependency_risk": bool(risk),
    }


def _historical() -> dict[str, float]:
    with HISTORICAL.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, float] = {}
    for method in ("A0T", "GAPS/A4"):
        matches = [row for row in rows if row["method"] == method and row["target"] == "C5"]
        if len(matches) != 1:
            raise RuntimeError(f"historical C5 row unavailable: {method}")
        result[method] = float(matches[0]["macro_f1"])
    return result


def _probabilities(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"prob_{class_id}"]) for class_id in range(4)] for row in rows],
        dtype=np.float64,
    )


def _evaluate_one(checkpoint: Path, endpoint: tuple[str, int], device: torch.device):
    by_scope: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for client in (1, 2, 5):
        rows, metrics = evaluate_checkpoint_stream(
            checkpoint,
            data_root=DATA_ROOT,
            target_client=client,
            split="test",
            device=device,
            batch_size=BATCH_SIZE,
            expected_endpoint=endpoint,
        )
        by_scope[f"C{client}"] = (rows, metrics)
    merged_rows = by_scope["C1"][0] + by_scope["C2"][0]
    by_scope["C1+C2"] = (
        merged_rows,
        classification_metrics(
            [int(row["true_class"]) for row in merged_rows],
            _probabilities(merged_rows),
        ),
    )
    return by_scope


def _adapt_all(output: Path, device: torch.device) -> None:
    source_model, _config, source_container = load_checkpoint_model(
        str(SOURCE_CHECKPOINT), device, BATCH_SIZE
    )
    if int(source_container.get("round", -1)) != 25:
        raise RuntimeError("source endpoint is not round 25")
    source_sha = sha256_file(SOURCE_CHECKPOINT)
    source_state_sha = ordered_state_fingerprint(source_container["model_state"])
    calibration_manifest = DATA_ROOT / "client_5/calibration_experiment_info.json"
    load_calibration_identity_manifest(
        calibration_manifest, expected_client=5, expected_count=320
    )
    source_arrays = load_domain_adaptation_arrays(
        [DATA_ROOT / "client_1", DATA_ROOT / "client_2"],
        strict=True,
        expected_window_shape=(50, 8),
    )
    target_arrays = load_domain_adaptation_arrays(
        [DATA_ROOT / "client_5"], strict=True, expected_window_shape=(50, 8)
    )
    for method in METHODS:
        method_dir = output / method
        method_dir.mkdir()
        source_loader = _loader(source_arrays, limit=500)
        target_loader = _loader(target_arrays, limit=None)
        with _RSSMonitor() as monitor:
            if method == "a4":
                adapted, diagnostics, system = a4_posthoc_adapt(
                    source_model,
                    source_loader,
                    target_loader,
                    device=device,
                    steps=STEPS,
                    seed=SEED,
                )
            else:
                adapted, diagnostics, system = supervised_ce_adapt(
                    source_model,
                    target_loader,
                    method=method,
                    device=device,
                    steps=STEPS,
                    lr=LR,
                    seed=SEED,
                )
        system["peak_rss_bytes"] = int(monitor.peak) if monitor.process is not None else None
        checkpoint = method_dir / f"posthoc_{method}_c5.pth"
        _save_checkpoint(
            checkpoint,
            adapted,
            method=method,
            source_sha=source_sha,
            source_state_sha=source_state_sha,
        )
        system["checkpoint_bytes"] = checkpoint.stat().st_size
        system["checkpoint_sha256"] = sha256_file(checkpoint)
        _write_csv(method_dir / "adaptation_diagnostics.csv", diagnostics)
        _json(method_dir / "system_metrics.json", system)
        manifest = {
            "schema_version": "iotj.canonical_v1.posthoc_g1.v1",
            "experiment_id": f"CAN-V1-MR-G1-{method.upper().replace('_', '-')}",
            "method": method,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": system["checkpoint_sha256"],
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": source_sha,
            "source_state_fingerprint": source_state_sha,
            "calibration_manifest": str(calibration_manifest.resolve()),
            "calibration_manifest_sha256": sha256_file(calibration_manifest),
            "calibration_count": 320,
            "target_test_opened": False,
            "steps": STEPS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "system_metrics": system,
        }
        _json(method_dir / "run_manifest.json", manifest)
        _json(
            method_dir / "fixed_endpoint_complete.json",
            {
                "status": "COMPLETE",
                "method": method,
                "step": STEPS,
                "checkpoint_sha256": system["checkpoint_sha256"],
                "source_state_fingerprint": source_state_sha,
                "target_test_opened": False,
            },
        )


def _evaluate_and_analyze(output: Path, device: torch.device) -> dict[str, Any]:
    gate = verify_adaptation_gate(output)
    test_manifest = DATA_ROOT / "client_5/test_experiment_info.json"
    _json(
        output / "SEALED_TEST_OPEN.json",
        {
            "status": "OPENED_AFTER_ALL_G1_ENDPOINTS_LOCKED",
            "opened_at_unix": time.time(),
            "target_test_manifest_sha256": sha256_file(test_manifest),
            "selection_performed": False,
        },
    )
    checkpoints = {"source_only": SOURCE_CHECKPOINT}
    checkpoints.update({method: Path(gate[method]["checkpoint"]) for method in METHODS})
    metric_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    metrics_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for method, checkpoint in checkpoints.items():
        endpoint = ("round", 25) if method == "source_only" else ("step", STEPS)
        evaluated = _evaluate_one(checkpoint, endpoint, device)
        metrics_by_method[method] = {}
        for scope, (rows, metrics) in evaluated.items():
            metrics_by_method[method][scope] = metrics
            metric_rows.append(
                {
                    "method": DISPLAY[method],
                    "scope": scope,
                    "N": metrics["N"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "nll": metrics["nll"],
                    "ece": metrics["ece"],
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
            for class_id in range(4):
                per_class_rows.append(
                    {
                        "method": DISPLAY[method],
                        "scope": scope,
                        "class_id": class_id,
                        "recall": metrics["per_class_recall"][str(class_id)],
                        "f1": metrics["per_class_f1"][str(class_id)],
                    }
                )
            if scope != "C1+C2":
                for row in rows:
                    prediction_rows.append({"method": DISPLAY[method], **row})
    _write_csv(output / "classification_metrics.csv", metric_rows)
    _write_csv(output / "POSTHOC_COMMISSIONING_PER_CLASS.csv", per_class_rows)
    _write_csv(output / "classification_predictions.csv", prediction_rows)

    source_c1 = metrics_by_method["source_only"]["C1"]["macro_f1"]
    source_c2 = metrics_by_method["source_only"]["C2"]["macro_f1"]
    retention: dict[str, float] = {}
    for method in METHODS:
        retention[method] = float(
            (
                metrics_by_method[method]["C1"]["macro_f1"] - source_c1
                + metrics_by_method[method]["C2"]["macro_f1"] - source_c2
            )
            / 2.0
        )
    systems = {
        method: json.loads((output / method / "system_metrics.json").read_text(encoding="utf-8"))
        for method in METHODS
    }
    main_rows: list[dict[str, Any]] = []
    for method in ("source_only", *METHODS):
        system = systems.get(method, {})
        main_rows.append(
            {
                "Method": DISPLAY[method],
                "C1 F1": metrics_by_method[method]["C1"]["macro_f1"],
                "C2 F1": metrics_by_method[method]["C2"]["macro_f1"],
                "C1+C2 F1": metrics_by_method[method]["C1+C2"]["macro_f1"],
                "C5 F1": metrics_by_method[method]["C5"]["macro_f1"],
                "C5 Accuracy": metrics_by_method[method]["C5"]["accuracy"],
                "C5 NLL": metrics_by_method[method]["C5"]["nll"],
                "C5 ECE": metrics_by_method[method]["C5"]["ece"],
                "Trainable params": system.get("trainable_parameters", 0),
                "Trainable param ratio": system.get("trainable_parameter_ratio", 0.0),
                "Adaptation time": system.get("adaptation_seconds", 0.0),
                "Peak RSS bytes": system.get("peak_rss_bytes", ""),
                "Checkpoint bytes": system.get("checkpoint_bytes", SOURCE_CHECKPOINT.stat().st_size),
                "Parameter delta": system.get("relative_parameter_displacement", 0.0),
                "Mean C1/C2 retention delta": retention.get(method, 0.0),
            }
        )
    _write_csv(output / "POSTHOC_COMMISSIONING_COMPARISON.csv", main_rows)
    historical = _historical()
    decision = decide_gate1(
        source_f1=metrics_by_method["source_only"]["C5"]["macro_f1"],
        a0t_f1=metrics_by_method["a0t_full"]["C5"]["macro_f1"],
        a4_f1=metrics_by_method["a4"]["C5"]["macro_f1"],
        head_f1=metrics_by_method["target_head"]["C5"]["macro_f1"],
        a0t_retention=retention["a0t_full"],
        a4_retention=retention["a4"],
        head_retention=retention["target_head"],
        a0t_seconds=systems["a0t_full"]["adaptation_seconds"],
        a4_seconds=systems["a4"]["adaptation_seconds"],
        head_seconds=systems["target_head"]["adaptation_seconds"],
        a0t_trainable=systems["a0t_full"]["trainable_parameters"],
        head_trainable=systems["target_head"]["trainable_parameters"],
        historical_a0t=historical["A0T"],
        historical_a4=historical["GAPS/A4"],
    )
    analysis_lines = [
        "# Gate 1 Post-hoc Commissioning Analysis",
        "",
        f"- Lifecycle verdict: `{decision['lifecycle']}`.",
        f"- A4 verdict: `{decision['a4']}`.",
        f"- Target-head verdict: `{decision['target_head']}`.",
        f"- Interleaved dependency risk: `{decision['interleaved_dependency_risk']}`.",
        "",
        "Operational source preservation is guaranteed because the source checkpoint remains immutable; source retention numbers here are only a diagnostic for hypothetical sharing of the personalized checkpoint.",
        "",
        "The A4 run preserves all registered coefficients. Client prototypes, client residuals, and interleaved state are unavailable from a CE-only source endpoint and were not fabricated; their associated post-hoc losses are therefore inactive when inputs are absent.",
        "",
        "Historical interleaved reference (not a single-factor comparison):",
        f"- A0T C5 Macro-F1: {historical['A0T']:.9f}",
        f"- A4 C5 Macro-F1: {historical['GAPS/A4']:.9f}",
    ]
    (output / "POSTHOC_G1_ANALYSIS.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    if decision["interleaved_dependency_risk"]:
        (output / "POSTHOC_UPDATE_BUDGET_DIAGNOSTIC_PROPOSAL.md").write_text(
            "# Post-hoc Update-budget Diagnostic Proposal\n\n"
            "The fixed 100-step post-hoc endpoints are all more than five percentage points below the corresponding interleaved references. A separately authorized 2500-step update-budget-matched diagnostic may test whether exposure timing or total update count explains the gap. This study did not run that diagnostic and did not tune any setting.\n",
            encoding="utf-8",
        )
    _json(output / "G1_DECISION.json", decision)
    _json(
        output / "protocol_manifest.json",
        {
            "status": "PASS",
            "source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
            "source_state_fingerprint": next(iter(gate.values()))["source_state_fingerprint"],
            "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
            "calibration_manifest_sha256": sha256_file(DATA_ROOT / "client_5/calibration_experiment_info.json"),
            "target_test_manifest_sha256": sha256_file(test_manifest),
            "target_test_opened_after_gate": True,
            "target_test_selection": False,
            "historical_interleaved": historical,
            "decision": decision,
        },
    )
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "sha256_index.json")
    _json(output / "sha256_index.json", {str(path.relative_to(output)): sha256_file(path) for path in files})
    return decision


def run(output: Path, device: torch.device) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED Gate-1 output exists: {output}")
    output.mkdir(parents=True)
    source_manifest = json.loads((SOURCE_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    if source_manifest["checkpoint_sha256"] != sha256_file(SOURCE_CHECKPOINT):
        raise RuntimeError("source checkpoint provenance mismatch")
    if source_manifest["protocol"].get("target_x") is not False or source_manifest["protocol"].get("target_y") is not False:
        raise RuntimeError("source checkpoint is not source-only")
    _adapt_all(output, device)
    decision = _evaluate_and_analyze(output, device)
    return {"status": "PASS", "output": str(output), **decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    print(json.dumps(run(args.output, device), indent=2))


if __name__ == "__main__":
    main()
