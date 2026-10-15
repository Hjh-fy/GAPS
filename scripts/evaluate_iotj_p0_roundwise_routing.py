"""Post-hoc P0 routing evaluation from 25 frozen P0-A source checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from federated_dataset import GasSensorWindowDataset, create_merged_calibration_loader  # noqa: E402
from gaps_flower.domain_adaptation import ServerDomainAdaptation  # noqa: E402
from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays  # noqa: E402
from gaps_flower.evaluate_checkpoint import evaluate_classification, load_checkpoint_model, make_loader  # noqa: E402
from utils import set_random_seed  # noqa: E402

ROUNDS = 25
SEED = 42
STEPS = 100
LR = 5e-4
BATCH_SIZE = 32
METHODS = ("source_only", "simple_target_ce", "full_target_adapter")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def enrich(metrics: dict[str, Any]) -> dict[str, Any]:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    f1s, recalls = [], {}
    for class_id in range(confusion.shape[0]):
        tp = int(confusion[class_id, class_id]); fn = int(confusion[class_id].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls[str(class_id)] = recall
    return {**metrics, "macro_f1": float(np.mean(f1s)), "per_class_recall": recalls}


def evaluate(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    result = enrich(evaluate_classification(model, test_loader, device, 4, 15))
    if result["num_examples"] != 1360 or not all(math.isfinite(float(result[k])) for k in ("accuracy", "macro_f1", "nll", "ece")):
        raise RuntimeError("FAIL_CLOSED invalid C5 test metrics")
    return result


def metric_row(round_id: int, method: str, metrics: dict[str, Any], checkpoint: Path, elapsed: float) -> dict[str, Any]:
    return {
        "source_round": round_id, "method": method,
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "nll": metrics["nll"], "ece": metrics["ece"],
        "num_examples": metrics["num_examples"],
        "per_class_recall": json.dumps(metrics["per_class_recall"], sort_keys=True),
        "checkpoint_sha256": sha256_file(checkpoint),
        "commissioning_steps": 0 if method == "source_only" else STEPS,
        "commissioning_lr": 0.0 if method == "source_only" else LR,
        "commissioning_time_seconds": elapsed,
        "target_label_access": "none" if method == "source_only" else "C5_calibration_320_labels",
        "selection_role": "post_hoc_diagnostic_only",
    }


def target_loader(data_root: Path, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    base = create_merged_calibration_loader(
        [data_root / "client_5"], batch_size=BATCH_SIZE, num_workers=0,
    )
    return DataLoader(
        base.dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def simple_commission(model: torch.nn.Module, data_root: Path, device: torch.device, source_round: int) -> tuple[torch.nn.Module, list[dict], float]:
    set_random_seed(SEED)
    adapted = copy.deepcopy(model).to(device)
    for parameter in adapted.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(adapted.parameters(), lr=LR)
    loader = target_loader(data_root, shuffle=True); iterator = iter(loader)
    rows, started = [], time.perf_counter(); adapted.train()
    for step in range(1, STEPS + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        x, y = batch[0].to(device), batch[1].to(device).long()
        optimizer.zero_grad(); logits, _, _ = adapted(x); loss = F.cross_entropy(logits, y)
        rows.append({"source_round": source_round, "step": step, "target_calibration_ce": float(loss.detach().item()), "target_calibration_accuracy": float((logits.detach().argmax(1) == y).float().mean().item())})
        loss.backward(); torch.nn.utils.clip_grad_norm_(adapted.parameters(), 5.0); optimizer.step()
    return adapted, rows, time.perf_counter() - started


def da_loader(arrays: tuple[np.ndarray, np.ndarray, np.ndarray], *, sample_limit: int = 500) -> DataLoader:
    features, classes, phases = arrays
    dataset = GasSensorWindowDataset(features, np.zeros((len(features), 4), np.float32), classes, phases, normalize=False, mean_std=None)
    sample_count = min(len(dataset), sample_limit)
    indices = np.random.RandomState(SEED).choice(len(dataset), size=sample_count, replace=False)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)


def full_hyperparams() -> dict[str, Any]:
    return {
        "USE_DEEP_CORAL": True, "USE_MMD_ALIGNMENT": True, "USE_ADVERSARIAL_DOMAIN": True,
        "MMD_OBJECTIVE": "mmd2", "STAGE_ALIGNMENT": "cross_domain_same_class_phase",
        "ADV_FEATURE_OBJECTIVE": "wasserstein_min", "CORAL_CLASS_CONDITIONAL": True,
        "LAMBDA_DEEP_CORAL": 0.5, "LAMBDA_GLOBAL_MMD": 0.5, "LAMBDA_CLASS_MMD": 0.5,
        "LAMBDA_STAGE_MMD": 0.2, "LAMBDA_ADV_DOMAIN": 0.5, "LAMBDA_PROTO_ANCHOR": 0.3,
        "LAMBDA_PROTO": 0.05, "LAMBDA_CONSISTENCY": 2.0, "LAMBDA_RES": 0.1,
        "LAMBDA_PROTO_MMD": 0.0, "LAMBDA_TARGET_CE": 0.0,
        "USE_ALIGN_REG_LEGACY": False, "LAMBDA_ALIGN_REG_LEGACY": 0.05,
        "USE_CONTRASTIVE_CONSISTENCY": True, "USE_PROTO_MMD": False,
        "USE_PROTO_DECOUPLING": True, "TARGET_CE_LABEL_SMOOTHING": 0.0,
        "TARGET_CE_CLASS_BALANCED": False, "SERVER_OPT_LR": LR, "HIDDEN_DIM2": 64,
        "NUM_CLASSES": 4, "MAX_VAL_BATCHES": 10, "ADV_DOMAIN_LR": 0.001,
        "ADV_CRITIC_ITERS": 3, "ADV_GRADIENT_PENALTY": 10.0,
        "ADV_CLASS_CONDITIONAL": True, "DA_LEARN_SEMANTIC_PROTOS": True,
        "RETURN_STEP_DIAGNOSTICS": True,
    }


def full_commission(model: torch.nn.Module, source_loader: DataLoader, calib_loader: DataLoader, device: torch.device, source_round: int) -> tuple[torch.nn.Module, list[dict], float, dict]:
    set_random_seed(SEED)
    trainer = ServerDomainAdaptation(model, source_loader, calib_loader, {}, device, full_hyperparams())
    started = time.perf_counter()
    adapted, summary = trainer.run_adaptation(num_steps=STEPS, client_mus=[{}, {}], client_counts=[{}, {}], client_weights=torch.tensor([0.5, 0.5], device=device), client_ids=[1, 2], client_residuals=[None, None])
    elapsed = time.perf_counter() - started
    per_step = summary.pop("step_diagnostics")
    rows = []
    names = list(per_step)
    for index in range(STEPS):
        row = {"source_round": source_round, "step": index + 1}
        row.update({name: per_step[name][index] for name in names})
        rows.append(row)
    return adapted, rows, elapsed, summary


def activity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        "coral_loss": (0.5, "weighted_coral_loss"), "mmd_global": (0.5, "weighted_mmd_global"),
        "mmd_class": (0.5, "weighted_mmd_class"), "stage_mmd_loss": (0.2, "weighted_stage_mmd_loss"),
        "adv_loss": (0.5, "weighted_adv_loss"), "proto_anchor": (0.3, "weighted_proto_anchor"),
        "proto_loss": (0.05, "weighted_proto_loss"), "consist_loss": (2.0, "weighted_consist_loss"),
        "residual_loss": (0.1, "weighted_residual_loss"), "mmd_proto_loss": (0.0, "weighted_mmd_proto_loss"),
        "target_ce_loss": (0.0, "weighted_target_ce_loss"),
    }
    result = []
    for raw, (configured_lambda, weighted) in mapping.items():
        raw_values = np.asarray([float(row[raw]) for row in rows]); weighted_values = np.asarray([float(row[weighted]) for row in rows])
        nonzero = int(np.count_nonzero(np.abs(weighted_values) > 1e-12))
        if configured_lambda == 0: status = "ZERO_BY_CONFIGURATION"
        elif nonzero == 0 and raw in {"proto_loss", "consist_loss", "residual_loss"}: status = "ZERO_NO_INPUT_STATISTICS"
        elif nonzero == 0: status = "ZERO_OBSERVED"
        else: status = "ACTIVE"
        result.append({"loss_name": raw, "configured_lambda": configured_lambda, "active_steps": len(rows), "nonzero_steps": nonzero, "mean_raw": float(raw_values.mean()), "mean_weighted": float(weighted_values.mean()), "median_weighted": float(np.median(weighted_values)), "max_abs_weighted": float(np.max(np.abs(weighted_values))), "activity_status": status})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", default="results/iotj_p0_routing_simplification_20260803", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source_run = args.source_run_dir.resolve(); data_root = args.data_root.resolve()
    manifest = json.loads((source_run / "protocol_manifest.json").read_text(encoding="utf-8"))
    if manifest["training"] != {"rounds": 25, "local_epochs": 1, "batch_size": 32, "client_lr": 0.0005, "optimizer": "Adam", "profile": "ce_only", "aggregation": "sample_weighted_FedAvg", "fedprox_mu": 0.0}:
        raise RuntimeError("FAIL_CLOSED P0-A training contract mismatch")
    checkpoint_dir = source_run / "remote_server"
    checkpoints = {r: checkpoint_dir / f"server_round_{r:03d}.pth" for r in range(1, ROUNDS + 1)}
    if not all(path.is_file() for path in checkpoints.values()): raise RuntimeError("FAIL_CLOSED expected exactly 25 source checkpoints")
    output_dir = (args.output_root / "P0B_ROUNDWISE_COMMISSIONING_S42").resolve(); output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    test = make_loader(data_root, 5, "test", BATCH_SIZE)
    source_arrays = load_domain_adaptation_arrays([data_root / "client_1", data_root / "client_2"], strict=True)
    target_arrays = load_domain_adaptation_arrays([data_root / "client_5"], strict=True)
    metrics_rows, simple_rows, full_rows, full_summaries = [], [], [], []

    # All target-test evaluations are retrospective and start only after P0-A completed.
    for round_id, checkpoint in checkpoints.items():
        model, _, _ = load_checkpoint_model(str(checkpoint), device, BATCH_SIZE); started = time.perf_counter()
        metrics_rows.append(metric_row(round_id, "source_only", evaluate(model, test, device), checkpoint, time.perf_counter() - started))
    for round_id, checkpoint in checkpoints.items():
        model, _, _ = load_checkpoint_model(str(checkpoint), device, BATCH_SIZE)
        adapted, diagnostics, elapsed = simple_commission(model, data_root, device, round_id); simple_rows += diagnostics
        if round_id == 25:
            saved = output_dir / "round25_simple_ce_adapted.pth"; torch.save({"round": 25, "model_state": adapted.state_dict(), "source_checkpoint_sha256": sha256_file(checkpoint)}, saved)
        metrics_rows.append(metric_row(round_id, "simple_target_ce", evaluate(adapted, test, device), checkpoint, elapsed))
    for round_id, checkpoint in checkpoints.items():
        model, _, _ = load_checkpoint_model(str(checkpoint), device, BATCH_SIZE)
        source_loader, calib_loader = da_loader(source_arrays), da_loader(target_arrays)
        adapted, diagnostics, elapsed, summary = full_commission(model, source_loader, calib_loader, device, round_id)
        full_rows += diagnostics; full_summaries.append({"source_round": round_id, **summary})
        if round_id == 25:
            saved = output_dir / "round25_full_target_adapter.pth"; torch.save({"round": 25, "model_state": adapted.state_dict(), "source_checkpoint_sha256": sha256_file(checkpoint)}, saved)
        metrics_rows.append(metric_row(round_id, "full_target_adapter", evaluate(adapted, test, device), checkpoint, elapsed))

    metrics_rows.sort(key=lambda row: (int(row["source_round"]), METHODS.index(str(row["method"]))))
    write_csv(output_dir / "roundwise_routing_metrics.csv", metrics_rows)
    write_csv(output_dir / "roundwise_source_only_metrics.csv", [row for row in metrics_rows if row["method"] == "source_only"])
    write_csv(output_dir / "simple_ce_commissioning_diagnostics.csv", simple_rows)
    write_csv(output_dir / "full_da_commissioning_diagnostics.csv", full_rows)
    write_csv(output_dir / "server_loss_activity_summary.csv", activity_summary(full_rows))
    write_csv(output_dir / "full_da_round_summaries.csv", full_summaries)
    round25 = []
    for row in metrics_rows:
        if row["source_round"] != 25: continue
        adapted_path = output_dir / ("round25_simple_ce_adapted.pth" if row["method"] == "simple_target_ce" else "round25_full_target_adapter.pth")
        size_path = checkpoints[25] if row["method"] == "source_only" else adapted_path
        round25.append({key: row[key] for key in ("method", "accuracy", "macro_f1", "nll", "ece", "per_class_recall", "commissioning_time_seconds")} | {"checkpoint_bytes": size_path.stat().st_size})
    write_csv(output_dir / "round25_routing_comparison.csv", round25)
    protocol = {
        "schema_version": "iotj.p0.roundwise_commissioning.v1", "seed": SEED,
        "source_run_manifest": str((source_run / "protocol_manifest.json").resolve()),
        "source_checkpoint_count": 25, "methods": list(METHODS), "commissioning_steps": STEPS,
        "commissioning_lr": LR, "formal_comparison_round": 25,
        "checkpoint_reload_policy": "each method/round starts from matching original source checkpoint",
        "adapted_checkpoint_inheritance": False, "target_test_role": "post_hoc_evaluation_only",
        "target_test_used_for_selection": False,
    }
    (output_dir / "protocol_manifest.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(metrics_rows), "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
