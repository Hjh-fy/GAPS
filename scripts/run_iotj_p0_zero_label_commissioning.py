"""Run the frozen P0-U zero-label target commissioning study."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaps_flower.domain_adaptation import wasserstein_feature_objective
from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays
from gaps_flower.evaluate_checkpoint import evaluate_classification, load_checkpoint_model, make_loader
from model import DomainDiscriminator
from scripts.evaluate_iotj_p0_roundwise_routing import da_loader, enrich
from utils import compute_mmd2, deep_coral_loss, set_random_seed

SEED = 42
STEPS = 100
MODEL_LR = 5e-4
BATCH_SIZE = 32
PSEUDO_THRESHOLD = 0.90
EXPECTED_CHECKPOINT_SHA256 = "4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c"
U1_WEIGHTS = {"source_ce": 1.0, "coral": 0.5, "global_mmd2": 0.5, "adversarial": 0.5}
CRITIC_LR = 1e-3
CRITIC_ITERS = 3
GRADIENT_PENALTY = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"FAIL_CLOSED empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class FeatureOnlyCalibrationDataset(Dataset):
    """Target calibration dataset whose public samples contain x and nothing else."""

    def __init__(self, client_dir: Path):
        self.feature_path = client_dir / "calibration_features.npy"
        self.features = np.load(self.feature_path, allow_pickle=False).astype(np.float32, copy=False)
        if self.features.shape != (320, 100, 8):
            raise RuntimeError(f"FAIL_CLOSED unexpected target feature shape: {self.features.shape}")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.features[index])


def feature_only_loader(client_dir: Path, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        FeatureOnlyCalibrationDataset(client_dir), batch_size=BATCH_SIZE,
        shuffle=shuffle, generator=generator if shuffle else None, num_workers=0,
    )


def require_x_only(batch: Any, *, method: str) -> torch.Tensor:
    if not isinstance(batch, torch.Tensor):
        raise RuntimeError(f"FAIL_CLOSED {method} target batch carries a non-feature object: {type(batch)!r}")
    if batch.ndim != 3 or tuple(batch.shape[1:]) != (100, 8):
        raise RuntimeError(f"FAIL_CLOSED {method} target feature shape: {tuple(batch.shape)}")
    return batch


def next_cycling(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def gradient_penalty(discriminator: torch.nn.Module, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    count = min(source.size(0), target.size(0))
    source = source[:count]; target = target[:count]
    alpha = torch.rand(count, 1, device=source.device)
    interpolated = (alpha * source + (1.0 - alpha) * target).requires_grad_(True)
    score = discriminator(interpolated)
    gradients = torch.autograd.grad(score, interpolated, torch.ones_like(score), create_graph=True, retain_graph=True)[0]
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def run_unsupervised_global_alignment(
    source_model: torch.nn.Module,
    source_loader: DataLoader,
    target_x_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, Any]], float]:
    """U1 training API: target input is feature-only by contract."""
    set_random_seed(SEED)
    model = copy.deepcopy(source_model).to(device)
    discriminator = DomainDiscriminator(feat_dim=64, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    critic_optimizer = torch.optim.Adam(discriminator.parameters(), lr=CRITIC_LR)
    source_iter, target_iter = iter(source_loader), iter(target_x_loader)
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter(); model.train(); discriminator.train()
    for step in range(1, STEPS + 1):
        source_batch, source_iter = next_cycling(source_iter, source_loader)
        target_batch, target_iter = next_cycling(target_iter, target_x_loader)
        x_s, y_s = source_batch[0].to(device), source_batch[1].to(device).long()
        x_t = require_x_only(target_batch, method="U1").to(device)
        optimizer.zero_grad(set_to_none=True)
        logits_s, feat_s, _ = model(x_s)
        _logits_t, feat_t, _ = model(x_t)
        for _ in range(CRITIC_ITERS):
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = -(
                discriminator(feat_s.detach()).mean() - discriminator(feat_t.detach()).mean()
            ) + GRADIENT_PENALTY * gradient_penalty(discriminator, feat_s.detach(), feat_t.detach())
            critic_loss.backward(); torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0); critic_optimizer.step()
        source_ce = F.cross_entropy(logits_s, y_s)
        coral = deep_coral_loss(feat_s, feat_t)
        global_mmd2 = compute_mmd2(feat_s, feat_t)
        adversarial = wasserstein_feature_objective(discriminator, feat_s, feat_t)
        weighted_coral = U1_WEIGHTS["coral"] * coral
        weighted_mmd = U1_WEIGHTS["global_mmd2"] * global_mmd2
        weighted_adv = U1_WEIGHTS["adversarial"] * adversarial
        total = source_ce + weighted_coral + weighted_mmd + weighted_adv
        total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        diagnostics.append({
            "step": step, "source_ce": float(source_ce.detach()), "coral_loss": float(coral.detach()),
            "global_mmd2": float(global_mmd2.detach()), "adversarial_loss": float(adversarial.detach()),
            "weighted_coral": float(weighted_coral.detach()), "weighted_global_mmd2": float(weighted_mmd.detach()),
            "weighted_adversarial": float(weighted_adv.detach()), "total_loss": float(total.detach()),
            "target_batch_size": int(x_t.size(0)), "target_label_object_present": False,
            "target_ce_status": "UNAVAILABLE", "class_conditional_coral_status": "DISABLED",
            "class_mmd_status": "DISABLED", "stage_mmd_status": "DISABLED",
            "target_proto_anchor_status": "UNAVAILABLE", "pseudo_label_status": "DISABLED",
        })
    return model, diagnostics, time.perf_counter() - started


def run_pseudo_label_self_training(
    source_model: torch.nn.Module,
    target_x_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, list[dict[str, Any]], float]:
    """U2 training API: pseudo labels come only from a frozen source teacher."""
    set_random_seed(SEED)
    teacher = copy.deepcopy(source_model).to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    student = copy.deepcopy(source_model).to(device).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=MODEL_LR)
    iterator = iter(target_x_loader); diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        target_batch, iterator = next_cycling(iterator, target_x_loader)
        x_t = require_x_only(target_batch, method="U2").to(device)
        with torch.no_grad():
            teacher_logits, _, _ = teacher(x_t)
            probabilities = F.softmax(teacher_logits, dim=1)
            confidence, pseudo = probabilities.max(dim=1)
            selected = confidence >= PSEUDO_THRESHOLD
        optimizer.zero_grad(set_to_none=True)
        student_logits, _, _ = student(x_t)
        if selected.any():
            loss = F.cross_entropy(student_logits[selected], pseudo[selected])
            loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0); optimizer.step()
            loss_value = float(loss.detach())
        else:
            loss_value = 0.0
        counts = torch.bincount(pseudo[selected].detach().cpu(), minlength=4)
        diagnostics.append({
            "record_type": "training_step", "step": step, "threshold": PSEUDO_THRESHOLD,
            "batch_size": int(x_t.size(0)), "selected_count": int(selected.sum()),
            "coverage": float(selected.float().mean()), "mean_confidence_all": float(confidence.mean()),
            "mean_confidence_selected": float(confidence[selected].mean()) if selected.any() else 0.0,
            "pseudo_class_0": int(counts[0]), "pseudo_class_1": int(counts[1]),
            "pseudo_class_2": int(counts[2]), "pseudo_class_3": int(counts[3]),
            "pseudo_ce": loss_value, "teacher_policy": "frozen_source_round25",
            "pseudo_label_origin": "teacher_argmax_only", "target_label_object_present": False,
            "posthoc_precision": "",
        })
    return student, teacher, diagnostics, time.perf_counter() - started


def save_adapted_checkpoint(path: Path, source_checkpoint: Path, model: torch.nn.Module, method: str) -> None:
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    payload["model_state"] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload["commissioning"] = {"method": method, "steps": STEPS, "lr": MODEL_LR, "seed": SEED, "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256}
    torch.save(payload, path)


def posthoc_pseudo_precision(teacher: torch.nn.Module, target_x_loader: DataLoader, client_dir: Path, device: torch.device) -> dict[str, Any]:
    """Open target truth only after training, for one non-selective diagnostic."""
    truth_path = client_dir / "calibration_classification_labels.npy"
    truth = np.load(truth_path, allow_pickle=False).astype(np.int64, copy=False)
    predictions, confidences = [], []
    teacher.eval()
    with torch.no_grad():
        for batch in target_x_loader:
            x = require_x_only(batch, method="U2-posthoc").to(device)
            logits, _, _ = teacher(x); probs = F.softmax(logits, dim=1); conf, pred = probs.max(dim=1)
            predictions.extend(pred.cpu().tolist()); confidences.extend(conf.cpu().tolist())
    pred = np.asarray(predictions); conf = np.asarray(confidences); mask = conf >= PSEUDO_THRESHOLD
    counts = np.bincount(pred[mask], minlength=4)
    return {
        "record_type": "posthoc_truth_audit", "step": "", "threshold": PSEUDO_THRESHOLD,
        "batch_size": len(truth), "selected_count": int(mask.sum()), "coverage": float(mask.mean()),
        "mean_confidence_all": float(conf.mean()), "mean_confidence_selected": float(conf[mask].mean()) if mask.any() else 0.0,
        "pseudo_class_0": int(counts[0]), "pseudo_class_1": int(counts[1]), "pseudo_class_2": int(counts[2]), "pseudo_class_3": int(counts[3]),
        "pseudo_ce": "", "teacher_policy": "frozen_source_round25", "pseudo_label_origin": "teacher_argmax_only",
        "target_label_object_present": "posthoc_truth_only", "posthoc_precision": float((pred[mask] == truth[mask]).mean()) if mask.any() else 0.0,
    }


def static_label_access_audit() -> dict[str, Any]:
    u1_signature = tuple(inspect.signature(run_unsupervised_global_alignment).parameters)
    u2_signature = tuple(inspect.signature(run_pseudo_label_self_training).parameters)
    checks = {
        "u1_has_no_target_label_parameter": not any("label" in name or name.startswith("y_t") for name in u1_signature),
        "u2_has_no_target_label_parameter": not any("label" in name or name.startswith("y_t") for name in u2_signature),
        "feature_dataset_returns_tensor_only": "return torch.from_numpy" in inspect.getsource(FeatureOnlyCalibrationDataset.__getitem__),
        "threshold_predeclared_0_90": PSEUDO_THRESHOLD == 0.90,
        "u2_pseudo_origin_is_teacher_argmax": "probabilities.max" in inspect.getsource(run_pseudo_label_self_training),
        "u1_target_conditioned_losses_absent": all(
            token not in inspect.getsource(run_unsupervised_global_alignment)
            for token in (
                "deep_coral_loss_class_conditional(",
                "cross_domain_same_class_phase_mmd2(",
                "y_t =",
                "pseudo =",
            )
        ),
    }
    if not all(checks.values()): raise RuntimeError(f"FAIL_CLOSED static label audit: {checks}")
    return {"status": "PASS", "checks": checks, "audited_at": utc_now()}


def evaluate(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    metrics = enrich(evaluate_classification(model, test_loader, device, 4, 15))
    if metrics["num_examples"] != 1360: raise RuntimeError("FAIL_CLOSED C5 test size")
    return metrics


def comparison_row(method: str, metrics: dict[str, Any], seconds: float, label_access: str, checkpoint_hash: str) -> dict[str, Any]:
    return {"method": method, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "nll": metrics["nll"], "ece": metrics["ece"], "commissioning_seconds": seconds, "target_label_access": label_access, "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "adapted_checkpoint_sha256": checkpoint_hash, "seed": SEED, "num_examples": 1360}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--existing-round25-comparison", required=True, type=Path)
    parser.add_argument("--output-root", default="results/iotj_p0_zero_label_commissioning_20260803", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); checkpoint = args.source_checkpoint.resolve(); data_root = args.data_root.resolve(); output = args.output_root.resolve()
    if output.exists(): raise FileExistsError(f"REFUSE_TO_OVERWRITE: {output}")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256: raise RuntimeError("FAIL_CLOSED source checkpoint hash mismatch")
    output.mkdir(parents=True); (output / "U1_UNSUPERVISED_GLOBAL_ALIGNMENT").mkdir(); (output / "U2_PSEUDO_LABEL_SELF_TRAINING").mkdir()
    ledger: list[dict[str, Any]] = []
    static_audit = static_label_access_audit(); (output / "label_access_static_preflight.json").write_text(json.dumps(static_audit, indent=2) + "\n", encoding="utf-8")
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    source_model, _, _ = load_checkpoint_model(str(checkpoint), device, BATCH_SIZE); ledger.append({"event": "source_checkpoint_loaded", "at": utc_now(), "sha256": EXPECTED_CHECKPOINT_SHA256})
    source_arrays = load_domain_adaptation_arrays([data_root / "client_1", data_root / "client_2"], strict=True)

    u1_target = feature_only_loader(data_root / "client_5", shuffle=True); ledger.append({"event": "u1_target_x_loader_created", "at": utc_now(), "target_labels_loaded": False})
    u1_model, u1_diag, u1_seconds = run_unsupervised_global_alignment(source_model, da_loader(source_arrays), u1_target, device)
    u1_checkpoint = output / "U1_UNSUPERVISED_GLOBAL_ALIGNMENT" / "u1_adapted.pth"; save_adapted_checkpoint(u1_checkpoint, checkpoint, u1_model, "unsupervised_global_alignment")
    ledger.append({"event": "u1_training_completed", "at": utc_now(), "steps": 100, "target_labels_loaded": False})

    u2_target = feature_only_loader(data_root / "client_5", shuffle=True); ledger.append({"event": "u2_target_x_loader_created", "at": utc_now(), "target_labels_loaded": False})
    u2_model, teacher, u2_diag, u2_seconds = run_pseudo_label_self_training(source_model, u2_target, device)
    u2_checkpoint = output / "U2_PSEUDO_LABEL_SELF_TRAINING" / "u2_adapted.pth"; save_adapted_checkpoint(u2_checkpoint, checkpoint, u2_model, "pseudo_label_self_training")
    ledger.append({"event": "u2_training_completed", "at": utc_now(), "steps": 100, "target_labels_loaded": False})

    audit_loader = feature_only_loader(data_root / "client_5", shuffle=False)
    posthoc = posthoc_pseudo_precision(teacher, audit_loader, data_root / "client_5", device); u2_diag.append(posthoc)
    ledger.append({"event": "calibration_truth_opened_posthoc", "at": utc_now(), "purpose": "pseudo_label_precision_only", "after_both_training": True})

    test_loader = make_loader(data_root, 5, "test", BATCH_SIZE); ledger.append({"event": "c5_test_opened", "at": utc_now(), "after_both_training": True, "purpose": "single_final_evaluation"})
    u1_metrics = evaluate(u1_model, test_loader, device); u2_metrics = evaluate(u2_model, test_loader, device)
    write_csv(output / "unsupervised_alignment_diagnostics.csv", u1_diag); write_csv(output / "pseudo_label_diagnostics.csv", u2_diag)

    with args.existing_round25_comparison.open(newline="", encoding="utf-8") as handle: existing = {row["method"]: row for row in csv.DictReader(handle)}
    comparison = []
    for method, label_access in (("source_only", "none"), ("simple_target_ce", "C5_calibration_true_labels")):
        row = existing[method]
        comparison.append({"method": method, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "nll": row["nll"], "ece": row["ece"], "commissioning_seconds": row["commissioning_time_seconds"], "target_label_access": label_access, "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "adapted_checkpoint_sha256": "not_applicable_or_existing_read_only", "seed": SEED, "num_examples": 1360})
    comparison.append(comparison_row("unsupervised_global_alignment", u1_metrics, u1_seconds, "none_x_only", sha256_file(u1_checkpoint)))
    comparison.append(comparison_row("pseudo_label_self_training", u2_metrics, u2_seconds, "none_x_only_pseudo_from_teacher", sha256_file(u2_checkpoint)))
    write_csv(output / "zero_label_commissioning_comparison.csv", comparison)

    manifest = {
        "schema_version": "iotj.p0u.zero_label.v1", "status": "completed_pending_strict_audit", "seed": SEED,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "dataset": str(data_root), "target_calibration_windows": 320, "target_test_windows": 1360,
        "steps": STEPS, "model_lr": MODEL_LR, "threshold": PSEUDO_THRESHOLD, "hyperparameter_search": False,
        "u1": {"target_api": "x_only", "weights": U1_WEIGHTS, "critic_lr": CRITIC_LR, "critic_iters": CRITIC_ITERS, "gradient_penalty": GRADIENT_PENALTY, "forbidden_target_losses": ["target_ce", "class_conditional_coral", "class_mmd", "same_class_phase_stage_mmd", "target_proto_anchor", "target_label_semantic_matching", "pseudo_labels"]},
        "u2": {"target_api": "x_only", "teacher": "frozen_source_round25", "pseudo_label_origin": "teacher_argmax", "threshold": PSEUDO_THRESHOLD, "truth_access": "posthoc_precision_only"},
        "label_access_ledger": ledger, "target_test_used_for_selection": False, "checkpoint_selection": "fixed_source_round25",
        "metrics": {"u1": u1_metrics, "u2": u2_metrics}, "commissioning_seconds": {"u1": u1_seconds, "u2": u2_seconds},
        "pseudo_label_posthoc": posthoc,
    }
    (output / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    label_report = f"""# P0-U Label Access Audit\n\n## Verdict: PASS pending experiment audit\n\n- U1/U2 training APIs receive `torch.Tensor` target features only; non-tensor batches fail closed.\n- No target class-label parameter exists in either adaptation function.\n- U1 target CE and target prototype anchor are unavailable; class-conditional CORAL, class MMD, same-class-phase MMD and pseudo labels are disabled.\n- U2 pseudo labels originate only from frozen source-teacher argmax predictions at the predeclared threshold 0.90.\n- Calibration truth was opened only after both 100-step training branches completed, for one post-hoc pseudo-label precision audit: coverage {posthoc['coverage']:.6f}, precision {posthoc['posthoc_precision']:.6f}.\n- C5 sealed test was opened after both branches and used only for final evaluation.\n- No early stopping, threshold selection, hyperparameter search, or checkpoint selection occurred.\n"""
    (output / "LABEL_ACCESS_AUDIT.md").write_text(label_report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "u1_macro_f1": u1_metrics["macro_f1"], "u2_macro_f1": u2_metrics["macro_f1"]}))


if __name__ == "__main__": main()
