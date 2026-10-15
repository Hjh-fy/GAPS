"""Execute Gate 3 C5 5%-labeled/15%-unlabeled SSDA commissioning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaps_flower.domain_adaptation_inputs import load_domain_adaptation_arrays  # noqa: E402
from gaps_flower.evaluate_checkpoint import load_checkpoint_model  # noqa: E402
from gaps_flower.posthoc_commissioning import (  # noqa: E402
    ordered_state_fingerprint,
    sha256_file,
    supervised_ce_adapt,
)
from gaps_flower.ssda import (  # noqa: E402
    G3Config,
    G3Request,
    UnlabeledTargetDataset,
    build_g3_partition,
    compute_frozen_class_prototypes,
    deterministic_two_fold,
    gaps_ssda_adapt,
    labeled_loader,
    macro_f1_nll,
    mme_compatible_adapt,
    posthoc_hidden_pseudo_diagnostic,
    predict_probabilities,
    unlabeled_loader,
)
from scripts.summarize_iotj_classification_ablation import (  # noqa: E402
    evaluate_checkpoint_stream,
)


DATA_ROOT = ROOT / "dataset/iotj_canonical_v1"
BUDGET_ROOT = ROOT / "results/iotj_canonical_v1_c5_budget_20260810/budget_data"
SOURCE_RUN = ROOT / "results/iotj_canonical_v1_scientific_validation_20260809/comparators/source_fl/CAN-V1-CMP-FEDAVG"
SOURCE_CHECKPOINT = SOURCE_RUN / "remote_server/server_latest.pth"
DEFAULT_OUTPUT = ROOT / "results/iotj_canonical_v1_method_redesign_20260811/gate3_ssda"
DOC_ROOT = ROOT / "docs/experiments/iotj_canonical_v1_final/method_redesign"
METHODS = ("a0t_5l", "mme_5l15u", "gaps_ssda_5l15u")
DISPLAY = {
    "a0t_5l": "A0T-5L",
    "mme_5l15u": "MME-compatible-5L15U",
    "gaps_ssda_5l15u": "GAPS-SSDA-5L15U",
}


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def decide_gate3(*, a0t_f1: float, mme_f1: float, gaps_f1: float) -> dict[str, Any]:
    threshold = 0.005
    best_ssda_gain = max(mme_f1, gaps_f1) - a0t_f1
    if best_ssda_gain < threshold:
        decision = "NO_SSDA_SPACE"
        reason = "Neither SSDA endpoint improves A0T-5L by 0.005 Macro-F1."
    elif mme_f1 - gaps_f1 >= threshold:
        decision = "MME_DOMINATES"
        reason = "MME-compatible exceeds GAPS-SSDA by at least 0.005 Macro-F1."
    elif gaps_f1 - a0t_f1 >= threshold and gaps_f1 >= mme_f1 - threshold:
        decision = "SSDA_COMPONENT_SUPPORTED"
        reason = "GAPS-SSDA improves A0T-5L and is within 0.005 of or above MME-compatible."
    else:
        decision = "SSDA_COMPONENT_NOT_SUPPORTED"
        reason = "GAPS-SSDA does not satisfy the pre-registered benefit/competitiveness rule."
    return {
        "decision": decision,
        "threshold_macro_f1": threshold,
        "a0t_macro_f1": float(a0t_f1),
        "mme_macro_f1": float(mme_f1),
        "gaps_macro_f1": float(gaps_f1),
        "mme_minus_a0t": float(mme_f1 - a0t_f1),
        "gaps_minus_a0t": float(gaps_f1 - a0t_f1),
        "gaps_minus_mme": float(gaps_f1 - mme_f1),
        "reason": reason,
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    method: str,
    source_sha: str,
    source_state_fingerprint: str,
    config: G3Config,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "step": config.steps,
        "model_state": model.state_dict(),
        "method": method,
        "seed": config.seed,
        "source_checkpoint_sha256": source_sha,
        "source_state_fingerprint": source_state_fingerprint,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _load_inputs() -> dict[str, Any]:
    calibration_dir = DATA_ROOT / "client_5"
    labeled_dir = BUDGET_ROOT / "client_5_budget_05"
    calibration_manifest = calibration_dir / "calibration_experiment_info.json"
    labeled_manifest = labeled_dir / "calibration_experiment_info.json"
    request = G3Request(SOURCE_CHECKPOINT, calibration_manifest, labeled_manifest)
    request.validate_static_boundary()
    partition = build_g3_partition(calibration_manifest, labeled_manifest)
    calibration_features = np.load(
        calibration_dir / "calibration_features.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    labeled_features = np.load(
        labeled_dir / "calibration_features.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    labeled_labels = np.load(
        labeled_dir / "calibration_classification_labels.npy", allow_pickle=False
    ).astype(np.int64, copy=False)
    if calibration_features.shape != (320, 50, 8):
        raise ValueError(f"unexpected C5 calibration feature shape: {calibration_features.shape}")
    if labeled_features.shape != (80, 50, 8) or labeled_labels.shape != (80,):
        raise ValueError("unexpected 5% labeled budget array shape")
    if not np.array_equal(labeled_features, calibration_features[list(partition.labeled_indices)]):
        raise RuntimeError("labeled feature arrays do not match canonical identity mapping")
    sparse_labels = np.full(320, -1, dtype=np.int64)
    sparse_labels[list(partition.labeled_indices)] = labeled_labels
    unlabeled_identities = tuple(
        partition.calibration_identities[index]
        for index in partition.unlabeled_indices
    )
    return {
        "partition": partition,
        "calibration_features": calibration_features,
        "sparse_labels": sparse_labels,
        "unlabeled_features": calibration_features[list(partition.unlabeled_indices)],
        "unlabeled_identities": unlabeled_identities,
        "calibration_manifest": calibration_manifest,
        "labeled_manifest": labeled_manifest,
    }


def _write_pre_run_evidence(
    output: Path,
    inputs: dict[str, Any],
    config: G3Config,
    *,
    source_sha: str,
    source_state_fingerprint: str,
) -> None:
    partition = inputs["partition"]
    _json(
        output / "G3_PRE_RUN_FREEZE.json",
        {
            "status": "FROZEN_BEFORE_TARGET_TEST_ACCESS",
            "gate": "G3",
            "dataset": "canonical-v1",
            "target": "C5",
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": source_sha,
            "source_state_fingerprint": source_state_fingerprint,
            "labeled_count": 80,
            "unlabeled_count": 240,
            "target_test_count": 1360,
            "steps_per_training_endpoint": config.steps,
            "optimizer": config.optimizer,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "gaps_grid": [
                {"tau": tau, "lambda_u": lambda_u}
                for tau, lambda_u in config.grid()
            ],
            "gaps_validation": "deterministic two-fold; one labeled window per stratum train and one validation",
            "lambda_proto": config.lambda_proto,
            "ema_alpha": config.ema_alpha,
            "mme_lambda": config.mme_lambda,
            "mme_identity": "MME-compatible existing linear head; not exact paper reproduction",
            "selection_primary": "mean validation Macro-F1 descending",
            "selection_secondary": "mean validation NLL ascending",
            "selection_tie_break": "grid declaration order",
            "target_test_selection": False,
            "decision_threshold_macro_f1": 0.005,
        },
    )
    unlabeled_rows = [
        {"physical_identity": identity, "role": "unlabeled_calibration"}
        for identity in inputs["unlabeled_identities"]
    ]
    _json(output / "unlabeled_x_only_manifest.json", unlabeled_rows)
    data_audit = [
        "# C5 SSDA Data Audit",
        "",
        "- Canonical calibration pool: 320 identities.",
        "- Labeled pool: 80 identities, exactly 2 in each of 40 pre-existing class×concentration strata.",
        "- Unlabeled pool: the identity complement, 240 identities, exactly 6 per stratum.",
        "- Labeled and unlabeled identity intersection: empty.",
        "- The unlabeled training dataset stores and returns only `x` and `physical_identity`; it has no class, phase, or concentration field.",
        "- The hidden-label array is not loaded before all final endpoints and the selected configuration are locked.",
        "- Target-test manifest and arrays are not opened by the adaptation or selection stage.",
        "- Stratum labels were used only to verify the already frozen nested calibration construction; they are not passed to any unlabeled loader, loss, sampler, selector, or checkpoint rule.",
        "",
        f"Calibration manifest SHA-256: `{sha256_file(inputs['calibration_manifest'])}`.",
        f"Labeled manifest SHA-256: `{sha256_file(inputs['labeled_manifest'])}`.",
        f"Unlabeled X tensor content SHA-256: `{_array_sha256(inputs['unlabeled_features'])}`.",
    ]
    (output / "C5_SSDA_DATA_AUDIT.md").write_text("\n".join(data_audit) + "\n", encoding="utf-8")
    (output / "SSDA_PHASE_OBSERVABILITY_AUDIT.md").write_text(
        "# SSDA Phase Observability Audit\n\n"
        "The first GAPS-SSDA endpoint uses class-only frozen source prototypes. Target phase is absent from both labeled and unlabeled adaptation APIs. This conservative choice avoids assuming that injection-relative phase metadata is available for an unknown gas during online commissioning. No target phase or concentration enters training or selection.\n",
        encoding="utf-8",
    )
    (output / "MME_IMPLEMENTATION_FEASIBILITY.md").write_text(
        "# MME Implementation Feasibility\n\n"
        "The original MME method uses a temperature-scaled cosine-similarity classifier and adversarially maximizes unlabeled conditional entropy with respect to the classifier while minimizing it with respect to the feature encoder. The frozen GAPS model exposes normalized features but retains a conventional biased linear classifier. Replacing that layer would change the registered architecture and its source endpoint semantics.\n\n"
        "This Gate therefore uses **MME-compatible (existing linear head)**: gradient reversal applies the minimax entropy direction through the existing classifier, with the official default entropy weight 0.1, while retaining the canonical backbone, head, Adam 5e-4 optimizer, and exactly 100 optimizer updates. It is an algorithm-compatible comparator, not an exact reproduction of the ICCV implementation. No target-test value selected this design or coefficient.\n\n"
        "Primary references:\n"
        "- https://openaccess.thecvf.com/content_ICCV_2019/html/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.html\n"
        "- https://github.com/VisionLearningGroup/SSDA_MME\n",
        encoding="utf-8",
    )
    if len(partition.labeled_indices) != 80 or len(partition.unlabeled_indices) != 240:
        raise RuntimeError("partition changed while writing pre-run evidence")


def _source_prototypes(
    source_model: torch.nn.Module,
    *,
    device: torch.device,
    config: G3Config,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source_features, source_labels, _source_phases = load_domain_adaptation_arrays(
        [DATA_ROOT / "client_1", DATA_ROOT / "client_2"],
        strict=True,
        expected_window_shape=(50, 8),
    )
    source_loader = labeled_loader(
        source_features,
        source_labels,
        tuple(range(len(source_features))),
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    prototypes = compute_frozen_class_prototypes(
        source_model, source_loader, device=device
    )
    return prototypes, {
        "source_rows": int(len(source_features)),
        "source_clients": [1, 2],
        "prototype_scope": "class_only",
        "prototype_tensor_sha256": _array_sha256(prototypes.cpu().numpy()),
        "source_class_counts": {
            str(class_id): int(np.sum(source_labels == class_id)) for class_id in range(4)
        },
    }


def _select_gaps_config(
    source_model: torch.nn.Module,
    inputs: dict[str, Any],
    prototypes: torch.Tensor,
    *,
    output: Path,
    device: torch.device,
    config: G3Config,
) -> tuple[float, float]:
    partition = inputs["partition"]
    folds = deterministic_two_fold(partition)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for grid_index, (tau, lambda_u) in enumerate(config.grid()):
        fold_scores: list[tuple[float, float]] = []
        for fold in folds:
            train_loader = labeled_loader(
                inputs["calibration_features"],
                inputs["sparse_labels"],
                fold.train_indices,
                batch_size=config.batch_size,
                seed=config.seed,
            )
            x_only_loader = unlabeled_loader(
                inputs["unlabeled_features"],
                inputs["unlabeled_identities"],
                batch_size=config.batch_size,
                seed=config.seed,
            )
            student, _teacher, diagnostics, system = gaps_ssda_adapt(
                source_model,
                train_loader,
                x_only_loader,
                prototypes,
                tau=tau,
                lambda_u=lambda_u,
                device=device,
                config=config,
            )
            probabilities = predict_probabilities(
                student,
                inputs["calibration_features"][list(fold.validation_indices)],
                device=device,
                batch_size=config.batch_size,
            )
            labels = inputs["sparse_labels"][list(fold.validation_indices)]
            macro_f1, nll = macro_f1_nll(labels, probabilities)
            fold_scores.append((macro_f1, nll))
            rows.append(
                {
                    "grid_index": grid_index,
                    "tau": tau,
                    "lambda_u": lambda_u,
                    "fold": fold.fold,
                    "train_labeled_N": len(fold.train_indices),
                    "validation_labeled_N": len(fold.validation_indices),
                    "unlabeled_N": len(partition.unlabeled_indices),
                    "validation_macro_f1": macro_f1,
                    "validation_nll": nll,
                    "mean_step_acceptance": float(
                        np.mean([row["pseudo_acceptance_rate"] for row in diagnostics])
                    ),
                    "adaptation_seconds": system["adaptation_seconds"],
                    "target_test_accessed": False,
                }
            )
            del student, _teacher
        summaries.append(
            {
                "grid_index": grid_index,
                "tau": tau,
                "lambda_u": lambda_u,
                "mean_validation_macro_f1": float(np.mean([value[0] for value in fold_scores])),
                "mean_validation_nll": float(np.mean([value[1] for value in fold_scores])),
            }
        )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -row["mean_validation_macro_f1"],
            row["mean_validation_nll"],
            row["grid_index"],
        ),
    )
    selected = ranked[0]
    for row in summaries:
        row["selected"] = row["grid_index"] == selected["grid_index"]
    _write_csv(output / "gaps_ssda_selection_folds.csv", rows)
    _write_csv(output / "gaps_ssda_selection_summary.csv", summaries)
    _json(
        output / "GAPS_SSDA_SELECTED_CONFIG.json",
        {
            "status": "FROZEN_BEFORE_TARGET_TEST_ACCESS",
            **selected,
            "selection_used_target_test": False,
            "candidate_count": len(config.grid()),
            "fold_count": len(folds),
        },
    )
    return float(selected["tau"]), float(selected["lambda_u"])


def _run_final_endpoints(
    source_model: torch.nn.Module,
    inputs: dict[str, Any],
    prototypes: torch.Tensor,
    *,
    output: Path,
    device: torch.device,
    config: G3Config,
    source_sha: str,
    source_state_fingerprint: str,
    tau: float,
    lambda_u: float,
) -> torch.nn.Module:
    systems: dict[str, dict[str, Any]] = {}
    final_teacher: torch.nn.Module | None = None
    for method in METHODS:
        method_dir = output / method
        method_dir.mkdir()
        labels = labeled_loader(
            inputs["calibration_features"],
            inputs["sparse_labels"],
            inputs["partition"].labeled_indices,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        x_only = unlabeled_loader(
            inputs["unlabeled_features"],
            inputs["unlabeled_identities"],
            batch_size=config.batch_size,
            seed=config.seed,
        )
        if method == "a0t_5l":
            model, diagnostics, system = supervised_ce_adapt(
                source_model,
                labels,
                method="a0t_full",
                device=device,
                steps=config.steps,
                lr=config.lr,
                seed=config.seed,
            )
            system["method"] = method
            system["unlabeled_used"] = False
        elif method == "mme_5l15u":
            model, diagnostics, system = mme_compatible_adapt(
                source_model, labels, x_only, device=device, config=config
            )
            system["method"] = method
            system["unlabeled_used"] = True
        else:
            model, final_teacher, diagnostics, system = gaps_ssda_adapt(
                source_model,
                labels,
                x_only,
                prototypes,
                tau=tau,
                lambda_u=lambda_u,
                device=device,
                config=config,
            )
            system["method"] = method
            system["unlabeled_used"] = True
        checkpoint = method_dir / f"{method}_step100.pth"
        _save_checkpoint(
            checkpoint,
            model,
            method=method,
            source_sha=source_sha,
            source_state_fingerprint=source_state_fingerprint,
            config=config,
            extra={"tau": tau, "lambda_u": lambda_u} if method == "gaps_ssda_5l15u" else None,
        )
        _write_csv(method_dir / "adaptation_diagnostics.csv", diagnostics)
        system.update(
            {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "source_checkpoint_sha256": source_sha,
                "source_state_fingerprint": source_state_fingerprint,
                "labeled_count": 80,
                "unlabeled_count_available": 240,
            }
        )
        _json(method_dir / "system_metrics.json", system)
        _json(
            method_dir / "fixed_endpoint_complete.json",
            {
                "status": "COMPLETE",
                "method": method,
                "step": config.steps,
                "checkpoint_sha256": system["checkpoint_sha256"],
                "source_state_fingerprint": source_state_fingerprint,
                "target_test_opened": False,
                "hidden_unlabeled_truth_opened": False,
            },
        )
        systems[method] = system
        del model
    if final_teacher is None:
        raise RuntimeError("GAPS-SSDA final teacher was not produced")
    teacher_path = output / "gaps_ssda_5l15u/ema_teacher_step100.pth"
    _save_checkpoint(
        teacher_path,
        final_teacher,
        method="gaps_ssda_ema_teacher_diagnostic_only",
        source_sha=source_sha,
        source_state_fingerprint=source_state_fingerprint,
        config=config,
        extra={"tau": tau, "lambda_u": lambda_u},
    )
    systems["gaps_ssda_5l15u"]["ema_teacher_checkpoint"] = str(teacher_path.resolve())
    systems["gaps_ssda_5l15u"]["ema_teacher_checkpoint_sha256"] = sha256_file(teacher_path)
    _json(output / "gaps_ssda_5l15u/system_metrics.json", systems["gaps_ssda_5l15u"])
    return final_teacher


def _verify_endpoint_gate(
    output: Path,
    *,
    config: G3Config,
    source_state_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        method_dir = output / method
        marker = json.loads(
            (method_dir / "fixed_endpoint_complete.json").read_text(encoding="utf-8")
        )
        system = json.loads((method_dir / "system_metrics.json").read_text(encoding="utf-8"))
        checkpoint = Path(system["checkpoint"])
        if marker.get("target_test_opened") is not False:
            raise RuntimeError(f"target test opened before {method} endpoint lock")
        if marker.get("hidden_unlabeled_truth_opened") is not False:
            raise RuntimeError(f"hidden labels opened before {method} endpoint lock")
        if int(marker.get("step", -1)) != config.steps:
            raise RuntimeError(f"wrong fixed endpoint for {method}")
        if marker.get("source_state_fingerprint") != source_state_fingerprint:
            raise RuntimeError(f"wrong source state for {method}")
        if not checkpoint.is_file() or sha256_file(checkpoint) != marker.get("checkpoint_sha256"):
            raise RuntimeError(f"checkpoint hash failure for {method}")
        result[method] = system
    return result


def _per_class_rows(method: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for class_id in range(4):
        true_positive = float(confusion[class_id, class_id])
        false_positive = float(confusion[:, class_id].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        rows.append(
            {
                "method": DISPLAY[method],
                "class_id": class_id,
                "precision": precision,
                "recall": metrics["per_class_recall"][str(class_id)],
                "f1": metrics["per_class_f1"][str(class_id)],
            }
        )
    return rows


def _evaluate_and_analyze(
    output: Path,
    inputs: dict[str, Any],
    final_teacher: torch.nn.Module,
    systems: dict[str, dict[str, Any]],
    *,
    device: torch.device,
    config: G3Config,
    tau: float,
) -> dict[str, Any]:
    test_manifest = DATA_ROOT / "client_5/test_experiment_info.json"
    test_rows = json.loads(test_manifest.read_text(encoding="utf-8"))
    test_identities = {str(row["physical_identity"]) for row in test_rows}
    calibration_identities = {
        str(identity) for identity in inputs["partition"].calibration_identities
    }
    if test_identities & calibration_identities or len(test_identities) != 1360:
        raise RuntimeError("canonical target test overlap/count failure")
    _json(
        output / "SEALED_TEST_OPEN.json",
        {
            "status": "OPENED_AFTER_ALL_G3_ENDPOINTS_AND_SELECTION_LOCKED",
            "opened_at_unix": time.time(),
            "target_test_manifest_sha256": sha256_file(test_manifest),
            "target_test_selection": False,
        },
    )
    metric_rows: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    metrics_by_method: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        checkpoint = Path(systems[method]["checkpoint"])
        rows, metrics = evaluate_checkpoint_stream(
            checkpoint,
            data_root=DATA_ROOT,
            target_client=5,
            split="test",
            device=device,
            batch_size=config.batch_size,
            expected_endpoint=("step", config.steps),
        )
        metrics_by_method[method] = metrics
        metric_rows.append(
            {
                "method": DISPLAY[method],
                "N": metrics["N"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "ece": metrics["ece"],
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        per_class.extend(_per_class_rows(method, metrics))
        prediction_rows.extend({"method": DISPLAY[method], **row} for row in rows)
        confidences = np.asarray([float(row["confidence"]) for row in rows])
        for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0):
            confidence_rows.append(
                {
                    "method": DISPLAY[method],
                    "quantile": quantile,
                    "confidence": float(np.quantile(confidences, quantile)),
                }
            )
    _write_csv(output / "C5_SSDA_COMPARISON.csv", metric_rows)
    _write_csv(output / "C5_SSDA_PER_CLASS.csv", per_class)
    _write_csv(output / "C5_SSDA_PREDICTIONS.csv", prediction_rows)
    _write_csv(output / "C5_SSDA_CONFIDENCE_DISTRIBUTION.csv", confidence_rows)

    # Hidden truth is first loaded here, after endpoint and selection locks.
    full_hidden_labels = np.load(
        DATA_ROOT / "client_5/calibration_classification_labels.npy", allow_pickle=False
    ).astype(np.int64, copy=False)
    hidden_unlabeled = full_hidden_labels[list(inputs["partition"].unlabeled_indices)]
    x_only_dataset = UnlabeledTargetDataset(
        inputs["unlabeled_features"], inputs["unlabeled_identities"]
    )
    pseudo_rows, pseudo_summary = posthoc_hidden_pseudo_diagnostic(
        final_teacher,
        x_only_dataset,
        hidden_unlabeled,
        tau=tau,
        device=device,
        batch_size=config.batch_size,
    )
    _write_csv(output / "GAPS_SSDA_PSEUDO_LABEL_POSTHOC.csv", pseudo_rows)
    _json(output / "GAPS_SSDA_PSEUDO_LABEL_POSTHOC.json", pseudo_summary)
    _json(
        output / "HIDDEN_LABEL_DIAGNOSTIC_OPEN.json",
        {
            "status": "OPENED_AFTER_ALL_ENDPOINTS_AND_SELECTION_LOCKED",
            "used_for_training": False,
            "used_for_selection": False,
            "used_for_checkpointing": False,
            "calibration_label_file_sha256": sha256_file(
                DATA_ROOT / "client_5/calibration_classification_labels.npy"
            ),
        },
    )
    decision = decide_gate3(
        a0t_f1=metrics_by_method["a0t_5l"]["macro_f1"],
        mme_f1=metrics_by_method["mme_5l15u"]["macro_f1"],
        gaps_f1=metrics_by_method["gaps_ssda_5l15u"]["macro_f1"],
    )
    _json(output / "G3_DECISION.json", decision)
    analysis = [
        "# Gate 3 C5 SSDA Result Analysis",
        "",
        f"Decision: `{decision['decision']}`.",
        "",
        "| Method | Accuracy | Macro-F1 | NLL | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        analysis.append(
            f"| {row['method']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | {row['nll']:.6f} | {row['ece']:.6f} |"
        )
    analysis.extend(
        [
            "",
            f"- MME-compatible minus A0T Macro-F1: `{decision['mme_minus_a0t']:+.6f}`.",
            f"- GAPS-SSDA minus A0T Macro-F1: `{decision['gaps_minus_a0t']:+.6f}`.",
            f"- GAPS-SSDA minus MME-compatible Macro-F1: `{decision['gaps_minus_mme']:+.6f}`.",
            f"- GAPS pseudo-label acceptance: `{pseudo_summary['acceptance_rate']:.6f}`.",
            f"- Accepted pseudo-label precision (post-hoc only): `{pseudo_summary['pseudo_label_precision_posthoc']}`.",
            "",
            decision["reason"],
            "",
            "All values are one fixed seed42 endpoint. The MME row is an explicitly labeled compatible implementation on the existing linear head, not an exact reproduction. Hidden unlabeled truth was opened only for the final offline diagnostic and did not affect training, selection, or checkpointing.",
        ]
    )
    (output / "G3_RESULT_ANALYSIS.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    audit_text = (
        "# Gate 3 Experiment Audit\n\n"
        "Status: `PASS`.\n\n"
        "- All three final methods independently deep-copied the same verified source-only round25 model state.\n"
        "- The 80L and 240U pools are disjoint and exhaust the frozen 320 calibration identities; calibration and test identities are disjoint.\n"
        "- The unlabeled dataset exposes only X and physical identity. Target phase, concentration, and hidden class truth are absent from training batches.\n"
        "- GAPS selection evaluated exactly six pre-registered configurations on two deterministic labeled folds; test data did not enter selection.\n"
        "- Each final endpoint used exactly 100 Adam updates at 5e-4, batch size 32, seed42.\n"
        "- MME is explicitly reported as compatible rather than exact because the frozen biased linear classifier is retained.\n"
        "- Hidden unlabeled truth opened only after endpoint locks and is used solely for the post-hoc pseudo-label diagnostic.\n"
        "- C5 test opened once after all endpoint and selection locks; no checkpoint or hyperparameter was selected from it.\n"
    )
    (output / "G3_EXPERIMENT_AUDIT.md").write_text(audit_text, encoding="utf-8")
    (DOC_ROOT / "G3_EXPERIMENT_AUDIT.md").write_text(audit_text, encoding="utf-8")
    story = "STORY_B" if decision["decision"] == "SSDA_COMPONENT_SUPPORTED" else "STORY_D"
    story_text = (
        "Standard federated source learning + semi-supervised new-node commissioning + regression/QC. The source-only prototype-DG component is unsupported, while G3 supports the commissioning component."
        if story == "STORY_B"
        else "Neither source-only prototype-DG nor GAPS-SSDA provides a supported advantage under the frozen gates. Position GAPS as a practical federated sensing lifecycle framework (source FL, post-hoc calibration, R84/FedRidge, QC, and edge deployment), and downgrade algorithmic-superiority claims."
    )
    decision_doc = (
        "# GAPS Method Redesign Decision\n\n"
        f"Final story gate: `{story}`.\n\n"
        f"- G1: `POSTHOC_LIFECYCLE_SUPPORTED`.\n"
        f"- G2: `SOURCE_DG_NOT_SUPPORTED`.\n"
        f"- G3: `{decision['decision']}`.\n\n"
        f"{story_text}\n\n"
        "G4 is not allowed because G2 and G3 are not both supported. G5 is not launched automatically in this execution.\n"
    )
    (output / "GAPS_METHOD_REDESIGN_DECISION.md").write_text(decision_doc, encoding="utf-8")
    (DOC_ROOT / "GAPS_METHOD_REDESIGN_DECISION.md").write_text(decision_doc, encoding="utf-8")
    return {**decision, "story": story, "pseudo_summary": pseudo_summary}


def run(output: Path, device: torch.device) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"FAIL_CLOSED Gate-3 output exists: {output}")
    output.mkdir(parents=True)
    config = G3Config()
    source_model, _model_config, source_container = load_checkpoint_model(
        str(SOURCE_CHECKPOINT), device, config.batch_size
    )
    if int(source_container.get("round", -1)) != 25:
        raise RuntimeError("G3 source checkpoint is not fixed round25")
    source_sha = sha256_file(SOURCE_CHECKPOINT)
    source_state_fingerprint = ordered_state_fingerprint(source_container["model_state"])
    source_manifest = json.loads((SOURCE_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("checkpoint_sha256") != source_sha:
        raise RuntimeError("source provenance SHA mismatch")
    if source_manifest["protocol"].get("target_x") is not False or source_manifest["protocol"].get("target_y") is not False:
        raise RuntimeError("G3 source checkpoint was not source-only")
    inputs = _load_inputs()
    _write_pre_run_evidence(
        output,
        inputs,
        config,
        source_sha=source_sha,
        source_state_fingerprint=source_state_fingerprint,
    )
    prototypes, prototype_manifest = _source_prototypes(
        source_model, device=device, config=config
    )
    _json(output / "FROZEN_SOURCE_PROTOTYPES.json", prototype_manifest)
    tau, lambda_u = _select_gaps_config(
        source_model,
        inputs,
        prototypes,
        output=output,
        device=device,
        config=config,
    )
    final_teacher = _run_final_endpoints(
        source_model,
        inputs,
        prototypes,
        output=output,
        device=device,
        config=config,
        source_sha=source_sha,
        source_state_fingerprint=source_state_fingerprint,
        tau=tau,
        lambda_u=lambda_u,
    )
    systems = _verify_endpoint_gate(
        output,
        config=config,
        source_state_fingerprint=source_state_fingerprint,
    )
    decision = _evaluate_and_analyze(
        output,
        inputs,
        final_teacher,
        systems,
        device=device,
        config=config,
        tau=tau,
    )
    _json(
        output / "protocol_manifest.json",
        {
            "status": "PASS",
            "dataset": "canonical-v1",
            "dataset_aggregate_sha256": "2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6",
            "source_checkpoint_sha256": source_sha,
            "source_state_fingerprint": source_state_fingerprint,
            "calibration_manifest_sha256": sha256_file(inputs["calibration_manifest"]),
            "labeled_manifest_sha256": sha256_file(inputs["labeled_manifest"]),
            "unlabeled_manifest_sha256": sha256_file(output / "unlabeled_x_only_manifest.json"),
            "target_test_manifest_sha256": sha256_file(DATA_ROOT / "client_5/test_experiment_info.json"),
            "target_test_opened_after_all_endpoints": True,
            "target_test_selection": False,
            "hidden_unlabeled_truth_posthoc_only": True,
            "selected_tau": tau,
            "selected_lambda_u": lambda_u,
            "decision": decision,
        },
    )
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "sha256_index.json"
    )
    _json(
        output / "sha256_index.json",
        {str(path.relative_to(output)): sha256_file(path) for path in files},
    )
    return {"status": "PASS", "output": str(output), **decision}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    requested = args.device
    device = torch.device(
        requested if not requested.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    print(json.dumps(run(args.output, device), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
