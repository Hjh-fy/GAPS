"""Evaluate a Flower server checkpoint on GAPS client datasets.

The default split is ``test`` so target-domain accuracy is measured on held-out
windows only. Calibration windows can be evaluated explicitly with
``--split calibration`` when checking confidence calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from federated_dataset import (
    create_client_full_test_loader,
    create_client_test_only_loader,
    create_merged_calibration_loader,
)
from gaps_flower.task import create_model, make_config
from utils import soft_aggregate_probs


SPLIT_CHOICES = ("test", "calibration", "full")


def parse_client_ids(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def load_checkpoint_model(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    config = make_config(device=str(device), local_epochs=1, batch_size=batch_size)
    model = create_model(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state")
    if state is None:
        raise ValueError(f"Checkpoint has no model_state: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, checkpoint


def make_loader(
    data_root: str | Path,
    client_id: int,
    split: str,
    batch_size: int,
) -> DataLoader:
    client_dir = Path(data_root) / f"client_{client_id}"
    if split == "test":
        return create_client_test_only_loader(client_dir, batch_size=batch_size)
    if split == "calibration":
        return create_merged_calibration_loader([client_dir], batch_size=batch_size)
    if split == "full":
        return create_client_full_test_loader(client_dir, batch_size=batch_size)
    raise ValueError(f"Unsupported split: {split}")


def expected_calibration_error(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    num_bins: int,
) -> float:
    """Compute top-label ECE with fixed-width confidence bins."""
    if confidences.numel() == 0:
        return 0.0
    ece = torch.tensor(0.0, dtype=torch.float64)
    conf = confidences.double()
    corr = correct.double()
    for idx in range(num_bins):
        lo = idx / num_bins
        hi = (idx + 1) / num_bins
        if idx == num_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += (
                mask.double().mean()
                * torch.abs(corr[mask].mean() - conf[mask].mean())
            )
    return float(ece.item())


def _parse_proto_key(key: str) -> str | None:
    text = str(key).strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    text = text.replace("_", ",")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    try:
        return f"({int(parts[0])},{int(parts[1])})"
    except ValueError:
        return None


def load_semantic_protos(
    checkpoint: dict[str, Any],
    semantic_protos_path: str | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load semantic prototypes from checkpoint or semantic_protos_latest.json."""
    raw = None
    if semantic_protos_path:
        payload = json.loads(Path(semantic_protos_path).read_text(encoding="utf-8"))
        raw = payload.get("semantic_protos", payload)
    elif "semantic_protos" in checkpoint:
        raw = checkpoint.get("semantic_protos")
    if not raw:
        return {}
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        parsed = _parse_proto_key(key)
        if parsed is None:
            continue
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().to(device).view(-1)
        else:
            tensor = torch.tensor(value, dtype=torch.float32, device=device).view(-1)
        out[parsed] = tensor
    return out


def evaluate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ece_bins: int,
    inference_mode: str = "logits",
    semantic_protos: dict[str, torch.Tensor] | None = None,
    soft_agg_temperature: float = 0.35,
) -> dict[str, Any]:
    """Return accuracy, NLL, ECE, margins, and confusion matrix."""
    model.eval()
    total = 0
    correct_count = 0
    nll_sum = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    all_confidences = []
    all_correct = []
    all_margins = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y_cls = batch[1].to(device).long()
            logits, cls_feat, _ = model(x)
            if inference_mode == "soft_agg":
                if not semantic_protos:
                    raise ValueError("--inference-mode soft_agg requires semantic prototypes from checkpoint or --semantic-protos")
                probs = soft_aggregate_probs(
                    cls_feat,
                    semantic_protos,
                    temperature=soft_agg_temperature,
                )
                true_probs = probs.gather(1, y_cls.view(-1, 1)).clamp_min(1e-12)
                nll_sum += float((-torch.log(true_probs)).sum().item())
            else:
                nll_sum += float(F.cross_entropy(logits, y_cls, reduction="sum").item())
                probs = F.softmax(logits, dim=1)
            top2 = torch.topk(probs, k=min(2, probs.size(1)), dim=1).values
            confidences, preds = probs.max(dim=1)
            margins = top2[:, 0] - top2[:, 1] if top2.size(1) == 2 else confidences
            batch_correct = preds.eq(y_cls)

            total += int(y_cls.numel())
            correct_count += int(batch_correct.sum().item())
            all_confidences.append(confidences.detach().cpu())
            all_correct.append(batch_correct.detach().cpu())
            all_margins.append(margins.detach().cpu())
            for true_id, pred_id in zip(y_cls.detach().cpu(), preds.detach().cpu()):
                confusion[int(true_id), int(pred_id)] += 1

    if total == 0:
        return {
            "num_examples": 0,
            "accuracy": 0.0,
            "macro_accuracy": 0.0,
            "nll": 0.0,
            "ece": 0.0,
            "mean_confidence": 0.0,
            "mean_margin": 0.0,
            "per_class_accuracy": {},
            "confusion_matrix": confusion.tolist(),
        }

    confidences = torch.cat(all_confidences)
    correct = torch.cat(all_correct)
    margins = torch.cat(all_margins)
    per_class_accuracy: dict[str, float | None] = {}
    valid_class_acc = []
    for cls_id in range(num_classes):
        denom = int(confusion[cls_id].sum().item())
        if denom == 0:
            per_class_accuracy[str(cls_id)] = None
            continue
        value = float(confusion[cls_id, cls_id].item() / denom)
        per_class_accuracy[str(cls_id)] = value
        valid_class_acc.append(value)

    return {
        "num_examples": int(total),
        "accuracy": float(correct_count / total),
        "macro_accuracy": (
            float(sum(valid_class_acc) / len(valid_class_acc)) if valid_class_acc else 0.0
        ),
        "nll": float(nll_sum / total),
        "ece": expected_calibration_error(confidences, correct, ece_bins),
        "mean_confidence": float(confidences.mean().item()),
        "mean_margin": float(margins.mean().item()),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a GAPS Flower checkpoint")
    parser.add_argument("--checkpoint", default="server_latest.pth")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--client-ids", default="1")
    parser.add_argument("--split", choices=SPLIT_CHOICES, default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--inference-mode", choices=("logits", "soft_agg"), default="logits")
    parser.add_argument("--semantic-protos", default="", help="Path to semantic_protos_latest.json for soft_agg; optional if checkpoint contains semantic_protos")
    parser.add_argument("--soft-agg-temperature", type=float, default=0.35)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, config, checkpoint = load_checkpoint_model(
        args.checkpoint,
        device,
        args.batch_size,
    )
    client_ids = parse_client_ids(args.client_ids)
    semantic_protos = load_semantic_protos(
        checkpoint, args.semantic_protos or None, device
    ) if args.inference_mode == "soft_agg" else {}

    rows = []
    total_examples = 0
    weighted_acc = 0.0
    weighted_nll = 0.0
    weighted_ece = 0.0
    for client_id in client_ids:
        loader = make_loader(args.data_root, client_id, args.split, config.BATCH_SIZE)
        metrics = evaluate_classification(
            model=model,
            loader=loader,
            device=device,
            num_classes=args.num_classes,
            ece_bins=args.ece_bins,
            inference_mode=args.inference_mode,
            semantic_protos=semantic_protos,
            soft_agg_temperature=args.soft_agg_temperature,
        )
        num_examples = int(metrics["num_examples"])
        accuracy = float(metrics["accuracy"])
        total_examples += num_examples
        weighted_acc += accuracy * num_examples
        weighted_nll += float(metrics["nll"]) * num_examples
        weighted_ece += float(metrics["ece"]) * num_examples
        rows.append({"client_id": client_id, **metrics})

    summary = {
        "checkpoint": str(args.checkpoint),
        "round": int(checkpoint.get("round", -1)),
        "adaptive": bool(checkpoint.get("adaptive", False)),
        "split": args.split,
        "inference_mode": args.inference_mode,
        "semantic_proto_count": int(len(semantic_protos)),
        "soft_agg_temperature": float(args.soft_agg_temperature),
        "device": str(device),
        "num_tensors": len(checkpoint.get("model_state", {})),
        "clients": rows,
        "weighted_accuracy": weighted_acc / total_examples if total_examples else 0.0,
        "weighted_nll": weighted_nll / total_examples if total_examples else 0.0,
        "weighted_ece": weighted_ece / total_examples if total_examples else 0.0,
        "total_examples": total_examples,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
