"""Export classification-backbone confidence and embedding features.

The first use case is the official F6 round-25 adapted Flower checkpoint.  The
helpers are kept small and testable so the CSV schema is stable before running
large checkpoint exports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from gaps_flower.evaluate_checkpoint import load_checkpoint_model, make_loader, resolve_device


def class_probability_metrics(probs: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(probs, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("probability vector is empty")
    order = np.argsort(-values)
    top1 = float(values[order[0]])
    top2 = float(values[order[1]]) if values.size > 1 else 0.0
    entropy = float(-(values * np.log(np.maximum(values, 1e-12))).sum())
    return {
        "pred_class": int(order[0]),
        "confidence": top1,
        "margin": round(float(top1 - top2), 12),
        "entropy": entropy,
    }


def feature_column_names(
    num_classes: int,
    cls_dim: int,
    reg_dim: int,
    pred_prefix: str = "f6_r25",
) -> list[str]:
    return [
        f"pred_class_{pred_prefix}",
        *[f"prob_{idx}" for idx in range(num_classes)],
        "confidence",
        "margin",
        "entropy",
        *[f"cls_feat_{idx:03d}" for idx in range(cls_dim)],
        *[f"reg_feat_{idx:03d}" for idx in range(reg_dim)],
    ]


def build_feature_row(
    *,
    client: str,
    split: str,
    sample_index: int,
    probs: np.ndarray,
    cls_feat: np.ndarray,
    reg_feat: np.ndarray,
    pred_prefix: str,
) -> dict[str, Any]:
    metrics = class_probability_metrics(probs)
    row: dict[str, Any] = {
        "client": client,
        "split": split,
        "sample_index": int(sample_index),
        f"pred_class_{pred_prefix}": metrics["pred_class"],
        "confidence": metrics["confidence"],
        "margin": metrics["margin"],
        "entropy": metrics["entropy"],
    }
    for idx, value in enumerate(np.asarray(probs, dtype=np.float64).reshape(-1)):
        row[f"prob_{idx}"] = float(value)
    for idx, value in enumerate(np.asarray(cls_feat, dtype=np.float64).reshape(-1)):
        row[f"cls_feat_{idx:03d}"] = float(value)
    for idx, value in enumerate(np.asarray(reg_feat, dtype=np.float64).reshape(-1)):
        row[f"reg_feat_{idx:03d}"] = float(value)
    return row


def parse_clients(text: str) -> list[int]:
    return [int(item.strip().upper().replace("C", "")) for item in text.split(",") if item.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_split(
    *,
    model: torch.nn.Module,
    data_root: str | Path,
    client_ids: list[int],
    split: str,
    batch_size: int,
    device: torch.device,
    pred_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for client_id in client_ids:
            loader = make_loader(data_root, client_id, split, batch_size)
            offset = 0
            for batch in loader:
                x = batch[0].to(device)
                logits, cls_feat, reg_feat = model(x)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                cls_np = cls_feat.detach().cpu().numpy()
                reg_np = reg_feat.detach().cpu().numpy()
                for local_idx in range(probs.shape[0]):
                    rows.append(
                        build_feature_row(
                            client=f"C{client_id}",
                            split=split,
                            sample_index=offset + local_idx,
                            probs=probs[local_idx],
                            cls_feat=cls_np[local_idx],
                            reg_feat=reg_np[local_idx],
                            pred_prefix=pred_prefix,
                        )
                    )
                offset += probs.shape[0]
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, config, checkpoint = load_checkpoint_model(args.checkpoint, device, args.batch_size)
    client_ids = parse_clients(args.clients)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    row_counts: dict[str, int] = {}
    feature_dims: dict[str, int] = {}
    for split in splits:
        rows = export_split(
            model=model,
            data_root=args.data_root,
            client_ids=client_ids,
            split=split,
            batch_size=config.BATCH_SIZE,
            device=device,
            pred_prefix=args.pred_prefix,
        )
        row_counts[split] = len(rows)
        if rows:
            feature_dims = {
                "num_classes": len([key for key in rows[0] if key.startswith("prob_")]),
                "cls_dim": len([key for key in rows[0] if key.startswith("cls_feat_")]),
                "reg_dim": len([key for key in rows[0] if key.startswith("reg_feat_")]),
            }
        write_csv(output_dir / f"backbone_features_{split}.csv", rows)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "round": int(checkpoint.get("round", -1)) if isinstance(checkpoint, dict) else -1,
        "adaptive": bool(checkpoint.get("adaptive", False)) if isinstance(checkpoint, dict) else False,
        "diagnostic_only": bool(args.diagnostic_only),
        "data_root": str(args.data_root),
        "clients": [f"C{client_id}" for client_id in client_ids],
        "splits": splits,
        "pred_prefix": args.pred_prefix,
        "row_counts": row_counts,
        "feature_dims": feature_dims,
        "outputs": [f"backbone_features_{split}.csv" for split in splits],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "row_counts": row_counts}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--clients", default="3,4,5")
    parser.add_argument("--splits", default="calibration,test")
    parser.add_argument("--pred-prefix", default="f6_r25")
    parser.add_argument("--output-dir", default="results/f6_r25_backbone_feature_export_20260630")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--diagnostic-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
