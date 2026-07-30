"""Train a centralized TCN baseline on one generated five-fold split.

This is a pipeline/label sanity baseline, not the final federated experiment.
It uses the existing GAPS classification backbone with 3 classes and 6 inputs,
and reports both window-level and exposure-level metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import ConcatDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_dataset import GasSensorWindowDataset  # noqa: E402
from gaps_flower.task import create_model, make_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "client_data_lab_3gas_5fold_nominal_v1",
    )
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "lab_three_gas_centralized_baseline",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


def load_split_dataset(
    client_dir: Path,
    prefix: str,
) -> GasSensorWindowDataset:
    features = np.load(client_dir / f"{prefix}_features.npy")
    labels = np.load(client_dir / f"{prefix}_classification_labels.npy")
    phases = np.load(client_dir / f"{prefix}_phase_labels.npy")
    return GasSensorWindowDataset(
        features=features,
        classification_labels=labels,
        phase_labels=phases,
        normalize=False,
    )


def load_manifest(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_loaders(
    fold_dir: Path,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[dict], List[dict]]:
    train_sets = []
    val_sets = []
    test_sets = []
    val_manifest: List[dict] = []
    test_manifest: List[dict] = []
    for platform in (1, 2, 3):
        client_dir = fold_dir / f"client_{platform}"
        train_sets.append(load_split_dataset(client_dir, "train"))
        val_sets.append(load_split_dataset(client_dir, "calibration"))
        test_sets.append(load_split_dataset(client_dir, "test"))
        val_manifest.extend(
            load_manifest(client_dir / "calibration_window_manifest.csv")
        )
        test_manifest.extend(load_manifest(client_dir / "test_window_manifest.csv"))
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ConcatDataset(val_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ConcatDataset(test_sets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader, val_manifest, test_manifest


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1, 2],
        ).tolist(),
        "n_samples": int(len(labels)),
    }


def exposure_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    manifest: Sequence[dict],
) -> dict:
    if len(labels) != len(manifest):
        raise ValueError("Prediction count and manifest length differ")
    grouped: Dict[str, dict] = {}
    for index, row in enumerate(manifest):
        exposure_id = row["exposure_id"]
        item = grouped.setdefault(
            exposure_id,
            {
                "label": int(row["gas_label"]),
                "probabilities": [],
                "platform": int(row["platform"]),
            },
        )
        if item["label"] != int(labels[index]):
            raise ValueError(f"Manifest label mismatch for {exposure_id}")
        item["probabilities"].append(probabilities[index])
    exposure_labels = np.asarray(
        [item["label"] for item in grouped.values()],
        dtype=np.int64,
    )
    exposure_probabilities = np.stack(
        [
            np.mean(np.stack(item["probabilities"], axis=0), axis=0)
            for item in grouped.values()
        ],
        axis=0,
    )
    result = classification_metrics(exposure_labels, exposure_probabilities)
    result["n_exposures"] = result.pop("n_samples")
    result["by_platform"] = {}
    items = list(grouped.values())
    present_platforms = sorted({item["platform"] for item in items})
    for platform in present_platforms:
        indices = [i for i, item in enumerate(items) if item["platform"] == platform]
        result["by_platform"][str(platform)] = classification_metrics(
            exposure_labels[indices],
            exposure_probabilities[indices],
        )
    return result


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_labels = []
    all_probabilities = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y_cls, _y_reg, _phase in loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            logits, _cls_feat, _reg_feat = model(x)
            loss_sum += float(F.cross_entropy(logits, y_cls, reduction="sum").item())
            count += int(len(y_cls))
            all_labels.append(y_cls.cpu().numpy())
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        loss_sum / max(count, 1),
    )


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)
    fold_dir = args.data_root / f"fold_{args.fold}"
    train_loader, val_loader, test_loader, val_manifest, test_manifest = make_loaders(
        fold_dir,
        args.batch_size,
    )

    config = make_config(
        device=str(device),
        local_epochs=1,
        batch_size=args.batch_size,
        profile="ce_only",
        seed=args.seed,
        num_classes=3,
        input_dim=6,
        num_clients=3,
        num_phases=1,
    )
    config.USE_REG_LOSS = False
    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y_cls, _y_reg, _phase in train_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _cls_feat, _reg_feat = model(x)
            loss = F.cross_entropy(logits, y_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(y_cls)
            train_count += int(len(y_cls))

        val_labels, val_probs, val_loss = predict(model, val_loader, device)
        val_window = classification_metrics(val_labels, val_probs)
        val_exposure = exposure_metrics(val_labels, val_probs, val_manifest)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "validation_loss": val_loss,
            "validation_window_macro_f1": val_window["macro_f1"],
            "validation_exposure_macro_f1": val_exposure["macro_f1"],
        }
        history.append(epoch_row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"val_window_f1={val_window['macro_f1']:.4f} "
            f"val_exposure_f1={val_exposure['macro_f1']:.4f}"
        )
        if val_exposure["macro_f1"] > best_val_f1:
            best_val_f1 = val_exposure["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    val_labels, val_probs, val_loss = predict(model, val_loader, device)
    test_labels, test_probs, test_loss = predict(model, test_loader, device)
    result = {
        "task": "lab_three_gas_centralized_baseline",
        "fold": args.fold,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "input_dim": 6,
        "num_classes": 3,
        "best_validation_exposure_macro_f1": best_val_f1,
        "validation": {
            "loss": val_loss,
            "window": classification_metrics(val_labels, val_probs),
            "exposure": exposure_metrics(val_labels, val_probs, val_manifest),
        },
        "test": {
            "loss": test_loss,
            "window": classification_metrics(test_labels, test_probs),
            "exposure": exposure_metrics(test_labels, test_probs, test_manifest),
        },
        "history": history,
    }

    run_dir = args.output_dir / f"fold_{args.fold}_seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model_config": {
                "num_classes": 3,
                "input_dim": 6,
                "seq_len": 100,
                "num_phases": 1,
            },
            "fold": args.fold,
            "seed": args.seed,
        },
        run_dir / "best_model.pth",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "test_window_macro_f1="
        f"{result['test']['window']['macro_f1']:.4f}, "
        "test_exposure_macro_f1="
        f"{result['test']['exposure']['macro_f1']:.4f}"
    )
    return result


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
