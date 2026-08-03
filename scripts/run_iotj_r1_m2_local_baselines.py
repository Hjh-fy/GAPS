"""Run the three non-distributed R1-M2 seed-42 baselines.

The target test split is loaded only after training/model selection is complete.
Direct Standardization (DS) selects its ridge penalty with calibration-only
group folds and then performs one sealed target-test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from client import Client
from federated_dataset import GasSensorWindowDataset, create_client_test_only_loader
from gaps_flower.evaluate_checkpoint import evaluate_classification, load_checkpoint_model
from gaps_flower.task import create_model, make_config
from utils import set_random_seed


EXPERIMENTS = {
    "target-only": "R1M2-TARGET-ONLY-S42",
    "central-source": "R1M2-CENTRAL-SOURCE-S42",
    "ds-fedavg": "R1M2-DS-FEDAVG-S42",
}
SEED = 42
ROUNDS = 25
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 5e-4
TARGET_STEPS = 2500
DS_ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)


class PeakRSS:
    def __init__(self, interval_seconds: float = 0.2):
        self.interval_seconds = interval_seconds
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, int(process.memory_info().rss))
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def load_split(client_dir: Path, prefix: str) -> tuple[np.ndarray, ...]:
    return (
        np.load(client_dir / f"{prefix}_features.npy", allow_pickle=False),
        np.load(client_dir / f"{prefix}_regression_labels.npy", allow_pickle=False),
        np.load(client_dir / f"{prefix}_classification_labels.npy", allow_pickle=False),
        np.load(client_dir / f"{prefix}_phase_labels.npy", allow_pickle=False),
    )


def make_loader(parts: list[tuple[np.ndarray, ...]], *, shuffle: bool) -> DataLoader:
    arrays = [np.concatenate(items, axis=0) for items in zip(*parts)]
    dataset = GasSensorWindowDataset(
        features=arrays[0],
        regression_labels=arrays[1],
        classification_labels=arrays[2],
        phase_labels=arrays[3],
        normalize=False,
    )
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def enrich_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    f1s = []
    recalls = {}
    for class_id in range(confusion.shape[0]):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[str(class_id)] = float(recall)
        f1s.append(float(f1))
    out = dict(metrics)
    out["macro_f1"] = float(np.mean(f1s))
    out["per_class_recall"] = recalls
    return out


def evaluate_model(model: torch.nn.Module, data_root: Path, device: torch.device) -> dict[str, Any]:
    # Deliberately open the target test split only at this final stage.
    loader = create_client_test_only_loader(
        data_root / "client_5", batch_size=BATCH_SIZE
    )
    return enrich_metrics(
        evaluate_classification(
            model,
            loader,
            device,
            num_classes=4,
            ece_bins=15,
        )
    )


def train_classifier(
    loader: DataLoader,
    *,
    device: torch.device,
    rounds: int,
    local_epochs: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_random_seed(SEED)
    config = make_config(
        device=str(device),
        local_epochs=local_epochs,
        batch_size=BATCH_SIZE,
        profile="ce_only",
        seed=SEED,
    )
    config.GLOBAL_ROUNDS = rounds
    config.LR_CLIENT = LR
    model = create_model(config)
    client = Client(client_id=0, config=config)
    client.set_model(model)
    client.update_dataloader(loader)
    started = time.perf_counter()
    for round_idx in range(1, rounds + 1):
        client.train_one_round(current_round=round_idx)
        print(
            json.dumps(
                {"event": "round_complete", "round": round_idx, "rounds": rounds},
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    steps = math.ceil(len(loader.dataset) / BATCH_SIZE) * local_epochs * rounds
    return model, {
        "training_seconds": float(elapsed),
        "optimizer_steps": int(steps),
        "samples": int(len(loader.dataset)),
        "rounds": int(rounds),
        "local_epochs": int(local_epochs),
    }


def save_classifier_checkpoint(
    path: Path,
    model: torch.nn.Module,
    experiment_id: str,
    training: dict[str, Any],
) -> None:
    torch.save(
        {
            "round": int(training["rounds"]),
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "run_name": experiment_id,
            "seed": SEED,
            "training": training,
        },
        path,
    )


def _info_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["classification_label"]),
        str(row["concentration_code"]),
        str(row["phase_label"]),
    )


def load_group_centroids(client_dir: Path) -> dict[tuple[int, str, str], np.ndarray]:
    features = np.load(client_dir / "calibration_features.npy", allow_pickle=False)
    info = json.loads(
        (client_dir / "calibration_experiment_info.json").read_text(encoding="utf-8")
    )
    if len(features) != len(info):
        raise ValueError(f"calibration row mismatch: {client_dir}")
    grouped: dict[tuple[int, str, str], list[np.ndarray]] = defaultdict(list)
    for feature, row in zip(features, info):
        grouped[_info_key(row)].append(np.asarray(feature, dtype=np.float64))
    return {key: np.mean(values, axis=0) for key, values in grouped.items()}


def fit_ds_mapping(
    target: dict[tuple[int, str, str], np.ndarray],
    source: dict[tuple[int, str, str], np.ndarray],
    keys: list[tuple[int, str, str]],
    alpha: float,
) -> np.ndarray:
    x = np.concatenate([target[key] for key in keys], axis=0)
    y = np.concatenate([source[key] for key in keys], axis=0)
    design = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)


def apply_ds(features: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    shape = features.shape
    flat = np.asarray(features, dtype=np.float64).reshape(-1, shape[-1])
    design = np.concatenate([flat, np.ones((len(flat), 1), dtype=np.float64)], axis=1)
    return (design @ mapping).reshape(shape).astype(np.float32)


def run_ds(
    data_root: Path,
    a0_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    source_parts = [load_group_centroids(data_root / f"client_{cid}") for cid in (1, 2)]
    source_keys = set(source_parts[0]) & set(source_parts[1])
    source = {
        key: np.mean([part[key] for part in source_parts], axis=0)
        for key in source_keys
    }
    target = load_group_centroids(data_root / "client_5")
    matched = sorted(set(source) & set(target))
    if len(matched) < 8:
        raise RuntimeError(f"DS requires at least 8 matched calibration strata, found {len(matched)}")

    folds = [matched[index::4] for index in range(4)]
    cv = {}
    for alpha in DS_ALPHAS:
        errors = []
        for held in folds:
            train_keys = [key for key in matched if key not in set(held)]
            mapping = fit_ds_mapping(target, source, train_keys, alpha)
            for key in held:
                prediction = apply_ds(target[key][None, ...], mapping)[0]
                errors.append(float(np.mean((prediction - source[key]) ** 2)))
        cv[str(alpha)] = float(np.sqrt(np.mean(errors)))
    selected_alpha = min(DS_ALPHAS, key=lambda value: (cv[str(value)], value))
    mapping = fit_ds_mapping(target, source, matched, selected_alpha)
    mapping_path = output_dir / "ds_mapping.npz"
    np.savez(
        mapping_path,
        mapping=mapping,
        alpha=np.asarray(selected_alpha),
        matched_strata=np.asarray(["|".join(map(str, key)) for key in matched]),
    )

    model, _config, _checkpoint = load_checkpoint_model(
        str(a0_checkpoint), device, BATCH_SIZE
    )
    # This is the first point at which target test features are opened.
    test_parts = load_split(data_root / "client_5", "test")
    transformed = apply_ds(test_parts[0], mapping)
    loader = make_loader([(transformed, *test_parts[1:])], shuffle=False)
    metrics = enrich_metrics(
        evaluate_classification(model, loader, device, num_classes=4, ece_bins=15)
    )
    adaptation = {
        "method": "regularized_affine_direct_standardization_target_to_source",
        "selected_alpha": float(selected_alpha),
        "calibration_cv_rmse": cv,
        "matched_strata": int(len(matched)),
        "mapping_shape": list(mapping.shape),
        "a0_checkpoint": str(a0_checkpoint.resolve()),
        "a0_checkpoint_sha256": sha256_file(a0_checkpoint),
    }
    return metrics, adaptation, mapping_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument(
        "--data-root",
        default="dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid",
    )
    parser.add_argument(
        "--output-root",
        default="results/iotj_r1_m2_baseline_fairness_seed42_20260803",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--a0-checkpoint", default="")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    data_root = (repo_root / args.data_root).resolve()
    experiment_id = EXPERIMENTS[args.experiment]
    output_dir = (repo_root / args.output_root / experiment_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty formal output: {output_dir}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    set_random_seed(SEED)

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_start = time.perf_counter()
    training: dict[str, Any] = {}
    checkpoint_path: Path | None = None
    adaptation: dict[str, Any] | None = None
    with PeakRSS() as memory:
        if args.experiment == "target-only":
            loader = make_loader([load_split(data_root / "client_5", "calibration")], shuffle=True)
            batches = math.ceil(len(loader.dataset) / BATCH_SIZE)
            epochs = TARGET_STEPS // batches
            if epochs * batches != TARGET_STEPS:
                raise RuntimeError("target optimizer-step budget is not exactly divisible")
            model, training = train_classifier(
                loader, device=device, rounds=1, local_epochs=epochs
            )
            checkpoint_path = output_dir / "server_latest.pth"
            save_classifier_checkpoint(checkpoint_path, model, experiment_id, training)
            metrics = evaluate_model(model, data_root, device)
        elif args.experiment == "central-source":
            loader = make_loader(
                [load_split(data_root / f"client_{cid}", "train") for cid in (1, 2)],
                shuffle=True,
            )
            model, training = train_classifier(
                loader, device=device, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS
            )
            checkpoint_path = output_dir / "server_latest.pth"
            save_classifier_checkpoint(checkpoint_path, model, experiment_id, training)
            metrics = evaluate_model(model, data_root, device)
        else:
            if not args.a0_checkpoint:
                raise ValueError("--a0-checkpoint is required for ds-fedavg")
            a0_checkpoint = Path(args.a0_checkpoint).resolve()
            if not a0_checkpoint.is_file():
                raise FileNotFoundError(a0_checkpoint)
            metrics, adaptation, checkpoint_path = run_ds(
                data_root, a0_checkpoint, output_dir, device
            )

    wall_seconds = time.perf_counter() - wall_start
    manifest = {
        "schema_version": "iotj.r1_m2.seed42.local.v1",
        "experiment_id": experiment_id,
        "status": "completed",
        "seed": SEED,
        "started_utc": started_utc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": git_commit(repo_root),
        "data_root": str(data_root),
        "device": str(device),
        "platform": platform.platform(),
        "protocol": {
            "batch_size": BATCH_SIZE,
            "client_lr": LR,
            "rounds": ROUNDS if args.experiment == "central-source" else None,
            "local_epochs": LOCAL_EPOCHS if args.experiment == "central-source" else None,
            "target_optimizer_steps": TARGET_STEPS if args.experiment == "target-only" else None,
            "target_test_used_for_selection": False,
        },
        "training": training,
        "adaptation": adaptation,
        "metrics": metrics,
        "cost": {
            "wall_seconds": float(wall_seconds),
            "peak_process_rss_bytes": int(memory.peak),
            "communication_rounds": 0,
            "transmitted_bytes": 0,
        },
        "artifact": str(checkpoint_path),
        "artifact_sha256": sha256_file(checkpoint_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
