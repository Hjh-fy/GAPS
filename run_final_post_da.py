"""Run final server-only domain adaptation after a Flower run.

This is used for communication-compression experiments:

1. Train with fewer Flower communication rounds, e.g. 10R strong DA.
2. Starting from the final adapted checkpoint, run extra server-side DA steps
   using source calibration and target calibration splits only.
3. Save a new checkpoint such as ``server_latest_postda300.pth``.

The script is posthoc and does not contact Flower clients, so post-DA steps do
not add communication rounds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from federated_dataset import GasSensorWindowDataset
from gaps_flower.domain_adaptation import ServerDomainAdaptation
from gaps_flower.task import create_model, make_config


FIXED_DA_STRONG_HP = {
    "USE_DEEP_CORAL": True,
    "USE_MMD_ALIGNMENT": True,
    "USE_ADVERSARIAL_DOMAIN": True,
    "CORAL_CLASS_CONDITIONAL": True,
    "LAMBDA_DEEP_CORAL": 0.5,
    "LAMBDA_GLOBAL_MMD": 0.5,
    "LAMBDA_CLASS_MMD": 0.5,
    "LAMBDA_PROTO_ANCHOR": 0.3,
    "LAMBDA_ADV_DOMAIN": 0.5,
    "LAMBDA_TARGET_CE": 0.0,
    "LAMBDA_PROTO": 0.05,
    "LAMBDA_CONSISTENCY": 2.0,
    "LAMBDA_RES": 0.1,
    "LAMBDA_PROTO_MMD": 0.2,
    "LAMBDA_STAGE_MMD": 0.2,
    "USE_ALIGN_REG_LEGACY": False,
    "LAMBDA_ALIGN_REG_LEGACY": 0.05,
    "USE_CONTRASTIVE_CONSISTENCY": True,
    "USE_PROTO_MMD": True,
    "USE_PROTO_DECOUPLING": True,
    "TARGET_CE_LABEL_SMOOTHING": 0.0,
    "TARGET_CE_CLASS_BALANCED": False,
    "SERVER_OPT_LR": 0.0005,
    "HIDDEN_DIM2": 64,
    "NUM_CLASSES": 4,
    "MAX_VAL_BATCHES": 10,
    "ADV_DOMAIN_LR": 0.001,
    "ADV_CRITIC_ITERS": 3,
    "ADV_GRADIENT_PENALTY": 10.0,
    "ADV_CLASS_CONDITIONAL": True,
    "DA_LEARN_SEMANTIC_PROTOS": True,
}


def parse_client_ids(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_data_dirs(data_root: str | Path, client_ids: list[int]) -> list[Path]:
    root = Path(data_root)
    return [root / f"client_{client_id}" for client_id in client_ids]


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def load_calibration_loader(
    data_dirs: list[Path],
    *,
    batch_size: int,
    sample_limit: int,
    strict_calibration_split: bool,
    seed: int,
) -> DataLoader:
    all_features = []
    all_cls_labels = []
    all_phase_labels = []

    for data_dir in data_dirs:
        if strict_calibration_split:
            prefix = "calibration_"
            feat_path = data_dir / "calibration_features.npy"
            if not feat_path.exists():
                raise FileNotFoundError(f"Missing calibration_features.npy: {data_dir}")
        else:
            prefix = ""
            for candidate in ("calibration_", "test_", "train_", ""):
                if (data_dir / f"{candidate}features.npy").exists():
                    prefix = candidate
                    break

        features = np.load(data_dir / f"{prefix}features.npy")
        cls_path = data_dir / f"{prefix}classification_labels.npy"
        if not cls_path.exists():
            cls_path = data_dir / "classification_labels.npy"
        cls_labels = np.load(cls_path)
        phase_path = data_dir / f"{prefix}phase_labels.npy"
        if phase_path.exists():
            phase_labels = np.load(phase_path, allow_pickle=True)
        else:
            phase_labels = np.full(len(features), -1, dtype=np.int64)

        all_features.append(features)
        all_cls_labels.append(cls_labels)
        all_phase_labels.append(phase_labels)
        print(f"[post-DA] {data_dir}: {len(features)} calibration samples")

    features = np.concatenate(all_features, axis=0)
    cls_labels = np.concatenate(all_cls_labels, axis=0)
    phase_labels = np.concatenate(all_phase_labels, axis=0)
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=np.zeros((len(features), 4), dtype=np.float32),
        classification_labels=cls_labels,
        phase_labels=phase_labels,
        normalize=False,
        mean_std=None,
    )
    if sample_limit > 0 and len(dataset) > sample_limit:
        indices = np.random.RandomState(seed).choice(len(dataset), size=sample_limit, replace=False)
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def tensorize_semantic_protos(raw: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in (raw or {}).items():
        if isinstance(value, torch.Tensor):
            out[str(key)] = value.detach().float().to(device).view(-1)
        else:
            out[str(key)] = torch.tensor(value, dtype=torch.float32, device=device).view(-1)
    return out


def load_semantic_protos(
    checkpoint: dict[str, Any],
    semantic_protos_path: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if semantic_protos_path:
        payload = json.loads(Path(semantic_protos_path).read_text(encoding="utf-8"))
        return tensorize_semantic_protos(payload.get("semantic_protos", payload), device)
    return tensorize_semantic_protos(checkpoint.get("semantic_protos", {}), device)


def checkpoint_delta(base_state: dict[str, torch.Tensor], adapted_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    changed_tensors = 0
    compared_tensors = 0
    max_abs_delta = 0.0
    mean_abs_delta_sum = 0.0
    for key, base_tensor in base_state.items():
        if key not in adapted_state:
            continue
        delta = (adapted_state[key].detach().cpu().float() - base_tensor.detach().cpu().float()).abs()
        tensor_max = float(delta.max().item()) if delta.numel() else 0.0
        tensor_mean = float(delta.mean().item()) if delta.numel() else 0.0
        compared_tensors += 1
        mean_abs_delta_sum += tensor_mean
        max_abs_delta = max(max_abs_delta, tensor_max)
        if tensor_max > 0.0:
            changed_tensors += 1
    return {
        "checkpoint_changed_tensors": int(changed_tensors),
        "checkpoint_compared_tensors": int(compared_tensors),
        "checkpoint_max_abs_delta": float(max_abs_delta),
        "checkpoint_mean_abs_delta": (
            float(mean_abs_delta_sum / compared_tensors) if compared_tensors else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final posthoc server-side DA on a Flower checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-client-ids", default="1,2")
    parser.add_argument("--target-client-ids", default="3,4,5")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-diagnostics", default="")
    parser.add_argument("--semantic-protos", default="")
    parser.add_argument("--profile", default="strong_cls")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--strict-calibration-split", type=lambda v: str(v).lower() in {"true", "1", "yes"}, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    state = checkpoint.get("model_state")
    if state is None:
        raise ValueError(f"Checkpoint has no model_state: {checkpoint_path}")

    config = make_config(device=str(device), local_epochs=1, batch_size=args.batch_size, profile=args.profile)
    model = create_model(config)
    model.load_state_dict(state, strict=True)
    model.to(device)

    source_dirs = parse_data_dirs(args.data_root, parse_client_ids(args.source_client_ids))
    target_dirs = parse_data_dirs(args.data_root, parse_client_ids(args.target_client_ids))
    val_loader = load_calibration_loader(
        source_dirs,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        strict_calibration_split=args.strict_calibration_split,
        seed=args.seed,
    )
    calib_loader = load_calibration_loader(
        target_dirs,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        strict_calibration_split=args.strict_calibration_split,
        seed=args.seed,
    )
    semantic_protos = load_semantic_protos(checkpoint, args.semantic_protos, device)

    trainer = ServerDomainAdaptation(
        model=model,
        val_loader=val_loader,
        calib_loader=calib_loader,
        semantic_protos=semantic_protos,
        device=device,
        hyperparams=dict(FIXED_DA_STRONG_HP),
    )
    adapted_model, diagnostics = trainer.run_adaptation(num_steps=int(args.steps))
    adapted_state = {key: value.detach().cpu().clone() for key, value in adapted_model.state_dict().items()}
    diagnostics.update(checkpoint_delta(state, adapted_state))
    diagnostics.update(
        {
            "post_da": True,
            "post_da_steps": int(args.steps),
            "source_checkpoint": str(checkpoint_path),
            "source_client_ids": parse_client_ids(args.source_client_ids),
            "target_client_ids": parse_client_ids(args.target_client_ids),
            "sample_limit": int(args.sample_limit),
            "strict_calibration_split": bool(args.strict_calibration_split),
            "device": str(device),
        }
    )

    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "round": int(checkpoint.get("round", -1)),
        "model_state": adapted_state,
        "parameter_keys": checkpoint.get("parameter_keys", list(adapted_state.keys())),
        "run_name": checkpoint.get("run_name", ""),
        "adaptive": True,
        "post_da": True,
        "post_da_steps": int(args.steps),
        "source_checkpoint": str(checkpoint_path),
        "diagnostics": diagnostics,
        "semantic_protos": {
            key: value.detach().cpu().clone()
            for key, value in trainer.get_semantic_protos().items()
        },
    }
    torch.save(payload, output_checkpoint)

    diag_path = Path(args.output_diagnostics) if args.output_diagnostics else output_checkpoint.with_suffix(".json")
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"checkpoint: {output_checkpoint}")
    print(f"diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
