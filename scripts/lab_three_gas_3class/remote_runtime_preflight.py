"""Small remote preflight for the content-addressed three-gas runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-src", type=Path, required=True)
    parser.add_argument("--role", choices=("server", "client"), required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--client-id", type=int)
    parser.add_argument(
        "--profile",
        choices=("strong_cls", "proto_replay"),
        default="strong_cls",
    )
    parser.add_argument("--input-dim", type=int, default=6)
    args = parser.parse_args()

    runtime_src = args.runtime_src.resolve()
    sys.path.insert(0, str(runtime_src))

    from gaps_flower.task import (  # noqa: PLC0415
        create_model,
        load_client_loaders,
        make_config,
    )

    config = make_config(
        device="cpu",
        local_epochs=3,
        batch_size=32,
        profile=args.profile,
        seed=42,
        num_classes=3,
        input_dim=args.input_dim,
        num_clients=3,
        num_phases=1,
    )
    model = create_model(config)
    payload = {
        "role": args.role,
        "runtime_src": str(runtime_src),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_classes": config.NUM_CLASSES,
        "input_dim": config.INPUT_DIM,
        "num_phases": config.NUM_PHASES,
    }
    if args.role == "client":
        if args.data_root is None or args.client_id is None:
            parser.error("client role requires --data-root and --client-id")
        train_loader, validation_loader = load_client_loaders(
            args.data_root,
            args.client_id,
            config,
            eval_split="calibration",
        )
        batch = next(iter(train_loader))
        if int(batch[0].shape[-1]) != args.input_dim:
            raise ValueError(
                f"Dataset input_dim={batch[0].shape[-1]} but expected "
                f"{args.input_dim}"
            )
        payload.update(
            {
                "client_id": args.client_id,
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "batch_shape": list(batch[0].shape),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
