"""Flower client entrypoint for a local PC or Raspberry Pi edge node."""

from __future__ import annotations

import argparse
import time

import flwr as fl

from gaps_flower.task import (
    create_model,
    evaluate,
    get_parameters,
    load_client_loaders,
    make_client,
    make_config,
    set_parameters,
    train_one_round,
)


class GapsFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, data_root: str, device: str, local_epochs: int, batch_size: int):
        self.client_id = client_id
        self.config = make_config(device=device, local_epochs=local_epochs, batch_size=batch_size)
        self.model = create_model(self.config)
        self.parameter_keys = get_parameters(self.model)[1]
        train_loader, test_loader = load_client_loaders(data_root, client_id, self.config)
        self.gaps_client = make_client(client_id, self.model, train_loader, self.config)
        self.test_loader = test_loader
        train_samples = len(train_loader.dataset)
        test_samples = len(test_loader.dataset)
        print(
            f"[GAPS client {client_id}] ready: train_samples={train_samples}, "
            f"test_samples={test_samples}, device={device}, local_epochs={local_epochs}"
        )

    def get_parameters(self, config):
        arrays, _keys = get_parameters(self.model)
        return arrays

    def fit(self, parameters, config):
        start = time.perf_counter()
        set_parameters(self.model, parameters, self.parameter_keys)
        round_idx = int(config.get("server_round", 1)) if config else 1
        arrays, num_examples, metrics = train_one_round(self.gaps_client, round_idx)
        elapsed = time.perf_counter() - start
        metrics.update({
            "fit_seconds": float(elapsed),
            "round": int(round_idx),
        })
        print(
            f"[GAPS client {self.client_id}] fit round={round_idx}, "
            f"samples={num_examples}, seconds={elapsed:.2f}"
        )
        return arrays, num_examples, metrics

    def evaluate(self, parameters, config):
        start = time.perf_counter()
        set_parameters(self.model, parameters, self.parameter_keys)
        loss, num_examples, metrics = evaluate(self.model, self.test_loader, self.config, client_id=self.client_id)
        elapsed = time.perf_counter() - start
        metrics["evaluate_seconds"] = float(elapsed)
        print(
            f"[GAPS client {self.client_id}] evaluate samples={num_examples}, "
            f"accuracy={metrics.get('accuracy', 0.0):.4f}, seconds={elapsed:.2f}"
        )
        return loss, num_examples, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GAPS Flower edge client")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    client = GapsFlowerClient(
        client_id=args.client_id,
        data_root=args.data_root,
        device=args.device,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
