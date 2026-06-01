"""Flower server entrypoint for Alibaba Cloud ECS."""

from __future__ import annotations

import argparse

import flwr as fl
from flwr.common import ndarrays_to_parameters

from gaps_flower.task import create_model, get_parameters, make_config


def fit_config(server_round: int) -> dict:
    return {"server_round": server_round}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAPS Flower server")
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=1)
    args = parser.parse_args()

    config = make_config(device="cpu", local_epochs=1, batch_size=32)
    model = create_model(config)
    initial_arrays, _keys = get_parameters(model)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=ndarrays_to_parameters(initial_arrays),
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
