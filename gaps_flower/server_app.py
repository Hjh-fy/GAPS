"""Flower server entrypoint for Alibaba Cloud ECS."""

from __future__ import annotations

import argparse

import flwr as fl
from flwr.common import ndarrays_to_parameters

from gaps_flower.strategy import CheckpointFedAvg, GapsStrategy, weighted_average
from gaps_flower.task import create_model, get_parameters, make_config

DEFAULT_STRATEGIES = ("fedavg", "gaps")


def fit_config(server_round: int) -> dict:
    return {"server_round": server_round}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAPS Flower server")
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=1)
    parser.add_argument("--output-dir", default="results/flower_server")
    parser.add_argument("--run-name", default="flower_smoke")
    parser.add_argument("--save-history", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="是否保存 history.json 记录 (True/False)")
    parser.add_argument("--strategy", choices=DEFAULT_STRATEGIES, default="fedavg",
                        help="聚合策略: fedavg (Flower默认FedAvg) 或 gaps (GAPS自定义聚合)")
    args = parser.parse_args()

    config = make_config(device="cpu", local_epochs=1, batch_size=32)
    model = create_model(config)
    initial_arrays, parameter_keys = get_parameters(model)

    strategy_kwargs = dict(
        parameter_keys=parameter_keys,
        reference_state=model.state_dict(),
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_history=args.save_history,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=ndarrays_to_parameters(initial_arrays),
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    if args.strategy == "gaps":
        strategy = GapsStrategy(**strategy_kwargs)
    else:
        strategy = CheckpointFedAvg(**strategy_kwargs)

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
