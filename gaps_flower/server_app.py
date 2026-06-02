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
    parser.add_argument("--proto-ema-alpha", type=float, default=0.8,
                        help="语义原型 EMA 平滑系数 (仅 --strategy gaps 生效)")
    parser.add_argument("--use-selective-agg", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="是否启用选择性聚合 (仅 --strategy gaps 生效)")
    parser.add_argument("--selective-warmup", type=int, default=3,
                        help="选择性聚合预热轮数，前 N 轮使用标准 FedAvg")
    parser.add_argument("--selective-min-scale", type=float, default=0.3,
                        help="选择性聚合最小缩放因子，防止权重归零")
    parser.add_argument("--use-proto-mmd", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="是否计算原型级域漂移诊断 (仅 --strategy gaps 生效)")
    parser.add_argument("--use-domain-adapt", type=lambda v: v.lower() in ("true", "1", "yes"), default=False,
                        help="是否启用服务端域适应 CORAL/MMD/对抗 (需 --server-val-data 和 --server-calib-data)")
    parser.add_argument("--server-val-data", type=str, default=None,
                        help="源域验证集目录: 源域 training client 的 calibration_features.npy (如 client_1,client_2)")
    parser.add_argument("--server-calib-data", type=str, default=None,
                        help="目标域校准集目录: 目标域 test client 的 calibration_features.npy (如 client_3)")
    parser.add_argument("--domain-adapt-steps", type=int, default=30,
                        help="域适应优化步数 K")
    parser.add_argument("--domain-adapt-warmup", type=int, default=3,
                        help="域适应预热轮数")
    parser.add_argument("--da-use-coral", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="域适应是否启用 Deep CORAL 损失")
    parser.add_argument("--da-use-mmd", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="域适应是否启用 MMD 对齐损失")
    parser.add_argument("--da-use-adversarial", type=lambda v: v.lower() in ("true", "1", "yes"), default=False,
                        help="域适应是否启用对抗训练 (WGAN-GP + GRL)")
    parser.add_argument("--da-device", type=str, default="cpu",
                        help="域适应计算设备 (cpu 或 cuda)")
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
        strategy = GapsStrategy(
            proto_ema_alpha=args.proto_ema_alpha,
            use_selective_agg=args.use_selective_agg,
            selective_warmup=args.selective_warmup,
            selective_min_scale=args.selective_min_scale,
            use_proto_mmd=args.use_proto_mmd,
            use_domain_adapt=args.use_domain_adapt,
            server_val_data=args.server_val_data,
            server_calib_data=args.server_calib_data,
            domain_adapt_steps=args.domain_adapt_steps,
            domain_adapt_warmup=args.domain_adapt_warmup,
            da_use_coral=args.da_use_coral,
            da_use_mmd=args.da_use_mmd,
            da_use_adversarial=args.da_use_adversarial,
            da_device=args.da_device,
            **strategy_kwargs,
        )
    else:
        strategy = CheckpointFedAvg(**strategy_kwargs)

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
