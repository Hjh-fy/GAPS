"""Flower server entrypoint for Alibaba Cloud ECS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import flwr as fl
from flwr.common import ndarrays_to_parameters

from gaps_flower.domain_adaptation_inputs import validate_domain_adaptation_request
from gaps_flower.strategy import CheckpointFedAvg, GapsStrategy, weighted_average
from gaps_flower.task import CLASSIFICATION_PROFILE_FLAGS, create_model, get_parameters, make_config

DEFAULT_STRATEGIES = ("fedavg", "gaps")
PROFILE_CHOICES = tuple(CLASSIFICATION_PROFILE_FLAGS) + (
    "smoke",
    "gaps_cls",
    "gaps",
    "gaps_classification",
    "classification",
    "strong_cls",
)
DA_PRESETS = ("none", "default", "fixed_da_strong")
MMD_OBJECTIVES = ("legacy_quartic", "mmd2")
STAGE_ALIGNMENTS = ("legacy_intra_domain", "cross_domain_same_class_phase")
ADV_FEATURE_OBJECTIVES = ("legacy_grl_plus", "wasserstein_min")
FIXED_DA_STRONG = {
    "domain_adapt_steps": 100,
    "domain_adapt_warmup": 0,
    "da_use_coral": True,
    "da_use_mmd": True,
    "da_use_adversarial": True,
    "da_coral_class_conditional": True,
    "da_lambda_coral": 0.5,
    "da_lambda_global_mmd": 0.5,
    "da_lambda_class_mmd": 0.5,
    "da_lambda_adv": 0.5,
    "da_server_opt_lr": 0.0005,
    "use_adapted_as_global": True,
}


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit = set()
    for item in argv:
        option = item.split("=", 1)[0]
        if option in option_to_dest:
            explicit.add(option_to_dest[option])
    return explicit


def apply_da_preset(args: argparse.Namespace, explicit_dests: set[str]) -> None:
    if args.da_preset != "fixed_da_strong":
        return
    for key, value in FIXED_DA_STRONG.items():
        if key not in explicit_dests:
            setattr(args, key, value)


def save_run_config(args: argparse.Namespace, explicit_dests: set[str]) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "argv": sys.argv[1:],
        "explicit_args": sorted(explicit_dests),
        "args": vars(args),
        "da_preset_effective": (
            {key: getattr(args, key) for key in FIXED_DA_STRONG}
            if args.da_preset == "fixed_da_strong"
            else {}
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fit_config(server_round: int) -> dict:
    return {"server_round": server_round}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAPS Flower server")
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=1)
    parser.add_argument("--output-dir", default="results/flower_server")
    parser.add_argument("--run-name", default="flower_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--num-phases", type=int, default=3)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="smoke",
                        help="Server model/config profile; keep this aligned with client --profile for matrix runs")
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
    parser.add_argument("--da-preset", choices=DA_PRESETS, default="default",
                        help="Optional server-side DA preset; explicit CLI flags override preset values")
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
    parser.add_argument("--da-mmd-objective", choices=MMD_OBJECTIVES, default="legacy_quartic")
    parser.add_argument("--da-stage-alignment", choices=STAGE_ALIGNMENTS, default="legacy_intra_domain")
    parser.add_argument("--da-adv-feature-objective", choices=ADV_FEATURE_OBJECTIVES, default="legacy_grl_plus")
    parser.add_argument("--da-coral-class-conditional", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="Deep CORAL 是否按类别分别对齐；单机强分类基座默认使用 class-conditional CORAL")
    parser.add_argument("--da-use-align-reg-legacy", type=lambda v: v.lower() in ("true", "1", "yes"), default=False,
                        help="兼容单机旧版 server_representation_learning 的直接 local-proto→semantic-proto 对齐项，仅用于复现实验诊断")
    parser.add_argument("--da-lambda-align-reg-legacy", type=float, default=0.05,
                        help="legacy align-reg 诊断项权重；只有 --da-use-align-reg-legacy true 时生效")
    parser.add_argument("--strict-calibration-split", type=lambda v: v.lower() in ("true", "1", "yes"), default=True,
                        help="服务端 DA 数据是否强制读取 calibration_* split，防止 fallback 到 test/train 造成泄漏")
    parser.add_argument("--da-device", type=str, default="cpu",
                        help="域适应计算设备 (cpu 或 cuda)")
    parser.add_argument("--use-adapted-as-global", type=lambda v: v.lower() in ("true", "1", "yes"), default=False,
                        help="Return domain-adapted parameters to clients in the next Flower round")
    parser.add_argument("--da-lambda-coral", type=float, default=0.1)
    parser.add_argument("--da-lambda-global-mmd", type=float, default=0.5)
    parser.add_argument("--da-lambda-class-mmd", type=float, default=0.5)
    parser.add_argument("--da-lambda-proto-anchor", type=float, default=0.3)
    parser.add_argument("--da-lambda-adv", type=float, default=0.1)
    parser.add_argument("--da-lambda-target-ce", type=float, default=0.0)
    parser.add_argument("--da-lambda-proto", type=float, default=0.05,
                        help="DA prototype learning loss weight, matching single-machine LAMBDA_PROTO")
    parser.add_argument("--da-lambda-consistency", type=float, default=2.0,
                        help="DA source consistency-to-prototypes loss weight")
    parser.add_argument("--da-lambda-residual", type=float, default=0.1,
                        help="DA device residual matching loss weight")
    parser.add_argument("--da-lambda-proto-mmd", type=float, default=0.2,
                        help="DA inter-client prototype MMD/consistency weight")
    parser.add_argument("--da-lambda-stage-mmd", type=float, default=0.2,
                        help="DA stage-wise MMD weight within each class")
    parser.add_argument("--da-target-ce-label-smoothing", type=float, default=0.0)
    parser.add_argument("--da-target-ce-class-balanced", type=lambda v: v.lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--da-server-opt-lr", type=float, default=1e-4)
    args = parser.parse_args()
    explicit_dests = _explicit_cli_dests(parser, sys.argv[1:])
    apply_da_preset(args, explicit_dests)
    if args.use_domain_adapt and not args.strict_calibration_split:
        parser.error(
            "--use-domain-adapt requires --strict-calibration-split true"
        )
    validate_domain_adaptation_request(
        args.strategy,
        args.use_domain_adapt,
        args.server_val_data,
        args.server_calib_data,
    )
    save_run_config(args, explicit_dests)

    config = make_config(
        device="cpu",
        local_epochs=1,
        batch_size=32,
        profile=args.profile,
        seed=args.seed,
        num_classes=args.num_classes,
        input_dim=args.input_dim,
        num_clients=args.num_clients,
        num_phases=args.num_phases,
    )
    model = create_model(config)
    initial_arrays, parameter_keys = get_parameters(model)

    strategy_kwargs = dict(
        parameter_keys=parameter_keys,
        reference_state=model.state_dict(),
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_history=args.save_history,
        model_config={
            "num_classes": config.NUM_CLASSES,
            "input_dim": config.INPUT_DIM,
            "num_clients": config.NUM_CLIENTS,
            "num_phases": config.NUM_PHASES,
            "seq_len": config.SEQ_LEN,
            "profile": args.profile,
        },
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
            da_mmd_objective=args.da_mmd_objective,
            da_stage_alignment=args.da_stage_alignment,
            da_adv_feature_objective=args.da_adv_feature_objective,
            da_coral_class_conditional=args.da_coral_class_conditional,
            da_use_align_reg_legacy=args.da_use_align_reg_legacy,
            da_lambda_align_reg_legacy=args.da_lambda_align_reg_legacy,
            strict_calibration_split=args.strict_calibration_split,
            da_device=args.da_device,
            da_lambda_coral=args.da_lambda_coral,
            da_lambda_global_mmd=args.da_lambda_global_mmd,
            da_lambda_class_mmd=args.da_lambda_class_mmd,
            da_lambda_proto_anchor=args.da_lambda_proto_anchor,
            da_lambda_adv=args.da_lambda_adv,
            da_lambda_target_ce=args.da_lambda_target_ce,
            da_lambda_proto=args.da_lambda_proto,
            da_lambda_consistency=args.da_lambda_consistency,
            da_lambda_residual=args.da_lambda_residual,
            da_lambda_proto_mmd=args.da_lambda_proto_mmd,
            da_lambda_stage_mmd=args.da_lambda_stage_mmd,
            da_target_ce_label_smoothing=args.da_target_ce_label_smoothing,
            da_target_ce_class_balanced=args.da_target_ce_class_balanced,
            da_server_opt_lr=args.da_server_opt_lr,
            use_adapted_as_global=args.use_adapted_as_global,
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
