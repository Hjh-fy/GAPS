"""Phase 5: 源域回归联邦客户端 (独立脚本)

用法:
    python -m gaps_flower.regression_client \
        --classifier_ckpt gaps_flower/outputs/classifier_global.pt \
        --client_id 1 --data_root gaps_flower/data \
        --output_dir gaps_flower/outputs/regression \
        --steps 100 --lr 1e-3 [--cpu]

功能:
    1. 加载训练好的分类模型, 继承 TCN 共享编码器权重
    2. 创建 FedGasMultiTaskModel 回归模型 B
    3. 加载源域客户端 1/2 的训练数据, 执行本地回归训练
    4. 保存本地训练好的 checkpoint
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaps_flower.regression_task import (
    build_source_regression_loaders,
    create_regression_model,
    init_regression_branch_from_classifier,
    load_classifier_weights,
    make_regression_config,
    train_regression_local,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("regression_client")


def main() -> None:
    parser = argparse.ArgumentParser(description="源域回归联邦客户端训练")
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="分类模型 checkpoint 路径 (Phase 4 输出)")
    parser.add_argument("--client_id", type=int, required=True,
                        help="源域客户端 ID (如 1, 2)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据根目录 (包含 client_1/, client_2/ 等)")
    parser.add_argument("--output_dir", type=str, default="gaps_flower/outputs/regression",
                        help="输出目录")
    parser.add_argument("--steps", type=int, default=100,
                        help="本地训练步数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="回归头学习率")
    parser.add_argument("--huber-delta", type=float, default=0.2,
                        help="SmoothL1/Huber beta for normalized concentration")
    parser.add_argument("--class-weights", type=str, default="",
                        help="Optional class weights, e.g. '1:2.0' to emphasize CO")
    parser.add_argument("--reg-head-depth", type=int, default=None,
                        help="回归头深度; 默认沿用 config.py 中的 REG_HEAD_DEPTH")
    parser.add_argument("--reg-output-mode", default=None, choices=["sigmoid", "linear"],
                        help="回归输出模式; sigmoid 为旧逻辑, linear 用于 T8a 消融")
    parser.add_argument("--reg-range-penalty", type=float, default=0.0,
                        help="linear 输出模式下的 [0,1] 越界惩罚权重")
    parser.add_argument("--reg-window-stats", action="store_true",
                        help="追加窗口响应统计特征到回归分支")
    parser.add_argument("--reg-window-stats-mode", default="global", choices=["global", "per_channel"],
                        help="窗口统计模式")
    parser.add_argument("--reg-window-stats-dim", type=int, default=8,
                        help="窗口统计特征投影维度")
    parser.add_argument("--reg-response-branch", default=None, choices=["none", "dct", "msconv"],
                        help="Regression response-shape branch")
    parser.add_argument("--reg-dct-k", type=int, default=None,
                        help="Number of low-frequency DCT coefficients")
    parser.add_argument("--reg-dct-gamma-init", type=float, default=None,
                        help="Initial residual scale for DCT branch")
    parser.add_argument("--reg-dct-dropout", type=float, default=None,
                        help="Dropout inside DCT branch")
    parser.add_argument("--reg-msconv-channels", type=int, default=None,
                        help="Hidden channels per temporal scale for msconv branch")
    parser.add_argument("--reg-msconv-kernels", default=None,
                        help="Comma-separated odd kernels for msconv branch, e.g. 3,7,15,31")
    parser.add_argument("--reg-msconv-gamma-init", type=float, default=None,
                        help="Initial residual scale for msconv branch")
    parser.add_argument("--reg-msconv-dropout", type=float, default=None,
                        help="Dropout inside msconv branch")
    parser.add_argument("--reg-tcn-adapter", action="store_true",
                        help="Enable regression-specific residual adapter on shared TCN features")
    parser.add_argument("--reg-tcn-adapter-kernel", type=int, default=None,
                        help="Odd kernel size for regression TCN adapter")
    parser.add_argument("--reg-tcn-adapter-gamma-init", type=float, default=None,
                        help="Initial residual scale for regression TCN adapter")
    parser.add_argument("--reg-tcn-adapter-dropout", type=float, default=None,
                        help="Dropout for regression TCN adapter")
    parser.add_argument("--reg-use-shared-trunk", action="store_true",
                        help="use shared concentration trunk + class residual heads (disables independent per-class heads)")
    parser.add_argument("--reg-shared-trunk-dim", type=int, default=None,
                        help="hidden dimension for shared concentration trunk; default 128")
    parser.add_argument("--reg-gas-emb-dim", type=int, default=None,
                        help="gas embedding dimension for shared trunk; default 16")
    parser.add_argument("--reg-residual-head-depth", type=int, default=None,
                        help="depth of per-class residual heads; default 2 (1/2/3)")
    parser.add_argument("--use-reg-ratio-branch", action="store_true",
                        help="enable cross-channel ratio response features")
    parser.add_argument("--reg-ratio-gamma-init", type=float, default=None,
                        help="ratio branch gamma init; default 0.0")
    parser.add_argument("--reg-ratio-dropout", type=float, default=None,
                        help="ratio branch dropout; default 0.05")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    args = parser.parse_args()

    # 设备选择
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"回归客户端: 使用设备={device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 创建回归配置
    config = make_regression_config(
        device=device.type,
        batch_size=args.batch_size,
        local_steps=args.steps,
        lr=args.lr,
        reg_head_depth=args.reg_head_depth,
        reg_output_mode=args.reg_output_mode,
        use_reg_window_stats=args.reg_window_stats,
        reg_window_stats_mode=args.reg_window_stats_mode,
        reg_window_stats_dim=args.reg_window_stats_dim,
        reg_response_branch=args.reg_response_branch,
        reg_dct_k=args.reg_dct_k,
        reg_dct_gamma_init=args.reg_dct_gamma_init,
        reg_dct_dropout=args.reg_dct_dropout,
        reg_msconv_channels=args.reg_msconv_channels,
        reg_msconv_kernels=args.reg_msconv_kernels,
        reg_msconv_gamma_init=args.reg_msconv_gamma_init,
        reg_msconv_dropout=args.reg_msconv_dropout,
        use_reg_tcn_adapter=args.reg_tcn_adapter,
        reg_tcn_adapter_kernel=args.reg_tcn_adapter_kernel,
        reg_tcn_adapter_gamma_init=args.reg_tcn_adapter_gamma_init,
        reg_tcn_adapter_dropout=args.reg_tcn_adapter_dropout,
        use_reg_shared_trunk=args.reg_use_shared_trunk if args.reg_use_shared_trunk else None,
        reg_shared_trunk_dim=args.reg_shared_trunk_dim,
        reg_gas_emb_dim=args.reg_gas_emb_dim,
        reg_residual_head_depth=args.reg_residual_head_depth,
        use_reg_ratio_branch=args.use_reg_ratio_branch if args.use_reg_ratio_branch else None,
        reg_ratio_gamma_init=args.reg_ratio_gamma_init,
        reg_ratio_dropout=args.reg_ratio_dropout,
    )
    logger.info(
        "回归客户端%s: REG_HEAD_DEPTH=%s, REG_OUTPUT_MODE=%s, REG_WINDOW_STATS=%s",
        args.client_id,
        config.REG_HEAD_DEPTH,
        config.REG_OUTPUT_MODE,
        config.REG_WINDOW_STATS,
    )

    # 2. 创建回归模型 B
    reg_model = create_regression_model(config).to(device)

    # 3. 从分类模型加载共享编码器权重
    load_classifier_weights(reg_model, args.classifier_ckpt)

    # 4. 用分类投影层初始化回归投影层
    init_regression_branch_from_classifier(reg_model)

    # 5. 构建数据加载器 (仅当前客户端)
    loaders, sample_counts = build_source_regression_loaders(
        data_root=args.data_root,
        client_ids=[args.client_id],
        batch_size=args.batch_size,
    )

    loader = loaders.get(args.client_id)
    n_samples = sample_counts.get(args.client_id, 0)
    if loader is None:
        logger.error(f"回归客户端: 客户端 {args.client_id} 无训练数据")
        sys.exit(1)

    logger.info(f"回归客户端 {args.client_id}: 训练样本数={n_samples}")

    # 6. 本地回归训练
    avg_loss = train_regression_local(
        reg_model, loader, device,
        steps=args.steps, lr=args.lr,
        huber_delta=args.huber_delta,
        class_weights_str=args.class_weights,
        reg_range_penalty=args.reg_range_penalty,
        stage_name=f"client{args.client_id}_source_reg",
    )

    # 7. 保存本地 checkpoint
    ckpt_path = output_dir / f"regression_source_client{args.client_id}_local.pth"
    torch.save(
        {
            "model_state": reg_model.state_dict(),
            "client_id": args.client_id,
            "avg_loss": avg_loss,
            "n_samples": n_samples,
            "model_config": {
                "reg_head_depth": int(config.REG_HEAD_DEPTH),
                "reg_output_mode": str(config.REG_OUTPUT_MODE),
                "reg_window_stats": bool(config.REG_WINDOW_STATS),
                "reg_window_stats_mode": str(config.REG_WINDOW_STATS_MODE),
                "reg_window_stats_dim": int(config.REG_WINDOW_STATS_DIM),
                "reg_response_branch": str(config.REG_RESPONSE_BRANCH),
                "reg_dct_k": int(config.REG_DCT_K),
                "reg_dct_gamma_init": float(config.REG_DCT_GAMMA_INIT),
                "reg_dct_dropout": float(config.REG_DCT_DROPOUT),
                "reg_msconv_channels": int(config.REG_MSCONV_CHANNELS),
                "reg_msconv_kernels": str(config.REG_MSCONV_KERNELS),
                "reg_msconv_gamma_init": float(config.REG_MSCONV_GAMMA_INIT),
                "reg_msconv_dropout": float(config.REG_MSCONV_DROPOUT),
                "reg_tcn_adapter": bool(config.REG_TCN_ADAPTER),
                "reg_tcn_adapter_kernel": int(config.REG_TCN_ADAPTER_KERNEL),
                "reg_tcn_adapter_gamma_init": float(config.REG_TCN_ADAPTER_GAMMA_INIT),
                "reg_tcn_adapter_dropout": float(config.REG_TCN_ADAPTER_DROPOUT),
                "reg_use_shared_trunk": bool(config.REG_USE_SHARED_TRUNK),
                "reg_shared_trunk_dim": int(config.REG_SHARED_TRUNK_DIM),
                "reg_gas_emb_dim": int(config.REG_GAS_EMB_DIM),
                "reg_residual_head_depth": int(config.REG_RESIDUAL_HEAD_DEPTH),
                "use_reg_ratio_branch": bool(config.USE_REG_RATIO_BRANCH),
                "reg_ratio_gamma_init": float(config.REG_RATIO_GAMMA_INIT),
                "reg_ratio_dropout": float(config.REG_RATIO_DROPOUT),
            },
            "train_config": {
                "steps": int(args.steps),
                "lr": float(args.lr),
                "huber_delta": float(args.huber_delta),
                "class_weights": args.class_weights,
                "reg_range_penalty": float(args.reg_range_penalty),
            },
        },
        ckpt_path,
    )
    logger.info(f"回归客户端 {args.client_id}: checkpoint 已保存至 {ckpt_path}")
    logger.info(f"回归客户端 {args.client_id}: 训练完成, 平均损失={avg_loss:.6f}")


if __name__ == "__main__":
    main()
