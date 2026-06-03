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
    config = make_regression_config(device=device.type, batch_size=args.batch_size,
                                     local_steps=args.steps, lr=args.lr)

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
        stage_name=f"client{args.client_id}_source_reg",
    )

    # 7. 保存本地 checkpoint
    ckpt_path = output_dir / f"regression_source_client{args.client_id}_local.pth"
    torch.save(
        {"model_state": reg_model.state_dict(), "client_id": args.client_id,
         "avg_loss": avg_loss, "n_samples": n_samples},
        ckpt_path,
    )
    logger.info(f"回归客户端 {args.client_id}: checkpoint 已保存至 {ckpt_path}")
    logger.info(f"回归客户端 {args.client_id}: 训练完成, 平均损失={avg_loss:.6f}")


if __name__ == "__main__":
    main()