"""Phase 5: 源域回归联邦服务端 (独立脚本)

使用 FedAvg 方法聚合多个源域客户端的回归训练结果,
模拟联邦学习中的服务端聚合过程。
由于是单机模拟, 服务端脚本负责:
    1. 收集所有客户端本地训练的 checkpoint
    2. 执行 FedAvg 加权聚合
    3. 更新全局回归模型
    4. 保存聚合后的全局 checkpoint

用法:
    python -m gaps_flower.regression_server \
        --classifier_ckpt gaps_flower/outputs/classifier_global.pt \
        --client_ckpts_dir gaps_flower/outputs/regression \
        --data_root gaps_flower/data \
        --client_ids 1,2 \
        --output_dir gaps_flower/outputs/regression \
        [--cpu]

聚合流程:
    - 收集各客户端本地 state_dict
    - 按样本数加权平均 (FedAvg)
    - 输出全局回归模型 checkpoint
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaps_flower.regression_task import (
    build_source_regression_loaders,
    create_regression_model,
    fedavg_regression_states,
    get_regression_state_keys,
    init_regression_branch_from_classifier,
    load_classifier_weights,
    make_regression_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("regression_server")


def aggregate_regression_checkpoints(
    classifier_ckpt: str,
    client_ckpt_dir: str,
    data_root: str,
    client_ids: list[int],
    device: torch.device,
    batch_size: int = 32,
    output_dir: str = "gaps_flower/outputs/regression",
) -> str:
    """收集客户端 checkpoint 并执行 FedAvg 聚合

    参数:
        classifier_ckpt: 分类模型 checkpoint 路径
        client_ckpt_dir: 客户端 checkpoint 目录
        data_root: 数据根目录
        client_ids: 源域客户端 ID 列表
        device: 计算设备
        batch_size: 批次大小
        output_dir: 输出目录

    返回:
        全局聚合 checkpoint 路径
    """
    # 1. 创建全局模型 (加载共享编码器)
    config = make_regression_config(device=device.type, batch_size=batch_size)
    global_model = create_regression_model(config).to(device)
    load_classifier_weights(global_model, classifier_ckpt)
    init_regression_branch_from_classifier(global_model)

    # 2. 获取样本数
    _, sample_counts = build_source_regression_loaders(
        data_root=data_root, client_ids=client_ids, batch_size=batch_size,
    )
    logger.info(f"回归 FedAvg: 客户端样本数={sample_counts}")

    # 3. 加载各客户端本地 state_dict
    ckpt_dir = Path(client_ckpt_dir)
    local_states: Dict[int, Dict[str, torch.Tensor]] = {}
    for cid in client_ids:
        ckpt_path = ckpt_dir / f"regression_source_client{cid}_local.pth"
        if not ckpt_path.exists():
            logger.warning(f"回归 FedAvg: 客户端 {cid} checkpoint 不存在: {ckpt_path}, 跳过")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        local_states[cid] = ckpt.get("model_state", ckpt)
        logger.info(f"回归 FedAvg: 加载客户端 {cid} 本地参数")

    if len(local_states) < 2:
        logger.warning(
            f"回归 FedAvg: 仅有 {len(local_states)} 个客户端, 聚合效果有限。"
            "至少需要 2 个客户端才能体现多源域协同优势。"
        )

    # 4. FedAvg 聚合
    state_keys = get_regression_state_keys(global_model)
    aggregated = fedavg_regression_states(
        local_states, sample_counts, state_keys, device
    )

    # 5. 更新全局模型
    new_state = {key: value.clone() for key, value in global_model.state_dict().items()}
    for key, value in aggregated.items():
        ref = new_state[key]
        new_state[key] = value.to(ref.device).type_as(ref)

    global_model.load_state_dict(new_state, strict=True)

    # 6. 保存全局 checkpoint
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    global_ckpt_path = output / "regression_fedavg_global.pt"
    torch.save(
        {
            "model_state": global_model.state_dict(),
            "aggregated_clients": list(local_states.keys()),
            "sample_counts": sample_counts,
        },
        global_ckpt_path,
    )
    logger.info(f"回归 FedAvg 全局模型已保存: {global_ckpt_path}")

    # 7. 使全局模型也保存为 regressor.pt (兼容 inference.py 路径)
    regressor_path = output / "regressor.pt"
    torch.save(
        {"model_state": global_model.state_dict()},
        regressor_path,
    )
    logger.info(f"回归模型已保存为 regressor.pt: {regressor_path}")

    return str(global_ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="源域回归联邦 FedAvg 聚合")
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="分类模型 checkpoint 路径 (Phase 4 输出)")
    parser.add_argument("--client_ckpts_dir", type=str,
                        default="gaps_flower/outputs/regression",
                        help="客户端 checkpoint 目录")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据根目录")
    parser.add_argument("--client_ids", type=str, default="1,2",
                        help="客户端 ID 列表, 逗号分隔, 如 '1,2'")
    parser.add_argument("--output_dir", type=str,
                        default="gaps_flower/outputs/regression",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"回归服务端: 使用设备={device}")

    client_ids = [int(x.strip()) for x in args.client_ids.split(",") if x.strip()]
    logger.info(f"回归服务端: 聚合客户端={client_ids}")

    global_ckpt = aggregate_regression_checkpoints(
        classifier_ckpt=args.classifier_ckpt,
        client_ckpt_dir=args.client_ckpts_dir,
        data_root=args.data_root,
        client_ids=client_ids,
        device=device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    logger.info(f"回归服务端: 聚合完成, 全局模型={global_ckpt}")


if __name__ == "__main__":
    main()