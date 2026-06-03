"""Phase 5: 源域回归联邦训练辅助函数

从 exp_improved.py 移植核心回归训练逻辑, 提供:
    - 回归配置创建 (USE_DUAL_PROJ=True, REG_GRAD_DETACH=True)
    - 回归模型创建与初始化 (从分类模型继承共享权重)
    - 源域数据加载器构建
    - 回归参数提取与 FedAvg 聚合
    - 本地回归训练循环 (HuberLoss, 类别加权, 浓度桶辅助任务)
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import FLConfig
from federated_dataset import create_train_loader
from model import FedGasMultiTaskModel
from utils import create_model_by_config, normalize_concentration, set_random_seed

logger = logging.getLogger(__name__)


def make_regression_config(
    device: str = "cpu",
    local_steps: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> FLConfig:
    """创建回归训练专用配置

    与分类训练的关键差异:
        - USE_REG_LOSS = True: 启用回归损失
        - USE_DUAL_PROJ = True: 使用独立回归投影层
        - REG_GRAD_DETACH = True: 阻断回归梯度向共享 TCN 回传
        - USE_ALIGN = False: 回归训练不需要原型对齐

    参数:
        device: 训练设备 (cpu/cuda)
        local_steps: 每轮本地训练步数
        batch_size: 批次大小
        lr: 学习率

    返回:
        回归训练配置 FLConfig
    """
    config = FLConfig()
    config.DEVICE = device
    config.LOCAL_EPOCHS = 1
    config.BATCH_SIZE = batch_size
    # 回归训练开关
    config.USE_REG_LOSS = True
    config.USE_DUAL_PROJ = True
    config.REG_GRAD_DETACH = True
    # 关闭分类相关损失
    config.USE_ALIGN = False
    config.USE_REPLAY_DISTILL = False
    config.USE_SERVER_OPT = False
    config.USE_LEARNABLE_AGG = False
    config.USE_SELECTIVE_AGG = False
    config.USE_PROTO_DECOUPLING = False
    config.USE_SOFT_AGGREGATION = False
    config.USE_SENSOR_AUG = False
    config.USE_MMD_REG = False
    config.USE_DEEP_CORAL = False
    config.USE_ADVERSARIAL_DOMAIN = False
    config.USE_MMD_ALIGNMENT = False
    # 回归超参数
    config.REG_HEAD_DEPTH = 3
    config.HUBER_DELTA = 0.2
    config.NUM_CONC_BUCKETS = 0  # 默认不启用浓度桶辅助任务
    config.LAMBDA_CONC_BUCKET = 0.0
    config.SEPARATE_REG_CLASS_WEIGHTS = ""
    config.SEPARATE_REG_HUBER_DELTAS = ""
    config.SEPARATE_REG_ALLOW_ENCODER_BACKPROP = True
    config.SEPARATE_REG_WEIGHT_DECAY = 1e-3
    set_random_seed(config.SEED)
    return config


def create_regression_model(config: FLConfig) -> FedGasMultiTaskModel:
    """创建回归模型 B (FedGasMultiTaskModel)

    使用 create_model_by_config + with_reg_head=True 创建,
    等价于 exp_improved.py 中 reg_model 的创建方式。

    参数:
        config: 训练配置 (USE_DUAL_PROJ 和 REG_GRAD_DETACH 必须为 True)

    返回:
        FedGasMultiTaskModel 实例 (未加载权重)
    """
    reg_config = copy.deepcopy(config)
    reg_config.USE_DUAL_PROJ = True
    reg_config.REG_GRAD_DETACH = True
    reg_config.USE_REG_LOSS = True
    # 确保 USE_PROTO_REG 为 False (使用标准 RegHead MLP)
    reg_config.USE_PROTO_REG = False
    model = create_model_by_config(reg_config, with_reg_head=True)
    return model


def init_regression_branch_from_classifier(model: FedGasMultiTaskModel) -> None:
    """用分类投影层初始化回归投影层 (当形状相同时)

    复制 cls_proj 的权重和偏置到 reg_proj,
    使回归分支从分类特征空间出发, 加速收敛。

    参数:
        model: FedGasMultiTaskModel 实例
    """
    if (
        getattr(model, "cls_proj", None) is not None
        and getattr(model, "reg_proj", None) is not None
    ):
        if model.cls_proj.weight.shape == model.reg_proj.weight.shape:
            with torch.no_grad():
                model.reg_proj.weight.copy_(model.cls_proj.weight)
                model.reg_proj.bias.copy_(model.cls_proj.bias)
            logger.info("回归分支: 已从分类投影层初始化 reg_proj")


def load_classifier_weights(model: FedGasMultiTaskModel, classifier_ckpt_path: str) -> None:
    """从分类模型 checkpoint 加载共享编码器权重

    使用 strict=False 加载, 只匹配共享参数键名,
    回归专属参数 (reg_proj, reg_heads 等) 保持随机初始化。

    参数:
        model: FedGasMultiTaskModel 实例
        classifier_ckpt_path: 分类模型 .pth checkpoint 路径
    """
    ckpt = torch.load(classifier_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    # 提取参数键列表, 排除 optimizer_state 等非模型键
    if isinstance(state, dict) and "optimizer_state" in state:
        state = state["model_state"]
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        logger.info(f"加载分类权重: {len(missing_keys)} 个缺失键 (回归专属参数, 正常)")
    if unexpected_keys:
        logger.warning(f"加载分类权重: {len(unexpected_keys)} 个多余键 (分类专属参数, 已忽略)")
    logger.info(f"回归模型: 已从 {classifier_ckpt_path} 加载共享编码器权重")


def build_source_regression_loaders(
    data_root: str,
    client_ids: List[int],
    batch_size: int = 32,
) -> Tuple[Dict[int, DataLoader], Dict[int, int]]:
    """为源域客户端构建回归训练 DataLoader

    每个客户端使用自己的 train_features.npy, 保持联邦数据隔离。

    参数:
        data_root: 数据根目录 (包含 client_1, client_2 等子目录)
        client_ids: 源域客户端 ID 列表
        batch_size: 批次大小

    返回:
        (loaders_dict, sample_counts_dict)
        - loaders_dict: {client_id: DataLoader}
        - sample_counts_dict: {client_id: 样本数}
    """
    data_root = Path(data_root)
    loaders: Dict[int, DataLoader] = {}
    sample_counts: Dict[int, int] = {}

    for cid in client_ids:
        client_dir = data_root / f"client_{cid}"
        feat_path = client_dir / "train_features.npy"
        if not feat_path.exists():
            logger.warning(f"回归训练: 客户端 {cid} 无 train_features.npy, 跳过")
            continue

        try:
            loader = create_train_loader(
                client_dir,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
        except Exception as e:
            logger.error(f"回归训练: 客户端 {cid} 创建 DataLoader 失败: {e}")
            continue

        n_samples = len(loader.dataset) if hasattr(loader, "dataset") else 0
        loaders[cid] = loader
        sample_counts[cid] = n_samples
        logger.info(f"回归训练: 源域客户端 {cid} 训练样本数={n_samples}")

    if not loaders:
        raise RuntimeError("回归训练: 无可用源域训练数据")
    return loaders, sample_counts


def get_regression_state_keys(model: FedGasMultiTaskModel) -> List[str]:
    """提取仅回归分支的参数键名 (用于 FedAvg 聚合范围)

    回归专属参数前缀:
        reg_proj., reg_transformer., reg_attn., reg_attn_linear.,
        reg_stats_proj., reg_heads.
    回归专属参数字面量:
        proto_scale, proto_bias, proto_conc, conc_directions, conc_scale, conc_bias

    参数:
        model: FedGasMultiTaskModel 实例

    返回:
        回归分支参数键名列表
    """
    prefixes = (
        "reg_proj.", "reg_transformer.", "reg_attn.",
        "reg_attn_linear.", "reg_stats_proj.", "reg_heads.",
    )
    exact_names = {
        "proto_scale", "proto_bias", "proto_conc",
        "conc_directions", "conc_scale", "conc_bias",
    }
    return [
        key for key in model.state_dict().keys()
        if key.startswith(prefixes) or key in exact_names
    ]


def fedavg_regression_states(
    local_states: Dict[int, Dict[str, torch.Tensor]],
    sample_counts: Dict[int, int],
    state_keys: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """FedAvg 加权聚合回归参数

    公式:
        aggregated[key] = Σ_c (n_c / N) * local_state_c[key]

    参数:
        local_states: {client_id: state_dict}
        sample_counts: {client_id: 样本数}
        state_keys: 要聚合的参数键列表
        device: 计算设备

    返回:
        聚合后的 state_dict (仅包含 state_keys 中的键)
    """
    total = float(sum(sample_counts.values()))
    if total <= 0:
        raise RuntimeError("回归 FedAvg: 无源域样本可聚合")

    aggregated: Dict[str, torch.Tensor] = {}
    for key in state_keys:
        weighted: Optional[torch.Tensor] = None
        for cid, state in local_states.items():
            weight = float(sample_counts.get(cid, 0)) / total
            if weight <= 0:
                continue
            value = state[key].to(device)
            contrib = value * weight
            weighted = contrib if weighted is None else weighted + contrib
        if weighted is not None:
            aggregated[key] = weighted
    return aggregated


def train_regression_local(
    model: FedGasMultiTaskModel,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    lr: float = 1e-3,
    feat_lr: float = 0.0,
    huber_delta: float = 0.2,
    class_weights_str: str = "",
    stage_name: str = "regression",
) -> float:
    """单客户端本地回归训练

    训练策略:
        - 冻结所有非回归参数 (requires_grad=False)
        - 仅训练回归分支参数: reg_proj, reg_transformer, reg_attn, reg_heads 等
        - 损失函数: SmoothL1Loss (Huber), beta=huber_delta
        - 支持按类别加权的 sample_weights

    参数:
        model: FedGasMultiTaskModel 实例
        loader: 训练 DataLoader
        device: 计算设备
        steps: 训练步数
        lr: 回归头学习率
        feat_lr: 特征提取器学习率 (0 表示冻结)
        huber_delta: Huber 损失 beta 参数
        class_weights_str: 类别权重字符串, 如 "1,2,3"
        stage_name: 阶段名称 (用于日志)

    返回:
        平均训练损失
    """
    if loader is None or steps <= 0:
        return 0.0

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻回归分支参数
    reg_params: List[torch.nn.Parameter] = []
    for name in [
        "reg_proj", "reg_transformer", "reg_attn",
        "reg_attn_linear", "reg_stats_proj",
    ]:
        module = getattr(model, name, None)
        if module is not None:
            reg_params.extend(
                p for p in module.parameters() if p.requires_grad is not False
            )
    if getattr(model, "reg_heads", None) is not None:
        reg_params.extend(model.reg_heads.parameters())
    for name in [
        "proto_scale", "proto_bias", "proto_conc",
        "conc_directions", "conc_scale", "conc_bias",
    ]:
        param = getattr(model, name, None)
        if param is not None and isinstance(param, torch.nn.Parameter):
            reg_params.append(param)

    for param in reg_params:
        param.requires_grad = True

    if not reg_params:
        raise RuntimeError("回归训练: 无可训练回归参数")

    optimizer = torch.optim.AdamW(reg_params, lr=lr, weight_decay=1e-3)
    model.train()

    iterator = iter(loader)
    running_loss = 0.0

    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        # batch: (x, y_cls, y_reg_full, y_phase) — GasSensorWindowDataset 四元组
        x = batch[0].to(device)
        y_cls = batch[1].to(device)
        y_reg_full = batch[2].to(device)
        y_phase = batch[3].to(device)

        # 提取当前类别的回归标签
        y_reg = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls].unsqueeze(1)
        # 归一化到 [0, 1]
        y_norm = normalize_concentration(y_reg, y_cls)

        optimizer.zero_grad()
        _, _, reg_feat = model(x)
        pred_norm = model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)

        # SmoothL1 (Huber) 损失
        loss = F.smooth_l1_loss(pred_norm, y_norm, beta=huber_delta)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / max(steps, 1)
    logger.info(f"回归训练 [{stage_name}]: steps={steps}, avg_loss={avg_loss:.6f}")
    return avg_loss


def train_federated_source_regression(
    model: FedGasMultiTaskModel,
    source_loaders: Dict[int, DataLoader],
    sample_counts: Dict[int, int],
    device: torch.device,
    total_steps_per_client: int = 100,
    source_rounds: int = 3,
    lr: float = 1e-3,
    save_dir: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[str, str]]:
    """联邦源域回归预训练 (FedAvg)

    模拟联邦学习流程:
        1. 每个 round: 各客户端独立本地训练 total_steps/rounds 步
        2. 聚合: weighted FedAvg (按样本数加权)
        3. 更新全局模型

    参数:
        model: 全局回归模型 (FedGasMultiTaskModel)
        source_loaders: {client_id: DataLoader}
        sample_counts: {client_id: 样本数}
        device: 计算设备
        total_steps_per_client: 每个客户端总训练步数
        source_rounds: 联邦通信轮数
        lr: 学习率
        save_dir: 保存目录 (可选)

    返回:
        (final_local_states, local_checkpoint_paths)
    """
    if total_steps_per_client <= 0:
        logger.warning("回归 FedAvg: total_steps_per_client<=0, 跳过训练")
        return {}, {}

    source_rounds = max(1, int(source_rounds))
    base_steps = total_steps_per_client // source_rounds
    extra_steps = total_steps_per_client % source_rounds

    # 提取回归分支参数键
    state_keys = get_regression_state_keys(model)
    logger.info(
        f"回归 FedAvg: 源域客户端={list(source_loaders.keys())}, "
        f"通信轮数={source_rounds}, 总步数/客户端={total_steps_per_client}, "
        f"聚合参数数={len(state_keys)}"
    )

    final_local_states: Dict[int, Dict[str, torch.Tensor]] = {}
    local_checkpoint_paths: Dict[str, str] = {}

    for round_idx in range(1, source_rounds + 1):
        local_steps = base_steps + (1 if round_idx <= extra_steps else 0)
        if local_steps <= 0:
            continue

        global_state = copy.deepcopy(model.state_dict())
        local_states: Dict[int, Dict[str, torch.Tensor]] = {}

        # 每个客户端独立训练
        for cid, loader in source_loaders.items():
            local_model = copy.deepcopy(model).to(device)
            local_model.load_state_dict(global_state, strict=True)

            train_regression_local(
                local_model, loader, device,
                steps=local_steps, lr=lr,
                stage_name=f"client{cid}_round{round_idx}",
            )

            # 保存本地状态 (CPU)
            local_state_cpu = {
                key: value.detach().cpu()
                for key, value in local_model.state_dict().items()
            }
            local_states[cid] = local_state_cpu

            # 最后一轮保存本地 checkpoint
            if round_idx == source_rounds:
                final_local_states[cid] = copy.deepcopy(local_state_cpu)
                if save_dir is not None:
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    ckpt_path = save_path / f"regression_source_client{cid}_local.pth"
                    torch.save(
                        {"model_state": local_state_cpu, "client_id": cid},
                        ckpt_path,
                    )
                    local_checkpoint_paths[str(cid)] = str(ckpt_path)

            del local_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # FedAvg 聚合
        averaged = fedavg_regression_states(
            local_states, sample_counts, state_keys, device
        )

        # 更新全局模型
        new_state = copy.deepcopy(global_state)
        for key, value in averaged.items():
            ref = new_state[key]
            new_state[key] = value.to(ref.device).type_as(ref)

        model.load_state_dict(new_state, strict=True)
        logger.info(
            f"回归 FedAvg: 完成第 {round_idx}/{source_rounds} 轮通信"
        )

    return final_local_states, local_checkpoint_paths