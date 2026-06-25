"""
工具函数模块：评估、可视化、日志、结果保存等
"""
import torch
import torch.nn.functional as F
import random
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import logging
from model import FedGasModel
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.colors import LogNorm

import logging
logger = logging.getLogger('gasfl')

# 浓度归一化统计量（基于训练集）
CONC_STATS = {
    0: {'min': 12.5, 'max': 125.0},   # 乙醇
    1: {'min': 25.0, 'max': 250.0},   # CO
    2: {'min': 12.5, 'max': 125.0},   # 乙烯
    3: {'min': 25.0, 'max': 250.0},   # 甲烷
}

def normalize_concentration(y_reg, y_cls):
    """按类别将浓度归一化到 [0,1]"""
    y_norm = torch.zeros_like(y_reg)
    for c in range(4):
        mask = (y_cls == c)
        if mask.sum() > 0:
            min_c = CONC_STATS[c]['min']
            max_c = CONC_STATS[c]['max']
            y_norm[mask] = (y_reg[mask] - min_c) / (max_c - min_c + 1e-6)
    return y_norm


def compute_conc_bucket_boundaries(conc_data, cls_data, num_classes, num_buckets):
    """根据训练数据计算浓度桶边界
    
    Args:
        conc_data: 浓度数据数组 (N,) 或 (N, 1)
        cls_data: 类别标签数组 (N,)
        num_classes: 类别数量
        num_buckets: 每个类别的桶数量
    
    Returns:
        boundaries: 字典，key为类别，value为边界列表 (K+1,)，可序列化
    """
    import numpy as np
    boundaries = {}
    conc_data = np.squeeze(conc_data)
    
    for c in range(num_classes):
        mask = (cls_data == c)
        conc_c = conc_data[mask]
        
        if len(conc_c) == 0:
            boundaries[c] = np.linspace(0, 1, num_buckets + 1).tolist()
            continue
        
        # 使用分位数作为边界
        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(conc_c, percentiles)
        
        # 确保边界严格递增
        bins = np.unique(bins)
        if len(bins) < num_buckets + 1:
            # 如果分位数相同，使用线性插值补充
            bins = np.linspace(conc_c.min(), conc_c.max(), num_buckets + 1)
        
        # 添加一点余量避免边界问题
        bins[0] = bins[0] - (bins[-1] - bins[0]) * 0.01
        bins[-1] = bins[-1] + (bins[-1] - bins[0]) * 0.01
        
        boundaries[c] = bins.tolist()
    
    return boundaries

def get_conc_bucket_labels(y_reg, y_cls, boundaries):
    """将连续浓度映射到桶索引 (0..K-1)
    
    Args:
        y_reg: 原始浓度值 (B, 1)
        y_cls: 类别标签 (B,)
        boundaries: 字典，key为类别，value为边界列表或数组 (K+1,)
    
    Returns:
        bucket_labels: 桶索引标签 (B,)
    """
    device = y_reg.device
    bucket_labels = torch.zeros_like(y_cls, dtype=torch.long)
    for c in range(len(boundaries)):
        mask = (y_cls == c)
        if mask.sum() > 0:
            conc_c = y_reg[mask].squeeze()
            bins = boundaries[c]   # K+1 个边界
            # 将列表或数组转换为 PyTorch 张量
            if isinstance(bins, list):
                bins = torch.tensor(bins, device=device, dtype=conc_c.dtype)
            elif isinstance(bins, np.ndarray):
                bins = torch.from_numpy(bins).to(device, dtype=conc_c.dtype)
            # 使用 searchsorted，只使用内部边界
            idx = torch.bucketize(conc_c, bins[1:-1])  # idx 范围 0..K-1
            bucket_labels[mask] = idx.to(device)
    return bucket_labels


def get_lambda_reg(current_round, total_rounds):
    """
    分阶段动态调整回归损失权重，解决分类与回归冲突。
    论文可描述为：Phase-Aware Dynamic Weighting for Multi-Task Balance.
    后期权重降至 0.8 以控制回归梯度对特征空间的污染，避免 Align Loss 发散。
    λ_reg ∈ [0.05, 0.8]，分类主导(0~20%) → 温和引入(20~70%) → 稳定保持(70~100%)。
    """
    t = current_round / total_rounds
    if t < 0.2:          # 前20%轮次：分类主导，建立稳定决策边界
        return 0.05
    elif t < 0.7:        # 中间50%轮次：线性引入回归，温和塑造有序特征空间
        return 0.05 + (0.8 - 0.05) * (t - 0.2) / 0.5
    else:                # 后30%轮次：回归权重稳定在 0.8，避免过度拉扯特征空间
        return 0.8


# ==================== CORAL 特征空间对齐 ====================

def coral_transform(source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
    """
    将 target_feats 对齐到 source_feats 的二阶统计量（协方差矩阵）
    source_feats: (N, D)
    target_feats: (M, D)
    返回对齐后的 target_feats: (M, D)
    """
    eps = 1e-4
    
    s_mean = source_feats.mean(dim=0, keepdim=True)
    t_mean = target_feats.mean(dim=0, keepdim=True)
    s_centered = source_feats - s_mean
    t_centered = target_feats - t_mean
    
    cov_s = torch.mm(s_centered.T, s_centered) / (len(source_feats) - 1)
    cov_t = torch.mm(t_centered.T, t_centered) / (len(target_feats) - 1)
    
    try:
        u_s, s_s, _ = torch.svd(cov_s)
        u_t, s_t, _ = torch.svd(cov_t)
        
        sqrt_cov_s = u_s @ torch.diag(torch.sqrt(s_s + eps)) @ u_s.T
        inv_sqrt_cov_t = u_t @ torch.diag(1.0 / torch.sqrt(s_t + eps)) @ u_t.T
        
        t_whitened = t_centered @ inv_sqrt_cov_t
        t_coral = t_whitened @ sqrt_cov_s + s_mean
    except Exception as e:
        # SVD 失败，回退为直接白化 + 目标均值
        t_std = t_centered.std(0, keepdim=True) + eps
        s_std = s_centered.std(0, keepdim=True) + eps
        t_coral = t_centered / t_std * s_std + s_mean
    
    return t_coral


def coral_transform_class_conditional(source_feats, source_labels, target_feats, target_labels):
    """
    按类别分别应用 CORAL 变换，使 target_feats 分各类别对齐到 source_feats
    """
    unique_classes = torch.unique(source_labels)
    aligned_feats = torch.zeros_like(target_feats)
    for cls in unique_classes:
        src_mask = (source_labels == cls)
        tgt_mask = (target_labels == cls)
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            aligned_feats[tgt_mask] = target_feats[tgt_mask]  # 无对齐时保持原样
        else:
            aligned_feats[tgt_mask] = coral_transform(
                source_feats[src_mask], target_feats[tgt_mask]
            )
    return aligned_feats


def deep_coral_loss(src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
    """
    计算源域和目标域特征的协方差矩阵对齐损失 (Frobenius 范数)。
    src_feats: (N, D)
    tgt_feats: (M, D)
    """
    if src_feats.size(0) < 2 or tgt_feats.size(0) < 2:
        return torch.tensor(0.0, device=src_feats.device)
    
    src_mean = src_feats.mean(0, keepdim=True)
    tgt_mean = tgt_feats.mean(0, keepdim=True)
    src_centered = src_feats - src_mean
    tgt_centered = tgt_feats - tgt_mean
    
    cov_src = torch.mm(src_centered.t(), src_centered) / (src_feats.size(0) - 1)
    cov_tgt = torch.mm(tgt_centered.t(), tgt_centered) / (tgt_feats.size(0) - 1)
    
    loss = torch.norm(cov_src - cov_tgt, p='fro').pow(2) / (4 * src_feats.size(1) ** 2)
    return loss


def deep_coral_loss_class_conditional(src_feats, src_labels, tgt_feats, tgt_labels, num_classes=4):
    """按类别计算深度 CORAL 损失并求和"""
    loss = 0.0
    count = 0
    for c in range(num_classes):
        src_mask = (src_labels == c)
        tgt_mask = (tgt_labels == c)
        if src_mask.sum() < 2 or tgt_mask.sum() < 2:
            continue
        loss += deep_coral_loss(src_feats[src_mask], tgt_feats[tgt_mask])
        count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=src_feats.device)


def extract_reg_features_batch(model, dataloader, device, max_samples=None):
    """提取回归专用特征 (reg_feat) 和分类标签"""
    model.eval()
    all_reg_feats = []
    all_labels = []
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x, y_cls, _, _ = batch
            else:
                continue
            x = x.to(device)
            _, _, reg_feat = model(x)
            all_reg_feats.append(reg_feat.cpu())
            all_labels.append(y_cls)
            total += x.size(0)
            if max_samples and total >= max_samples:
                break
    reg_feats = torch.cat(all_reg_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if max_samples:
        reg_feats = reg_feats[:max_samples]
        labels = labels[:max_samples]
    return reg_feats.numpy(), labels.numpy()


from torch.utils.data import TensorDataset, DataLoader


def create_aligned_feat_loader(aligned_feats, y_cls, y_reg_full, y_phase, batch_size):
    """创建对齐特征的 DataLoader"""
    dataset = TensorDataset(
        torch.from_numpy(aligned_feats).float(),
        torch.from_numpy(y_cls).long(),
        torch.from_numpy(y_reg_full).float(),
        torch.from_numpy(y_phase).long()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def soft_aggregate_probs(feats, semantic_protos, temperature=0.3):
    """
    软聚合推理，计算类别概率
    
    Args:
        feats: 特征向量 (B, D)
        semantic_protos: 语义原型字典
        temperature: 温度参数
        
    Returns:
        类别概率 (B, C)
    """
    proto_keys = list(semantic_protos.keys())
    proto_matrix = torch.stack([semantic_protos[k].to(feats.device) for k in proto_keys])
    proto_classes = torch.tensor([int(k.strip('()').split(',')[0]) for k in proto_keys], device=feats.device)
    num_classes = len(torch.unique(proto_classes))
    
    feats_norm = F.normalize(feats, dim=1)
    proto_norm = F.normalize(proto_matrix, dim=1)
    sim = torch.mm(feats_norm, proto_norm.T) / temperature
    weights = F.softmax(sim, dim=1)
    scores = torch.zeros(feats.size(0), num_classes, device=feats.device)
    # === Sum 聚合：每个类别聚合该类别内所有原型的权重 ===
    scores.scatter_add_(1, proto_classes.unsqueeze(0).expand(feats.size(0), -1), weights)
    probs = F.softmax(scores, dim=1)
    return probs


def soft_aggregate_probs_mahalanobis(reg_feats, semantic_protos, semantic_proto_vars,
                                     temperature=1.0, eps=1e-4, min_var=0.01):
    """
    基于马氏距离的软聚合推理（P0 核心新增）。
    
    数学原理（详见 config 注释）：
    设特征 f ∈ R^D，原型均值 μ_k ∈ R^D，对角方差 σ²_k ∈ R^D₊。
    马氏距离 d_k² = Σⱼ (fⱼ − μ_{k,j})² / (σ²_{k,j} + ε)
    判别函数 g_k(f) = −d_k²  （负马氏距离平方正比于对数后验概率）
    类别概率 p(y=k|f) = softmax_k(g_k(f) / τ)
    
    与余弦相似度的关键区别：
    - 余弦：仅关心方向，丢失幅度信息，假定各向同性 → 对传感器漂移的异方差不鲁棒
    - 马氏：通过 Σ⁻¹ 白化恢复各向同性，高方差维度自动降权 → 漂移场景更鲁棒
    
    Args:
        reg_feats: 未归一化特征 (B, D)，来自模型 reg_feat
        semantic_protos: 语义原型均值字典 {str_key: Tensor(D,)}
        semantic_proto_vars: 各原型的对角方差字典 {str_key: Tensor(D,)}
        temperature: 温度参数 τ，控制 softmax 的锐度
        eps: 防止除零的小量
        min_var: 最小方差约束，防止低方差维度过分主导
    
    Returns:
        (probs, sim_raw): 类别概率 (B, C) 和原始距离 (B, K)
        
    时间复杂度: O(B·K·D)，与余弦版本相同
    """
    if semantic_proto_vars is None:
        # 退化：无方差信息时回退到 L2 距离（非余弦）
        feats_norm = F.normalize(reg_feats, dim=1)
        proto_keys = list(semantic_protos.keys())
        proto_matrix = torch.stack([semantic_protos[k].to(reg_feats.device) for k in proto_keys])
        proto_classes = torch.tensor([int(k.strip('()').split(',')[0]) for k in proto_keys],
                                      device=reg_feats.device)
        sim = torch.mm(feats_norm, proto_matrix.T) / temperature
        weights = F.softmax(sim, dim=1)
        num_classes = len(torch.unique(proto_classes))
        scores = torch.zeros(reg_feats.size(0), num_classes, device=reg_feats.device)
        scores.scatter_add_(1, proto_classes.unsqueeze(0).expand(reg_feats.size(0), -1), weights)
        probs = F.softmax(scores, dim=1)
        return probs, sim

    device = reg_feats.device
    proto_keys = list(semantic_protos.keys())
    proto_list = [semantic_protos[k].to(device) for k in proto_keys]
    proto_tensor = torch.stack(proto_list)                         # (K, D)
    proto_classes = torch.tensor([int(k.strip('()').split(',')[0]) for k in proto_keys],
                                  device=device)
    num_classes = len(torch.unique(proto_classes))

    # 构建方差矩阵 [K, D]
    var_list = []
    for key in proto_keys:
        str_key = f"({key[0]},{key[1]})" if isinstance(key, tuple) else key
        if str_key in semantic_proto_vars:
            var_vec = semantic_proto_vars[str_key].to(device)
            var_vec = torch.clamp(var_vec, min=min_var)
        else:
            var_vec = torch.ones(proto_tensor.shape[1], device=device)
        var_list.append(var_vec)
    var_tensor = torch.stack(var_list)                             # (K, D)

    # 马氏距离平方: d² = Σ_j (f_j - μ_{k,j})² / (σ²_{k,j} + ε)
    diff = reg_feats.unsqueeze(1) - proto_tensor.unsqueeze(0)     # (B, K, D)
    mahal_sq = (diff ** 2) / (var_tensor.unsqueeze(0) + eps)     # (B, K, D)
    d_mahal_sq = mahal_sq.sum(dim=-1)                             # (B, K)

    # 判别函数: g_k = -d_k² / τ
    sim_raw = -d_mahal_sq / temperature                           # (B, K)
    weights = F.softmax(sim_raw, dim=-1)                          # (B, K)

    # Sum 聚合：同类原型的权重求和
    scores = torch.zeros(reg_feats.size(0), num_classes, device=device)
    scores.scatter_add_(1, proto_classes.unsqueeze(0).expand(reg_feats.size(0), -1), weights)

    # 类别归一化：消除原型数量偏置
    class_counts = torch.bincount(proto_classes, minlength=num_classes).float().to(device)
    class_counts = torch.clamp(class_counts, min=1.0)
    scores = scores / class_counts.unsqueeze(0)

    probs = F.softmax(scores, dim=1)
    return probs, sim_raw


def extract_features_batch(model, dataloader, device, max_samples=None):
    """提取特征，支持截断"""
    model.eval()
    all_feats = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x, y_cls, _, _ = batch
            else:
                x = batch[0]
                y_cls = batch[1] if len(batch) > 1 else None
            
            x = x.to(device)
            _, feats, _ = model(x)  # 取归一化特征
            all_feats.append(feats.cpu())
            if y_cls is not None:
                all_labels.append(y_cls)
            
            total_samples += x.size(0)
            if max_samples and total_samples >= max_samples:
                break
    
    all_feats = torch.cat(all_feats, dim=0)
    if max_samples:
        all_feats = all_feats[:max_samples]
    
    if all_labels:
        all_labels = torch.cat(all_labels, dim=0)
        if max_samples:
            all_labels = all_labels[:max_samples]
        return all_feats.numpy(), all_labels.numpy()
    return all_feats.numpy(), None


# ==================== BN自适应TTA函数 ====================

def adapt_bn_statistics(model: torch.nn.Module, 
                       loader: DataLoader, 
                       device: torch.device, 
                       num_epochs: int = 1) -> None:
    """
    用无标签数据更新模型中所有 BatchNorm 层的 running_mean 和 running_var。
    不更新任何可学习参数，完全符合零样本数据隔离规则。
    
    Args:
        model: 要适应的模型
        loader: 目标域无标签数据加载器
        device: 设备
        num_epochs: 遍历数据的轮数（通常 1 足够）
    """
    was_training = model.training
    model.train()
    
    # 临时关闭所有 Dropout 层
    dropout_states = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_states[name] = module.training
            module.eval()
    
    with torch.no_grad():
        for _ in range(num_epochs):
            for batch in loader:
                if len(batch) == 4:
                    x = batch[0]
                elif len(batch) == 2 and isinstance(batch[1], tuple):
                    x = batch[0]
                else:
                    x = batch[0]
                x = x.to(device)
                _ = model(x)
    
    # 恢复 Dropout 原始状态
    for name, module in model.named_modules():
        if name in dropout_states:
            if dropout_states[name]:
                module.train()
    
    if not was_training:
        model.eval()


# ==================== 核心评估函数 ====================
def soft_aggregate_inference(model, dataloader, device, semantic_protos,
                             device_residuals=None, temperature=0.5, prior_weight=0.1, num_classes=4):
    """
    使用软聚合推理评估模型
    - semantic_protos: Dict[str, Tensor], 键为 "(c,p)", 值为特征向量
    - device_residuals: 可选，若提供则为目标设备的残差向量
    - temperature: 温度参数，增大使分布更平滑
    - prior_weight: 类别先验权重
    """
    model.eval()
    correct = 0
    total = 0
    
    # 准备原型矩阵
    proto_matrix, proto_classes = prepare_proto_matrix(semantic_protos, device, residual=device_residuals)
    
    if proto_matrix is None:
        # 回退到标准分类器
        return evaluate_model(model, dataloader, device)
    
    # 计算类别先验（可从训练集统计，这里简化为均匀分布）
    class_prior = torch.ones(num_classes, device=device) / num_classes
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 4:
                x, y_cls, _, _ = batch
            elif len(batch) == 2 and isinstance(batch[1], tuple):
                # 兼容旧格式
                x, (_, y_cls) = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)}")
            x = x.to(device)
            y_cls = y_cls.to(device)
            
            logits, cls_feat, reg_feat = model(x)  # cls_feat: 归一化特征, reg_feat: 未归一化特征
            
            # 计算相似度
            feats_norm = F.normalize(cls_feat, dim=-1)
            proto_norm = F.normalize(proto_matrix, dim=-1)
            sim_raw = torch.matmul(feats_norm, proto_norm.T)          # (B, K)
            sim_raw = torch.clamp(sim_raw, min=-10.0, max=10.0)      # 裁剪极端值
            sim = sim_raw / temperature                               # 温度缩放
            sim_weight = F.softmax(sim, dim=-1)                          # (B, K)
            
            # 回归引导的推理增强（RGPR）
            # 1. 获取回归预测（仅当模型支持时）
            reg_pred = None
            conc_weight = torch.ones_like(sim_weight, device=device)
            if hasattr(model, 'forward_reg'):
                probs = F.softmax(logits, dim=1)
                reg_pred = model.forward_reg(reg_feat, probs=probs).squeeze(1)
                
                # 2. 计算每个原型的浓度权重（简化版：使用类别平均浓度）
                # 假设每个原型的浓度为对应类别的平均浓度
                # 在实际应用中，可以为每个原型存储其对应的浓度值
                for i, c in enumerate(proto_classes):
                    # 简单策略：基于类别浓度范围的归一化
                    # 实际应用中可以使用更复杂的策略
                    conc_weight[:, i] = 1.0  # 这里使用默认权重，后续可以扩展
            
            # 3. 融合特征相似度与浓度权重
            final_weight = sim_weight * (0.7 + 0.3 * conc_weight)
            final_weight = final_weight / final_weight.sum(dim=-1, keepdim=True)  # 重新归一化
            
            # 聚合类别分数
            scores = torch.zeros(feats.size(0), num_classes, device=device)
            # === Sum 聚合：每个类别聚合该类别内所有原型的权重 ===
            scores.scatter_add_(1, proto_classes.unsqueeze(0).expand(feats.size(0), -1), final_weight)
            
            # 类别归一化：消除原型数量偏置
            class_counts = torch.bincount(proto_classes, minlength=num_classes).float().to(device)
            class_counts = torch.clamp(class_counts, min=1.0)         # 防止除零
            scores = scores / class_counts.unsqueeze(0)
            
            # 融合类别先验
            scores = (1 - prior_weight) * scores + prior_weight * class_prior.unsqueeze(0)
            
            preds = scores.argmax(dim=-1)
            total += y_cls.size(0)
            correct += (preds == y_cls).sum().item()
    
    return correct / total if total > 0 else 0.0


def evaluate_model(model: FedGasModel, data_loader: torch.utils.data.DataLoader,
                   device: torch.device, semantic_protos=None, device_residuals=None,
                   use_soft_agg=False, soft_agg_temp=0.5, prior_weight=0.1, num_classes=4) -> float:
    """评估模型
    
    评估模型在给定数据加载器上的准确率
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 设备
        semantic_protos: 语义原型，用于软聚合推理
        device_residuals: 设备残差，用于软聚合推理
        use_soft_agg: 是否使用软聚合推理
        soft_agg_temp: 软聚合温度参数
        prior_weight: 类别先验权重
        num_classes: 类别数量
        
    Returns:
        准确率
    """
    if use_soft_agg and semantic_protos is not None:
        return soft_aggregate_inference(model, data_loader, device,
                                        semantic_protos, device_residuals,
                                        soft_agg_temp, prior_weight, num_classes)
    else:
        # 原有标准评估逻辑
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 4:
                    x, y_cls, _, _ = batch
                elif len(batch) == 2 and isinstance(batch[1], tuple):
                    # 兼容旧格式
                    x, (_, y_cls) = batch
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)}")
                x = x.to(device)
                y_cls = y_cls.to(device)

                logits, _, _ = model(x)
                _, predicted = logits.max(1)
                total += y_cls.size(0)
                correct += predicted.eq(y_cls).sum().item()

        if total == 0:
            return 0.0
        acc = correct / total
        return acc


def compute_mmd(features1: torch.Tensor, features2: torch.Tensor, seed: int = 42) -> torch.Tensor:
    r"""计算最大均值差异
    
    计算两个特征分布之间的最大均值差异（MMD），使用高斯核。
    \f[
        \text{MMD}^2(P, Q) = \mathbb{E}_{x,x'}[k(x,x')] + \mathbb{E}_{y,y'}[k(y,y')] - 2\mathbb{E}_{x,y}[k(x,y)]
    \f]
    其中 k 为高斯核: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    
    重要：返回值保留为 Tensor（非 Python 标量），以保持梯度流。
    调用方若仅需日志值，需自行 .item()。
    
    Args:
        features1: 第一个特征分布，形状 (N1, D)
        features2: 第二个特征分布，形状 (N2, D)
        seed: 随机种子，保证可复现
        
    Returns:
        MMD值（保留计算图的 Tensor，可反向传播）
    """
    torch.manual_seed(seed)
    
    def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        dist = torch.cdist(x, y) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    batch_size = min(1000, len(features1), len(features2))
    indices1 = torch.randperm(len(features1))[:batch_size]
    indices2 = torch.randperm(len(features2))[:batch_size]

    x = features1[indices1]
    y = features2[indices2]

    k_xx = gaussian_kernel(x, x).mean()
    k_yy = gaussian_kernel(y, y).mean()
    k_xy = gaussian_kernel(x, y).mean()

    mmd = k_xx + k_yy - 2 * k_xy
    return mmd  # 不调用 .item()，保留梯度流


def evaluate_model_with_phase_and_soft_agg(model: FedGasModel, data_loader: torch.utils.data.DataLoader,
                                          device: torch.device, semantic_protos=None, device_residuals=None,
                                          soft_agg_temp=0.5, prior_weight=0.1, num_classes=4,
                                          coral_source_feats=None, use_class_conditional_coral=False, global_test_loader=None, source_model=None,
                                          source_feats_provided=None, source_labels_provided=None,
                                          proto_matrix=None, proto_classes=None,
                                          proto_temperatures=None,
                                          use_mahalanobis_inference: bool = False,
                                          semantic_proto_vars: dict = None) -> Dict[str, float]:
    """评估模型在各阶段的准确率，支持软聚合推理和CORAL特征对齐

    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 设备
        semantic_protos: 语义原型，用于软聚合推理
        device_residuals: 设备残差，用于软聚合推理
        soft_agg_temp: 软聚合温度参数 (统一时使用)
        prior_weight: 类别先验权重
        num_classes: 类别数量
        coral_source_feats: CORAL源域特征，用于特征对齐
        use_class_conditional_coral: 是否使用类条件CORAL变换
        global_test_loader: 全局测试加载器，用于类条件CORAL变换
        source_model: 用于提取源域特征的模型，默认为None（使用model）
        source_feats_provided: 已提取的源域特征，用于避免重复提取
        source_labels_provided: 已提取的源域标签，用于避免重复提取
        proto_matrix: 预先计算的原型矩阵（加速推理）
        proto_classes: 预先计算的原型类别（加速推理）
        proto_temperatures: 每原型自适应温度 (K,) 或 None 使用统一温度
        use_mahalanobis_inference: True=马氏距离推理，False=余弦相似度推理
        semantic_proto_vars: 各原型的对角方差，用于马氏距离 (仅 mahalanobis 模式)

    Returns:
        各阶段的准确率
    """
    model.eval()
    phase_correct = {0: 0, 1: 0, 2: 0}
    phase_total = {0: 0, 1: 0, 2: 0}
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # 准备原型矩阵：优先使用传入的缓存，否则回退到动态构建
    # 马氏模式下跳过 proto_matrix 缓存（直接在 semantic_protos 上计算）
    if use_mahalanobis_inference:
        if semantic_protos is None:
            return evaluate_model_with_phase(model, data_loader, device)
    elif proto_matrix is None or proto_classes is None:
        proto_matrix, proto_classes = prepare_proto_matrix(semantic_protos, device, residual=device_residuals)
        if proto_matrix is None:
            return evaluate_model_with_phase(model, data_loader, device)
    else:
        proto_matrix = proto_matrix.to(device)
        proto_classes = proto_classes.to(device)
    
    class_prior = torch.ones(num_classes, device=device) / num_classes
    
    with torch.no_grad():
        all_feats = []
        all_y_cls = []
        all_y_p = []
        
        for batch in data_loader:
            if len(batch) == 4:
                x, y_cls, _, y_p = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)}")
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_p = y_p.to(device)
            
            if use_mahalanobis_inference:
                _, _, reg_feat = model(x)               # 取未归一化特征 reg_feat
                all_feats.append(reg_feat.cpu())
            else:
                _, feats, _ = model(x)                  # 取归一化特征 cls_feat
                all_feats.append(feats.cpu())
            all_y_cls.append(y_cls.cpu())
            all_y_p.append(y_p.cpu())
        
        all_feats = torch.cat(all_feats, dim=0)
        all_y_cls = torch.cat(all_y_cls, dim=0)
        all_y_p = torch.cat(all_y_p, dim=0)
        
        # CORAL 对齐仅在余弦模式下生效（马氏模式使用原始 reg_feat 空间）
        if not use_mahalanobis_inference and coral_source_feats is not None:
            if use_class_conditional_coral:
                if source_feats_provided is not None and source_labels_provided is not None:
                    source_feats = source_feats_provided.to(device)
                    source_labels = source_labels_provided.to(device)
                    all_feats = coral_transform_class_conditional(
                        source_feats, source_labels, all_feats, all_y_cls
                    )
                else:
                    if global_test_loader is not None and source_model is not None:
                        extract_model = source_model if source_model is not None else model
                        source_feats_np, source_labels_np = extract_features_batch(extract_model, global_test_loader, device, max_samples=1000)
                        source_feats = torch.from_numpy(source_feats_np).float().to(device)
                        source_labels = torch.from_numpy(source_labels_np).long().to(device)
                        all_feats = coral_transform_class_conditional(
                            source_feats, source_labels, all_feats, all_y_cls
                        )
                    else:
                        all_feats = coral_transform(coral_source_feats, all_feats)
            else:
                all_feats = coral_transform(coral_source_feats, all_feats)
        
        all_feats = all_feats.to(device)
        
        # ====== 推理分派：马氏距离 vs 余弦相似度 ======
        if use_mahalanobis_inference:
            # 马氏距离推理：使用 reg_feat + 原型方差
            probs, _ = soft_aggregate_probs_mahalanobis(
                all_feats, semantic_protos, semantic_proto_vars,
                temperature=soft_agg_temp
            )
            preds = probs.argmax(dim=-1)
        else:
            # 余弦相似度推理（原有逻辑）
            feats_norm = F.normalize(all_feats, dim=-1)
            proto_norm = F.normalize(proto_matrix, dim=-1)
            sim_raw = torch.matmul(feats_norm, proto_norm.T)          # (B, K)
            sim_raw = torch.clamp(sim_raw, min=-10.0, max=10.0)      # 裁剪极端值
            
            if proto_temperatures is not None and proto_temperatures.numel() == sim_raw.size(1):
                temperatures = proto_temperatures.to(device).unsqueeze(0)  # (1, K)
                sim = sim_raw / temperatures
            else:
                sim = sim_raw / soft_agg_temp                               # 统一温度
            weights = F.softmax(sim, dim=-1)                          # (B, K)
            
            scores = torch.zeros(all_feats.size(0), num_classes, device=device)
            scores.scatter_add_(1, proto_classes.unsqueeze(0).expand(all_feats.size(0), -1), weights)
            
            class_counts = torch.bincount(proto_classes, minlength=num_classes).float().to(device)
            class_counts = torch.clamp(class_counts, min=1.0)
            scores = scores / class_counts.unsqueeze(0)
            
            scores = (1 - prior_weight) * scores + prior_weight * class_prior.unsqueeze(0)
            preds = scores.argmax(dim=-1)
        
        for i in range(len(all_y_p)):
            phase = all_y_p[i].item()
            cls = all_y_cls[i].item()
            if phase in phase_total:
                phase_total[phase] += 1
                if preds[i] == all_y_cls[i]:
                    phase_correct[phase] += 1
            if cls in class_total:
                class_total[cls] += 1
                if preds[i] == all_y_cls[i]:
                    class_correct[cls] += 1
    
    early_total = phase_total.get(0, 0)
    middle_total = phase_total.get(1, 0)
    late_total = phase_total.get(2, 0)
    total_samples = sum(phase_total.values())
    
    early_acc = phase_correct.get(0, 0) / early_total if early_total > 0 else 0.0
    middle_acc = phase_correct.get(1, 0) / middle_total if middle_total > 0 else 0.0
    late_acc = phase_correct.get(2, 0) / late_total if late_total > 0 else 0.0
    overall_acc = sum(phase_correct.values()) / total_samples if total_samples > 0 else 0.0
    
    # 计算类别准确率
    class_accuracies = {}
    for cls in range(num_classes):
        cls_total = class_total.get(cls, 0)
        cls_acc = class_correct.get(cls, 0) / cls_total if cls_total > 0 else 0.0
        class_accuracies[f'class_{cls}'] = cls_acc
    
    return {
        'early': early_acc,
        'middle': middle_acc,
        'late': late_acc,
        'global': overall_acc,
        **class_accuracies
    }


def estimate_device_residual(model, dataloader, device, semantic_protos):
    """用全部样本估计设备残差"""
    model.eval()
    residuals = []
    with torch.no_grad():
        for batch in dataloader:
            # 处理不同批次格式
            if len(batch) == 4:
                x, y_cls, _, y_p = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)}")
            x = x.to(device)
            _, feats, _ = model(x)  # 取归一化特征
            for i in range(len(x)):
                key = f"({y_cls[i].item()},{y_p[i].item()})"
                if key in semantic_protos:
                    mu_sem = semantic_protos[key].to(device)
                    residuals.append(feats[i] - mu_sem)
    if residuals:
        return torch.stack(residuals).mean(dim=0)
    else:
        # 如果没有足够的样本，返回零向量
        return torch.zeros(model.feat_proj.out_features, device=device)


def evaluate_model_with_phase(model: torch.nn.Module, loader: DataLoader, device: str, 
                              semantic_protos=None, device_residuals=None,
                              use_soft_agg=False, soft_agg_temp=0.1, num_classes=4) -> Dict[str, float]:
    """评估模型性能
    
    返回全局准确率和各阶段准确率
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        semantic_protos: 语义原型，用于软聚合推理
        device_residuals: 设备残差，用于软聚合推理
        use_soft_agg: 是否使用软聚合推理
        soft_agg_temp: 软聚合温度参数
        num_classes: 类别数量
        
    Returns:
        包含全局准确率和各阶段准确率的字典
    """
    from collections import defaultdict
    model.eval()
    correct = 0
    total = 0
    phase_correct = defaultdict(int)
    phase_total = defaultdict(int)
    
    if use_soft_agg and semantic_protos is not None:
        # 准备原型矩阵
        proto_matrix, proto_classes = prepare_proto_matrix(semantic_protos, device, residual=device_residuals)
        
        if proto_matrix is None:
            # 回退到标准分类器
            return evaluate_model_with_phase(model, loader, device)
        
        with torch.no_grad():
            for batch in loader:
                # 处理不同批次格式
                if len(batch) == 4:
                    x, y_cls, _, phase = batch
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)}")
                x = x.to(device)
                y_cls = y_cls.to(device)
                
                _, feats, _ = model(x)  # (B, D)
                
                # 计算相似度
                feats_norm = F.normalize(feats, dim=-1)
                proto_norm = F.normalize(proto_matrix, dim=-1)
                sim = torch.matmul(feats_norm, proto_norm.T) / soft_agg_temp  # (B, K)
                weights = F.softmax(sim, dim=-1)  # (B, K)
                
                # 聚合类别分数
                scores = torch.zeros(feats.size(0), num_classes, device=device)
                scores.scatter_add_(1, proto_classes.unsqueeze(0).expand(feats.size(0), -1), weights)
                
                predicted = scores.argmax(dim=-1)
                
                total += y_cls.size(0)
                correct += (predicted == y_cls).sum().item()
                
                # 统计各阶段的准确率
                for i in range(len(phase)):
                    phase_idx = phase[i].item()
                    phase_total[phase_idx] += 1
                    if predicted[i] == y_cls[i]:
                        phase_correct[phase_idx] += 1
    else:
        # 原有标准评估逻辑
        with torch.no_grad():
            for batch in loader:
                # 处理不同批次格式
                if len(batch) == 4:
                    x, y_cls, _, phase = batch
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)}")
                x = x.to(device)
                y_cls = y_cls.to(device)
                
                logits, _, _ = model(x)
                _, predicted = logits.max(1)
                
                total += y_cls.size(0)
                correct += predicted.eq(y_cls).sum().item()
                
                # 统计各阶段的准确率
                for i in range(len(phase)):
                    phase_idx = phase[i].item()
                    phase_total[phase_idx] += 1
                    if predicted[i] == y_cls[i]:
                        phase_correct[phase_idx] += 1
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0.0
    phase_accuracy = {}
    phase_map = {0: 'early', 1: 'middle', 2: 'late'}
    
    for phase_idx, phase_name in phase_map.items():
        if phase_total.get(phase_idx, 0) > 0:
            phase_accuracy[phase_name] = phase_correct.get(phase_idx, 0) / phase_total[phase_idx]
        else:
            phase_accuracy[phase_name] = 0.0
    
    return {
        'global': accuracy,
        'early': phase_accuracy['early'],
        'middle': phase_accuracy['middle'],
        'late': phase_accuracy['late']
    }

import torch.nn.functional as F

def contrastive_loss_with_protos(feats, labels, phase_labels, proto_dict, temperature=0.1):
    """
    feats: (B, D) 特征向量
    labels: (B,) 分类标签
    phase_labels: (B,) 阶段标签
    proto_dict: Dict[Tuple[int,int] or Tuple[int,int,int], Tensor(D,)] 原型字典
    temperature: 温度系数
    """
    B, D = feats.shape
    device = feats.device
    
    # 收集所有原型及对应的键
    proto_list = []
    proto_keys = []
    for key, proto in proto_dict.items():
        proto_list.append(proto.to(device))
        proto_keys.append(key)
    proto_matrix = torch.stack(proto_list)  # (K, D)
    
    # 获取每个样本的正原型索引
    pos_indices = []
    for i in range(B):
        c = labels[i].item()
        p = phase_labels[i].item()
        
        # 查找正原型
        best_pos_idx = -1
        best_sim = -float('inf')
        
        for j, key in enumerate(proto_keys):
            # 解析键
            if isinstance(key, tuple):
                if len(key) == 3:
                    proto_c, proto_p, _ = key
                else:
                    proto_c, proto_p = key
            else:
                # 处理字符串格式，可能是 "(0, 0, 0)" 或 "0_0_0"
                if key.startswith('(') and key.endswith(')'):
                    # 处理 "(0, 0, 0)" 格式
                    key_clean = key.strip('()')
                    parts = key_clean.split(',')
                    proto_c = int(parts[0].strip())
                    proto_p = int(parts[1].strip())
                else:
                    # 处理 "0_0_0" 格式
                    parts = key.split('_')
                    proto_c = int(parts[0])
                    proto_p = int(parts[1])
            
            # 统一按 (class, phase) 精确匹配
            if c == proto_c and p == proto_p:
                best_pos_idx = j
                break
        
        pos_indices.append(best_pos_idx)
    pos_indices = torch.tensor(pos_indices, device=device)
    
    # 归一化
    feats_norm = F.normalize(feats, dim=-1)
    proto_norm = F.normalize(proto_matrix, dim=-1)
    
    # 相似度矩阵
    sim = torch.matmul(feats_norm, proto_norm.T) / temperature  # (B, K)
    
    # 构建正样本 mask
    pos_mask = torch.zeros_like(sim)
    valid_mask = pos_indices >= 0
    if valid_mask.any():
        pos_mask[valid_mask, pos_indices[valid_mask]] = 1.0
    
    # InfoNCE 损失
    exp_sim = torch.exp(sim)
    pos_exp = (exp_sim * pos_mask).sum(dim=-1)
    loss = -torch.log(pos_exp / exp_sim.sum(dim=-1) + 1e-8)
    
    # 仅对有效正样本计算平均损失
    if valid_mask.any():
        loss = loss[valid_mask].mean()
        # 添加调试日志
        logger = logging.getLogger('gasfl')
        logger.debug(f"Contrastive loss: {loss.item():.4f}, Valid samples: {valid_mask.sum().item()}/{B}")
    else:
        loss = torch.tensor(0.0, device=device)
        logger = logging.getLogger('gasfl')
        logger.debug("Warning: No valid positive prototypes found, contrastive loss is 0")
    
    return loss


def compute_calibration_params(model, data_loader, device, num_bins=5):
    """Estimate piecewise affine calibration in ppm space.

    The model predicts normalized concentration in [0, 1], but evaluation metrics use ppm.
    Therefore both binning and affine fitting are performed on ppm values:
        pred_ppm -> true_ppm
    This avoids the previous scale mismatch where params fitted on pred_norm were applied
    to pred_ppm again, producing huge calibrated predictions.
    """
    model.eval()
    all_y_true = {c: [] for c in range(4)}
    all_pred_ppm = {c: [] for c in range(4)}

    with torch.no_grad():
        for batch in data_loader:
            x, y_cls, y_reg_full, _ = batch
            x = x.to(device)
            y_cls = y_cls.to(device)
            logits, _, reg_feat = model(x)
            probs = F.softmax(logits, dim=1)
            if not hasattr(model, 'forward_reg'):
                continue

            pred_norm = model.forward_reg(reg_feat, probs=probs)
            y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls.cpu()].unsqueeze(1).to(device)

            for c in range(4):
                mask = (y_cls == c)
                if mask.sum() == 0:
                    continue
                min_c = CONC_STATS[c]['min']
                max_c = CONC_STATS[c]['max']
                pred_ppm = pred_norm[mask] * (max_c - min_c) + min_c
                all_y_true[c].append(y_reg[mask].detach().cpu().numpy())
                all_pred_ppm[c].append(pred_ppm.detach().cpu().numpy())

    calib_params = {}
    for c in range(4):
        if not all_y_true[c]:
            continue
        y_true = np.concatenate(all_y_true[c], axis=0).flatten()
        pred_ppm = np.concatenate(all_pred_ppm[c], axis=0).flatten()
        valid = np.isfinite(y_true) & np.isfinite(pred_ppm)
        y_true = y_true[valid]
        pred_ppm = pred_ppm[valid]
        if len(y_true) < 2:
            continue

        bin_edges = np.percentile(pred_ppm, np.linspace(0, 100, num_bins + 1))
        bin_edges = np.maximum.accumulate(bin_edges)
        params = []
        for i in range(num_bins):
            if i == num_bins - 1:
                mask = (pred_ppm >= bin_edges[i]) & (pred_ppm <= bin_edges[i + 1])
            else:
                mask = (pred_ppm >= bin_edges[i]) & (pred_ppm < bin_edges[i + 1])

            if mask.sum() < 2 or np.ptp(pred_ppm[mask]) < 1e-6:
                params.append({'scale': 1.0, 'shift': 0.0})
                continue

            x_bin = pred_ppm[mask]
            y_bin = y_true[mask]
            scale, shift = np.polyfit(x_bin, y_bin, deg=1)
            scale = float(np.clip(scale, 0.1, 10.0))
            shift = float(shift)
            params.append({'scale': scale, 'shift': shift})

        calib_params[c] = {
            'space': 'ppm',
            'bin_edges': bin_edges,
            'params': params,
            'clip_min': float(CONC_STATS[c]['min']),
            'clip_max': float(CONC_STATS[c]['max'])
        }
    return calib_params


def apply_calibration(pred_norm, c, calib_params):
    """Apply piecewise affine calibration and return ppm predictions."""
    if np.any(np.isnan(pred_norm)):
        return pred_norm
    min_c = CONC_STATS[c]['min']
    max_c = CONC_STATS[c]['max']
    pred_ppm = pred_norm * (max_c - min_c) + min_c

    if c not in calib_params or not calib_params[c]:
        return pred_ppm

    class_params = calib_params[c]
    bin_edges = class_params['bin_edges']
    params = class_params['params']
    pred_calib = pred_ppm.copy()

    for i, param in enumerate(params):
        if i == len(params) - 1:
            mask = (pred_ppm >= bin_edges[i]) & (pred_ppm <= bin_edges[i + 1])
        else:
            mask = (pred_ppm >= bin_edges[i]) & (pred_ppm < bin_edges[i + 1])
        if mask.any():
            pred_calib[mask] = param['scale'] * pred_ppm[mask] + param['shift']

    clip_min = class_params.get('clip_min', min_c)
    clip_max = class_params.get('clip_max', max_c)
    pred_calib = np.clip(pred_calib, clip_min, clip_max)
    return pred_calib

def save_calib_params(calib_params, path):
    """保存分段校准参数到 .npz 文件"""
    import numpy as np
    save_dict = {}
    for c in range(4):
        if c in calib_params:
            save_dict[f'bin_edges_{c}'] = calib_params[c]['bin_edges']
            for i, p in enumerate(calib_params[c]['params']):
                save_dict[f'scale_{c}_{i}'] = np.array(p['scale'])
                save_dict[f'shift_{c}_{i}'] = np.array(p['shift'])
    np.savez(path, **save_dict)


def load_calib_params(path):
    """从 .npz 文件加载分段校准参数"""
    import numpy as np
    data = np.load(path, allow_pickle=True)
    calib_params = {}
    for c in range(4):
        if f'bin_edges_{c}' in data:
            params = []
            for i in range(5):
                if f'scale_{c}_{i}' in data:
                    params.append({'scale': float(data[f'scale_{c}_{i}']), 
                                   'shift': float(data[f'shift_{c}_{i}'])})
                else:
                    break
            if params:
                calib_params[c] = {'bin_edges': data[f'bin_edges_{c}'], 'params': params}
    return calib_params if calib_params else None

def evaluate_regression_metrics(model, data_loader, device, tolerance=0.1, enable_calibration=False, calib_params=None):
    """计算各类别的回归指标 R², RMSE, MAE

    Args:
        model: 训练好的模型（需包含回归头）
        data_loader: 目标域数据加载器
        device: 设备
        tolerance: 未使用 (保留接口兼容)
        enable_calibration: 是否启用目标域回归后校准
        calib_params: 预计算的校准参数
    """
    model.eval()
    all_y_true = {c: [] for c in range(4)}
    all_y_pred = {c: [] for c in range(4)}
    all_pred_norm = {c: [] for c in range(4)}

    with torch.no_grad():
        for batch in data_loader:
            x, y_cls, y_reg_full, _ = batch
            x = x.to(device)
            logits, cls_feat, reg_feat = model(x)
            probs = F.softmax(logits, dim=1)

            if not hasattr(model, 'forward_reg'):
                logger.warning("Model does not have forward_reg method.")
                empty_metrics = {c: {'R2': 0.0, 'RMSE': 0.0, 'MAE': 0.0, 'n_samples': 0} for c in range(4)}
                return empty_metrics, empty_metrics

            pred_norm = model.forward_reg(reg_feat, probs=probs)
            y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1)

            for c in range(4):
                mask = (y_cls == c)
                if mask.sum() > 0:
                    min_c = CONC_STATS[c]['min']
                    max_c = CONC_STATS[c]['max']
                    reg_pred_ppm = pred_norm[mask] * (max_c - min_c) + min_c
                    all_y_true[c].append(y_reg[mask].cpu().numpy())
                    all_y_pred[c].append(reg_pred_ppm.cpu().numpy())
                    all_pred_norm[c].append(pred_norm[mask].cpu().numpy())
    
    # 目标域回归后校准（优先使用预计算的calib_params）
    if enable_calibration:
        calibrated_y_pred = {c: [] for c in range(4)}
        if calib_params is not None:
            # 使用预先计算的校准参数（方案3）
            # apply_calibration 输出已经是 ppm 单位，直接使用
            for c in range(4):
                if all_pred_norm[c]:
                    for pred_norm_batch in all_pred_norm[c]:
                        calibrated_pred = apply_calibration(pred_norm_batch, c, calib_params)
                        calibrated_y_pred[c].append(calibrated_pred)
                else:
                    calibrated_y_pred[c] = all_y_pred[c]
        else:
            # 使用传统校准方式（单段线性）
            for c in range(4):
                if all_y_true[c] and all_pred_norm[c]:
                    y_true = np.concatenate(all_y_true[c], axis=0).flatten()
                    pred_norm = np.concatenate(all_pred_norm[c], axis=0).flatten()
                    
                    # 计算校准参数
                    min_c = CONC_STATS[c]['min']
                    max_c = CONC_STATS[c]['max']
                    
                    # 计算真实值和预测值的范围
                    true_min, true_max = y_true.min(), y_true.max()
                    pred_min, pred_max = pred_norm.min(), pred_norm.max()
                    
                    # 计算缩放因子和偏移量
                    if pred_max > pred_min:
                        scale = (true_max - true_min) / (pred_max - pred_min)
                        shift = true_min - scale * pred_min
                        
                        # 应用校准
                        for pred_norm_batch in all_pred_norm[c]:
                            calibrated_pred = scale * pred_norm_batch + shift
                            calibrated_y_pred[c].append(calibrated_pred)
                    else:
                        # 如果预测值范围太小，使用原始预测
                        calibrated_y_pred[c] = all_y_pred[c]
                else:
                    calibrated_y_pred[c] = all_y_pred[c]
        # 使用校准后的预测值
        all_y_pred = calibrated_y_pred
    
    metrics = {}
    overall_y_true = []
    overall_y_pred = []
    
    for c in range(4):
        if all_y_true[c]:
            y_true = np.concatenate(all_y_true[c], axis=0).flatten()
            y_pred = np.concatenate(all_y_pred[c], axis=0).flatten()
            overall_y_true.extend(y_true)
            overall_y_pred.extend(y_pred)
            
            # 打印预测值统计信息
            print(f"Class {c}: y_true mean={np.mean(y_true):.4f}, y_pred mean={np.mean(y_pred):.4f}, y_pred std={np.std(y_pred):.4f}")
            
            # 过滤 NaN/Inf
            valid_mask = np.isfinite(y_pred)
            if valid_mask.sum() < 2:
                metrics[c] = {'R2': -999, 'RMSE': -1, 'MAE': -1, 'n_samples': len(y_true)}
                continue
            if valid_mask.sum() < len(y_pred):
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]

            metrics[c] = {
                'R2': r2_score(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'n_samples': len(y_true)
            }
    
    if overall_y_true:
        overall_y_true = np.array(overall_y_true)
        overall_y_pred = np.array(overall_y_pred)
        valid_mask = np.isfinite(overall_y_pred)
        if valid_mask.sum() < 2:
            overall = {'R2': -999, 'RMSE': -1, 'MAE': -1, 'Acc': 0.0, 'n_samples': len(overall_y_true)}
        else:
            if valid_mask.sum() < len(overall_y_pred):
                overall_y_true = overall_y_true[valid_mask]
                overall_y_pred = overall_y_pred[valid_mask]
            overall = {
                'R2': r2_score(overall_y_true, overall_y_pred),
                'RMSE': np.sqrt(mean_squared_error(overall_y_true, overall_y_pred)),
                'MAE': mean_absolute_error(overall_y_true, overall_y_pred),
                'n_samples': len(overall_y_true)
            }
    else:
        overall = {}
    
    return metrics, overall

# ==================== 日志与随机种子 ====================

def setup_logging(log_dir: str, log_filename: Optional[str] = None) -> logging.Logger:
    """设置日志记录器
    
    同时输出到控制台和文件
    
    Args:
        log_dir: 日志目录
        log_filename: 日志文件名
        
    Returns:
        日志记录器
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('gasfl')
    logger.setLevel(logging.INFO)
    
    # 移除所有现有处理器（避免重复，兼容所有 Python 版本）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    
    # 文件处理器 - 如果未指定文件名，使用时间戳
    if log_filename is None:
        log_filename = f'run_{time.strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_path / log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger


def set_random_seed(seed: int = 42) -> None:
    """固定随机种子
    
    确保实验可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== 结果保存与加载 ====================

def save_results(results: Dict, save_dir: str, filename: str = "results.json") -> None:
    """保存实验结果
    
    以JSON格式保存实验结果，并添加环境快照
    
    Args:
        results: 实验结果字典
        save_dir: 保存目录
        filename: 文件名
    """
    import subprocess
    import sys
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 添加环境信息
    env_info = {}
    try:
        import git
        try:
            repo = git.Repo(search_parent_directories=True)
            env_info['git_commit'] = repo.head.commit.hexsha
            env_info['git_branch'] = repo.active_branch.name
        except Exception:
            # 当前目录不是 git 仓库或无提交记录（如 sandbox 环境）
            env_info['git_commit'] = 'unknown'
    except ImportError:
        # 如果没有 gitpython，尝试命令行
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            env_info['git_commit'] = commit
        except:
            env_info['git_commit'] = 'unknown'
    env_info['python_version'] = sys.version
    # 可以进一步添加 pip freeze，但可能过大，可选
    results['environment'] = env_info
    
    final_path = save_path / filename
    tmp_path = save_path / f".{filename}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp_path.replace(final_path)
    print(f"Results saved to {final_path}")


def load_results(save_dir: str, filename: str = "results.json") -> Dict:
    """加载实验结果
    
    从JSON文件加载实验结果
    
    Args:
        save_dir: 保存目录
        filename: 文件名
        
    Returns:
        实验结果字典
    """
    with open(Path(save_dir) / filename, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== 可视化函数 ====================
def plot_training_curves(history: List[Dict], save_dir: str, filename: str = "training_curves.png") -> None:
    """绘制训练曲线，支持多个测试客户端，修复x轴对齐问题"""
    if not history:
        print("No history data to plot.")
        return
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 重新组织数据：按轮次聚合
    rounds = sorted(set(h['round'] for h in history))
    round_data = {r: {} for r in rounds}
    for h in history:
        r = h['round']
        round_data[r]['test_acc'] = h.get('test_acc', 0)
        round_data[r]['align_loss'] = h.get('align_loss', 0)
        cid = h.get('test_client_id')
        if cid is not None:
            round_data[r][f'client_{cid}_acc'] = h.get('test_client_acc', 0)

    # 提取对齐后的数据
    test_acc = [round_data[r].get('test_acc', 0) for r in rounds]
    align_loss = [round_data[r].get('align_loss', 0) for r in rounds]

    # 获取所有测试客户端ID
    client_ids = set()
    for h in history:
        if 'test_client_id' in h:
            client_ids.add(h['test_client_id'])
    client_ids = sorted(client_ids)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：准确率
    axes[0].plot(rounds, test_acc, 'b-', linewidth=2, label='Global Test Accuracy')
    # 使用colormap避免颜色硬编码
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, cid in enumerate(client_ids):
        client_acc = [round_data[r].get(f'client_{cid}_acc', float('nan')) for r in rounds]
        axes[0].plot(rounds, client_acc, '--', color=colors[i % len(colors)],
                     marker='.', markersize=3, label=f'Client {cid}')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Classification Accuracy')
    axes[0].legend(loc='best', fontsize='small')
    axes[0].grid(True, alpha=0.3)

    # 右图：对齐损失
    axes[1].plot(rounds, align_loss, 'g-', linewidth=2)
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Align Loss')
    axes[1].set_title('Domain Alignment Loss')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path / filename}")


def plot_aggregation_weights(weights_history: List[Dict], save_dir: str, filename: str = "aggregation_weights.png") -> None:
    """绘制聚合权重曲线
    
    绘制客户端聚合权重随轮次的变化
    
    Args:
        weights_history: 每轮的权重字典，如 {'client_1': 0.25, 'client_2': 0.25, ...}
        save_dir: 保存目录
        filename: 文件名
    """
    if not weights_history:
        print("No weight history to plot.")
        return
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    rounds = list(range(1, len(weights_history) + 1))
    # 提取所有客户端ID
    client_ids = set()
    for w_dict in weights_history:
        client_ids.update(w_dict.keys())
    client_ids = sorted(client_ids)

    plt.figure(figsize=(10, 6))
    for cid in client_ids:
        weights = [w_dict.get(cid, 0) for w_dict in weights_history]
        plt.plot(rounds, weights, label=cid, marker='o', markersize=3)
    plt.xlabel('Round')
    plt.ylabel('Aggregation Weight')
    plt.title('Learnable Aggregation Weights')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / filename, dpi=150)
    plt.close()
    print(f"Aggregation weights plot saved to {save_path / filename}")


def plot_phase_curves(history: List[Dict], forgetting_dict: Dict, save_dir: str, filename: str = "phase_curves.png") -> None:
    """绘制分阶段准确率曲线，遗忘信息置于图外右侧避免遮挡"""
    if not history:
        print("No history data to plot.")
        return
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    rounds = [h['round'] for h in history]
    acc_early = [h.get('acc_early', 0) for h in history]
    acc_middle = [h.get('acc_middle', 0) for h in history]
    acc_late = [h.get('acc_late', 0) for h in history]

    plt.figure(figsize=(12, 6))
    plt.plot(rounds, acc_early, 'b-', label='Early Phase', marker='o', markersize=4)
    plt.plot(rounds, acc_middle, 'g-', label='Middle Phase', marker='s', markersize=4)
    plt.plot(rounds, acc_late, 'r-', label='Late Phase', marker='^', markersize=4)
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.title('Phase-wise Test Accuracy over Rounds')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    # 构建遗忘信息文本
    text_lines = []
    if 'peak_early' in forgetting_dict:
        text_lines.append(f"Peak Early: {forgetting_dict['peak_early']:.4f}")
    if 'peak_middle' in forgetting_dict:
        text_lines.append(f"Peak Middle: {forgetting_dict['peak_middle']:.4f}")
    if 'early' in forgetting_dict:
        text_lines.append(f"Forgetting Early: {forgetting_dict['early']:.4f}")
    if 'middle' in forgetting_dict:
        text_lines.append(f"Forgetting Middle: {forgetting_dict['middle']:.4f}")

    if text_lines:
        text_str = '\n'.join(text_lines)
        # 将文本框置于图外右侧
        plt.gcf().text(0.78, 0.85, text_str, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top')

    plt.tight_layout(rect=(0, 0, 0.75, 1))  # 为右侧文本框留出空间，使用元组类型
    plt.savefig(save_path / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Phase curves saved to {save_path / filename}")

def plot_tsne_features(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loaders: Dict[int, DataLoader],
    device: str,
    save_dir: str,
    filename: str = "tsne_features.png",
    max_samples_per_set: int = 1000,
    random_state: int = 42
) -> None:
    """
    使用 t-SNE 可视化模型提取的特征分布

    Args:
        model: 训练好的模型
        train_loader: 训练集数据加载器（用于展示训练分布）
        test_loaders: 测试客户端数据加载器字典 {client_id: loader}
        device: 计算设备
        save_dir: 保存目录
        filename: 保存文件名
        max_samples_per_set: 每个数据集最大采样数量（避免 t-SNE 过慢）
        random_state: 随机种子
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 收集训练集特征和标签
    train_features, train_labels = [], []
    with torch.no_grad():
        for batch in train_loader:
            if len(batch) == 4:
                x, y_cls, _, _ = batch
            else:
                continue
            x = x.to(device)
            _, feats, _ = model(x)
            train_features.append(feats.cpu().numpy())
            train_labels.append(y_cls.numpy())

    if train_features:
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        if len(train_features) > max_samples_per_set:
            idx = np.random.choice(len(train_features), max_samples_per_set, replace=False)
            train_features = train_features[idx]
            train_labels = train_labels[idx]
    else:
        train_features, train_labels = np.array([]), np.array([])

    # 收集各测试客户端特征和标签
    test_data = {}
    for cid, loader in test_loaders.items():
        feats_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    x, y_cls, _, _ = batch
                else:
                    continue
                x = x.to(device)
                _, feats, _ = model(x)
                feats_list.append(feats.cpu().numpy())
                labels_list.append(y_cls.numpy())
        if feats_list:
            feats = np.concatenate(feats_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            if len(feats) > max_samples_per_set:
                idx = np.random.choice(len(feats), max_samples_per_set, replace=False)
                feats = feats[idx]
                labels = labels[idx]
            test_data[cid] = (feats, labels)

    # 合并所有特征用于 t-SNE 拟合
    all_features = [train_features] if len(train_features) > 0 else []
    for feats, _ in test_data.values():
        all_features.append(feats)
    if not all_features:
        print("No features to plot for t-SNE.")
        return
    all_features = np.concatenate(all_features, axis=0)

    # t-SNE 降维
    print(f"Running t-SNE on {len(all_features)} samples...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    all_2d = tsne.fit_transform(all_features)

    # 分割回各数据集
    idx = 0
    train_2d = None
    if len(train_features) > 0:
        train_2d = all_2d[idx:idx+len(train_features)]
        idx += len(train_features)
    test_2d_dict = {}
    for cid, (feats, _) in test_data.items():
        test_2d_dict[cid] = all_2d[idx:idx+len(feats)]
        idx += len(feats)

    # 绘图：每个子图对应一个数据集
    n_plots = (1 if len(train_features) > 0 else 0) + len(test_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    if train_2d is not None:
        ax = axes[plot_idx]
        for cls in np.unique(train_labels):
            mask = train_labels == cls
            ax.scatter(train_2d[mask, 0], train_2d[mask, 1],
                       c=colors[int(cls)], label=gas_names[int(cls)],
                       alpha=0.5, s=5)
        ax.set_title(f"Training Set (n={len(train_features)})")
        ax.legend(markerscale=3, fontsize='small')
        plot_idx += 1

    for cid, (feats, labels) in test_data.items():
        ax = axes[plot_idx]
        feat_2d = test_2d_dict[cid]
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(feat_2d[mask, 0], feat_2d[mask, 1],
                       c=colors[int(cls)], label=gas_names[int(cls)],
                       alpha=0.5, s=5)
        ax.set_title(f"Test Client {cid} (n={len(feats)})")
        ax.legend(markerscale=3, fontsize='small')
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {save_path / filename}")

def plot_concentration_feature_correlation(
    model, data_loader, device, save_dir, filename="conc_feat_corr.png"
):
    """
    绘制浓度与特征的相关性散点图，验证特征空间的有序化
    
    Args:
        model: 训练好的模型
        data_loader: 目标域数据加载器（如 Unit5）
        device: 设备
        save_dir: 保存目录
        filename: 文件名
    """
    model.eval()
    
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    all_feats = {c: [] for c in range(4)}
    all_concs = {c: [] for c in range(4)}
    
    with torch.no_grad():
        for batch in data_loader:
            x, y_cls, y_reg_full, _ = batch
            x = x.to(device)
            _, feats, _ = model(x)
            
            y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls]
            
            for c in range(4):
                mask = (y_cls == c)
                if mask.sum() > 0:
                    all_feats[c].append(feats[mask].cpu().numpy())
                    all_concs[c].append(y_reg[mask].cpu().numpy())
    
    # 防御性检查：确保有足够的有效数据
    total_samples = sum(len(np.concatenate(all_feats[c], axis=0)) if all_feats[c] else 0 for c in range(4))
    if total_samples == 0:
        print("Warning: No features extracted, skipping concentration-feature correlation plot.")
        return
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr
    
    for c in range(4):
        ax = axes[c]
        if not all_feats[c] or len(np.concatenate(all_feats[c], axis=0)) < 2:
            ax.text(0.5, 0.5, 'No/Insufficient samples', ha='center', va='center')
            ax.set_title(gas_names[c])
            continue
        
        feats = np.concatenate(all_feats[c], axis=0)
        conc = np.concatenate(all_concs[c], axis=0)
        
        # 检查特征是否全零
        if np.allclose(feats, 0):
            ax.text(0.5, 0.5, 'Features are zeros', ha='center', va='center')
            ax.set_title(gas_names[c])
            continue
        
        pca = PCA(n_components=1)
        feat_1d = pca.fit_transform(feats).flatten()
        
        corr, pval = pearsonr(feat_1d, conc)
        
        ax.scatter(conc, feat_1d, c=colors[c], alpha=0.5, s=10)
        ax.set_xlabel('Concentration (ppm)')
        ax.set_ylabel('Feature PC1')
        ax.set_title(f'{gas_names[c]}: r = {corr:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Concentration-feature correlation plot saved to {save_path}")

def plot_regression_scatter(
    model, data_loader, device, save_dir, filename="reg_scatter.png", tolerance=0.1
):
    """绘制预测浓度 vs 真实浓度散点图，显示 R² 和 RMSE

    Args:
        model: 训练好的模型（需包含回归头）
        data_loader: 目标域数据加载器
        device: 设备
        save_dir: 保存目录
        filename: 文件名
        tolerance: 相对误差容忍度
    """
    model.eval()

    all_y_true = {c: [] for c in range(4)}
    all_y_pred = {c: [] for c in range(4)}

    with torch.no_grad():
        for batch in data_loader:
            x, y_cls, y_reg_full, _ = batch
            x = x.to(device)
            logits, cls_feat, reg_feat = model(x)
            probs = F.softmax(logits, dim=1)

            if not hasattr(model, 'forward_reg'):
                logger.warning("Model does not have forward_reg method.")
                return

            pred_norm = model.forward_reg(reg_feat, probs=probs)
            y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1)

            for c in range(4):
                mask = (y_cls == c)
                if mask.sum() > 0:
                    min_c = CONC_STATS[c]['min']
                    max_c = CONC_STATS[c]['max']
                    reg_pred_ppm = pred_norm[mask] * (max_c - min_c) + min_c
                    all_y_true[c].append(y_reg[mask].cpu().numpy())
                    all_y_pred[c].append(reg_pred_ppm.cpu().numpy())

    total_samples = sum(len(np.concatenate(all_y_true[c], axis=0)) if all_y_true[c] else 0 for c in range(4))
    if total_samples == 0:
        print("Warning: No samples collected for regression scatter plot.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    from sklearn.metrics import r2_score

    for c in range(4):
        ax = axes[c]
        if not all_y_true[c] or len(np.concatenate(all_y_true[c], axis=0)) < 2:
            ax.text(0.5, 0.5, 'No/Insufficient samples', ha='center', va='center')
            ax.set_title(gas_names[c])
            continue

        y_true = np.concatenate(all_y_true[c], axis=0).flatten()
        y_pred = np.concatenate(all_y_pred[c], axis=0).flatten()

        if np.std(y_pred) < 1e-6:
            ax.text(0.5, 0.5, f'Constant prediction\n({y_pred[0]:.4f})', ha='center', va='center')
            ax.set_title(gas_names[c])
            continue

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        ax.scatter(y_true, y_pred, c=colors[c], alpha=0.5, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=1)
        ax.set_xlabel('True Concentration (ppm)')
        ax.set_ylabel('Predicted Concentration (ppm)')
        ax.set_title(f'{gas_names[c]}: R²={r2:.3f}, RMSE={rmse:.1f}, MAE={mae:.1f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Regression scatter plot saved to {save_path}")


def plot_source_classification_acc(source_cls_accs, save_dir, filename="source_classification_acc.png"):
    """绘制训练客户端本地测试集的分类准确率柱状图"""
    if not source_cls_accs:
        return
    import matplotlib.pyplot as plt
    from pathlib import Path
    clients = sorted(source_cls_accs.keys())
    accs = [source_cls_accs[c] for c in clients]
    plt.figure(figsize=(6, 4))
    bars = plt.bar([f"Client {c}" for c in clients], accs, color=['#1f77b4', '#ff7f0e'])
    plt.xlabel('Training Client')
    plt.ylabel('Accuracy')
    plt.title('Source Domain Classification Accuracy on Local Test Sets')
    plt.ylim(0, 1.05)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.4f}', ha='center', va='bottom')
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Source classification accuracy bar chart saved to {save_path}")

def plot_umap_features(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loaders: Dict[int, DataLoader],
    device: str,
    save_dir: str,
    filename: str = "umap_features.png",
    max_samples_per_set: int = 1000,
    random_state: int = 42
) -> None:
    """
    使用 UMAP 可视化模型提取的特征分布

    Args:
        model: 训练好的模型
        train_loader: 训练集数据加载器（用于展示训练分布）
        test_loaders: 测试客户端数据加载器字典 {client_id: loader}
        device: 计算设备
        save_dir: 保存目录
        filename: 保存文件名
        max_samples_per_set: 每个数据集最大采样数量（避免 UMAP 过慢）
        random_state: 随机种子（保证可复现）
    """
    import umap.umap_ as umap
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 收集训练集特征和标签
    train_features, train_labels = [], []
    with torch.no_grad():
        for batch in train_loader:
            if len(batch) == 4:
                x, y_cls, _, _ = batch
            else:
                continue
            x = x.to(device)
            _, feats, _ = model(x)
            train_features.append(feats.cpu().numpy())
            train_labels.append(y_cls.numpy())

    if train_features:
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        if len(train_features) > max_samples_per_set:
            idx = np.random.choice(len(train_features), max_samples_per_set, replace=False)
            train_features = train_features[idx]
            train_labels = train_labels[idx]
    else:
        train_features, train_labels = np.array([]), np.array([])

    # 收集各测试客户端特征和标签
    test_data = {}
    for cid, loader in test_loaders.items():
        feats_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    x, y_cls, _, _ = batch
                else:
                    continue
                x = x.to(device)
                _, feats, _ = model(x)
                feats_list.append(feats.cpu().numpy())
                labels_list.append(y_cls.numpy())
        if feats_list:
            feats = np.concatenate(feats_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            if len(feats) > max_samples_per_set:
                idx = np.random.choice(len(feats), max_samples_per_set, replace=False)
                feats = feats[idx]
                labels = labels[idx]
            test_data[cid] = (feats, labels)

    # 合并所有特征用于 UMAP 拟合
    all_features = [train_features] if len(train_features) > 0 else []
    for feats, _ in test_data.values():
        all_features.append(feats)
    if not all_features:
        print("No features to plot for UMAP.")
        return
    all_features = np.concatenate(all_features, axis=0)

    # UMAP 降维
    print(f"Running UMAP on {len(all_features)} samples...")
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    all_2d = reducer.fit_transform(all_features)

    # 分割回各数据集
    idx = 0
    train_2d = None
    if len(train_features) > 0:
        train_2d = all_2d[idx:idx+len(train_features)]
        idx += len(train_features)
    test_2d_dict = {}
    for cid, (feats, _) in test_data.items():
        test_2d_dict[cid] = all_2d[idx:idx+len(feats)]
        idx += len(feats)

    # 绘图：每个子图对应一个数据集
    n_plots = (1 if len(train_features) > 0 else 0) + len(test_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    if train_2d is not None:
        ax = axes[plot_idx]
        for cls in np.unique(train_labels):
            mask = train_labels == cls
            ax.scatter(train_2d[mask, 0], train_2d[mask, 1],
                       c=colors[int(cls)], label=gas_names[int(cls)],
                       alpha=0.5, s=5)
        ax.set_title(f"Training Set (n={len(train_features)})")
        ax.legend(markerscale=3, fontsize='small')
        plot_idx += 1

    for cid, (feats, labels) in test_data.items():
        ax = axes[plot_idx]
        feat_2d = test_2d_dict[cid]
        for cls in np.unique(labels):
            mask = labels == cls
            ax.scatter(feat_2d[mask, 0], feat_2d[mask, 1],
                       c=colors[int(cls)], label=gas_names[int(cls)],
                       alpha=0.5, s=5)
        ax.set_title(f"Test Client {cid} (n={len(feats)})")
        ax.legend(markerscale=3, fontsize='small')
        plot_idx += 1

    plt.tight_layout()
    plt.savefig(save_path / filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"UMAP plot saved to {save_path / filename}")

# ==================== 辅助工具 ====================

def set_global_seed(seed):
    """设置全局随机种子，确保实验可重复性
    
    Args:
        seed: 随机种子值
    """
    import random
    import numpy as np
    import torch
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_model_by_config(config, with_reg_head=False):
    """根据配置创建模型实例
    
    Args:
        config: 实验配置
        with_reg_head: 是否包含回归头
    
    Returns:
        创建的模型实例
    """
    from model import FedGasModel, FedGasMultiTaskModel, FedGasProtoRegMultiTaskModel
    model_kwargs = {
        'num_classes': config.NUM_CLASSES,
        'num_sensors': config.INPUT_DIM,
        'feat_dim': config.HIDDEN_DIM2,
        'use_mixstyle': getattr(config, 'USE_MIXSTYLE', False),
        'mixstyle_prob': getattr(config, 'MIXSTYLE_PROB', 0.5),
        'mixstyle_alpha': getattr(config, 'MIXSTYLE_ALPHA', 0.5),
        'noise_std': getattr(config, 'FEATURE_NOISE_STD', 0.01),
        'encoder_type': getattr(config, 'USE_TRANSFORMER_ENCODER', False) and 'transformer' or 'tcn',
        'transformer_d_model': getattr(config, 'TRANSFORMER_D_MODEL', 48),
        'transformer_nhead': getattr(config, 'TRANSFORMER_NHEAD', 4),
        'transformer_num_layers': getattr(config, 'TRANSFORMER_NUM_LAYERS', 2),
        'transformer_ff_dim': getattr(config, 'TRANSFORMER_FF_DIM', 96),
        'tcn_norm': getattr(config, 'TCN_NORM', 'instance'),
    }
    
    if with_reg_head:
        num_conc_buckets = getattr(config, 'NUM_CONC_BUCKETS', 0)
        use_dual_proj = getattr(config, 'USE_DUAL_PROJ', False)
        use_proto = getattr(config, 'USE_PROTO_REG', False)
        reg_grad_detach = getattr(config, 'REG_GRAD_DETACH', False)
        use_reg_window_stats = getattr(config, 'REG_WINDOW_STATS', False)
        reg_window_stats_mode = getattr(config, 'REG_WINDOW_STATS_MODE', 'global')
        reg_window_stats_dim = getattr(config, 'REG_WINDOW_STATS_DIM', 8)
        reg_output_mode = getattr(config, 'REG_OUTPUT_MODE', 'sigmoid')
        reg_response_branch = getattr(config, 'REG_RESPONSE_BRANCH', 'none')
        reg_dct_k = getattr(config, 'REG_DCT_K', 8)
        reg_dct_gamma_init = getattr(config, 'REG_DCT_GAMMA_INIT', 0.0)
        reg_dct_dropout = getattr(config, 'REG_DCT_DROPOUT', 0.1)
        reg_msconv_channels = getattr(config, 'REG_MSCONV_CHANNELS', 16)
        reg_msconv_kernels = getattr(config, 'REG_MSCONV_KERNELS', '3,7,15,31')
        reg_msconv_gamma_init = getattr(config, 'REG_MSCONV_GAMMA_INIT', 0.0)
        reg_msconv_dropout = getattr(config, 'REG_MSCONV_DROPOUT', 0.1)
        use_reg_tcn_adapter = getattr(config, 'REG_TCN_ADAPTER', False)
        reg_tcn_adapter_kernel = getattr(config, 'REG_TCN_ADAPTER_KERNEL', 3)
        reg_tcn_adapter_gamma_init = getattr(config, 'REG_TCN_ADAPTER_GAMMA_INIT', 0.0)
        reg_tcn_adapter_dropout = getattr(config, 'REG_TCN_ADAPTER_DROPOUT', 0.05)
        use_reg_shared_trunk = getattr(config, 'REG_USE_SHARED_TRUNK', False)
        reg_shared_trunk_dim = getattr(config, 'REG_SHARED_TRUNK_DIM', 128)
        reg_gas_emb_dim = getattr(config, 'REG_GAS_EMB_DIM', 16)
        reg_residual_head_depth = getattr(config, 'REG_RESIDUAL_HEAD_DEPTH', 2)
        use_reg_ratio_branch = getattr(config, 'USE_REG_RATIO_BRANCH', False)
        reg_ratio_gamma_init = getattr(config, 'REG_RATIO_GAMMA_INIT', 0.0)
        reg_ratio_dropout = getattr(config, 'REG_RATIO_DROPOUT', 0.05)
        if getattr(config, 'USE_PROTO_REG', False) and not hasattr(config, 'OVERRIDE_PROTO_REG'):
            model_kwargs.update({
                'reg_head_depth': getattr(config, 'REG_HEAD_DEPTH', 3),
                'use_quantile': getattr(config, 'USE_QUANTILE_LOSS', False),
                'num_phases': config.NUM_PHASES,
                'num_conc_buckets': num_conc_buckets,
                'use_dual_proj': use_dual_proj,
                'use_proto_reg': use_proto,
                'reg_grad_detach': reg_grad_detach,
                'use_reg_window_stats': use_reg_window_stats,
                'reg_window_stats_mode': reg_window_stats_mode,
                'reg_window_stats_dim': reg_window_stats_dim,
                'reg_output_mode': reg_output_mode,
                'reg_response_branch': reg_response_branch,
                'reg_dct_k': reg_dct_k,
                'reg_dct_gamma_init': reg_dct_gamma_init,
                'reg_dct_dropout': reg_dct_dropout,
                'reg_msconv_channels': reg_msconv_channels,
                'reg_msconv_kernels': reg_msconv_kernels,
                'reg_msconv_gamma_init': reg_msconv_gamma_init,
                'reg_msconv_dropout': reg_msconv_dropout,
                'use_reg_tcn_adapter': use_reg_tcn_adapter,
                'reg_tcn_adapter_kernel': reg_tcn_adapter_kernel,
                'reg_tcn_adapter_gamma_init': reg_tcn_adapter_gamma_init,
                'reg_tcn_adapter_dropout': reg_tcn_adapter_dropout,
                'use_reg_shared_trunk': use_reg_shared_trunk,
                'reg_shared_trunk_dim': reg_shared_trunk_dim,
                'reg_gas_emb_dim': reg_gas_emb_dim,
                'reg_residual_head_depth': reg_residual_head_depth,
                'use_reg_ratio_branch': use_reg_ratio_branch,
                'reg_ratio_gamma_init': reg_ratio_gamma_init,
                'reg_ratio_dropout': reg_ratio_dropout
            })
            return FedGasMultiTaskModel(**model_kwargs)
        elif getattr(config, 'USE_PROTO_REG', False):
            use_residual = getattr(config, 'USE_REG_RESIDUAL', False)
            return FedGasProtoRegMultiTaskModel(**model_kwargs, use_residual=use_residual, num_conc_buckets=num_conc_buckets)
        else:
            model_kwargs.update({
                'reg_head_depth': getattr(config, 'REG_HEAD_DEPTH', 3),
                'use_quantile': getattr(config, 'USE_QUANTILE_LOSS', False),
                'num_phases': config.NUM_PHASES,
                'num_conc_buckets': num_conc_buckets,
                'use_dual_proj': use_dual_proj,
                'use_proto_reg': False,
                'reg_grad_detach': reg_grad_detach,
                'use_reg_window_stats': use_reg_window_stats,
                'reg_window_stats_mode': reg_window_stats_mode,
                'reg_window_stats_dim': reg_window_stats_dim,
                'reg_output_mode': reg_output_mode,
                'reg_response_branch': reg_response_branch,
                'reg_dct_k': reg_dct_k,
                'reg_dct_gamma_init': reg_dct_gamma_init,
                'reg_dct_dropout': reg_dct_dropout,
                'reg_msconv_channels': reg_msconv_channels,
                'reg_msconv_kernels': reg_msconv_kernels,
                'reg_msconv_gamma_init': reg_msconv_gamma_init,
                'reg_msconv_dropout': reg_msconv_dropout,
                'use_reg_tcn_adapter': use_reg_tcn_adapter,
                'reg_tcn_adapter_kernel': reg_tcn_adapter_kernel,
                'reg_tcn_adapter_gamma_init': reg_tcn_adapter_gamma_init,
                'reg_tcn_adapter_dropout': reg_tcn_adapter_dropout,
                'use_reg_shared_trunk': use_reg_shared_trunk,
                'reg_shared_trunk_dim': reg_shared_trunk_dim,
                'reg_gas_emb_dim': reg_gas_emb_dim,
                'reg_residual_head_depth': reg_residual_head_depth,
                'use_reg_ratio_branch': use_reg_ratio_branch,
                'reg_ratio_gamma_init': reg_ratio_gamma_init,
                'reg_ratio_dropout': reg_ratio_dropout
            })
            return FedGasMultiTaskModel(**model_kwargs)
    else:
        if getattr(config, 'USE_DUAL_PROJ', False):
            model_kwargs = dict(model_kwargs)
            model_kwargs['use_cls_proj'] = True
        return FedGasModel(**model_kwargs)


def load_shared_weights(model, state_dict, strict=False):
    """加载共享参数，忽略不匹配的键
    
    Args:
        model: 模型实例
        state_dict: 状态字典
        strict: 是否严格匹配
    """
    model.load_state_dict(state_dict, strict=strict)


def apply_sensor_aug(x, config):
    """传感器数据增强
    
    Args:
        x: 输入张量 (batch, seq_len, channels)
        config: 配置对象，包含增强参数
    Returns:
        增强后的张量
    """
    import torch.nn.functional as F
    orig_T = x.size(1)
    gain = torch.randn(1, 1, 1, device=x.device) * config.SENSOR_AUG_GAIN_STD + 1.0
    bias = torch.randn(1, 1, 1, device=x.device) * config.SENSOR_AUG_BIAS_STD
    x = x * gain + bias
    ch_gain = torch.randn(1, 1, x.size(2), device=x.device) * config.SENSOR_AUG_CH_GAIN_STD + 1.0
    x = x * ch_gain
    if torch.rand(1).item() < config.SENSOR_AUG_TIME_PROB:
        scale = torch.rand(1).item() * config.SENSOR_AUG_TIME_SCALE_RANGE * 2 + (1 - config.SENSOR_AUG_TIME_SCALE_RANGE)
        new_T = int(orig_T * scale)
        x_temp = x.permute(0, 2, 1)
        x_temp = F.interpolate(x_temp, size=new_T, mode='linear', align_corners=False)
        x = x_temp.permute(0, 2, 1)
        if new_T > orig_T:
            x = x[:, :orig_T, :]
        elif new_T < orig_T:
            pad_size = orig_T - new_T
            x = F.pad(x, (0, 0, 0, pad_size), mode='reflect')
    return x

def few_shot_finetune_regression(model, loader, device, config, num_steps=60, lr=5e-4,
                                 finetune_feat_lr=5e-4, weight_decay=1e-3, force_aug=False,
                                 aligned_feats_loader=None):
    """
    分阶段微调回归头：先优化回归头和原型偏置，再解冻特征层微调。
    loader 提供少量有标签样本（每类1-10个即可）。
    
    Args:
        model: 模型实例
        loader: 数据加载器
        device: 设备
        config: 配置对象
        num_steps: 微调步数
        lr: 回归头学习率
        finetune_feat_lr: 特征提取器学习率，0表示不微调特征提取器
        weight_decay: 权重衰减，用于防止过拟合
        force_aug: 是否强制启用数据增强（即使 config.USE_SENSOR_AUG 为 False）
        aligned_feats_loader: 可选的已对齐回归特征加载器，当提供时跳过模型前向
    """
    model.eval()
    with torch.no_grad():
        cls_conc_sums = defaultdict(float)
        cls_conc_counts = defaultdict(int)
        phase_cls_conc_sums = defaultdict(lambda: defaultdict(float))
        phase_cls_conc_counts = defaultdict(lambda: defaultdict(int))
        for x, y_cls, y_reg_full, phase in loader:
            y_cls_np = y_cls.cpu().numpy()
            y_reg_np = y_reg_full.cpu().numpy()
            phase_np = phase.cpu().numpy()
            for idx in range(len(y_cls_np)):
                c = y_cls_np[idx]
                p = phase_np[idx]
                conc = y_reg_np[idx, int(c)]
                cls_conc_sums[c] += conc
                cls_conc_counts[c] += 1
                if p >= 0:
                    phase_cls_conc_sums[c][p] += conc
                    phase_cls_conc_counts[c][p] += 1
    if hasattr(model, 'proto_conc'):
        model.proto_conc.requires_grad = True   # 允许微调中更新原型浓度偏置

    was_training = model.training
    original_reg_grad_detach = getattr(model, 'reg_grad_detach', None)
    if finetune_feat_lr > 0 and original_reg_grad_detach is not None:
        model.reg_grad_detach = False
    model.train()

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻回归相关参数（使用统一接口）
    if not model.has_regression:
        raise RuntimeError("No regression head found in the model.")

    reg_params = model.get_regression_params()
    for p in reg_params:
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = True
        elif hasattr(p, 'requires_grad'):
            p.requires_grad = True

    optimizer = torch.optim.Adam(reg_params, lr=lr, weight_decay=weight_decay)

    # 对齐特征迭代器
    aligned_iter = None
    if aligned_feats_loader is not None:
        aligned_iter = iter(aligned_feats_loader)

    # 第一阶段：只优化回归头（前 10 步）
    for step in range(10):
        if aligned_feats_loader is not None:
            try:
                reg_feat_aligned, y_cls_b, y_reg_full_b, y_p_b = next(aligned_iter)
            except StopIteration:
                aligned_iter = iter(aligned_feats_loader)
                reg_feat_aligned, y_cls_b, y_reg_full_b, y_p_b = next(aligned_iter)
            reg_feat_aligned = reg_feat_aligned.to(device)
            y_cls_b = y_cls_b.to(device)
            y_p_b = y_p_b.to(device)
            y_reg_b = y_reg_full_b[torch.arange(y_cls_b.size(0)), y_cls_b].unsqueeze(1).to(device)
            pred_norm = model.forward_reg(reg_feat_aligned, y_cls_b, y_phase=y_p_b)
            if hasattr(model, 'use_quantile') and model.use_quantile and pred_norm.shape[1] == 3:
                pred_norm = pred_norm[:, 1:2]
            y_reg_norm = normalize_concentration(y_reg_b, y_cls_b)
            loss = F.smooth_l1_loss(pred_norm, y_reg_norm, beta=getattr(config, 'HUBER_DELTA', 0.2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            for x, y_cls, y_reg_full, y_p in loader:
                if force_aug or config.USE_SENSOR_AUG:
                    x = apply_sensor_aug(x, config)
                x, y_cls, y_p = x.to(device), y_cls.to(device), y_p.to(device)
                y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1).to(device)
                logits, cls_feat, reg_feat = model(x)
                pred_norm = model.forward_reg(reg_feat, y_cls, y_phase=y_p)
                if hasattr(model, 'use_quantile') and model.use_quantile and pred_norm.shape[1] == 3:
                    pred_norm = pred_norm[:, 1:2]
                y_reg_norm = normalize_concentration(y_reg, y_cls)
                loss = F.smooth_l1_loss(pred_norm, y_reg_norm, beta=getattr(config, 'HUBER_DELTA', 0.2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # 第二阶段（如果有特征层微调且没有使用对齐特征）
    if finetune_feat_lr > 0 and aligned_feats_loader is None:
        for param in model.tcn[-1].parameters():
            param.requires_grad = True
        for param in model.self_attn.parameters():
            param.requires_grad = True
        for param in model.attn_linear.parameters():
            param.requires_grad = True
        for param in model.feat_proj.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam([
            {'params': reg_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': list(model.tcn[-1].parameters()) + list(model.self_attn.parameters()) + list(model.attn_linear.parameters()) + list(model.feat_proj.parameters()), 'lr': finetune_feat_lr, 'weight_decay': weight_decay}
        ])
        warmup_steps = max(1, (num_steps - 10) // 10)
        for step in range(10, num_steps):
            if step - 10 < warmup_steps:
                current_lr = lr * (step - 10 + 1) / warmup_steps
                current_feat_lr = finetune_feat_lr * (step - 10 + 1) / warmup_steps
            else:
                current_lr = lr
                current_feat_lr = finetune_feat_lr
            for param_group in optimizer.param_groups:
                if param_group['lr'] == lr or param_group['lr'] == current_lr:
                    param_group['lr'] = current_lr
                else:
                    param_group['lr'] = current_feat_lr

            for x, y_cls, y_reg_full, y_p in loader:
                if force_aug or config.USE_SENSOR_AUG:
                    x = apply_sensor_aug(x, config)
                x, y_cls, y_p = x.to(device), y_cls.to(device), y_p.to(device)
                y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1).to(device)
                logits, cls_feat, reg_feat = model(x)
                pred_norm = model.forward_reg(reg_feat, y_cls, y_phase=y_p)
                if hasattr(model, 'use_quantile') and model.use_quantile and pred_norm.shape[1] == 3:
                    pred_norm = pred_norm[:, 1:2]
                y_reg_norm = normalize_concentration(y_reg, y_cls)
                loss = F.smooth_l1_loss(pred_norm, y_reg_norm, beta=getattr(config, 'HUBER_DELTA', 0.2))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    elif aligned_feats_loader is not None:
        # 对齐特征模式下继续微调回归头
        for step in range(10, num_steps):
            try:
                reg_feat_aligned, y_cls_b, y_reg_full_b, y_p_b = next(aligned_iter)
            except StopIteration:
                aligned_iter = iter(aligned_feats_loader)
                reg_feat_aligned, y_cls_b, y_reg_full_b, y_p_b = next(aligned_iter)
            reg_feat_aligned = reg_feat_aligned.to(device)
            y_cls_b = y_cls_b.to(device)
            y_p_b = y_p_b.to(device)
            y_reg_b = y_reg_full_b[torch.arange(y_cls_b.size(0)), y_cls_b].unsqueeze(1).to(device)
            pred_norm = model.forward_reg(reg_feat_aligned, y_cls_b, y_phase=y_p_b)
            if hasattr(model, 'use_quantile') and model.use_quantile and pred_norm.shape[1] == 3:
                pred_norm = pred_norm[:, 1:2]
            y_reg_norm = normalize_concentration(y_reg_b, y_cls_b)
            loss = F.smooth_l1_loss(pred_norm, y_reg_norm, beta=getattr(config, 'HUBER_DELTA', 0.2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for param in model.parameters():
        param.requires_grad = True
    if original_reg_grad_detach is not None:
        model.reg_grad_detach = original_reg_grad_detach
    model.train(was_training)


def few_shot_finetune_classification(model, loader, device, epochs=5, lr=1e-3, finetune_feat_lr=0.0,
                                     class_weights=None, focal_gamma=0.0,
                                     aug_config=None, aug_prob=0.0,
                                     cost_matrix=None, cost_weight=0.0):
    """
    冻结模型特征提取器，仅微调分类头。
    loader 提供少量有标签样本（每类1-5个即可）。
    
    Args:
        model: 模型实例
        loader: 数据加载器
        device: 设备
        epochs: 微调轮数
        lr: 分类头学习率
        finetune_feat_lr: 特征提取器学习率，0表示不微调特征提取器
    """
    # 保存原始训练状态
    was_training = model.training
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 解冻分类头
    for param in model.classifier.parameters():
        param.requires_grad = True
    # 可选解冻特征提取器最后部分
    feature_modules = []
    if finetune_feat_lr > 0:
        if hasattr(model, 'tcn') and model.tcn is not None:
            feature_modules.append(model.tcn[-1])
        if hasattr(model, 'self_attn') and model.self_attn is not None:
            feature_modules.append(model.self_attn)
        if hasattr(model, 'attn_linear') and model.attn_linear is not None:
            feature_modules.append(model.attn_linear)
        if hasattr(model, 'cls_proj') and model.cls_proj is not None:
            feature_modules.append(model.cls_proj)
        elif hasattr(model, 'feat_proj') and model.feat_proj is not None:
            feature_modules.append(model.feat_proj)
        for module in feature_modules:
            for param in module.parameters():
                param.requires_grad = True
    
    # 构建参数组
    param_groups = [{'params': model.classifier.parameters(), 'lr': lr}]
    if finetune_feat_lr > 0:
        feat_params = [p for module in feature_modules for p in module.parameters()]
        if feat_params:
            param_groups.append({'params': feat_params, 'lr': finetune_feat_lr})
    optimizer = torch.optim.Adam(param_groups)
    if class_weights is not None:
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    focal_gamma = float(focal_gamma or 0.0)
    aug_prob = float(aug_prob or 0.0)
    cost_weight = float(cost_weight or 0.0)
    cost_matrix_tensor = None
    if cost_matrix is not None and cost_weight > 0:
        cost_matrix_tensor = torch.as_tensor(cost_matrix, dtype=torch.float32, device=device)
        if cost_matrix_tensor.dim() != 2 or cost_matrix_tensor.size(0) != cost_matrix_tensor.size(1):
            raise ValueError('cost_matrix must be a square 2D matrix')
    
    model.train()
    for _ in range(epochs):
        for x, y_cls, _, _ in loader:
            x, y_cls = x.to(device), y_cls.to(device)
            if aug_config is not None and aug_prob > 0 and torch.rand(1).item() < aug_prob:
                x = apply_sensor_aug(x, aug_config)
            logits, _, _ = model(x)
            ce = F.cross_entropy(logits, y_cls, weight=class_weights, reduction='none')
            if focal_gamma > 0:
                probs = torch.softmax(logits, dim=1)
                pt = probs.gather(1, y_cls.view(-1, 1)).squeeze(1).clamp_min(1e-6)
                loss = (((1.0 - pt) ** focal_gamma) * ce).mean()
            else:
                loss = ce.mean()
            if cost_matrix_tensor is not None and cost_weight > 0:
                if cost_matrix_tensor.size(0) != logits.size(1):
                    raise ValueError(
                        f'cost_matrix size {cost_matrix_tensor.size(0)} does not match logits size {logits.size(1)}'
                    )
                probs = torch.softmax(logits, dim=1)
                route_cost = (probs * cost_matrix_tensor[y_cls]).sum(dim=1).mean()
                loss = loss + cost_weight * route_cost
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 恢复 requires_grad
    for param in model.parameters():
        param.requires_grad = True
    # 恢复原始训练状态
    model.train(was_training)


def get_device_residual(args, server, cid, test_client_loaders, semantic_protos, device):
    """根据参数决定是否估计设备残差并返回残差张量
    
    Args:
        args: 命令行参数
        server: 服务器实例
        cid: 客户端ID
        test_client_loaders: 测试客户端加载器
        semantic_protos: 语义原型
        device: 设备
    
    Returns:
        设备残差张量
    """
    if not args.few_shot_residual:
        return server.compute_generic_residual() if server.config.USE_SOFT_AGGREGATION else None
    try:
        from utils import estimate_device_residual
        loader = test_client_loaders[cid]
        residual = estimate_device_residual(
            server.global_model, loader, device,
            semantic_protos=semantic_protos
        )
        return residual
    except Exception as e:
        logger = logging.getLogger('gasfl')
        logger.warning(f"Failed to estimate device residual: {e}")
        return server.compute_generic_residual() if server.config.USE_SOFT_AGGREGATION else None


def prepare_proto_matrix(semantic_protos, device, residual=None):
    """
    将语义原型字典转换为矩阵和类别标签列表，可选添加设备残差
    
    Args:
        semantic_protos: 语义原型字典
        device: 设备
        residual: 设备残差
    
    Returns:
        (proto_matrix, proto_classes)
    """
    proto_list = []
    proto_classes = []
    for str_key, mu_sem in semantic_protos.items():
        mu = mu_sem.clone().to(device)
        if residual is not None:
            mu = mu + residual.to(device)
        proto_list.append(mu)
        c = int(str_key.strip('()').split(',')[0])
        proto_classes.append(c)
    if not proto_list:
        return None, None
    proto_matrix = torch.stack(proto_list)
    proto_classes = torch.tensor(proto_classes, device=device)
    return proto_matrix, proto_classes


def print_experiment_config(config: Any) -> None:
    """打印实验配置
    
    打印dataclass或字典格式的实验配置
    
    Args:
        config: 实验配置
    """
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    if hasattr(config, '__dict__'):
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")
    else:
        for key, value in config.items():
            print(f"{key}: {value}")
    print("=" * 60)


def print_training_summary(summary: Dict) -> None:
    """打印训练摘要
    
    打印训练结果摘要
    
    Args:
        summary: 训练摘要字典
    """
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 60)


def format_time(seconds: float) -> str:
    """格式化时间
    
    将秒数转换为HH:MM:SS格式
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_regression_loss_combined(
    reg_feat: torch.Tensor,
    y_cls: torch.Tensor,
    y_reg: torch.Tensor,
    model,
    config,
    semantic_protos=None,
    y_phase=None,
    current_round=None,
):
    """回归损失（重构版）: 纯 Huber 损失 + 可选轻量对数相对误差

    重构原则:
    - 单核优先: Huber 损失作为唯一主力，直接作用于归一化浓度 [0,1]
    - 空间统一: pred 和 target 均在线性归一化空间
    - 移除冗余: 排序损失/分位数/浓度桶/类别自适应权重/低浓度加权全部删除
    - 阶段信息完整: y_phase 传入 forward_reg 以学习原型浓度先验

    Args:
        reg_feat: 回归分支特征 (B, D)
        y_cls: 类别标签 (B,)
        y_reg: 浓度标签 (B, 1)
        model: 模型 (需有 forward_reg 方法)
        config: 配置对象
        semantic_protos: 未使用 (保留接口兼容)
        y_phase: 阶段标签 (B,) 【关键：用于原型浓度先验】
        current_round: 未使用 (保留接口兼容)
    """
    if not config.USE_REG_LOSS:
        return torch.tensor(0.0, device=reg_feat.device)

    eps = 1e-6
    y_reg_norm = normalize_concentration(y_reg, y_cls)

    # 模型预测 (线性空间, [0,1])
    pred_norm = model.forward_reg(reg_feat, y_cls, y_reg, y_phase=y_phase)

    # 核心: Huber 损失
    if getattr(config, 'USE_HUBER_LOSS', True):
        huber_delta = getattr(config, 'HUBER_DELTA', 0.1)
        per_sample_loss = F.smooth_l1_loss(pred_norm, y_reg_norm, beta=huber_delta, reduction='none').view(-1)
    else:
        per_sample_loss = F.mse_loss(pred_norm, y_reg_norm, reduction='none').view(-1)

    tail_weight = float(getattr(config, 'REG_TAIL_WEIGHT', 1.0))
    tail_threshold = float(getattr(config, 'REG_TAIL_THRESHOLD', 1.0))
    if tail_weight > 1.0 and tail_threshold < 1.0:
        sample_weights = torch.ones_like(per_sample_loss)
        sample_weights = torch.where(
            y_reg_norm.view(-1) >= tail_threshold,
            sample_weights.new_tensor(tail_weight),
            sample_weights,
        )
        reg_loss = (per_sample_loss * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)
    else:
        reg_loss = per_sample_loss.mean()

    # 可选: 轻量对数相对误差
    if getattr(config, 'USE_LOG_REL_LOSS', False):
        lambda_log_rel = getattr(config, 'LAMBDA_LOG_REL', 0.05)
        log_rel = torch.abs(
            torch.log(pred_norm + eps) - torch.log(y_reg_norm + eps)
        )
        log_rel = torch.clamp(log_rel, max=2.0)
        reg_loss = reg_loss + lambda_log_rel * log_rel.mean()

    return reg_loss


def ranking_loss(feats, y_cls, y_reg, semantic_protos, margin_adaptive=True,
                 margin_quantile=0.5, min_margin=0.01, class_weights=None):
    loss = 0.0
    count = 0
    for c in range(4):
        mask = (y_cls == c)
        if mask.sum() < 2:
            continue
        feats_c = feats[mask]
        conc_c = y_reg[mask].squeeze()
        
        # 多原型融合
        proto_list = [semantic_protos[f'({c},{p})'] for p in range(3) if f'({c},{p})' in semantic_protos]
        if not proto_list:
            continue
        proto = torch.stack(proto_list).mean(dim=0).to(feats_c.device)
        
        # 相似度计算
        sim = F.cosine_similarity(F.normalize(feats_c, dim=1),
                                  F.normalize(proto.unsqueeze(0).expand(len(feats_c), -1), dim=1))
        
        # 生成所有组合
        idx = torch.combinations(torch.arange(len(feats_c)), r=2)
        diff_conc = conc_c[idx[:, 0]] - conc_c[idx[:, 1]]
        diff_sim = sim[idx[:, 0]] - sim[idx[:, 1]]
        
        # 自适应 margin
        if margin_adaptive:
            abs_diff = torch.abs(diff_conc)
            margin = torch.quantile(abs_diff, margin_quantile) if abs_diff.numel() > 0 else min_margin
            margin = max(margin, min_margin)
        else:
            margin = 0.1
        
        valid = torch.abs(diff_conc) > margin
        if valid.sum() == 0:
            continue
        class_weight = class_weights.get(c, 1.0) if class_weights is not None else 1.0
        loss += class_weight * F.relu(margin - torch.sign(diff_conc) * diff_sim)[valid].mean()
        count += 1
    
    return loss / count if count > 0 else torch.tensor(0.0, device=feats.device)

def regression_contrastive_loss(feats, y_cls, y_reg, temperature=0.1,
                                 pos_margin=0.15, num_negatives=16):
    """
    基于浓度有序性的回归对比损失（Rank-N-Contrast 思想）。
    正样本：同类别中浓度差在 pos_margin 以内的样本对
    负样本：同类别中浓度差最大的样本
    """
    B = feats.size(0)
    device = feats.device
    loss = torch.tensor(0.0, device=device)
    count = 0

    for c in range(4):
        mask = (y_cls == c)
        if mask.sum() < 4:
            continue
        feats_c = feats[mask]
        conc_c = y_reg[mask].squeeze()

        feats_norm = F.normalize(feats_c, dim=1)
        sim_matrix = feats_norm @ feats_norm.T   # (N_c, N_c)

        conc_diff = torch.abs(conc_c.unsqueeze(1) - conc_c.unsqueeze(0))

        # 正样本：不同样本，且浓度差 < margin
        pos_mask = (conc_diff < pos_margin) & ~torch.eye(len(feats_c), dtype=torch.bool, device=device)

        # 负样本：每个样本取浓度差最大的 num_negatives 个
        neg_mask = torch.zeros_like(pos_mask)
        for i in range(len(feats_c)):
            _, neg_idx = torch.topk(conc_diff[i], k=min(num_negatives, len(feats_c) - 1))
            neg_mask[i, neg_idx] = True

        pos_sim = sim_matrix[pos_mask].exp().sum() if pos_mask.any() else torch.tensor(0.0, device=device)
        neg_sim = sim_matrix[neg_mask].exp().sum() if neg_mask.any() else torch.tensor(0.0, device=device)

        if pos_sim > 0:
            loss += -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=device)

def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """创建实验目录
    
    创建实验目录及其子目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        
    Returns:
        实验目录的Path对象
    """
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)
    return exp_dir


def compute_class_separability(feats, labels):
    """计算特征空间的类别可分离性指标
    
    Args:
        feats: 特征矩阵，形状为 (N, D)
        labels: 标签向量，形状为 (N,)
    
    Returns:
        dict: 包含类内散布 Sw、类间散布 Sb 和可分离性 J = Sb / Sw 的字典
    """
    import numpy as np
    
    # 获取唯一类别
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    num_samples, feat_dim = feats.shape
    
    # 计算全局均值
    global_mean = np.mean(feats, axis=0)
    
    # 计算类内散布 Sw
    Sw = np.zeros((feat_dim, feat_dim))
    for c in unique_classes:
        class_feats = feats[labels == c]
        class_mean = np.mean(class_feats, axis=0)
        class_scatter = np.dot((class_feats - class_mean).T, (class_feats - class_mean))
        Sw += class_scatter
    
    # 计算类间散布 Sb
    Sb = np.zeros((feat_dim, feat_dim))
    for c in unique_classes:
        class_feats = feats[labels == c]
        class_mean = np.mean(class_feats, axis=0)
        n_c = len(class_feats)
        mean_diff = (class_mean - global_mean).reshape(-1, 1)
        Sb += n_c * np.dot(mean_diff, mean_diff.T)
    
    # 计算可分离性指标 J
    # 为了避免除以零，添加一个小的epsilon
    epsilon = 1e-10
    Sw_inv = np.linalg.inv(Sw + epsilon * np.eye(feat_dim))
    J = np.trace(np.dot(Sw_inv, Sb))
    
    return {
        'Sw': Sw,
        'Sb': Sb,
        'J': J,
        'num_classes': num_classes,
        'num_samples': num_samples
    }


def plot_proto_similarity_matrix(proto_dict, save_dir, filename='proto_similarity_matrix.png'):
    """绘制原型相似度矩阵热力图
    
    Args:
        proto_dict: 原型字典，键为类别，值为原型向量
        save_dir: 保存目录
        filename: 保存文件名
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    import os
    
    # 提取原型向量
    classes = sorted(proto_dict.keys())
    protos = [proto_dict[c].detach().cpu().numpy() for c in classes]
    protos = np.array(protos)
    
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(protos)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(im)
    
    # 设置标签
    class_names = {0: 'Ethanol', 1: 'CO', 2: 'Ethylene', 3: 'Methane'}
    labels = [class_names.get(c, f'Class {c}') for c in classes]
    plt.xticks(range(len(classes)), labels, rotation=45)
    plt.yticks(range(len(classes)), labels)
    
    # 添加标题
    plt.title('Prototype Similarity Matrix')
    
    # 保存图
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# ==================== 通用可视化函数（跨实验复用） ====================

def plot_coral_tsne_comparison(hard_client_ids, plots_dir, use_coral=True, logger=None):
    """绘制 CORAL 前后的 t-SNE 对比图（从 tsne_features 目录读取已保存特征）
    Args:
        hard_client_ids: 困难客户端 ID 列表
        plots_dir: 图片保存根目录，其下应有 tsne_features 子目录
        use_coral: 仅当 True 且文件存在时才绘图
    """
    import os, numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    if not hard_client_ids or not use_coral:
        return

    tsne_dir = os.path.join(plots_dir, "tsne_features")
    if not os.path.exists(tsne_dir):
        return

    gas_names = {0: 'Ethanol', 1: 'CO', 2: 'Ethylene', 3: 'Methane'}
    for cid in hard_client_ids:
        coral_file = os.path.join(tsne_dir, f"tsne_c{cid}_coral_before_after.npz")
        if not os.path.exists(coral_file):
            if logger:
                logger.info(f"CORAL feature file not found for client {cid}")
            continue
        try:
            data = np.load(coral_file)
            raw_feats = data['raw']
            coral_feats = data['coral']
            labels = data['labels']

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            raw_2d = tsne.fit_transform(raw_feats)
            coral_2d = tsne.fit_transform(coral_feats)

            plt.figure(figsize=(14, 6))
            plt.subplot(1, 2, 1)
            for cls in np.unique(labels):
                mask = labels == cls
                plt.scatter(raw_2d[mask, 0], raw_2d[mask, 1],
                            label=gas_names.get(int(cls), f'Class {cls}'), alpha=0.6, s=5)
            plt.title(f'Client {cid}: Before CORAL')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(markerscale=3)

            plt.subplot(1, 2, 2)
            for cls in np.unique(labels):
                mask = labels == cls
                plt.scatter(coral_2d[mask, 0], coral_2d[mask, 1],
                            label=gas_names.get(int(cls), f'Class {cls}'), alpha=0.6, s=5)
            plt.title(f'Client {cid}: After CORAL')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(markerscale=3)

            plt.tight_layout()
            save_path = os.path.join(plots_dir, f'tsne_c{cid}_coral_before_after.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            if logger:
                logger.info(f"CORAL t-SNE comparison plot saved to {save_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to plot CORAL t-SNE for client {cid}: {e}")


def plot_finetune_tsne_comparison(hard_client_ids, plots_dir, logger=None):
    """绘制微调前后的 t-SNE 对比图（需先执行分类微调并保存特征文件）
    Args:
        hard_client_ids: 困难客户端 ID 列表
        plots_dir: 图片保存根目录，其下应有 tsne_features 子目录
    """
    import os, numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    if not hard_client_ids:
        return

    tsne_dir = os.path.join(plots_dir, "tsne_features")
    if not os.path.exists(tsne_dir):
        return

    for cid in hard_client_ids:
        before_file = os.path.join(tsne_dir, f"tsne_c{cid}_before_finetune.npz")
        after_file = os.path.join(tsne_dir, f"tsne_c{cid}_after_finetune.npz")
        if not os.path.exists(before_file) or not os.path.exists(after_file):
            continue
        try:
            before_data = np.load(before_file)
            after_data = np.load(after_file)
            before_feats = before_data['feats']
            before_labels = before_data['labels']
            after_feats = after_data['feats']
            after_labels = after_data['labels']

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            before_2d = tsne.fit_transform(before_feats)
            after_2d = tsne.fit_transform(after_feats)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(before_2d[:, 0], before_2d[:, 1], c=before_labels, cmap='viridis', s=5)
            plt.title(f'Client {cid}: Before Fine-tuning')
            plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.scatter(after_2d[:, 0], after_2d[:, 1], c=after_labels, cmap='viridis', s=5)
            plt.title(f'Client {cid}: After Fine-tuning')
            plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
            plt.colorbar()

            plt.tight_layout()
            save_path = os.path.join(plots_dir, f'tsne_c{cid}_before_after_finetune.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            if logger:
                logger.info(f"Fine-tune t-SNE comparison plot saved to {save_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to plot fine-tune t-SNE for client {cid}: {e}")


def plot_classifier_weight_analysis(hard_client_ids, plots_dir, logger=None):
    """绘制分类头权重变化热力图与相似度矩阵（需先执行分类微调并保存权重文件）
    Args:
        hard_client_ids: 困难客户端 ID 列表
        plots_dir: 图片保存根目录，其下应有 classifier_weights 子目录
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity

    weights_dir = os.path.join(plots_dir, "classifier_weights")
    if not os.path.exists(weights_dir) or not hard_client_ids:
        return

    for cid in hard_client_ids:
        before_file = os.path.join(weights_dir, f"classifier_weight_before_finetune_c{cid}.npz")
        after_file = os.path.join(weights_dir, f"classifier_weight_after_finetune_c{cid}.npz")
        if not os.path.exists(before_file) or not os.path.exists(after_file):
            continue
        try:
            before_weight = np.load(before_file)['weight']
            after_weight = np.load(after_file)['weight']

            sim_matrix = cosine_similarity(before_weight, after_weight)
            plt.figure(figsize=(8, 6))
            plt.imshow(sim_matrix, cmap='viridis')
            plt.colorbar()
            plt.title(f'Classifier Weight Similarity (Before vs After) - Client {cid}')
            plt.xlabel('Class (After)'); plt.ylabel('Class (Before)')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'classifier_weight_similarity_c{cid}.png'), dpi=150)
            plt.close()

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(before_weight, cmap='viridis')
            axes[0].set_title(f'Before Fine-tuning - Client {cid}')
            axes[0].set_xlabel('Feature'); axes[0].set_ylabel('Class')
            axes[1].imshow(after_weight, cmap='viridis')
            axes[1].set_title(f'After Fine-tuning - Client {cid}')
            axes[1].set_xlabel('Feature'); axes[1].set_ylabel('Class')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'classifier_weights_heatmap_c{cid}.png'), dpi=150)
            plt.close()

            if logger:
                logger.info(f"Classifier weight analysis plots saved for client {cid}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to plot classifier weight for client {cid}: {e}")


def plot_class_separability_analysis(hard_client_ids, plots_dir, logger=None):
    """从 tsne_features 读取特征，计算并绘制类别可分离性 J 指标（CORAL前后、微调前后）
    Args:
        hard_client_ids: 困难客户端 ID 列表
        plots_dir: 图片保存根目录，其下应有 tsne_features 子目录
    """
    import os, numpy as np, matplotlib.pyplot as plt

    tsne_dir = os.path.join(plots_dir, "tsne_features")
    if not os.path.exists(tsne_dir) or not hard_client_ids:
        return

    for cid in hard_client_ids:
        coral_file = os.path.join(tsne_dir, f"tsne_c{cid}_coral_before_after.npz")
        before_file = os.path.join(tsne_dir, f"tsne_c{cid}_before_finetune.npz")
        after_file = os.path.join(tsne_dir, f"tsne_c{cid}_after_finetune.npz")

        scores = {}
        if os.path.exists(coral_file):
            try:
                data = np.load(coral_file)
                raw_sep = compute_class_separability(data['raw'], data['labels'])
                coral_sep = compute_class_separability(data['coral'], data['labels'])
                scores['Raw'] = raw_sep['J']
                scores['CORAL'] = coral_sep['J']
            except:
                pass
        if os.path.exists(before_file) and os.path.exists(after_file):
            try:
                before_data = np.load(before_file)
                after_data = np.load(after_file)
                before_sep = compute_class_separability(before_data['feats'], before_data['labels'])
                after_sep = compute_class_separability(after_data['feats'], after_data['labels'])
                scores['Before FT'] = before_sep['J']
                scores['After FT'] = after_sep['J']
            except:
                pass

        if scores:
            plt.figure(figsize=(8, 5))
            labels = list(scores.keys())
            values = list(scores.values())
            bars = plt.bar(labels, values, color='skyblue', edgecolor='black')
            plt.xlabel('Stage')
            plt.ylabel('Separability J')
            plt.title(f'Feature Class Separability - Client {cid}')
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            save_path = os.path.join(plots_dir, f'class_separability_c{cid}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()
            if logger:
                logger.info(f"Class separability plot saved to {save_path}")


def plot_regression_visualizations(server, test_client_loaders, test_client_ids,
                                   config, best_rounds=None, best_test_accs=None,
                                   model_selection=False, logger=None):
    """为每个测试客户端生成回归散点图和浓度-特征相关性图
    Args:
        server: 服务器实例（包含 global_model）
        test_client_loaders: {cid: DataLoader}
        test_client_ids: 测试客户端 ID 列表
        config: 实验配置对象（需包含 DEVICE, USE_REG_LOSS, MODEL_SAVE_DIR 等）
        best_rounds: {cid: best_round} 用于加载最佳模型（可选）
        best_test_accs: {cid: best_acc}（可选）
        model_selection: 是否使用模型选择（默认 False）
    """
    if not config.USE_REG_LOSS or server is None or test_client_loaders is None:
        return
    from pathlib import Path
    import torch

    for cid in test_client_ids:
        loader = test_client_loaders.get(cid)
        if loader is None:
            continue
        eval_model = create_model_by_config(config, with_reg_head=True).to(config.DEVICE)
        if model_selection and best_rounds and best_test_accs and cid in best_rounds:
            best_model_path = Path(config.MODEL_SAVE_DIR) / f"best_model_client{cid}.pth"
            if best_model_path.exists():
                eval_model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
            else:
                load_shared_weights(eval_model, server.global_model.state_dict(), strict=False)
        else:
            load_shared_weights(eval_model, server.global_model.state_dict(), strict=False)

        plot_regression_scatter(eval_model, loader, config.DEVICE, config.PLOT_SAVE_DIR,
                                filename=f"reg_scatter_client{cid}.png", tolerance=0.1)
        plot_concentration_feature_correlation(eval_model, loader, config.DEVICE,
                                               config.PLOT_SAVE_DIR,
                                               filename=f"conc_feat_corr_client{cid}.png")
        if logger:
            logger.info(f"Regression plots generated for Client {cid}")


def save_accuracy_table(history, results_dir, logger=None):
    """将训练历史保存为 CSV 和 Excel 文件
    Args:
        history: 字典列表，每个字典包含轮次、准确率、损失等字段
        results_dir: 保存目录
    """
    import os, pandas as pd
    if not history:
        return
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(history)
    csv_path = os.path.join(results_dir, "accuracy_table.csv")
    excel_path = os.path.join(results_dir, "accuracy_table.xlsx")
    try:
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
        if logger:
            logger.info(f"Accuracy table saved to {csv_path} and {excel_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save accuracy table: {e}")


def post_train_regression_refinement(model, train_loader, device, config, num_steps=15, lr=1e-4):
    """Refine regression heads after source-domain training.

    Uses Huber loss in normalized concentration space, not log-space MSE.

    Args:
        model: Client model.
        train_loader: Local training dataloader.
        device: Device.
        config: Config object.
        num_steps: Fine-tuning steps.
        lr: Learning rate.
    """
    was_training = model.training
    model.train()
    
    for param in model.parameters():
        param.requires_grad = False

    # 只解冻回归相关参数（使用统一接口）
    if not model.has_regression:
        raise RuntimeError("No regression head found.")

    params_to_optimize = model.get_regression_params()
    for p in params_to_optimize:
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = True
        elif hasattr(p, 'requires_grad'):
            p.requires_grad = True
    if hasattr(model, 'proto_conc'):
        model.proto_conc.requires_grad_(False)
    
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    
    step = 0
    while step < num_steps:
        for x, y_cls, y_reg_full, _ in train_loader:
            if step >= num_steps:
                break
            x, y_cls = x.to(device), y_cls.to(device)
            y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1).to(device)
            
            logits, cls_feat, reg_feat = model(x)
            pred_norm = model.forward_reg(reg_feat, y_cls)
            
            if hasattr(model, 'use_quantile') and model.use_quantile and pred_norm.shape[1] == 3:
                pred_norm = pred_norm[:, 1:2]
            
            y_reg_norm = normalize_concentration(y_reg, y_cls)
            
            loss = F.smooth_l1_loss(pred_norm, y_reg_norm, beta=getattr(config, 'HUBER_DELTA', 0.2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
    
    model.train(was_training)
    return model

def calibrate_regression_bias(model, loader, device, config, num_steps=10, lr=1e-3):
    """对困难客户端学习专用回归偏置参数（proto_scale + proto_bias）
    
    完全冻结特征提取器和回归头权重，只优化 proto_scale 和 proto_bias。
    这两个参数作为"客户端专属校准器"，校正跨域偏移导致的系统误差。
    
    Args:
        model: 模型实例
        loader: 客户端数据加载器
        device: 设备
        config: 配置对象
        num_steps: 优化步数
        lr: 学习率
    """
    was_training = model.training
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'conc_scale'):
        model.conc_scale.requires_grad = True
        model.conc_bias.requires_grad = True
        params = [model.conc_scale, model.conc_bias]
    elif hasattr(model, 'proto_scale'):
        model.proto_scale.requires_grad = True
        model.proto_bias.requires_grad = True
        params = [model.proto_scale, model.proto_bias]
    else:
        model.train(was_training)
        return model

    if hasattr(model, 'proto_conc'):
        model.proto_conc.requires_grad = False

    optimizer = torch.optim.Adam(params, lr=lr)

    step = 0
    while step < num_steps:
        for x, y_cls, y_reg_full, y_p in loader:
            if step >= num_steps:
                break
            x, y_cls, y_p = x.to(device), y_cls.to(device), y_p.to(device)
            y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1).to(device)
            logits, cls_feat, reg_feat = model(x)
            pred_norm = model.forward_reg(reg_feat, y_cls, y_phase=y_p)
            if hasattr(model, 'use_quantile') and model.use_quantile and pred_norm.shape[1] == 3:
                pred_norm = pred_norm[:, 1:2]
            y_reg_norm = normalize_concentration(y_reg, y_cls)
            loss = F.smooth_l1_loss(pred_norm, y_reg_norm, beta=getattr(config, 'HUBER_DELTA', 0.2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

    model.train(was_training)
    return model
