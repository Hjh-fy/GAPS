import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import soft_aggregate_probs, apply_sensor_aug

class LocalAdaptationEngine:
    """
    客户端本地在线适应引擎
    核心功能：
    - TENT风格预测熵最小化（主要损失，无需原型/伪标签）
    - 软原型锚定正则（辅助损失，防止灾难性遗忘）
    - 教师-学生EMA自训练
    - 保守参数更新（冻结骨干，仅更新头部和少量投影层）
    - 可配置的参数解冻级别
    - 联邦校准支持（接收服务器下发的校正量）
    """
    def __init__(self, student_model, global_protos, config, unfreeze_level='basic', anchor_weight=None, proto_vars=None):
        self.student = student_model
        self.teacher = copy.deepcopy(student_model)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.protos = global_protos  # dict, e.g., {"(0,0)": tensor, ...}
        self.proto_vars = proto_vars  # dict[str_key → Tensor(D,)] 各原型的对角方差, None 时自动用余弦回退
        self.config = config
        self.ema_decay = config.MT_EMA_DECAY
        # TENT熵最小化权重（主要损失）
        self.tent_weight = getattr(config, 'TENT_WEIGHT', 1.0)
        # 软锚定正则权重（从1.0降至0.05，仅作为轻量正则）
        self.anchor_weight = anchor_weight if anchor_weight is not None else getattr(config, 'ANCHOR_REG_WEIGHT', 0.05)
        self.conf_thresh = config.MT_CONF_THRESH
        self.unfreeze_level = unfreeze_level
        self.pending_corrections = None  # 待应用的服务器校正量

        # 确保学生模型处于训练模式（从检查点加载后默认是eval模式）
        self.student.train()

        # 根据解冻级别配置参数
        self._configure_parameter_freeze()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=config.ADAPT_LR
        )
        self.fisher_mask = None  # 可选

    def _configure_parameter_freeze(self):
        """根据解冻级别配置哪些参数可训练"""
        for name, param in self.student.named_parameters():
            if self.unfreeze_level == 'aggressive':
                # 仅冻结第一层 TCN，其他全部可训练
                if 'tcn1' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif self.unfreeze_level == 'medium':
                # 冻结除 tcn3, attention 外的主干，解冻头部和部分特征提取层
                if any(x in name for x in ['tcn3', 'attention', 'classifier', 'reg_heads',
                                           'reg_proj', 'cls_proj', 'proto_scale', 'proto_bias',
                                           'conc_directions', 'conc_scale', 'conc_bias',
                                           'feat_proj', 'bn', 'layer_norm']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:  # basic
                # 仅解冻头部和投影层，冻结所有 TCN 和 attention
                if any(x in name for x in ['classifier', 'reg_heads', 'reg_proj', 'cls_proj',
                                           'proto_scale', 'proto_bias', 'conc_directions',
                                           'conc_scale', 'conc_bias', 'bn', 'layer_norm']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.student.parameters())
        print(f"[Adaptation Engine] Unfreeze level: {self.unfreeze_level}")
        print(f"[Adaptation Engine] Trainable params: {trainable_params}/{total_params} ({100*trainable_params/total_params:.1f}%)")

    def set_corrections(self, corrections):
        """设置服务器下发的校正量"""
        self.pending_corrections = corrections

    def apply_corrections(self):
        """应用服务器下发的校正量到本地原型"""
        if self.pending_corrections is not None:
            for key, delta in self.pending_corrections.items():
                if key in self.protos:
                    self.protos[key] = self.protos[key] + delta.to(self.protos[key].device)
            self.pending_corrections = None

    def adapt_step(self, unlabeled_batch, log_stats=False):
        """
        单步在线适应：TENT熵最小化（主要损失）+ 轻量软原型锚定（正则）
        
        TENT核心：直接优化分类器输出的预测熵，迫使模型在目标域数据上
        做出更自信的预测。不需要原型、不需要伪标签，梯度直接来自分类器。
        
        与旧版软原型吸引的区别：
        - 旧版：特征拉向原型 → 原型固定 → 特征已接近原型 → 梯度≈0
        - TENT：优化 logits 的熵 → 决策边界在无标签数据上自适应移动
        
        Args:
            unlabeled_batch: (x, y_cls, y_reg_full, y_phase)，仅使用 x
            log_stats: 是否打印统计信息
        Returns:
            总损失值
        """
        self.apply_corrections()
        self.student.train()

        x = unlabeled_batch[0]
        x_strong = apply_sensor_aug(x, self.config)

        logits_s, feats_s, _ = self.student(x_strong)

        # ====== 主要损失：TENT预测熵最小化 ======
        # 物理意义：让模型对目标域样本的预测更自信（概率分布更尖锐）
        probs = F.softmax(logits_s, dim=1)
        loss_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        # ====== 辅助正则：轻量软原型锚定 ======
        # 防止特征空间过度漂移导致灾难性遗忘
        # 权重从1.0降至0.05，仅作为安全带而非主力
        loss_anchor = self._compute_soft_anchor_loss(feats_s)

        # ====== 投影一致性约束（可选） ======
        loss_proj = 0.0
        if self.config.USE_PROJ_CONSISTENCY:
            if hasattr(self.student, 'feat_proj') and self.student.feat_proj is not None:
                with torch.no_grad():
                    init_proj = self.teacher.feat_proj.weight.data.clone()
                loss_proj = F.mse_loss(self.student.feat_proj.weight, init_proj)

        total_loss = (self.tent_weight * loss_entropy 
                      + self.anchor_weight * loss_anchor 
                      + self.config.PROJ_CONSISTENCY_WEIGHT * loss_proj)

        self.optimizer.zero_grad()
        if total_loss.grad_fn is None:
            return 0.0

        total_loss.backward()
        if self.fisher_mask is not None:
            self._apply_fisher_penalty(self.fisher_mask)
        self.optimizer.step()

        self._update_teacher()

        if log_stats:
            print(f"[Adapt Step] entropy={loss_entropy.item():.4f} "
                  f"anchor={loss_anchor.item():.4f} total={total_loss.item():.4f}")
            if hasattr(self, 'logger'):
                self.logger.info(f"[Adapt Step] entropy={loss_entropy.item():.4f} "
                                f"anchor={loss_anchor.item():.4f} total={total_loss.item():.4f}")

        return total_loss.item(), loss_entropy.item(), loss_anchor.item()

    def _compute_soft_anchor_loss(self, feats_s):
        """
        软原型锚定损失 — 自动选择余弦相似度或马氏距离。
        
        马氏距离模式启用条件：self.proto_vars 非空 且 config.USE_MAHALANOBIS_ANCHOR=True。
        若不满足则自动回退到余弦相似度。
        """
        use_mahal = (
            getattr(self.config, 'USE_MAHALANOBIS_ANCHOR', False)
            and self.proto_vars is not None
            and len(self.proto_vars) > 0
        )
        if use_mahal:
            return self._compute_mahalanobis_anchor_loss(feats_s)
        else:
            return self._compute_cosine_anchor_loss(feats_s)

    def _compute_cosine_anchor_loss(self, feats_s):
        """
        基于余弦相似度的软原型锚定损失（原始实现）。
        
        数学原理：
        设特征 x ∈ R^D，原型集合 {μ_k} ⊂ R^D。
        余弦相似度 s_k = (x·μ_k) / (||x||·||μ_k||)
        软权重 w_k = softmax(s_k / τ), τ=0.1
        软锚点 a = Σ_k w_k · μ_k
        L = ||x - a||²₂
        """
        device = feats_s.device

        proto_list = [v.to(device) for v in self.protos.values()]
        proto_tensor = torch.stack(proto_list)  # (K, D)

        feats_norm = F.normalize(feats_s, dim=-1)
        proto_norm = F.normalize(proto_tensor, dim=-1)
        sim = torch.matmul(feats_norm, proto_norm.T)  # (B, K)

        max_sim, _ = sim.max(dim=1)
        threshold = torch.quantile(max_sim, 0.25)
        mask = max_sim > threshold
        if mask.sum() < 8:
            mask = max_sim > max_sim.min()
        if mask.sum() < 1:
            return (feats_s * 0).mean()

        feats_sel = feats_s[mask]
        sim_sel = sim[mask]

        proto_weights = F.softmax(sim_sel / 0.1, dim=1)  # τ=0.1
        soft_anchor = torch.matmul(proto_weights, proto_tensor)  # (B', D)
        loss = F.mse_loss(feats_sel, soft_anchor)
        return loss

    def _compute_mahalanobis_anchor_loss(self, feats_s):
        """
        基于马氏距离的软原型锚定损失（新增变体）。
        
        数学原理：
        设特征 x ∈ R^D，原型均值 μ_k ∈ R^D，对角方差 σ²_k ∈ R^D₊。
        马氏距离 d_k² = Σⱼ (xⱼ − μ_{k,j})² / (σ²_{k,j} + ε)
        转换为相似度 s_k = −d_k²
        软权重 w_k = softmax(s_k / τ_mahal)
        软锚点 a = Σ_k w_k · μ_k
        L = ||x − a||²₂
        
        核心优势 vs 余弦相似度：
        1. 方差感知：高方差维度被自动降权，不惩罚在该维度上的合理偏差
        2. 范数感知：保留特征幅值信息（余弦归一化会丢失）
        3. 与服务器端修剪一致：服务器已使用马氏距离做原型修剪
        
        Args:
            feats_s: 学生模型输出的特征 (B, D)，未归一化
        
        Returns:
            标量损失值
        """
        device = feats_s.device
        eps = getattr(self.config, 'MAHALANOBIS_EPS', 1e-4)
        min_var = getattr(self.config, 'MAHALANOBIS_MIN_VAR', 0.01)
        temp = getattr(self.config, 'MAHALANOBIS_ANCHOR_TEMP', 1.0)

        # 1. 构建原型均值张量 [K, D] 和对应方差
        proto_keys = list(self.protos.keys())
        proto_list = [self.protos[k].to(device) for k in proto_keys]
        proto_tensor = torch.stack(proto_list)  # (K, D)

        # 提取各原型的方差向量 (使用与 server 一致的 key 格式)
        var_list = []
        for key in proto_keys:
            str_key = f"({key[0]},{key[1]})" if isinstance(key, tuple) else key
            if str_key in self.proto_vars:
                var_vec = self.proto_vars[str_key].to(device)
                var_vec = torch.clamp(var_vec, min=min_var)  # 最小方差约束防除零
            else:
                # 该原型无方差记录 → 使用单位方差（退化为欧氏距离/范数）
                var_vec = torch.ones(proto_tensor.shape[1], device=device)
            var_list.append(var_vec)
        var_tensor = torch.stack(var_list)  # (K, D)

        # 2. 计算马氏距离: d²_k = Σ_j (x_j - μ_{k,j})² / (σ²_{k,j} + ε)
        # feats_s: (B, D), proto_tensor: (K, D), var_tensor: (K, D)
        diff = feats_s.unsqueeze(1) - proto_tensor.unsqueeze(0)  # (B, K, D)
        mahal_sq = (diff ** 2) / (var_tensor.unsqueeze(0) + eps)  # (B, K, D)
        d_mahal_sq = mahal_sq.sum(dim=-1)  # (B, K), 马氏距离平方

        # 3. 距离→相似度: s_k = -d²_k
        sim = -d_mahal_sq  # (B, K)

        # 4. softmax 获得软权重（使用专用的马氏温度缩放）
        proto_weights = F.softmax(sim / temp, dim=1)  # (B, K)

        # 5. 计算软锚点 = 权重加权原型
        soft_anchor = torch.matmul(proto_weights, proto_tensor)  # (B, D)

        # 6. 无需选择性过滤: 马氏距离天然区分近/远原型, 远原型权重自然趋近0
        #    仅过滤掉与所有原型都无意义地远的噪声样本
        min_d_mahal, _ = d_mahal_sq.min(dim=1)  # (B,)
        d_threshold = torch.quantile(min_d_mahal, 0.9)  # 保留90%的样本
        mask = min_d_mahal <= d_threshold
        if mask.sum() < 4:
            mask = slice(None)  # 全部保留

        feats_sel = feats_s[mask] if isinstance(mask, torch.Tensor) else feats_s
        soft_anchor_sel = soft_anchor[mask] if isinstance(mask, torch.Tensor) else soft_anchor

        loss = F.mse_loss(feats_sel, soft_anchor_sel)
        return loss

    def _compute_dynamic_anchor_loss(self, feats_s, pseudo_labels, feats_t=None):
        """
        动态原型锚定损失：
        对于每个样本，根据教师伪标签和当前特征与各阶段原型的相似度，软选择最匹配阶段的原型。
        可选：直接使用教师模型对原型的软聚合权重作为原型权重。
        """
        device = feats_s.device
        loss = 0.0
        count = 0
        # 构建原型张量 [K, D]
        proto_list, proto_keys = [], []
        for k, v in self.protos.items():
            proto_list.append(v.to(device))
            proto_keys.append(k)
        proto_tensor = torch.stack(proto_list)  # (K, D)

        # 计算学生特征与所有原型的余弦相似度
        feats_norm = F.normalize(feats_s, dim=-1)
        proto_norm = F.normalize(proto_tensor, dim=-1)
        sim = torch.matmul(feats_norm, proto_norm.T)  # (B, K)

        # 对于每个样本，构建目标锚点：
        # 方案A：软加权原型（所有类别的原型，但用教师预测类别过滤）
        for i, c in enumerate(pseudo_labels):
            # 找到属于类别c的原型索引
            c_mask = torch.tensor([int(key.strip('()').split(',')[0]) == c for key in proto_keys], device=device)
            if c_mask.sum() == 0:
                continue
            # 只考虑类别c的原型，用相似度softmax加权
            proto_c_sim = sim[i][c_mask]  # 该类内各阶段原型的相似度
            proto_c_weights = F.softmax(proto_c_sim / 0.1, dim=0)  # 温度0.1加强区分
            proto_c = proto_tensor[c_mask]  # (num_phase, D)
            anchor = (proto_c_weights.unsqueeze(0) @ proto_c).squeeze(0)  # (D,)
            loss += F.mse_loss(feats_s[i], anchor)
            count += 1

        # 方案B（简单）：直接用教师模型对原型的软聚合结果作为锚点，但教师特征不更新，需另行计算。
        # 我们这里采用方案A，因为它直接基于学生特征与各阶段原型的亲和度，更灵活。

        return loss / count if count > 0 else 0.0

    def _update_teacher(self):
        # EMA更新教师参数
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data = self.ema_decay * param_t.data + (1 - self.ema_decay) * param_s.data

    def _apply_fisher_penalty(self, fisher_mask):
        for name, param in self.student.named_parameters():
            if param.requires_grad and param.grad is not None and name in fisher_mask:
                param.grad.data *= fisher_mask[name]

    def compute_head_fisher(self, pseudo_loader):
        """计算头部参数的Fisher信息，用于后续掩码保护"""
        # 实现略，与之前讨论一致
        pass

    def compute_adaptation_metrics(self, loader):
        """
        计算适应指标：预测熵和特征到原型的平均距离
        """
        self.student.eval()
        total_entropy = 0.0
        total_proto_dist = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.config.DEVICE)
                logits, feats, _ = self.student(x)
                
                # 计算预测熵
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                total_entropy += entropy.mean().item()

                # 计算特征到最近原型的平均距离（余弦距离）
                feats_norm = F.normalize(feats, dim=-1)
                proto_list = [v.to(self.config.DEVICE) for v in self.protos.values()]
                proto_tensor = torch.stack(proto_list)  # (K, D)
                proto_norm = F.normalize(proto_tensor, dim=-1)
                sim = torch.matmul(feats_norm, proto_norm.T)  # (B, K)
                max_sim, _ = sim.max(dim=1)  # 每个特征到最近原型的相似度
                dist = 1 - max_sim  # 余弦距离 = 1 - 余弦相似度
                total_proto_dist += dist.mean().item()

                count += 1

        if count == 0:
            return 0.0, 0.0

        avg_entropy = total_entropy / count
        avg_proto_dist = total_proto_dist / count
        return avg_entropy, avg_proto_dist