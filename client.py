"""
联邦学习客户端：本地训练、特征均值计算、对齐正则、差分隐私
"""
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, Tuple, Optional, List, Any
from torch.utils.data import DataLoader
from model import FedGasModel
from config import FLConfig
from utils import get_lambda_reg, apply_sensor_aug


class Client:
    """联邦学习客户端类
    
    负责本地训练、特征均值计算、对齐正则、差分隐私等功能
    
    Attributes:
        client_id: 客户端ID
        config: 配置对象
        device: 设备
        model: 本地模型
        train_loader: 训练数据加载器
        prev_model: 上一轮模型（用于蒸馏）
    """
    def __init__(self, client_id: int, config: FLConfig):
        """初始化客户端
        
        Args:
            client_id: 客户端ID
            config: 配置对象
        """
        self.client_id = client_id
        self.config = config
        self.device = config.DEVICE
        self.model: Optional[FedGasModel] = None
        self.train_loader: Optional[DataLoader] = None
        self.prev_model: Optional[FedGasModel] = None   # 上一轮模型（用于蒸馏）
        
        # 用于区分预热和正式训练
        self.is_warmup = False

        # 部署阶段相关属性
        self.online_loader: Optional[DataLoader] = None
        self.adaptation_engine: Optional[Any] = None

    def set_model(self, model: FedGasModel) -> None:
        """设置模型
        
        Args:
            model: 要设置的模型
        """
        self.model = model.to(self.device)

    def set_parameters(self, state_dict: Dict[str, torch.Tensor], skip_reg: bool = False,
                       partial_personalize: bool = False) -> None:
        """Load global/shared parameters into the local model.

        Args:
            state_dict: Model state dict to load.
            skip_reg: If True, keep all local regression parameters untouched.
            partial_personalize: If True, only overwrite shared regression body
                (reg_proj, reg_transformer, reg_heads, etc.) but keep local
                calibration parameters (proto_scale, proto_bias, proto_conc, etc.).
                Requires PERSONALIZED_REG=True & SHARE_REG_HEAD=True in config.
        """
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: model not set")

        # Do not mutate the shared dict from the server; every client receives the same object.
        load_state = dict(state_dict)
        if skip_reg:
            keys_to_skip = set()
            mlp_keys = {'reg_heads', 'reg_proj', 'reg_transformer', 'reg_attn', 'reg_attn_linear',
                        'proto_scale', 'proto_bias', 'proto_conc', 'conc_bucket_classifier'}
            proto_reg_keys = {'conc_directions', 'conc_scale', 'conc_bias', 'residual_heads',
                              'conc_bucket_classifier'}
            for k in list(load_state.keys()):
                base_name = k.split('.')[0]
                if base_name in mlp_keys or base_name in proto_reg_keys:
                    keys_to_skip.add(k)
            for k in keys_to_skip:
                del load_state[k]
        elif partial_personalize:
            # 模式 3：下发共享回归体（reg_proj, reg_transformer, reg_heads），
            # 但跳过个性化校准参数（proto_scale, proto_bias, proto_conc 等），
            # 避免服务器初始/聚合值覆盖本地训练好的校准参数
            keys_to_skip = set()
            personalized_calib_keys = {
                'proto_scale', 'proto_bias', 'proto_conc',
                'conc_scale', 'conc_bias', 'residual_heads'
            }
            for k in list(load_state.keys()):
                base_name = k.split('.')[0]
                if base_name in personalized_calib_keys:
                    keys_to_skip.add(k)
            for k in keys_to_skip:
                del load_state[k]
        self.model.load_state_dict(load_state, strict=False)

    def init_online_adaptation(self, global_protos, unfreeze_level='basic', anchor_weight=None, proto_vars=None):
        """部署阶段初始化本地适应引擎
        
        Args:
            global_protos: 全局语义原型 dict[str_key → Tensor(D,)]
            unfreeze_level: 解冻级别 ('basic'|'medium'|'full')
            anchor_weight: 锚定损失权重, None 使用默认值
            proto_vars: 各原型的对角方差 dict[str_key → Tensor(D,)], 用于马氏距离锚定
        """
        from adaptation import LocalAdaptationEngine
        self.adaptation_engine = LocalAdaptationEngine(self.model, global_protos, self.config, 
                                                      unfreeze_level=unfreeze_level, 
                                                      anchor_weight=anchor_weight,
                                                      proto_vars=proto_vars)

    def online_adapt_step(self, unlabeled_batch):
        """执行一次在线适应，返回损失值"""
        return self.adaptation_engine.adapt_step(unlabeled_batch)

    def upload_online_statistics(self):
        """
        周期性上传统计量：
        - 类条件特征均值（基于教师伪标签，高置信度过滤）
        - 平均预测熵（用于联邦校准过滤异常客户端）
        """
        from utils import soft_aggregate_probs
        if self.adaptation_engine is None or self.online_loader is None:
            return None

        self.model.eval()
        sum_feats = defaultdict(lambda: torch.zeros(self.config.HIDDEN_DIM2, device=self.device))
        count_samples = defaultdict(int)
        confidence_sum = 0.0
        entropy_sum = 0.0
        entropy_count = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.online_loader:
                x = batch[0].to(self.device)
                # 教师前向: (logits, cls_feat[归一化], reg_feat[未归一化])
                _, feats_t, reg_feat_t = self.adaptation_engine.teacher(x)
                probs_t = soft_aggregate_probs(feats_t, self.adaptation_engine.protos,
                                               temperature=self.config.SOFT_AGG_TEMPERATURE)
                max_conf, pseudo_labels = probs_t.max(dim=1)

                # 计算预测熵
                entropy = -torch.sum(probs_t * torch.log(probs_t + 1e-8), dim=1)
                entropy_sum += entropy.sum().item()
                entropy_count += len(entropy)

                # 高置信度过滤
                mask = max_conf > self.adaptation_engine.conf_thresh
                if mask.sum() == 0:
                    continue
                feats_filtered = feats_t[mask]
                labels_filtered = pseudo_labels[mask]
                conf_filtered = max_conf[mask]

                # 马氏距离阶段分配：若 proto_vars 可用则使用未归一化特征 + 马氏距离
                proto_vars_available = (
                    self.adaptation_engine.proto_vars is not None
                    and len(self.adaptation_engine.proto_vars) > 0
                )
                if proto_vars_available:
                    reg_filtered = reg_feat_t[mask]  # 未归一化特征用于马氏距离

                for i in range(len(feats_filtered)):
                    feat = feats_filtered[i]
                    c = labels_filtered[i].item()
                    # 收集类别c的各阶段原型
                    proto_keys_c = [k for k in self.adaptation_engine.protos.keys()
                                    if int(k.strip('()').split(',')[0]) == c]
                    if not proto_keys_c:
                        continue

                    if proto_vars_available:
                        # 马氏距离阶段分配: d² = Σ_j (f_j - μ_j)² / (σ²_j + ε)
                        reg_f = reg_filtered[i]
                        best_dist = float('inf')
                        best_phase = None
                        eps = getattr(self.config, 'MAHALANOBIS_EPS', 1e-4)
                        min_var = getattr(self.config, 'MAHALANOBIS_MIN_VAR', 0.01)
                        for pk in proto_keys_c:
                            proto_mu = self.adaptation_engine.protos[pk].to(self.device)
                            if pk in self.adaptation_engine.proto_vars:
                                var_vec = torch.clamp(
                                    self.adaptation_engine.proto_vars[pk].to(self.device),
                                    min=min_var
                                )
                                mahal_sq = ((reg_f - proto_mu) ** 2 / (var_vec + eps)).sum()
                            else:
                                mahal_sq = ((reg_f - proto_mu) ** 2).sum()  # 回退到 L2
                            if mahal_sq < best_dist:
                                best_dist = mahal_sq
                                best_phase = int(pk.strip('()').split(',')[1])
                    else:
                        # 余弦相似度阶段分配（原有逻辑）
                        proto_list_c = [self.adaptation_engine.protos[k].to(self.device) for k in proto_keys_c]
                        proto_tensor_c = torch.stack(proto_list_c)  # (num_phases, D)
                        sim = F.cosine_similarity(feat.unsqueeze(0), proto_tensor_c)  # (num_phases,)
                        best_phase_idx = sim.argmax().item()
                        best_phase = int(proto_keys_c[best_phase_idx].strip('()').split(',')[1])

                    key = (c, best_phase)
                    sum_feats[key] += feat
                    count_samples[key] += 1

                confidence_sum += conf_filtered.sum().item()
                total_samples += mask.sum()

        if total_samples == 0:
            return None

        class_feat_means = {}
        counts = {}
        for key in sum_feats:
            class_feat_means[key] = sum_feats[key] / count_samples[key]
            counts[key] = count_samples[key]

        avg_confidence = confidence_sum / total_samples
        avg_entropy = entropy_sum / entropy_count if entropy_count > 0 else 1.0

        if self.config.USE_DP:
            for key in class_feat_means:
                class_feat_means[key] += torch.randn_like(class_feat_means[key]) * self.config.DP_SIGMA

        return {
            'client_id': self.client_id,
            'class_feat_means': class_feat_means,
            'counts': counts,
            'confidence': avg_confidence,
            'avg_entropy': avg_entropy
        }

    def set_prev_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """加载上一轮模型参数
        
        用于特征蒸馏，防止模型遗忘
        
        Args:
            state_dict: 上一轮模型的状态字典
        """
        if self.prev_model is None:
            # 创建与当前模型结构相同的模型实例
            if hasattr(self.model, 'conc_directions'):
                # ProtoReg 模型
                from model import FedGasProtoRegMultiTaskModel
                self.prev_model = FedGasProtoRegMultiTaskModel(
                    num_classes=self.config.NUM_CLASSES,
                    num_sensors=self.config.INPUT_DIM,
                    feat_dim=self.config.HIDDEN_DIM2,
                    use_residual=getattr(self.model, 'use_residual', False),
                    num_conc_buckets=getattr(self.config, 'NUM_CONC_BUCKETS', 0)
                ).to(self.device)
            elif hasattr(self.model, 'reg_heads'):
                # 如果当前模型有回归头，则创建 FedGasMultiTaskModel
                from model import FedGasMultiTaskModel
                self.prev_model = FedGasMultiTaskModel(
                    num_classes=self.config.NUM_CLASSES,
                    num_sensors=self.config.INPUT_DIM,
                    feat_dim=self.config.HIDDEN_DIM2,
                    reg_head_depth=getattr(self.config, 'REG_HEAD_DEPTH', 3),
                    use_quantile=getattr(self.config, 'USE_QUANTILE_LOSS', False),
                    num_conc_buckets=getattr(self.config, 'NUM_CONC_BUCKETS', 0),
                    use_dual_proj=getattr(self.config, 'USE_DUAL_PROJ', False)
                ).to(self.device)
            else:
                # 否则创建基础模型
                self.prev_model = FedGasModel(
                    num_classes=self.config.NUM_CLASSES,
                    num_sensors=self.config.INPUT_DIM,
                    feat_dim=self.config.HIDDEN_DIM2
                ).to(self.device)
        # 严格模式为False，允许缺少回归头
        self.prev_model.load_state_dict(state_dict, strict=False)
        self.prev_model.eval()   # 旧模型不参与梯度更新

    def get_parameters(self, skip_reg=False, partial_personalize=False) -> Dict[str, torch.Tensor]:
        """Return local parameters for federated aggregation.

        Args:
            skip_reg: If True, exclude all regression-related parameters.
            partial_personalize: If True, upload shared regression body but exclude calibration parameters.
        """
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: model not set")
        state = self.model.state_dict()

        if skip_reg or partial_personalize:
            keys_to_skip = set()
            full_reg_keys = {
                'reg_proj', 'reg_transformer', 'reg_attn', 'reg_attn_linear',
                'reg_heads', 'proto_scale', 'proto_bias', 'proto_conc',
                'conc_directions', 'conc_scale', 'conc_bias', 'residual_heads',
                'conc_bucket_classifier'
            }
            personalized_calib_keys = {
                'proto_scale', 'proto_bias', 'proto_conc',
                'conc_scale', 'conc_bias', 'residual_heads'
            }
            skip_set = full_reg_keys if skip_reg else personalized_calib_keys

            for k in list(state.keys()):
                base_name = k.split('.')[0]
                if base_name in skip_set:
                    keys_to_skip.add(k)
            for k in keys_to_skip:
                del state[k]
        return {k: v.detach().cpu() for k, v in state.items()}

    def update_dataloader(self, loader: DataLoader) -> None:
        """更新数据加载器
        
        每轮更新数据加载器，实现时序漂移
        
        Args:
            loader: 新的数据加载器
        """
        self.train_loader = loader

    def _compute_feature_means(self, return_var: bool = False) -> Tuple[Dict[Tuple[int, int], torch.Tensor],
                                              Dict[Tuple[int, int], int],
                                              Optional[Dict[Tuple[int, int], torch.Tensor]]]:
        """计算特征均值（及可选对角方差）

        用未归一化特征 reg_feat 计算方差（马氏距离需要原始空间分布）。
        
        Args:
            return_var: 是否返回方差字典 (P2-1 马氏修剪需要)

        Returns:
            (均值字典, 样本数字典, [方差字典或None])
        """
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: model not set")
        self.model.eval()
        sum_feats = defaultdict(lambda: torch.zeros(self.config.HIDDEN_DIM2, device=self.device))
        sum_feats_sq = defaultdict(lambda: torch.zeros(self.config.HIDDEN_DIM2, device=self.device))
        count_samples = defaultdict(int)

        if self.train_loader is None:
            print(f"Client {self.client_id}: train_loader is None, returning empty dictionaries")
            return ({}, {}, {}) if return_var else ({}, {})

        has_data = False
        with torch.no_grad():
            for batch in self.train_loader:
                has_data = True
                x, y_cls, y_reg, y_p = batch
                x = x.to(self.device)
                y_cls = y_cls.to(self.device)
                y_p = y_p.to(self.device)

                if return_var:
                    _, cls_feat, reg_feat = self.model(x)
                    raw = reg_feat  # 未归一化特征
                else:
                    _, cls_feat, _ = self.model(x)
                    raw = cls_feat  # 归一化特征

                for i in range(len(x)):
                    key = (y_cls[i].item(), y_p[i].item())
                    sum_feats[key] += raw[i]
                    if return_var:
                        sum_feats_sq[key] += raw[i] ** 2
                    count_samples[key] += 1

        if not has_data:
            print(f"Client {self.client_id}: train_loader has no data, returning empty dictionaries")
            return ({}, {}, {}) if return_var else ({}, {})

        mean_dict = {k: sum_feats[k] / count_samples[k] for k in sum_feats}
        
        if return_var:
            var_dict = {}
            for k in sum_feats:
                n = float(count_samples[k])
                var = sum_feats_sq[k] / n - mean_dict[k] ** 2
                var = torch.clamp(var, min=getattr(self.config, 'MAHALANOBIS_MIN_VAR', 0.01))
                var_dict[k] = var
            return mean_dict, count_samples, var_dict
        
        return mean_dict, count_samples

    def train_warmup(self, epochs: int = 3) -> None:
        """热身训练
        
        增量设备热身训练（仅主任务，无对齐）
        
        Args:
            epochs: 训练轮数
        """
        self.is_warmup = True
        if self.train_loader is None:
            raise RuntimeError(f"Client {self.client_id}: train_loader not set")
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: model not set")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LR_CLIENT)
        
        # 学习率调度
        if self.config.LR_SCHEDULER == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif self.config.LR_SCHEDULER == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.LR_STEP_SIZE, gamma=self.config.LR_GAMMA)
        else:
            scheduler = None
        
        for _ in range(epochs):
            for batch in self.train_loader:
                x, y_cls, _, _ = batch  # 显式忽略回归标签和阶段标签
                x = x.to(self.device)
                y_cls = y_cls.to(self.device)

                # === 设备级传感器增强（仅预热阶段） ===
                if self.config.USE_SENSOR_AUG:
                    orig_T = x.size(1)
                    
                    # ① 设备级增益和偏置（batch 作为整体，模拟一个虚拟设备）
                    gain = torch.randn(1, 1, 1, device=self.device) * self.config.SENSOR_AUG_GAIN_STD + 1.0
                    bias = torch.randn(1, 1, 1, device=self.device) * self.config.SENSOR_AUG_BIAS_STD
                    x = x * gain + bias
                    
                    # ② 通道独立增益（模拟不同传感器通道老化差异）
                    ch_gain = torch.randn(1, 1, x.size(2), device=self.device) * self.config.SENSOR_AUG_CH_GAIN_STD + 1.0
                    x = x * ch_gain
                    
                    # ③ 时间轴扰动（模拟响应速度差异）
                    if torch.rand(1).item() < self.config.SENSOR_AUG_TIME_PROB:
                        scale = torch.rand(1).item() * self.config.SENSOR_AUG_TIME_SCALE_RANGE * 2 + (1 - self.config.SENSOR_AUG_TIME_SCALE_RANGE)
                        new_T = int(orig_T * scale)
                        
                        x_temp = x.permute(0, 2, 1)
                        x_temp = F.interpolate(x_temp, size=new_T, mode='linear', align_corners=False)
                        x = x_temp.permute(0, 2, 1)
                        
                        # 用原始长度对齐
                        if new_T > orig_T:
                            x = x[:, :orig_T, :]
                        elif new_T < orig_T:
                            pad_size = orig_T - new_T
                            x = F.pad(x, (0, 0, 0, pad_size), mode='reflect')  # reflect 更平滑

                optimizer.zero_grad()
                logits, _, _ = self.model(x)
                loss_cls = F.cross_entropy(logits, y_cls)
                loss = loss_cls
                loss.backward()
                # 梯度裁剪（提高阈值）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
            
            # 学习率调度
            if scheduler:
                scheduler.step()
        
        self.is_warmup = False
    
    def compute_means_only(self) -> Tuple[Dict[Tuple[int, int], torch.Tensor], Dict[Tuple[int, int], int]]:
        """仅计算特征均值
        
        不更新模型，仅计算特征均值
        
        Returns:
            均值字典和样本数字典
        """
        return self._compute_feature_means()
    
    def _compute_global_feature_mean(self) -> torch.Tensor:
        """计算全局特征均值
        
        计算整个数据集的平均特征向量 F_i
        
        Returns:
            全局平均特征向量
        """
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: model not set")
        if self.train_loader is None:
            return torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
        self.model.eval()
        sum_feat = torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
        total = 0
        with torch.no_grad():
            for batch in self.train_loader:
                x, _, _, _ = batch
                x = x.to(self.device)
                _, feats, _ = self.model(x)   # 取归一化特征
                sum_feat += feats.sum(dim=0)
                total += x.size(0)
        return sum_feat / total if total > 0 else torch.zeros_like(sum_feat)

    def train_one_round(self, current_round: int, global_protos: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
                     semantic_protos: Optional[Dict[str, torch.Tensor]] = None
                     ) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[int, int], torch.Tensor],
                                Dict[Tuple[int, int], int], torch.Tensor, Optional[torch.Tensor],
                                Dict[Tuple[int, int], torch.Tensor]]:
        """本地一轮训练
        
        1. 本地训练+对齐正则
        2. 后计算μ+DP噪声
        
        Args:
            global_protos: 全局可学习原型
            semantic_protos: 全局语义原型，用于计算设备残差
            
        Returns:
            模型参数, 加噪均值, 样本数, 特征统计量F_i, 设备残差估计值, 加噪方差
        """
        if self.train_loader is None:
            raise RuntimeError(f"Client {self.client_id}: train_loader not set")
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: model not set")
        self.model.train()

        config = self.config
        stagewise = getattr(config, 'STAGEWISE_TRAINING', False)

        # ---- 分阶段控制逻辑 ----
        if stagewise:
            phase1_end = config.PHASE1_END_ROUND
            in_phase1 = (current_round <= phase1_end)

            # 回归权重：第一阶段微暖，第二阶段渐变
            if in_phase1:
                lambda_reg = config.PHASE1_REG_WARMUP_WEIGHT
            else:
                phase2_round = current_round - phase1_end
                ramp = min(1.0, phase2_round / max(1, config.PHASE2_REG_RAMPUP_ROUNDS))
                lambda_reg = ramp * config.PHASE2_REG_WEIGHT

            # 分区参数组
            shared_params = list(self.model.tcn.parameters()) + list(self.model.self_attn.parameters()) + list(self.model.attn_linear.parameters())
            cls_proj_params = list(self.model.cls_proj.parameters()) if getattr(self.model, 'use_dual_proj', False) and self.model.cls_proj is not None else []
            reg_params = self.model.get_regression_params()

            for p in self.model.parameters():
                p.requires_grad = True

            if in_phase1:
                # 第一阶段：分类分支正常学习，回归分支极慢速预热，允许 reg_proj 缓慢适应
                optimizer = torch.optim.Adam([
                    {'params': shared_params, 'lr': config.LR_CLIENT},
                    {'params': cls_proj_params, 'lr': config.LR_CLIENT},
                    {'params': self.model.classifier.parameters(), 'lr': config.LR_CLIENT},
                    {'params': reg_params, 'lr': config.LR_CLIENT * 0.1},
                ])
            else:
                # 第二阶段：冻结分类头，cls_proj 极低学习率，共享层降速，回归分支正常学习
                for param in self.model.classifier.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.Adam([
                    {'params': shared_params, 'lr': config.PHASE2_SHARED_LR},
                    {'params': cls_proj_params, 'lr': config.PHASE2_CLS_PROJ_LR},
                    {'params': reg_params, 'lr': config.PHASE2_LR_CLIENT},
                ])
        else:
            # 原有联合训练逻辑
            lambda_reg = None  # 使用动态权重
            optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LR_CLIENT)

        # 学习率调度器：分阶段模式下停用（因为参数组动态变化）
        scheduler = None

        # ========= 1. 本地训练 =========
        for _ in range(self.config.LOCAL_EPOCHS):
            for batch in self.train_loader:
                # 解包时获取 4 维回归标签
                x, y_cls, y_reg_full, y_p = batch
                x = x.to(self.device)
                y_cls = y_cls.to(self.device)
                y_reg_full = y_reg_full.to(self.device)
                y_p = y_p.to(self.device)
                
                # 训练阶段传感器数据增强（概率由 AUG_PROB_TRAIN 控制）
                if self.config.USE_SENSOR_AUG and torch.rand(1).item() < getattr(self.config, 'AUG_PROB_TRAIN', 1.0):
                    x = apply_sensor_aug(x, self.config)

                # 从 4 维向量中提取当前气体对应的浓度
                # y_cls: (batch_size,) 每个样本的类别标签
                # y_reg_full: (batch_size, 4)
                y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1)  # (batch_size, 1)

                # 噪声已移至 model.forward 特征空间，此处不再需要任何输入扰动

                optimizer.zero_grad()
                logits, cls_feat, reg_feat = self.model(x)   # cls_feat: 归一化特征, reg_feat: 未归一化特征

                # 计算各部分损失
                loss_cls = self._compute_classification_loss(logits, y_cls)
                loss_align = self._compute_align_loss(cls_feat, y_cls, y_p, global_protos)
                loss_distill = self._compute_distill_loss(x, cls_feat)
                loss_reg = self._compute_regression_loss(reg_feat, y_cls, y_reg, semantic_protos, y_p, current_round)

                # 总损失（不再包含 inter_loss）
                if lambda_reg is None:
                    lambda_reg = get_lambda_reg(current_round, self.config.GLOBAL_ROUNDS)
                total_loss = loss_cls + loss_align + loss_distill + lambda_reg * loss_reg
                # 回归对比损失
                if getattr(self.config, 'USE_REG_CONTRASTIVE', False):
                    from utils import regression_contrastive_loss
                    loss_reg_contrast = regression_contrastive_loss(
                        reg_feat, y_cls, y_reg,
                        temperature=self.config.REG_CONTRASTIVE_TEMP,
                        pos_margin=self.config.REG_CONTRASTIVE_POS_MARGIN,
                    )
                    total_loss = total_loss + self.config.LAMBDA_REG_CONTRASTIVE * loss_reg_contrast
                                
                total_loss.backward()
                # 梯度裁剪（提高阈值）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
            
            # 学习率调度
            if scheduler:
                scheduler.step()

        # ========= 3. 训练后均值 + 方差 + 差分隐私噪声  =========
        post_mu, count_dict, post_var = self._compute_feature_means(return_var=True)
        noisy_mu = {}
        noisy_var = {}
        for key, feat in post_mu.items():
            if self.config.USE_DP:
                feat = feat + torch.randn_like(feat) * self.config.DP_SIGMA
            feat = torch.clamp(feat, min=-5.0, max=5.0)
            noisy_mu[key] = feat
        for key, var in post_var.items():
            if self.config.USE_DP:
                var = var + torch.abs(torch.randn_like(var) * self.config.DP_SIGMA * 0.1)
            var = torch.clamp(var, min=getattr(self.config, 'MAHALANOBIS_MIN_VAR', 0.01), max=10.0)
            noisy_var[key] = var

        # 计算特征统计量F_i（全局无条件均值）
        F_i = self._compute_global_feature_mean()
        
        # ========= 新增：计算本地设备残差估计 =========
        device_residual = None
        if self.config.USE_PROTO_DECOUPLING and semantic_protos is not None:
            residual_sum = torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
            residual_count = 0
            for key, mu_local in post_mu.items():
                # 统一键格式为无空格的格式，与服务器存储的格式一致
                c, p = key
                str_key = f"({c},{p})"
                # semantic_protos 传入时键为字符串格式
                if str_key in semantic_protos:
                    mu_sem = semantic_protos[str_key].to(self.device)
                    residual_sum += (mu_local - mu_sem)
                    residual_count += 1
            if residual_count > 0:
                device_residual = (residual_sum / residual_count).detach().cpu()
            else:
                device_residual = torch.zeros(self.config.HIDDEN_DIM2)
        
        skip_reg = False
        partial_personalize = False
        if hasattr(self.config, 'PERSONALIZED_REG') and hasattr(self.config, 'SHARE_REG_HEAD'):
            if self.config.PERSONALIZED_REG and self.config.SHARE_REG_HEAD:
                # 部分个性化: 共享方向, 个性偏置
                partial_personalize = True
            elif self.config.PERSONALIZED_REG and not self.config.SHARE_REG_HEAD:
                skip_reg = True
        elif hasattr(self.config, 'PERSONALIZED_REG'):
            skip_reg = self.config.PERSONALIZED_REG
        return self.get_parameters(skip_reg=skip_reg, partial_personalize=partial_personalize), noisy_mu, count_dict, F_i, device_residual, noisy_var
    
    def _compute_classification_loss(self, logits, y_cls):
        """计算分类损失
        
        Args:
            logits: 模型输出的logits
            y_cls: 真实类别标签
        
        Returns:
            分类损失值
        """
        return F.cross_entropy(logits, y_cls)
    
    def _compute_align_loss(self, feats, y_cls, y_p, global_protos):
        """计算对齐损失
        
        Args:
            feats: 模型提取的特征
            y_cls: 真实类别标签
            y_p: 相位标签
            global_protos: 全局原型
        
        Returns:
            对齐损失值
        """
        loss_align = torch.tensor(0.0, device=self.device)
        if self.config.USE_ALIGN and global_protos:
            if self.config.USE_CONTRASTIVE_ALIGN:
                from utils import contrastive_loss_with_protos
                # 转换全局原型为字符串键格式
                protos_str_key = {str(k): v for k, v in global_protos.items()}
                loss_align = contrastive_loss_with_protos(
                    feats, y_cls, y_p, protos_str_key,
                    temperature=self.config.CONTRAST_TEMPERATURE
                )
            else:
                # 批量化原型匹配，避免逐样本字典查找
                batch_size = len(feats)
                if batch_size > 0:
                    # 构建原型张量 [num_classes, num_phases, feat_dim]
                    num_classes = self.config.NUM_CLASSES
                    num_phases = self.config.NUM_PHASES
                    feat_dim = feats.size(1)
                    
                    # 初始化原型张量为零
                    proto_tensor = torch.zeros(num_classes, num_phases, feat_dim, device=self.device)
                    
                    # 填充原型张量
                    for (cls, phase), proto in global_protos.items():
                        if cls < num_classes and phase < num_phases:
                            proto_tensor[cls, phase] = proto.to(self.device)
                    
                    # 批量获取对应的原型
                    selected_protos = proto_tensor[y_cls, y_p]
                    
                    # 计算批量损失
                    align_terms = torch.norm(feats - selected_protos, p=2, dim=1).pow(2)
                    loss_align = align_terms.mean()
            loss_align *= self.config.LAMBDA_ALIGN
        return loss_align
    
    def _compute_distill_loss(self, x, feats):
        """计算蒸馏损失
        
        Args:
            x: 输入数据
            feats: 当前模型提取的特征
        
        Returns:
            蒸馏损失值
        """
        loss_distill = torch.tensor(0.0, device=self.device)
        if (self.config.USE_REPLAY_DISTILL and self.prev_model is not None 
                and self.prev_model.training is False):
            with torch.no_grad():
                _, feats_prev, _ = self.prev_model(x)   # 旧模型特征，不更新梯度
            # MSE 损失，迫使当前特征接近旧特征
            loss_distill = F.mse_loss(feats, feats_prev)
            loss_distill = loss_distill * self.config.LAMBDA_REPLAY_DISTILL
        return loss_distill
    
    def _compute_regression_loss(self, feats, y_cls, y_reg, semantic_protos=None, y_phase=None, current_round=None):
        """计算回归损失（复用 utils.py 中的统一函数）

        Args:
            feats: 模型提取的特征
            y_cls: 真实类别标签
            y_reg: 真实回归标签
            semantic_protos: 语义原型（用于排序损失）
            y_phase: 阶段标签（用于原型浓度先验）
            current_round: 当前训练轮次，用于排序损失预热

        Returns:
            回归损失值
        """
        from utils import compute_regression_loss_combined
        return compute_regression_loss_combined(
            reg_feat=feats,
            y_cls=y_cls,
            y_reg=y_reg,
            model=self.model,
            config=self.config,
            semantic_protos=semantic_protos,
            y_phase=y_phase,
            current_round=current_round,
        )
