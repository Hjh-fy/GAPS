"""
联邦学习服务器：可学习聚合、全局均值更新、原型重放、增量设备EMA
"""
import torch
import torch.nn.functional as F
import copy
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader
from model import FedGasModel, DomainDiscriminator, GradientReversalLayer
from config import FLConfig
from utils import compute_mmd, deep_coral_loss, get_lambda_reg
import torch.nn as nn

class Server:
    """联邦学习服务器类
    
    负责模型聚合、可学习原型管理、服务器联合优化等核心功能
    
    Attributes:
        global_model: 全局模型
        val_loader: 验证集数据加载器
        test_loader: 测试集数据加载器
        unit5_loader: Unit5测试集数据加载器
        config: 配置对象
        device: 设备
        logger: 日志记录器
        client_weights: 客户端聚合权重
        prev_model: 上一轮模型（用于蒸馏）
        global_protos: 全局可学习原型
        global_feature_Fg: 全局特征滑动平均
        optimizer: 服务器联合优化器
    """
    def __init__(self, global_model: FedGasModel, val_loader: DataLoader,
                 test_loader: DataLoader, unit5_loader: Optional[DataLoader],
                 config: FLConfig, logger: logging.Logger, calib_loader=None):
        """初始化服务器
        
        Args:
            global_model: 全局模型
            val_loader: 验证集数据加载器
            test_loader: 测试集数据加载器
            unit5_loader: Unit5测试集数据加载器（可以为None）
            config: 配置对象
            logger: 日志记录器
            calib_loader: 深度CORAL校准集数据加载器（可选）
        """
        self.global_model = global_model.to(config.DEVICE)
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.unit5_loader = unit5_loader  # 可以为 None
        self.calib_loader = calib_loader  # 深度CORAL校准集
        self._calib_feature_mean: Optional[torch.Tensor] = None  # 校准集特征均值缓存 (目标感知聚合)
        self.config = config
        self.device = config.DEVICE
        self.logger = logger

        # 全局状态
        self.client_weights: torch.Tensor = torch.ones(config.NUM_CLIENTS, device=self.device) / config.NUM_CLIENTS
        self.prev_model_state: Optional[Dict] = None
        self.prev_model: Optional[FedGasModel] = None   # 用于特征蒸馏
        self.shared_reg_state: Dict[str, torch.Tensor] = {}  # aggregated shared regression initialization

        self.global_feature_Fg: Optional[torch.Tensor] = None  # 滑动平均特征
        self.optimizer: Optional[torch.optim.Optimizer] = None  # 服务器联合优化器

        self.semantic_protos: nn.ParameterDict = nn.ParameterDict()  # 语义原型 μ^{sem}
        self.device_residuals: Dict[int, nn.Parameter] = {}          # 设备残差 δ_dev (训练客户端ID)
        self.use_decoupling = config.USE_PROTO_DECOUPLING

        # 原型矩阵缓存（加速推理）
        self._cached_proto_matrix: Optional[torch.Tensor] = None   # (K, D)
        self._cached_proto_classes: Optional[torch.Tensor] = None  # (K,)
        self._proto_cache_dirty = True

        # ========= 自适应温度推理 & 马氏距离 =========
        self._proto_temperatures: Optional[torch.Tensor] = None  # (K,) 每原型自适应温度
        self._proto_temperatures_dirty = True
        self._proto_spreads: Dict[str, float] = {}  # 每原型特征散布统计

        # 马氏距离修剪 (P2-1)
        self.semantic_proto_vars: Dict[str, torch.Tensor] = {}  # 对角方差, key同 semantic_protos

        # 域对抗训练（服务器端，不违反联邦隐私约束）
        self.domain_discriminator: Optional[DomainDiscriminator] = None
        self.disc_optimizer: Optional[torch.optim.Optimizer] = None
        self.grl: Optional[GradientReversalLayer] = None
        if getattr(config, 'USE_ADVERSARIAL_DOMAIN', False):
            feat_dim = getattr(config, 'HIDDEN_DIM2', 64)
            self.domain_discriminator = DomainDiscriminator(
                feat_dim=feat_dim, hidden_dim=feat_dim // 2
            ).to(self.device)
            self.disc_optimizer = torch.optim.Adam(
                self.domain_discriminator.parameters(),
                lr=getattr(config, 'ADV_DOMAIN_LR', 0.001)
            )
            self.grl = GradientReversalLayer(
                lambda_grl=1.0  # GRL仅负责反转梯度方向，幅度由外部LAMBDA_ADV_DOMAIN控制
            )
            self.logger.info(f"域对抗训练已启用: lambda={config.LAMBDA_ADV_DOMAIN}, "
                           f"disc_lr={config.ADV_DOMAIN_LR}, "
                           f"class_conditional={config.ADV_CLASS_CONDITIONAL}")


    # def deployment_aggregation_round(self, client_uploads):
    #     """
    #     client_uploads: list of dict from clients
    #     """
    #     # 0. 预处理上传中的键：将元组键转换为字符串键
    #     processed_uploads = []
    #     for u in client_uploads:
    #         new_means = {}
    #         new_counts = {}
    #         for k, v in u['class_feat_means'].items():
    #             if isinstance(k, tuple):
    #                 str_key = f"({k[0]},{k[1]})"
    #             else:
    #                 str_key = k
    #             new_means[str_key] = v
    #             if k in u.get('counts', {}):
    #                 new_counts[str_key] = u['counts'][k]
    #         u['class_feat_means'] = new_means
    #         u['counts'] = new_counts
    #         processed_uploads.append(u)

    #     # 1. 过滤异常上传（简单：检查上传的置信度）
    #     valid_uploads = [u for u in processed_uploads if u['confidence'] > self.config.MIN_PROTO_UPDATE_CONF]
    #     if not valid_uploads:
    #         return

    #     # 2. 原型EMA更新
    #     alpha = self.config.PROTO_EMA_ALPHA
    #     # 所有有效客户端的类均值平均
    #     for key in self.semantic_protos:
    #         aggregated_mean = torch.zeros_like(self.semantic_protos[key].data)
    #         total_cnt = 0
    #         for u in valid_uploads:
    #             if key in u['class_feat_means']:
    #                 cnt = u['counts'].get(key, 1)
    #                 aggregated_mean += cnt * u['class_feat_means'][key].to(self.device)
    #                 total_cnt += cnt
    #         if total_cnt > 0:
    #             aggregated_mean /= total_cnt
    #             # EMA更新
    #             self.semantic_protos[key].data = alpha * self.semantic_protos[key].data + (1-alpha) * aggregated_mean

    #     # 3. 可选：更新开集阈值（利用上传的高置信度已知样本特征）
    #     # 可收集所有高置信度样本的特征批量更新OSR阈值，需要服务器端特征提取，可简化

    #     # 4. 广播原型和阈值给所有客户端（通过返回值或单独下发）
    #     self._proto_cache_dirty = True  # 使得客户端获取最新原型

    def broadcast_online_update(self):
        """返回当前原型，用于下发"""
        return self.semantic_protos, None

    def _init_protos_from_clients(self, all_mus: List[Dict], all_counts: List[Dict], 
                                weights: torch.Tensor, client_ids: List[int] = None,
                                all_vars: List[Dict] = None):
        """
        新增参数 client_ids: 与 all_mus 顺序对应的客户端 ID 列表
        新增参数 all_vars: 客户端上传的对角方差列表 (P2-1)
        """
        # 1. 计算加权平均作为候选初始值（语义原型）
        temp_means = {}
        temp_vars = {}
        all_keys = set().union(*all_mus)
        for key in all_keys:
            numerator = torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
            denominator = 0.0
            for i, (mu_dict, cnt_dict) in enumerate(zip(all_mus, all_counts)):
                if key in mu_dict:
                    w = weights[i].item()
                    n = cnt_dict[key]
                    numerator += w * n * mu_dict[key].to(self.device)
                    denominator += w * n
            if denominator > 0:
                temp_means[key] = numerator / denominator

        # 1b. 聚合方差 (P2-1)
        if all_vars is not None:
            for key in all_keys:
                numerator = torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
                denominator = 0.0
                for i, (var_dict, cnt_dict) in enumerate(zip(all_vars, all_counts)):
                    if key in var_dict:
                        w = weights[i].item()
                        n = cnt_dict[key]
                        numerator += w * n * var_dict[key].to(self.device)
                        denominator += w * n
                if denominator > 0:
                    temp_vars[key] = numerator / denominator
                else:
                    temp_vars[key] = torch.ones(self.config.HIDDEN_DIM2, device=self.device)

        # 外推增强（仅对语义原型做外推）
        EXTRAP_ALPHA = getattr(self.config, 'PROTO_EXTRAP_ALPHA', 0.6)
        for key, val in temp_means.items():
            c, p = key
            if p == 2:
                key_early = (c, 0)
                key_mid = (c, 1)
                if key_early in temp_means and key_mid in temp_means:
                    mu_e = temp_means[key_early]
                    mu_m = temp_means[key_mid]
                    extrapolated = mu_m + (mu_m - mu_e)
                    val = EXTRAP_ALPHA * extrapolated + (1 - EXTRAP_ALPHA) * val
                    temp_means[key] = val

        # 补全缺失的原型（确保所有 4 类 × 3 阶段的组合都存在）
        for c in range(self.config.NUM_CLASSES):
            for p in range(self.config.NUM_PHASES):
                key = (c, p)
                if key not in temp_means:
                    # 收集该类别所有已有阶段的均值作为默认值
                    existing = [v for k, v in temp_means.items() if k[0] == c]
                    if existing:
                        temp_means[key] = torch.stack(existing).mean(dim=0)
                    else:
                        temp_means[key] = torch.zeros(self.config.HIDDEN_DIM2, device=self.device)

        # 2. 注册语义原型（不使用EMA更新，完全依赖梯度优化）
        for key, val in temp_means.items():
            c, p = key
            str_key = f"({c},{p})"
            if str_key in self.semantic_protos:
                pass
            else:
                self.semantic_protos[str_key] = nn.Parameter(val.clone().detach().to(self.device))
                self.logger.info(f"Registered semantic prototype for {key}")
        
        # 2b. 注册/更新方差 (P2-1, EMA)
        if all_vars is not None and temp_vars:
            var_ema = getattr(self.config, 'MAHALANOBIS_VAR_EMA', 0.9)
            for key, val in temp_vars.items():
                c, p = key
                str_key = f"({c},{p})"
                if str_key in self.semantic_proto_vars:
                    self.semantic_proto_vars[str_key] = (
                        var_ema * self.semantic_proto_vars[str_key] + (1 - var_ema) * val.detach()
                    )
                else:
                    self.semantic_proto_vars[str_key] = val.clone().detach()
        
        # 3. 若启用解耦，初始化或更新设备残差
        if self.use_decoupling and client_ids is not None:
            for i, cid in enumerate(client_ids):
                # 计算该客户端的平均残差
                mu_dict = all_mus[i]
                residual_sum = torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
                count = 0
                for key, mu_local in mu_dict.items():
                    # 统一键格式为无空格的格式，与服务器存储的格式一致
                    c, p = key
                    str_key = f"({c},{p})"
                    if str_key in self.semantic_protos:
                        mu_sem = self.semantic_protos[str_key].detach()
                        residual_sum += (mu_local.to(self.device) - mu_sem)
                        count += 1
                if count > 0:
                    residual = residual_sum / count
                else:
                    residual = torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
                
                if cid not in self.device_residuals:
                    self.device_residuals[cid] = nn.Parameter(residual)
                    self.logger.info(f"Initialized device residual for client {cid}")
                else:
                    # 不再进行 EMA 更新，完全依赖梯度优化
                    # （可选：仅记录统计值，但不覆盖参数）
                    pass
        
    def get_full_protos(self, device_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        获取完整原型。若提供 device_id 则加对应残差，否则仅返回语义原型。
        """
        full_protos = {}
        for str_key, mu_sem in self.semantic_protos.items():
            mu = mu_sem
            if self.use_decoupling and device_id is not None and device_id in self.device_residuals:
                mu = mu + self.device_residuals[device_id]
            full_protos[str_key] = mu
        return full_protos
    
    def get_global_protos_detached(self, client_id: Optional[int] = None) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        下发原型的 detached 版本，若启用解耦则下发该客户端的完整原型
        """
        protos = {}
        full_protos = self.get_full_protos(device_id=client_id)
        for str_key, param in full_protos.items():
            try:
                key_tuple = tuple(map(int, str_key.strip('()').split(',')))
                protos[key_tuple] = param.detach().cpu()
            except Exception:
                protos[str_key] = param.detach().cpu()
        return protos


    def _aggregate_params(self, client_params: List[Dict], weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """加权聚合模型参数
        
        Args:
            client_params: 客户端参数列表
            weights: 客户端权重
            
        Returns:
            聚合后的参数字典
        """
        agg_params = {}
        # 只聚合服务器模型和客户端模型共有的参数
        server_keys = set(self.global_model.state_dict().keys())
        for key in client_params[0].keys():
            if key not in server_keys:
                if 'reg_heads' in key:
                    # 回归参数由 aggregate_shared_regression_state 单独处理，此处跳过属正常行为
                    pass
                continue
            # 初始化聚合参数为Float类型，确保类型一致
            agg_params[key] = torch.zeros_like(client_params[0][key], device=self.device, dtype=torch.float32)
            for i, params in enumerate(client_params):
                if key in params:
                    # 确保参数类型为Float
                    param = params[key].to(self.device).float()
                    agg_params[key] += weights[i] * param
        return agg_params


    def _is_regression_param(self, key: str) -> bool:
        base_name = key.split('.')[0]
        shared_reg_keys = {
            'reg_proj', 'reg_transformer', 'reg_attn', 'reg_attn_linear',
            'reg_heads', 'conc_directions', 'conc_bucket_classifier'
        }
        personalized_reg_keys = {
            'proto_scale', 'proto_bias', 'proto_conc',
            'conc_scale', 'conc_bias', 'residual_heads'
        }
        if base_name in shared_reg_keys:
            return True
        if not getattr(self.config, 'PERSONALIZED_REG', False) and base_name in personalized_reg_keys:
            return True
        return False

    def aggregate_shared_regression_state(self, client_params: List[Dict], weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_state = {}
        if not client_params:
            self.shared_reg_state = shared_state
            return shared_state
        keys = sorted({key for params in client_params for key in params.keys() if self._is_regression_param(key)})
        for key in keys:
            present = [(i, params[key]) for i, params in enumerate(client_params) if key in params]
            if not present:
                continue
            weight_indices = torch.tensor([i for i, _ in present], device=weights.device, dtype=torch.long)
            weight_sum = weights[weight_indices].sum()
            if weight_sum.item() <= 0:
                continue
            agg = torch.zeros_like(present[0][1], device=self.device, dtype=torch.float32)
            for i, param in present:
                agg += (weights[i] / weight_sum) * param.to(self.device).float()
            shared_state[key] = agg.detach().cpu()
        self.shared_reg_state = shared_state
        return shared_state

    def _compute_val_loss_on_model(self, model: FedGasModel) -> torch.Tensor:
        """计算模型在验证集上的损失
        
        计算模型在验证集上的交叉熵损失，限制批次数以避免内存溢出
        
        Args:
            model: 要评估的模型
            
        Returns:
            验证损失值
        """
        val_loss = torch.tensor(0.0, device=self.device)
        val_batches = 0
        max_val_batches = 10  # 限制验证批次数，避免内存溢出
        for i, batch in enumerate(self.val_loader):
            if i >= max_val_batches:
                break
            x, y_cls, y_reg_full, y_p = batch
            x, y_cls, y_reg_full = x.to(self.device), y_cls.to(self.device), y_reg_full.to(self.device)
            y_p = y_p.to(self.device)
            
            # 从 4 维向量中提取当前气体对应的浓度
            y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1)  # (batch_size, 1)
            
            logits, cls_feat, reg_feat = model(x)
            batch_loss = F.cross_entropy(logits, y_cls)
            
            val_loss += batch_loss
            val_batches += 1
        if val_batches > 0:
            val_loss /= val_batches
        return val_loss

    def _compute_weight_gradients(self, model: FedGasModel, client_params: List[Dict]) -> torch.Tensor:
        """计算权重梯度
        
        计算聚合模型梯度与客户端参数偏差的点积，用于可学习聚合权重更新
        
        Args:
            model: 聚合模型
            client_params: 客户端参数列表
            
        Returns:
            权重梯度张量
        """
        model.zero_grad()
        val_loss = self._compute_val_loss_on_model(model)
        val_loss.backward()
        # 收集聚合模型梯度
        g_agg_flat = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        
        # 计算聚合模型参数 θ_agg 的展平向量
        theta_agg_flat = torch.cat([p.data.view(-1) for p in model.parameters()])
        
        grad_ws = []
        for params in client_params:
            # 只使用服务器模型和客户端模型共有的参数
            common_params = [(name, p) for name, p in model.named_parameters() if name in params]
            if len(common_params) == 0:
                # 如果没有共同参数，使用默认梯度
                grad_ws.append(0.0)
                continue
            
            theta_i_flat = torch.cat([params[name].to(self.device).view(-1) for name, _ in common_params])
            theta_agg_common_flat = torch.cat([p.data.view(-1) for _, p in common_params])
            
            # 确保长度匹配
            min_len = min(len(theta_i_flat), len(theta_agg_common_flat))
            if min_len > 0:
                diff = theta_i_flat[:min_len] - theta_agg_common_flat[:min_len]
                g_agg_common = g_agg_flat[:min_len]
                grad_ws.append(torch.dot(g_agg_common, diff).item())
            else:
                grad_ws.append(0.0)
        return torch.tensor(grad_ws, device=self.device)

    def learnable_aggregate(self, client_params: List[Dict], prev_weights: torch.Tensor) -> torch.Tensor:
        """可学习自适应聚合
        
        实现可学习自适应聚合算法，根据模型性能动态调整客户端权重
        
        Args:
            client_params: 客户端参数列表
            prev_weights: 上一轮权重
            
        Returns:
            更新后的客户端权重
        """
        # 1. 用旧权重聚合得到临时模型
        temp_params = self._aggregate_params(client_params, prev_weights)
        # 使用与全局模型相同的类型创建临时模型（支持多任务头结构）
        temp_model = copy.deepcopy(self.global_model)
        temp_model.load_state_dict(temp_params, strict=False)  # 仅加载共享层参数

        # 2. 计算权重梯度
        grad_ws = self._compute_weight_gradients(temp_model, client_params)
        self.logger.info(f"Weight gradients Δ: {grad_ws.tolist()}")  # 调试日志

        # 3. 更新权重 (Eq.3.8, 3.9)
        w_new = prev_weights - self.config.ETA_W * grad_ws
        w_new = torch.softmax(w_new / self.config.TAU, dim=0)
        
        # 仅保留最小权重约束，移除最大权重裁剪以避免破坏归一化
        w_new = torch.clamp(w_new, min=self.config.WEIGHT_MIN)
        w_new = w_new / w_new.sum()

        # 4. 用新权重重新聚合得到最终全局模型
        final_params = self._aggregate_params(client_params, w_new)
        self.global_model.load_state_dict(final_params, strict=False)

        return w_new

    def get_prev_model_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """返回上一轮全局模型的状态字典
        
        用于客户端进行特征蒸馏
        
        Returns:
            上一轮模型的状态字典，如果不存在则返回None
        """
        if self.prev_model is not None:
            return {k: v.cpu() for k, v in self.prev_model.state_dict().items()}
        return None

    def _compute_align_loss(self) -> torch.Tensor:
        """计算对齐损失用于日志"""
        self.global_model.eval()
        align_loss = torch.tensor(0.0, device=self.device)
        align_batches = 0
        max_align_batches = 10  # 限制对齐损失计算批次数
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_align_batches:
                    break
                x, y_cls, y_reg, y_p = batch
                x, y_cls, y_p = x.to(self.device), y_cls.to(self.device), y_p.to(self.device)
                _, feats, _ = self.global_model(x)  # 取归一化特征
                for i in range(len(x)):
                    c = y_cls[i].item()
                    p = y_p[i].item()
                    str_key = f"({c},{p})"
                    if str_key in self.semantic_protos:
                        align_loss += torch.norm(feats[i] - self.semantic_protos[str_key], p=2).pow(2)
                align_batches += 1
        if align_batches > 0:
            align_loss /= align_batches
        return align_loss

    def _update_proto_cache(self):
        """更新语义原型矩阵缓存"""
        keys = sorted(self.semantic_protos.keys())
        proto_list = []
        class_list = []
        for k in keys:
            proto_list.append(self.semantic_protos[k].detach())
            # 解析键中的类别信息 (格式: "(c,p)")
            if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
                c = int(k.strip('()').split(',')[0])
                class_list.append(c)
        self._cached_proto_matrix = torch.stack(proto_list) if proto_list else None
        self._cached_proto_classes = torch.tensor(class_list, device=self.device) if class_list else None
        self._proto_cache_dirty = False

    def get_cached_protos(self):
        """获取缓存的语义原型矩阵和类别列表"""
        if self._proto_cache_dirty or self._cached_proto_matrix is None:
            self._update_proto_cache()
        return self._cached_proto_matrix, self._cached_proto_classes

    def server_representation_learning(self, client_mus: List[Dict], client_counts: List[Dict],
                                    client_weights: torch.Tensor, K: int, client_ids: List[int] = None,
                                    client_residuals: List[Optional[torch.Tensor]] = None, current_round: int = 0):
        """服务器联合优化
        
        联合优化模型参数 θ、语义原型 μ^{sem} 和设备残差 δ_dev
        
        Args:
            client_mus: 客户端本地原型均值
            client_counts: 客户端本地原型计数
            client_weights: 客户端权重
            K: 优化步数
            client_ids: 与 client_mus 顺序对应的客户端 ID 列表
            client_residuals: 客户端上传的本地残差估计
            current_round: 当前训练轮次
            
        Returns:
            float: 平均对齐损失
        """
        self.global_model.train()
        
        # 构建优化器：模型参数 + 语义原型参数 (+ 设备残差参数)
        opt_params = list(self.global_model.parameters()) + list(self.semantic_protos.values())
        if self.use_decoupling:
            opt_params.extend(list(self.device_residuals.values()))
        self.optimizer = torch.optim.Adam(opt_params, lr=self.config.SERVER_OPT_LR)
        
        source_iter = iter(self.val_loader)
        if self.calib_loader is not None and self.config.USE_DEEP_CORAL:
            target_iter = iter(self.calib_loader)
        else:
            target_iter = None
        
        for step in range(K):
            self.optimizer.zero_grad()
            
            # ---- 1. 验证损失 ----
            val_loss = torch.tensor(0.0, device=self.device)
            val_batches = 0
            max_val_batches = 10  # 限制验证批次数，避免显存溢出或超时
            for batch in self.val_loader:
                if val_batches >= max_val_batches:
                    break
                x, y_cls, y_reg_full, y_p = batch
                x, y_cls, y_reg_full = x.to(self.device), y_cls.to(self.device), y_reg_full.to(self.device)
                y_p = y_p.to(self.device)
                
                # 从 4 维向量中提取当前气体对应的浓度
                y_reg = y_reg_full[torch.arange(y_cls.size(0)), y_cls].unsqueeze(1)  # (batch_size, 1)
                
                logits, cls_feat, reg_feat = self.global_model(x)
                loss_cls = F.cross_entropy(logits, y_cls)
                val_loss += loss_cls
                
                val_batches += 1
            if val_batches > 0:
                val_loss /= val_batches
            
            # ---- 2. 原型学习损失（使用完整原型）----
            proto_loss_terms = []
            for i, (mu_dict, cnt_dict) in enumerate(zip(client_mus, client_counts)):
                w = client_weights[i]
                cid = client_ids[i] if client_ids else i
                full_protos = self.get_full_protos(device_id=cid)  # 该客户端对应的完整原型
                for key, mu_i in mu_dict.items():
                    # 统一键格式为无空格的格式，与服务器存储的格式一致
                    c, p = key
                    str_key = f"({c},{p})"
                    if str_key in full_protos:
                        n = torch.tensor(cnt_dict[key], device=self.device, dtype=torch.float32)
                        loss = torch.norm(full_protos[str_key] - mu_i.to(self.device), p=2).pow(2)
                        proto_loss_terms.append(w * n * loss)
            if proto_loss_terms:
                proto_loss = torch.sum(torch.stack(proto_loss_terms)) / len(proto_loss_terms)
            else:
                proto_loss = torch.tensor(0.0, device=self.device)
            
            # ---- 3. 一致性损失（使用语义原型，强制特征向语义中心靠拢）----
            consist_loss = torch.tensor(0.0, device=self.device)
            if self.config.USE_CONTRASTIVE_CONSISTENCY:
                from utils import contrastive_loss_with_protos
                # 收集整个验证集的特征和标签
                all_feats, all_labels, all_phases = [], [], []
                max_batches = 20  # 限制最大批次数，避免显存溢出
                batch_count = 0
                for batch in self.val_loader:
                    if batch_count >= max_batches:
                        break
                    x, y_cls, _, y_p = batch
                    x, y_cls, y_p = x.to(self.device), y_cls.to(self.device), y_p.to(self.device)
                    _, feats, _ = self.global_model(x)
                    all_feats.append(feats)
                    all_labels.append(y_cls)
                    all_phases.append(y_p)
                    batch_count += 1
                
                if len(all_feats) > 0:
                    feats_cat = torch.cat(all_feats, dim=0)
                    labels_cat = torch.cat(all_labels, dim=0)
                    phases_cat = torch.cat(all_phases, dim=0)
                    consist_loss = contrastive_loss_with_protos(
                        feats_cat, labels_cat, phases_cat,
                        self.semantic_protos,
                        temperature=self.config.CONTRAST_TEMPERATURE
                    )
            else:
                # 原有 L2 计算
                consist_samples = 0
                max_batches = 20  # 限制最大批次数，避免显存溢出
                batch_count = 0
                for batch in self.val_loader:
                    if batch_count >= max_batches:
                        break
                    x, y_cls, _, y_p = batch
                    x, y_cls, y_p = x.to(self.device), y_cls.to(self.device), y_p.to(self.device)
                    _, feats, _ = self.global_model(x)
                    for i in range(len(x)):
                        key = (y_cls[i].item(), y_p[i].item())
                        str_key = str(key)
                        if str_key in self.semantic_protos:
                            mu_sem = self.semantic_protos[str_key]
                            consist_loss += torch.norm(feats[i] - mu_sem, p=2).pow(2)
                            consist_samples += 1
                    batch_count += 1
                if consist_samples > 0:
                    consist_loss /= consist_samples
            
            # ---- 5. 设备残差损失（新增）----
            residual_loss = torch.tensor(0.0, device=self.device)
            if self.use_decoupling and client_residuals is not None:
                res_count = 0
                for i, cid in enumerate(client_ids):
                    if client_residuals[i] is not None and cid in self.device_residuals:
                        residual_loss += torch.norm(self.device_residuals[cid] - client_residuals[i].to(self.device), p=2).pow(2)
                        res_count += 1
                if res_count > 0:
                    residual_loss /= res_count
            

            # ---- 域泛化正则：原型对齐损失（替代 MMD） ----
            align_reg_loss = torch.tensor(0.0, device=self.device)
            if self.config.USE_MMD_REG and len(client_mus) >= 2:
                count = 0
                for mu_dict in client_mus:
                    for key, mu_local in mu_dict.items():
                        # 统一键格式为无空格的格式，与服务器存储的格式一致
                        c, p = key
                        str_key = f"({c},{p})"
                        if str_key in self.semantic_protos:
                            mu_sem = self.semantic_protos[str_key]
                            align_reg_loss += torch.norm(mu_local.to(self.device) - mu_sem, p=2).pow(2)
                            count += 1
                if count > 0:
                    align_reg_loss = align_reg_loss / count
            
            # 原型对比学习已移除
            
            # 原型级 MMD 对齐损失：不同客户端对同一 (c,p) 原型的分布一致性约束
            mmd_proto_loss = torch.tensor(0.0, device=self.device)
            if self.config.USE_PROTO_MMD and len(client_mus) >= 2:
                proto_dict = defaultdict(list)
                for mu_dict in client_mus:
                    for key, mu in mu_dict.items():
                        proto_dict[key].append(mu)
                count = 0
                for key, mu_list in proto_dict.items():
                    if len(mu_list) < 2:
                        continue
                    for i in range(len(mu_list)):
                        for j in range(i+1, len(mu_list)):
                            mmd_proto_loss += compute_mmd(mu_list[i].unsqueeze(0), mu_list[j].unsqueeze(0))
                            count += 1
                if count > 0:
                    mmd_proto_loss /= count
            
            # ---- MMD对齐损失（新方案）----
            global_mmd_loss, class_mmd_loss, proto_anchor_loss, stage_mmd_loss = self._compute_mmd_alignment_loss()
            
            # ---- 深度CORAL损失 + 域对抗训练 ----
            # 这两个损失共享源域和目标域特征，统一获取以复用前向传播
            coral_loss = torch.tensor(0.0, device=self.device)
            adv_loss = torch.tensor(0.0, device=self.device)
            # 域对抗训练仅在同时有calib_loader和判别器时启用
            use_adv = (self.domain_discriminator is not None and self.calib_loader is not None)
            need_target_features = (target_iter is not None or use_adv)
            
            cls_feat_s = None
            cls_feat_t = None
            x_s = x_t = None
            y_s = y_t = None
            
            # 统一获取目标域特征（供CORAL和域对抗训练共用）
            # 统一使用 cls_feat（L2归一化），避免 reg_feat/cls_feat 两种空间在共享TCN上产生间接冲突
            if need_target_features:
                if target_iter is not None:
                    try:
                        target_batch = next(target_iter)
                    except StopIteration:
                        target_iter = iter(self.calib_loader)
                        target_batch = next(target_iter)
                else:
                    if not hasattr(self, '_adv_target_iter'):
                        self._adv_target_iter = iter(self.calib_loader)
                    try:
                        target_batch = next(self._adv_target_iter)
                    except StopIteration:
                        self._adv_target_iter = iter(self.calib_loader)
                        target_batch = next(self._adv_target_iter)
                if len(target_batch) >= 2:
                    x_t, y_t = target_batch[0].to(self.device), target_batch[1].to(self.device)
                else:
                    x_t = target_batch[0].to(self.device)
                    y_t = None
                _, cls_feat_t, _ = self.global_model(x_t)
            
            # 统一获取源域特征
            if self.config.USE_DEEP_CORAL or use_adv:
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(self.val_loader)
                    source_batch = next(source_iter)
                if len(source_batch) >= 2:
                    x_s, y_s = source_batch[0].to(self.device), source_batch[1].to(self.device)
                else:
                    x_s = source_batch[0].to(self.device)
                    y_s = None
                _, cls_feat_s, _ = self.global_model(x_s)
            
            # 深度CORAL损失（统一使用 cls_feat，与域对抗训练保持特征空间一致）
            if target_iter is not None and self.config.USE_DEEP_CORAL and cls_feat_s is not None and cls_feat_t is not None:
                if self.config.CORAL_CLASS_CONDITIONAL and y_s is not None and y_t is not None:
                    from utils import deep_coral_loss_class_conditional
                    coral_loss = deep_coral_loss_class_conditional(cls_feat_s, y_s, cls_feat_t, y_t, self.config.NUM_CLASSES)
                else:
                    coral_loss = deep_coral_loss(cls_feat_s, cls_feat_t)
            
            # ---- 域对抗训练（Wasserstein GAN-GP + 梯度反转层）----
            # 使用 cls_feat（L2归一化后），与分类决策直接相关
            # n_critic 次判别器更新 + 梯度惩罚，充分训练判别器
            if use_adv and cls_feat_s is not None and cls_feat_t is not None:
                n_critic = getattr(self.config, 'ADV_CRITIC_ITERS', 3)
                gp_lambda = getattr(self.config, 'ADV_GRADIENT_PENALTY', 10.0)
                
                for critic_step in range(n_critic):
                    self.disc_optimizer.zero_grad()
                    
                    if (self.config.ADV_CLASS_CONDITIONAL and 
                        y_s is not None and y_t is not None):
                        disc_loss = torch.tensor(0.0, device=self.device)
                        gp_loss = torch.tensor(0.0, device=self.device)
                        valid_classes = 0
                        for c in range(self.config.NUM_CLASSES):
                            src_mask = (y_s == c)
                            tgt_mask = (y_t == c)
                            if src_mask.sum() > 0 and tgt_mask.sum() > 0:
                                d_src = self.domain_discriminator(cls_feat_s[src_mask].detach())
                                d_tgt = self.domain_discriminator(cls_feat_t[tgt_mask].detach())
                                w_dist_c = d_src.mean() - d_tgt.mean()
                                disc_loss -= w_dist_c
                                valid_classes += 1
                                if gp_lambda > 0:
                                    gp_c = self._compute_gradient_penalty(
                                        cls_feat_s[src_mask].detach(),
                                        cls_feat_t[tgt_mask].detach()
                                    )
                                    gp_loss += gp_c
                        if valid_classes > 0:
                            disc_loss = disc_loss / valid_classes
                            if gp_lambda > 0:
                                disc_loss = disc_loss + gp_lambda * gp_loss / valid_classes
                    else:
                        d_src_detach = self.domain_discriminator(cls_feat_s.detach())
                        d_tgt_detach = self.domain_discriminator(cls_feat_t.detach())
                        disc_loss = -(d_src_detach.mean() - d_tgt_detach.mean())
                        if gp_lambda > 0:
                            gp = self._compute_gradient_penalty(
                                cls_feat_s.detach(), cls_feat_t.detach()
                            )
                            disc_loss = disc_loss + gp_lambda * gp
                    
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.domain_discriminator.parameters(), max_norm=1.0)
                    self.disc_optimizer.step()
                
                # GRL反转梯度：特征提取器最小化域距离
                if (self.config.ADV_CLASS_CONDITIONAL and 
                    y_s is not None and y_t is not None):
                    adv_loss = torch.tensor(0.0, device=self.device)
                    valid_classes = 0
                    for c in range(self.config.NUM_CLASSES):
                        src_mask = (y_s == c)
                        tgt_mask = (y_t == c)
                        if src_mask.sum() > 0 and tgt_mask.sum() > 0:
                            feat_s_grl = self.grl(cls_feat_s[src_mask])
                            feat_t_grl = self.grl(cls_feat_t[tgt_mask])
                            w_dist_c = (self.domain_discriminator(feat_s_grl).mean() 
                                       - self.domain_discriminator(feat_t_grl).mean())
                            adv_loss += w_dist_c
                            valid_classes += 1
                    if valid_classes > 0:
                        adv_loss = self.config.LAMBDA_ADV_DOMAIN * adv_loss / valid_classes
                else:
                    feat_s_grl = self.grl(cls_feat_s)
                    feat_t_grl = self.grl(cls_feat_t)
                    w_dist_grl = (self.domain_discriminator(feat_s_grl).mean() 
                                 - self.domain_discriminator(feat_t_grl).mean())
                    adv_loss = self.config.LAMBDA_ADV_DOMAIN * w_dist_grl
            
            # ---- MMD对齐损失项（新方案）----
            mmd_alignment_loss = (self.config.LAMBDA_GLOBAL_MMD * global_mmd_loss +
                                self.config.LAMBDA_CLASS_MMD * class_mmd_loss +
                                self.config.LAMBDA_PROTO_ANCHOR * proto_anchor_loss +
                                self.config.LAMBDA_STAGE_MMD * stage_mmd_loss)
            
            # 域对抗启用时的损失组合
            # 保留 consistency（降权）和 align_reg（锚定原型），丢弃冗余的 mmd_proto 和 residual
            if self.domain_discriminator is not None:
                reduced_consist_weight = 0.2 * self.config.LAMBDA_CONSISTENCY
                total_loss = (val_loss + 
                            self.config.LAMBDA_PROTO * proto_loss + 
                            reduced_consist_weight * consist_loss +
                            self.config.LAMBDA_MMD * align_reg_loss +
                            self.config.LAMBDA_DEEP_CORAL * coral_loss +
                            adv_loss +
                            mmd_alignment_loss)
            else:
                total_loss = (val_loss + 
                            self.config.LAMBDA_PROTO * proto_loss + 
                            self.config.LAMBDA_CONSISTENCY * consist_loss + 
                            (self.config.LAMBDA_RES * residual_loss if self.use_decoupling else 0) +
                            self.config.LAMBDA_MMD * align_reg_loss +
                            self.config.LAMBDA_PROTO_MMD * mmd_proto_loss +
                            self.config.LAMBDA_DEEP_CORAL * coral_loss +
                            adv_loss +
                            mmd_alignment_loss
                            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, max_norm=5.0)
            self.optimizer.step()
        
        # 计算对齐损失用于日志
        align_loss = self._compute_align_loss()
        self.logger.info(f"Server optimization done for {K} steps. Loss: {total_loss.item():.4f}, Align Loss: {align_loss.item():.4f}")
        self.logger.info(f"Server optimization: val_loss={val_loss.item():.4f}, proto_loss={proto_loss.item():.4f}, align_reg={align_reg_loss.item():.4f}, mmd_proto={mmd_proto_loss.item():.4f}, coral={coral_loss.item():.4f}, adv={adv_loss.item():.4f}")
        # 新方案MMD对齐损失日志（处理可能是float或tensor的情况）
        def get_loss_val(loss):
            return loss.item() if hasattr(loss, 'item') else float(loss)
        self.logger.info(f"MMD Alignment Losses: global={get_loss_val(global_mmd_loss):.4f}, class={get_loss_val(class_mmd_loss):.4f}, proto_anchor={get_loss_val(proto_anchor_loss):.4f}, stage={get_loss_val(stage_mmd_loss):.4f}")

        # 标记原型缓存失效（因为原型参数已被优化器更新）
        self._proto_cache_dirty = True
        
        return align_loss.item()

    def _extract_features(self, loader):
        """从数据加载器中提取特征
        
        Args:
            loader: 数据加载器
            
        Returns:
            tuple: (特征张量, 标签张量, 阶段张量) 或 (特征张量, None, None) 当无标签时
        """
        all_feats, all_labels, all_phases = [], [], []
        has_labels = True
        max_batches = 20
        batch_count = 0
        
        for batch in loader:
            if batch_count >= max_batches:
                break
            if len(batch) == 4:
                x, y_cls, _, y_p = batch
            elif len(batch) == 2:
                x, y_cls = batch
                y_p = None
            elif len(batch) == 1:
                x = batch[0]
                y_cls = None
                y_p = None
                has_labels = False
            else:
                x, y_cls = batch[0], batch[1] if len(batch) > 1 else (batch[0], None)
                y_p = batch[3] if len(batch) > 3 else None
            
            x = x.to(self.device)
            if y_cls is not None:
                y_cls = y_cls.to(self.device)
            if y_p is not None:
                y_p = y_p.to(self.device)
            
            _, feats, _ = self.global_model(x)
            all_feats.append(feats)
            if y_cls is not None:
                all_labels.append(y_cls)
            else:
                all_labels.append(torch.zeros(len(x), dtype=torch.long, device=self.device))
            if y_p is not None:
                all_phases.append(y_p)
            else:
                all_phases.append(torch.zeros(len(x), dtype=torch.long, device=self.device))
            
            batch_count += 1
        
        if len(all_feats) > 0:
            feats_cat = torch.cat(all_feats, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            phases_cat = torch.cat(all_phases, dim=0)
            if not has_labels:
                return feats_cat, None, phases_cat
            return feats_cat, labels_cat, phases_cat
        return None, None, None

    def _compute_mmd_alignment_loss(self):
        """计算MMD对齐损失（新方案）
        
        包含：
        1. 全局MMD损失：源域和目标域整体分布对齐
        2. 类别条件MMD损失：每类内部的域对齐
        3. 原型锚定正则：约束类别中心向语义原型靠拢
        4. 阶段间MMD损失：同一类别不同阶段的分布一致性
        
        Returns:
            tuple: (全局MMD损失, 类别条件MMD损失, 原型锚定损失, 阶段间MMD损失)
        """
        global_mmd_loss = torch.tensor(0.0, device=self.device)
        class_mmd_loss = torch.tensor(0.0, device=self.device)
        proto_anchor_loss = torch.tensor(0.0, device=self.device)
        stage_mmd_loss = torch.tensor(0.0, device=self.device)
        
        # 检查是否需要计算MMD对齐损失
        if not self.config.USE_MMD_ALIGNMENT or self.calib_loader is None:
            return global_mmd_loss, class_mmd_loss, proto_anchor_loss, stage_mmd_loss
        
        # 提取源域验证集特征
        src_feats, src_labels, src_phases = self._extract_features(self.val_loader)
        if src_feats is None:
            return global_mmd_loss, class_mmd_loss, proto_anchor_loss, stage_mmd_loss
        
        # 提取目标域校准集特征
        tgt_feats, tgt_labels, tgt_phases = self._extract_features(self.calib_loader)
        if tgt_feats is None:
            return global_mmd_loss, class_mmd_loss, proto_anchor_loss, stage_mmd_loss
        
        # 无标签模式: 仅计算全局MMD + stage_mmd, 跳过类别条件项
        use_class_conditional = (tgt_labels is not None and src_labels is not None)
        
        # 1. 全局MMD损失（整体分布对齐）
        global_mmd_loss = compute_mmd(src_feats, tgt_feats) ** 2
        
        # 2. 类别条件MMD损失（类内对齐）- 仅标签可用时计算
        if use_class_conditional:
            class_count = 0
            for c in range(self.config.NUM_CLASSES):
                src_mask = (src_labels == c)
                tgt_mask = (tgt_labels == c)
                if src_mask.sum() > 0 and tgt_mask.sum() > 0:
                    class_mmd_loss += compute_mmd(src_feats[src_mask], tgt_feats[tgt_mask]) ** 2
                    class_count += 1
            if class_count > 0:
                class_mmd_loss = class_mmd_loss / self.config.NUM_CLASSES
        
        # 3. 原型锚定正则 - 仅标签可用时计算
        if use_class_conditional:
            anchor_count = 0
            for c in range(self.config.NUM_CLASSES):
                src_mask = (src_labels == c)
                tgt_mask = (tgt_labels == c)
                if src_mask.sum() > 0 and tgt_mask.sum() > 0:
                    mu_src = src_feats[src_mask].mean(dim=0)
                    mu_tgt = tgt_feats[tgt_mask].mean(dim=0)
                    proto_key = f"({c},0)"
                    if proto_key in self.semantic_protos:
                        mu_sem = self.semantic_protos[proto_key]
                        proto_anchor_loss += torch.norm(mu_src - mu_sem, p=2).pow(2)
                        proto_anchor_loss += torch.norm(mu_tgt - mu_sem, p=2).pow(2)
                        anchor_count += 1
            if anchor_count > 0:
                proto_anchor_loss /= anchor_count
        
        # 4. 阶段间MMD损失（同一类别不同阶段的分布一致性）
        stage_count = 0
        for c in range(self.config.NUM_CLASSES):
            # 获取该类别的所有阶段特征
            class_mask = (src_labels == c)
            if class_mask.sum() < 2:
                continue
            
            class_feats = src_feats[class_mask]
            class_phases = src_phases[class_mask]
            
            # 获取所有不同的阶段
            unique_phases = torch.unique(class_phases)
            if len(unique_phases) < 2:
                continue
            
            # 计算两两阶段之间的MMD
            phase_list = unique_phases.tolist()
            for i in range(len(phase_list)):
                for j in range(i + 1, len(phase_list)):
                    phase_i_mask = (class_phases == phase_list[i])
                    phase_j_mask = (class_phases == phase_list[j])
                    
                    if phase_i_mask.sum() > 0 and phase_j_mask.sum() > 0:
                        stage_mmd_loss += compute_mmd(class_feats[phase_i_mask], class_feats[phase_j_mask]) ** 2
                        stage_count += 1
        
        if stage_count > 0:
            stage_mmd_loss /= stage_count
        
        return global_mmd_loss, class_mmd_loss, proto_anchor_loss, stage_mmd_loss

    def compute_drift_and_K(self, client_Fs: List[torch.Tensor]) -> int:
        """计算数据漂移程度并确定优化步数
        
        根据客户端上传的特征计算数据漂移程度，并根据漂移程度动态调整服务器优化步数
        
        Args:
            client_Fs: 客户端特征列表
            
        Returns:
            服务器优化步数
        """
        if not client_Fs:
            return self.config.SERVER_OPT_STEPS_BASE
        
        current_F = torch.stack(client_Fs).mean(dim=0)  # 平均特征
        
        if self.global_feature_Fg is None:
            self.global_feature_Fg = current_F
            return self.config.SERVER_OPT_STEPS_BASE
        
        # === 关键修复：特征归一化后计算 drift，并限制最大值 ===
        current_F_norm = F.normalize(current_F, dim=0, p=2)
        global_F_norm = F.normalize(self.global_feature_Fg, dim=0, p=2)
        drift = torch.norm(current_F_norm - global_F_norm, p=2).pow(2).item()
        drift = min(drift, 10.0)  # 硬限制
        
        # EMA 更新全局特征
        alpha = self.config.EMA_ALPHA_F
        self.global_feature_Fg = alpha * self.global_feature_Fg + (1 - alpha) * current_F
        
        K = self.config.SERVER_OPT_STEPS_BASE + int(self.config.DRIFT_GAMMA * drift)
        K = min(K, 100)  # 基础部分上限【从50放宽至100，配合大K步数实验】
        
        domain_adapt_bonus = getattr(self.config, 'DOMAIN_ADAPT_K_BONUS', 10)
        if self.domain_discriminator is not None or self.calib_loader is not None:
            K += domain_adapt_bonus
            K = min(K, 250)  # 总步数硬上限
            self.logger.info(f"Drift={drift:.4f}, K={K} (含域适应加成+{domain_adapt_bonus})")
        else:
            K = min(K, 120)  # 无域适应时的上限
            self.logger.info(f"Drift={drift:.4f}, K={K}")
        return K
    
    def _compute_gradient_penalty(self, feat_src: torch.Tensor, feat_tgt: torch.Tensor) -> torch.Tensor:
        src_size = feat_src.size(0)
        tgt_size = feat_tgt.size(0)
        min_size = min(src_size, tgt_size)
        if min_size < 2:
            return torch.tensor(0.0, device=self.device)
        
        feat_src_sample = feat_src[:min_size]
        feat_tgt_sample = feat_tgt[:min_size]
        
        alpha = torch.rand(min_size, 1, device=self.device)
        interpolates = alpha * feat_src_sample + (1 - alpha) * feat_tgt_sample
        interpolates = interpolates.detach().requires_grad_(True)
        
        disc_interpolates = self.domain_discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_norm = gradients.view(min_size, -1).norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
        return gradient_penalty
    
    def compute_generic_residual(self) -> torch.Tensor:
        """计算所有训练客户端设备残差的加权平均，作为通用残差"""
        if not self.device_residuals:
            return torch.zeros(self.config.HIDDEN_DIM2, device=self.device)
        residuals = [r for r in self.device_residuals.values()]
        return torch.stack(residuals).mean(dim=0).detach()

    def _compute_proto_spreads_from_calib(self):
        r"""从校准集计算每原型的特征散布度，用于自适应温度推理
        
        对校准集中每个样本，计算其 cls_feat 与对应 (类,阶段) 原型的
        余弦相似度标准差，作为该原型"特征紧致度"的估计。
        
        散布度定义：spread(c,p) = 1 - mean_{x∈calib} cos(x, μ_{c,p})
        散布大 → 该原型类内方差大 → 需要更高温度（更软的分配）
        
        数学背景：
        设 d_cp = 1 - cos(f, μ_{c,p})，对应该原型的角偏差。
        在超球面上，cos 距离与欧氏距离单调相关但非线性。
        用 1-cos 而非 L2 距离是为了保持在归一化特征空间内操作。
        
        Returns:
            bool: 是否成功计算
        """
        if self.calib_loader is None or not self.semantic_protos:
            self._proto_temperatures = None
            return False
        
        self.global_model.eval()
        proto_keys = list(self.semantic_protos.keys())
        # 建立 str_key → 索引 的映射
        key_to_idx = {k: i for i, k in enumerate(proto_keys)}
        spread_sums = defaultdict(float)
        spread_counts = defaultdict(int)
        
        max_samples = getattr(self.config, 'CORAL_MAX_SAMPLES', 500)
        samples_processed = 0
        
        with torch.no_grad():
            for batch in self.calib_loader:
                if samples_processed >= max_samples:
                    break
                if len(batch) >= 4:
                    x, y_cls, _, y_p = batch
                elif len(batch) >= 2:
                    x, y_cls = batch[0], batch[1]
                    y_p = torch.zeros_like(y_cls)
                else:
                    continue
                x, y_cls, y_p = x.to(self.device), y_cls.to(self.device), y_p.to(self.device)
                _, cls_feat, _ = self.global_model(x)
                
                for i in range(len(x)):
                    str_key = f"({y_cls[i].item()},{y_p[i].item()})"
                    if str_key in key_to_idx:
                        proto = self.semantic_protos[str_key]
                        cos_sim = F.cosine_similarity(cls_feat[i:i+1], proto.unsqueeze(0)).item()
                        spread_sums[str_key] += (1.0 - cos_sim)
                        spread_counts[str_key] += 1
                samples_processed += len(x)
        
        self.global_model.train()
        
        if not spread_sums:
            self._proto_temperatures = None
            return False
        
        self._proto_spreads.clear()
        for str_key in proto_keys:
            cnt = spread_counts.get(str_key, 0)
            avg_spread = spread_sums[str_key] / cnt if cnt > 0 else 0.5
            self._proto_spreads[str_key] = avg_spread
        
        self._proto_temperatures_dirty = True
        self.logger.debug(f"Proto spreads computed from calib: "
                         f"{ {k: round(v,4) for k,v in list(self._proto_spreads.items())[:4]} }...")
        return True

    def get_adaptive_temperatures(self, proto_keys: List[str]) -> Optional[torch.Tensor]:
        """获取每原型的自适应温度向量
        
        温度计算公式:
        τ_{c,p} = clamp(τ_base × (1 + β × (spread_{c,p}/spr̄ead - 1)), τ_min, τ_max)
        
        其中 spr̄ead 为所有原型的平均散布。
        散布高于平均的原型获得更高温度（更软、更不自信的分配）。
        
        Args:
            proto_keys: 原型键列表 (按 proto_matrix 行序)
            
        Returns:
            温度向量 (K,), 若未启用自适应温度则返回 None
        """
        if not getattr(self.config, 'USE_ADAPTIVE_TEMPERATURE', False):
            return None
        
        if self._proto_temperatures_dirty or self._proto_temperatures is None:
            if not self._proto_spreads:
                return None
            
            base_temp = self.config.SOFT_AGG_TEMPERATURE
            beta = getattr(self.config, 'ADAPTIVE_TEMP_BETA', 0.5)
            t_min = getattr(self.config, 'ADAPTIVE_TEMP_MIN', 0.1)
            t_max = getattr(self.config, 'ADAPTIVE_TEMP_MAX', 2.0)
            
            spreads = []
            for k in proto_keys:
                spreads.append(self._proto_spreads.get(k, 0.5))
            spread_tensor = torch.tensor(spreads, device=self.device)
            mean_spread = spread_tensor.mean().clamp(min=1e-6)
            
            ratios = spread_tensor / mean_spread
            temperatures = base_temp * (1.0 + beta * (ratios - 1.0))
            temperatures = temperatures.clamp(min=t_min, max=t_max)
            
            self._proto_temperatures = temperatures
            self._proto_temperatures_dirty = False
            self.logger.info(
                f"Adaptive temperatures: base={base_temp:.3f}, "
                f"range=[{temperatures.min().item():.3f}, {temperatures.max().item():.3f}], "
                f"mean_spread={mean_spread.item():.4f}"
            )
        
        return self._proto_temperatures

    def compute_selective_weights(self, client_mus: List[Dict], base_weights: torch.Tensor, current_round: int,
                                  client_Fs: List[torch.Tensor] = None) -> torch.Tensor:
        """
        基于原型相似度 + 目标域特征相似度计算选择性聚合权重
        
        核心改进 (#方案1: 目标感知聚合):
        - 原有: 仅按客户端局部原型与全局语义原型的余弦相似度降权
        - 新增: 额外乘以客户端特征均值与校准集特征均值的余弦相似度
          使得特征分布接近目标域的源客户端获得更高权重
        
        数学原理:
        w_i = w_base_i · (proto_sim_i)^{1-α} · (target_sim_i)^{α} / Z
        其中 α = TARGET_INFORMED_WEIGHT, Z 为归一化因子
        
        Args:
            client_mus: 客户端本地原型均值列表
            base_weights: 基础权重
            current_round: 当前训练轮次
            client_Fs: 客户端全局特征均值列表 (用于目标感知聚合)
            
        Returns:
            调整后的权重张量
        """
        if not self.config.USE_SELECTIVE_AGG or current_round < self.config.SELECTIVE_AGG_WARMUP:
            return base_weights
        
        # ---- 阶段1: 原型相似度 (原有逻辑 + 马氏距离可选) ----
        proto_sims = []
        use_mahal = (getattr(self.config, 'USE_MAHALANOBIS_INFERENCE', False) 
                     and self.semantic_proto_vars 
                     and len(self.semantic_proto_vars) > 0)
        feat_dim = self.config.HIDDEN_DIM2
        for mu_dict in client_mus:
            sim_sum = 0.0
            count = 0
            for key, mu_local in mu_dict.items():
                c, p = key
                str_key = f"({c},{p})"
                if str_key in self.semantic_protos:
                    mu_global = self.semantic_protos[str_key].detach()
                    if use_mahal and str_key in self.semantic_proto_vars:
                        # 马氏距离→相似度: sim = 1/(1 + d_mahal²/D)
                        # 当 d²=D 时 sim=0.5，与余弦的基线一致
                        var_vec = torch.clamp(self.semantic_proto_vars[str_key].to(self.device), 
                                              min=getattr(self.config, 'MAHALANOBIS_MIN_VAR', 0.01))
                        diff = mu_local.to(self.device) - mu_global
                        mahal_sq = (diff ** 2 / (var_vec + 1e-6)).sum()
                        sim = 1.0 / (1.0 + mahal_sq.item() / feat_dim)
                    else:
                        sim = F.cosine_similarity(mu_local.to(self.device), mu_global, dim=0).item()
                    sim_sum += sim
                    count += 1
            avg_sim = sim_sum / count if count > 0 else 0.5
            proto_sims.append(avg_sim)
        proto_scales = [max(self.config.SELECTIVE_AGG_MIN_SCALE, min(1.0, s)) for s in proto_sims]
        
        # ---- 阶段2: 目标域特征相似度 (方案1新增) ----
        if (getattr(self.config, 'USE_TARGET_INFORMED_AGG', False) 
            and client_Fs is not None 
            and self._calib_feature_mean is not None):
            target_scales = []
            calib_mean = self._calib_feature_mean.to(self.device)
            target_weight = getattr(self.config, 'TARGET_INFORMED_WEIGHT', 0.6)
            for F_i in client_Fs:
                F_i_device = F_i.to(self.device)
                target_sim = F.cosine_similarity(
                    F_i_device.view(1, -1), calib_mean.view(1, -1), dim=1
                ).item()
                target_scales.append(max(self.config.SELECTIVE_AGG_MIN_SCALE, min(1.0, target_sim)))
            
            alpha = target_weight
            combined_scales = [(1 - alpha) * ps + alpha * ts for ps, ts in zip(proto_scales, target_scales)]
            self.logger.info(
                f"Target-informed agg: proto_scales={[round(s,3) for s in proto_scales]}, "
                f"target_scales={[round(s,3) for s in target_scales]}, "
                f"combined={[round(s,3) for s in combined_scales]}"
            )
        else:
            combined_scales = proto_scales
        
        adjusted_weights = [w * s for w, s in zip(base_weights.tolist(), combined_scales)]
        total = sum(adjusted_weights) + 1e-10
        adjusted_weights = [w / total for w in adjusted_weights]
        
        return torch.tensor(adjusted_weights, device=self.device)
    
    def _compute_calib_feature_mean(self) -> Optional[torch.Tensor]:
        """计算校准集全局特征均值 (用于目标感知聚合)
        
        提取校准集中所有样本的 cls_feat (L2归一化后)，取均值作为目标域分布参考。
        首次计算后冻结缓存，后续调用直接返回缓存值。
        这防止了随着模型演化，校准特征漂移导致的相似度退化。
        
        Returns:
            校准集特征均值张量 (64维)，若校准集不可用则返回 None
        """
        if self.calib_loader is None:
            self._calib_feature_mean = None
            return None
        if self._calib_feature_mean is not None:
            return self._calib_feature_mean
        
        self.global_model.eval()
        all_feats = []
        max_samples = getattr(self.config, 'CORAL_MAX_SAMPLES', 500)
        with torch.no_grad():
            for batch in self.calib_loader:
                if len(all_feats) * self.config.BATCH_SIZE >= max_samples:
                    break
                x = batch[0].to(self.device)
                _, cls_feat, _ = self.global_model(x)
                all_feats.append(cls_feat)
        self.global_model.train()
        
        if all_feats:
            feats_cat = torch.cat(all_feats, dim=0)[:max_samples]
            self._calib_feature_mean = feats_cat.mean(dim=0)
            self.logger.debug(f"Calib feature mean computed: shape={self._calib_feature_mean.shape}, "
                            f"norm={self._calib_feature_mean.norm().item():.4f}")
        else:
            self._calib_feature_mean = None
        return self._calib_feature_mean
    
    def deployment_aggregation_round(self, client_uploads):
        """
        部署阶段联邦校准：修剪均值、有界更新、异常窗口过滤
        
        Args:
            client_uploads: list of dict，每个 dict 包含 'class_feat_means', 'counts', 'confidence', 'avg_entropy'
        """
        if not client_uploads:
            return
        
        # 1. 过滤异常上传：置信度低或熵高
        min_conf = getattr(self.config, 'DEPLOY_CONF_THRESH', 0.5)
        valid_uploads = [u for u in client_uploads 
                         if u['confidence'] > min_conf 
                         and u.get('avg_entropy', 1.0) < 0.8]
        if not valid_uploads:
            self.logger.info("No valid uploads after filtering")
            return

        # 2. 对每个 (c,p) 收集所有客户端的本地均值
        proto_dict = defaultdict(list)
        for u in valid_uploads:
            for key, mu in u['class_feat_means'].items():
                proto_dict[key].append(mu.to(self.device))

        # 3. 修剪均值求全局参考 (马氏距离或L2范数)
        trim_fraction = getattr(self.config, 'TRIMMED_MEAN_FRACTION', 0.2)
        use_mahal = getattr(self.config, 'USE_MAHALANOBIS_TRIM', False)
        global_ref = {}
        for key, mu_list in proto_dict.items():
            if len(mu_list) < 3:
                global_ref[key] = torch.stack(mu_list).mean(dim=0)
            else:
                stacked = torch.stack(mu_list)
                k = max(1, int(len(mu_list) * trim_fraction))
                
                if use_mahal and self.semantic_proto_vars:
                    # 马氏距离修剪: 利用类内方差缩放各维度
                    str_key = f"({key[0]},{key[1]})" if isinstance(key, tuple) else key
                    if str_key in self.semantic_proto_vars and str_key in self.semantic_protos:
                        vars_vec = self.semantic_proto_vars[str_key].to(self.device)
                        ref_mu = self.semantic_protos[str_key].data.detach()
                        diff = stacked - ref_mu.unsqueeze(0)
                        mahal_sq = (diff ** 2 / (vars_vec.unsqueeze(0) + 1e-6)).sum(dim=1)
                        _, idx = mahal_sq.sort()
                    else:
                        norms = stacked.norm(dim=1)
                        _, idx = norms.sort()
                else:
                    # L2范数修剪 (原方案)
                    norms = stacked.norm(dim=1)
                    _, idx = norms.sort()
                
                trimmed = stacked[idx[:len(mu_list)-k]] if len(idx) > k else stacked[idx]
                global_ref[key] = trimmed.mean(dim=0)

        # 4. 计算每个客户端的漂移估计，生成有界校正
        Cmax = getattr(self.config, 'CALIB_CMAX', 0.5)
        for u in valid_uploads:
            corrections = {}
            for key, mu_local in u['class_feat_means'].items():
                if key in global_ref:
                    delta = global_ref[key] - mu_local.to(self.device)
                    delta = torch.clamp(delta, -Cmax, Cmax)
                    corrections[key] = delta
            u['corrections'] = corrections

        # 5. EMA 更新全局原型（使用修剪后的参考）
        alpha = getattr(self.config, 'PROTO_EMA_ALPHA', 0.9)
        for key, ref in global_ref.items():
            str_key = f"({key[0]},{key[1]})" if isinstance(key, tuple) else key
            if str_key in self.semantic_protos:
                self.semantic_protos[str_key].data = alpha * self.semantic_protos[str_key].data + (1 - alpha) * ref

        # 6. 保存校正量供客户端拉取
        self._latest_corrections = {u['client_id']: u['corrections'] for u in valid_uploads}
        self._proto_cache_dirty = True
        
        self.logger.info(f"Federated calibration completed: {len(global_ref)} protos updated")
    
