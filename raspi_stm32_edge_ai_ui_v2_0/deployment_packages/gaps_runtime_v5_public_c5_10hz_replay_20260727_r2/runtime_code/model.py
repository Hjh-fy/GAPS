"""
FedGas模型：时序CNN + GRU + Attention池化，输出分类、回归和64维特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class MixStyle(nn.Module):
    """MixStyle 激活层，用于时序特征图 (B, C, T)"""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or np.random.rand() > self.p:
            return x
        
        B, C, T = x.shape
        if B < 2:
            return x  # batch size 至少为 2
        
        # 随机打乱 batch 顺序，用于混合
        idx = torch.randperm(B, device=x.device)
        
        # 计算每个样本的通道均值和标准差 (风格统计量)
        mu = x.mean(dim=[2], keepdim=True)  # (B, C, 1)
        var = x.var(dim=[2], keepdim=True, unbiased=False)  # (B, C, 1)
        sig = (var + self.eps).sqrt()
        
        mu_mix = mu[idx]  # (B, C, 1)
        sig_mix = sig[idx]
        
        # 从 Beta 分布采样混合系数
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        lam = torch.tensor(lam, device=x.device)
        
        # 混合统计量
        mu_new = lam * mu + (1 - lam) * mu_mix
        sig_new = lam * sig + (1 - lam) * sig_mix
        
        # 实例归一化 + 风格注入
        x_norm = (x - mu) / sig
        return x_norm * sig_new + mu_new


class TemporalTransformerEncoder(nn.Module):
    """轻量时序 Transformer 编码器，用于回归分支的长程轨迹建模"""
    def __init__(self, seq_len=100, d_model=48, nhead=4, num_layers=2, ff_dim=96, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x)
        return x


# ===============================
# Depthwise Separable Conv
# ===============================
class DSConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=dilation, dilation=dilation, groups=in_ch
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ===============================
# TCN Block（使用 InstanceNorm 替代 GroupNorm）
# ===============================
class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, tcn_norm: str = 'instance'):
        super().__init__()
        self.conv = DSConv1d(in_ch, out_ch, kernel_size=3, dilation=dilation)
        # 关键修改：InstanceNorm1d 消除 batch 依赖，适合跨设备泛化
        norm_key = str(tcn_norm).lower()
        if norm_key == 'instance':
            self.norm = nn.InstanceNorm1d(out_ch, affine=True)
        elif norm_key == 'batch':
            self.norm = nn.BatchNorm1d(out_ch)
        elif norm_key in ('none', 'identity'):
            self.norm = nn.Identity()
        else:
            raise ValueError(f'Unsupported TCN norm: {tcn_norm}')
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.norm(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ===============================
# 主模型（泛化优化版）
# ===============================
class FedGasBaseModel(nn.Module):
    def __init__(self, num_classes=4, num_sensors=8, feat_dim=64, 
                 use_mixstyle=False, mixstyle_prob=0.5, mixstyle_alpha=0.1, noise_std=0.01,
                 encoder_type='tcn', transformer_d_model=48, transformer_nhead=4,
                 transformer_num_layers=2, transformer_ff_dim=96,
                 use_cls_proj=False, tcn_norm='instance'):
        super().__init__()
        self.encoder_type = encoder_type
        self.use_cls_proj = use_cls_proj
        self.tcn_norm = tcn_norm

        if encoder_type == 'transformer':
            # Transformer编码器路径：传感器交互建模
            self.transformer_encoder = SensorInteractionTransformerEncoder(
                num_sensors=num_sensors, d_model=transformer_d_model,
                nhead=transformer_nhead, num_layers=transformer_num_layers,
                dim_feedforward=transformer_ff_dim, seq_len=100,
                dropout=0.1, feat_dim=feat_dim
            )
            # 保留通道注意力作为兼容层（非Transformer路径使用）
            self.channel_attn = None
            self.tcn = None
            self.tcn_layers = None
            self.self_attn = None
            self.attn_linear = None
            self.feat_proj = None  # Transformer自带feat_proj
            self.cls_proj = nn.Linear(transformer_d_model, feat_dim) if use_cls_proj else None
        else:
            # TCN编码器路径（原有逻辑）
            self.tcn1 = TCNBlock(num_sensors, 32, dilation=1, tcn_norm=tcn_norm)
            self.tcn2 = TCNBlock(32, 48, dilation=2, tcn_norm=tcn_norm)
            self.tcn3 = TCNBlock(48, 48, dilation=4, tcn_norm=tcn_norm)
            self.tcn_layers = nn.ModuleList([self.tcn1, self.tcn2, self.tcn3])
            self.tcn = nn.Sequential(self.tcn1, self.tcn2, self.tcn3)

            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(8, 8),
                nn.Sigmoid()
            )

            self.self_attn = nn.MultiheadAttention(embed_dim=48, num_heads=4, batch_first=True)
            self.attn_linear = nn.Linear(48, 1)
            self.feat_proj = nn.Linear(48, feat_dim)
            self.cls_proj = nn.Linear(48, feat_dim) if use_cls_proj else None
            self.transformer_encoder = None
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(feat_dim, num_classes)
        
        # MixStyle 配置
        self.use_mixstyle = use_mixstyle
        if use_mixstyle:
            self.mixstyle = MixStyle(p=mixstyle_prob, alpha=mixstyle_alpha)
        
        # 特征噪声标准差
        self.noise_std = noise_std
        
        self._init_weights()

    @property
    def has_regression(self):
        """返回模型是否具有回归能力"""
        return hasattr(self, 'reg_heads') or hasattr(self, 'conc_directions')

    def get_regression_params(self):
        """返回回归相关参数列表，供微调和校准使用"""
        params = []
        if hasattr(self, 'conc_directions'):
            params.append(self.conc_directions)
            params.append(self.conc_scale)
            params.append(self.conc_bias)
            if self.use_residual:
                for head in self.residual_heads:
                    params.extend(head.parameters())
        elif hasattr(self, 'reg_heads'):
            params.extend(self.reg_heads.parameters())
            if hasattr(self, 'proto_scale'):
                params.append(self.proto_scale)
            if hasattr(self, 'proto_bias'):
                params.append(self.proto_bias)
        return params

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def attention_pooling(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        weights = self.attn_linear(attn_out)
        weights = torch.softmax(weights, dim=1)
        return (weights * x).sum(dim=1)

    def forward(self, x):
        # x: (B, 100, 8) — (batch, seq_len, num_sensors)
        if self.encoder_type == 'transformer':
            # Transformer路径：直接处理(B, 100, 8)，内置通道注意力和池化
            if self.use_cls_proj and self.cls_proj is not None:
                temporal = self.transformer_encoder.encode_sequence(x)
                pooled = self.transformer_encoder.pool_sequence(temporal)
                raw = self.cls_proj(pooled)
            else:
                raw = self.transformer_encoder(x)  # (B, feat_dim)
            if self.training:
                raw = raw + torch.randn_like(raw) * self.noise_std
            raw = self.dropout(raw)
            cls_feat = F.normalize(raw, dim=1, p=2)
            logits = self.classifier(cls_feat)
            reg_feat = raw
            return logits, cls_feat, reg_feat

        # TCN路径（原有逻辑）
        x = x.permute(0, 2, 1)          # (B, 8, 100)
        x = self.tcn(x)                 # (B, C, T)  例如 (B, 48, 100)
        if self.use_mixstyle:
            x = self.mixstyle(x)        # 风格混合，形状不变
        x = x.permute(0, 2, 1)          # (B, T, C)
        pooled = self.attention_pooling(x)  # (B, 48)
        proj = self.cls_proj if self.use_cls_proj and self.cls_proj is not None else self.feat_proj
        raw = proj(pooled)    # (B, 64)

        # === 特征空间噪声：防止过拟合源域，保持泛化鲁棒性 ===
        if self.training:
            raw = raw + torch.randn_like(raw) * self.noise_std  # 标准差从配置中获取

        raw = self.dropout(raw)         # ← 先 dropout

        cls_feat = F.normalize(raw, dim=1, p=2)   # ← 归一化用于分类
        logits = self.classifier(cls_feat)
        reg_feat = raw                  # 保留幅度，不归一化
        return logits, cls_feat, reg_feat


# ===============================
# 共享浓度主干 (Shared Concentration Trunk)
# 设计目标: 让所有气体共享一个浓度预测主干,
#           减少独立 per-class head 的结构碎片化,
#           同时保留类别残差头做精细化修正。
#
# 输入: reg_feat (B, feat_dim) + gas_emb (B, gas_emb_dim)
# 输出: shared_pred (B, 1)
# ===============================
class SharedConcentrationTrunk(nn.Module):
    """共享浓度预测主干

    所有气体类别共享同一个 MLP 主干,
    通过 gas_embedding 注入类别信息,
    输出共享的归一化浓度基础预测。

    参数:
        feat_dim: 回归特征维度 (默认 64)
        gas_emb_dim: 气体嵌入维度 (默认 16)
        shared_dim: 共享主干隐藏层维度 (默认 128)
        dropout: Dropout 比例
    """
    def __init__(self, feat_dim: int = 64, gas_emb_dim: int = 16,
                 shared_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        input_dim = feat_dim + gas_emb_dim
        mid_dim = shared_dim // 2

        self.fc1 = nn.Linear(input_dim, shared_dim)
        self.ln1 = nn.LayerNorm(shared_dim)
        self.fc2 = nn.Linear(shared_dim, mid_dim)
        self.ln2 = nn.LayerNorm(mid_dim)
        self.fc3 = nn.Linear(mid_dim, 1)

        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(dropout)

        # 残差投影: 将输入投影到各隐藏层维度
        self.proj1 = nn.Linear(input_dim, shared_dim) if input_dim != shared_dim else nn.Identity()
        self.proj2 = nn.Linear(input_dim, mid_dim) if input_dim != mid_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, feat_dim + gas_emb_dim)
        identity1 = self.proj1(x)
        identity2 = self.proj2(x)

        out = self.selu(self.ln1(self.fc1(x)))
        out = out + identity1
        out = self.dropout(out)

        out = self.selu(self.ln2(self.fc2(out)) + identity2)
        out = self.fc3(out)  # (B, 1)
        return out


# ===============================
# 类别残差头 (Class Residual Head)
# 设计目标: 每个气体类别有一个轻量残差头,
#           学习类别特定的浓度偏移 delta,
#           与共享主干的 shared_pred 相加得到最终预测。
#
# 输入: reg_feat (B, feat_dim)
# 输出: delta (B, 1), 表示类别特定的浓度修正量
# ===============================
class ClassResidualHead(nn.Module):
    """类别特定残差头

    轻量级 MLP, 学习类别特定的浓度偏移 delta,
    最终预测 = shared_pred + delta。

    参数:
        feat_dim: 回归特征维度 (默认 64)
        depth: 残差头深度 (默认 2, 可选 1 或 3)
        dropout: Dropout 比例
    """
    def __init__(self, feat_dim: int = 64, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.depth = depth

        if depth == 1:
            # 最轻量: 单层线性
            self.fc1 = nn.Linear(feat_dim, 1)
        elif depth == 3:
            self.fc1 = nn.Linear(feat_dim, 64)
            self.ln1 = nn.LayerNorm(64)
            self.fc2 = nn.Linear(64, 32)
            self.ln2 = nn.LayerNorm(32)
            self.fc3 = nn.Linear(32, 1)
            self.proj1 = nn.Linear(feat_dim, 64) if feat_dim != 64 else nn.Identity()
            self.proj2 = nn.Linear(feat_dim, 32) if feat_dim != 32 else nn.Identity()
            self.selu = nn.SELU()
            self.dropout = nn.AlphaDropout(dropout)
        else:
            # depth=2 (默认): 两层隐藏
            self.fc1 = nn.Linear(feat_dim, 32)
            self.ln1 = nn.LayerNorm(32)
            self.fc2 = nn.Linear(32, 1)
            self.proj1 = nn.Linear(feat_dim, 32) if feat_dim != 32 else nn.Identity()
            self.selu = nn.SELU()
            self.dropout = nn.AlphaDropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depth == 1:
            return self.fc1(x)

        if self.depth == 3:
            identity1 = self.proj1(x)
            identity2 = self.proj2(x)
            out = self.selu(self.ln1(self.fc1(x)))
            out = out + identity1
            out = self.dropout(out)
            out = self.selu(self.ln2(self.fc2(out)) + identity2)
            out = self.fc3(out)
            return out

        # depth=2
        identity1 = self.proj1(x)
        out = self.selu(self.ln1(self.fc1(x)))
        out = out + identity1
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class RegHead(nn.Module):
    """深度残差回归头，支持可配置深度和分位数输出模式"""
    def __init__(self, feat_dim, depth=3, quantile_mode=False):
        super().__init__()
        self.quantile_mode = quantile_mode
        out_dim = 3 if quantile_mode else 1  # 0.25, 0.5, 0.75 分位数
        
        if depth == 2:
            self._build_depth2(feat_dim, out_dim)
        elif depth == 4:
            self._build_depth4(feat_dim, out_dim)
        else:
            self._build_depth3(feat_dim, out_dim)

    def _build_depth2(self, feat_dim, out_dim):
        self.fc1 = nn.Linear(feat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(32)
        self.dropout = nn.Dropout(0.1)
        self.proj1 = nn.Linear(feat_dim, 64) if feat_dim != 64 else nn.Identity()
        self.proj2 = nn.Linear(feat_dim, 32) if feat_dim != 32 else nn.Identity()
        self.depth = 2

    def _build_depth3(self, feat_dim, out_dim):
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, out_dim)
        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(0.1)
        self.proj1 = nn.Linear(feat_dim, 128) if feat_dim != 128 else nn.Identity()
        self.proj2 = nn.Linear(feat_dim, 64) if feat_dim != 64 else nn.Identity()
        self.proj3 = nn.Linear(feat_dim, 32) if feat_dim != 32 else nn.Identity()
        self.depth = 3

    def _build_depth4(self, feat_dim, out_dim):
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, out_dim)
        self.selu = nn.SELU()
        self.dropout = nn.AlphaDropout(0.1)
        self.proj1 = nn.Linear(feat_dim, 256) if feat_dim != 256 else nn.Identity()
        self.proj2 = nn.Linear(feat_dim, 128) if feat_dim != 128 else nn.Identity()
        self.proj3 = nn.Linear(feat_dim, 64) if feat_dim != 64 else nn.Identity()
        self.proj4 = nn.Linear(feat_dim, 32) if feat_dim != 32 else nn.Identity()
        self.depth = 4

    def forward(self, x):
        if self.depth == 2:
            return self._forward_depth2(x)
        elif self.depth == 4:
            return self._forward_depth4(x)
        else:
            return self._forward_depth3(x)

    def _forward_depth2(self, x):
        identity1 = self.proj1(x)
        identity2 = self.proj2(x)
        out = F.relu(self.ln1(self.fc1(x)))
        out = out + identity1
        out = self.dropout(out)
        out = F.relu(self.ln2(self.fc2(out)) + identity2)
        out = self.fc3(out)
        return out

    def _forward_depth3(self, x):
        identity1 = self.proj1(x)          # (B, 128) 残差路径1
        identity2 = self.proj2(x)          # (B, 64)  残差路径2
        identity3 = self.proj3(x)          # (B, 32)  残差路径3
        out = self.selu(self.fc1(x))       # (B, 128) 主路径1
        out = out + identity1              # 残差连接1: (B, 128)
        out = self.dropout(out)
        out = self.selu(self.fc2(out))     # (B, 64)  主路径2
        out = out + identity2              # 残差连接2: (B, 64)
        out = self.dropout(out)
        out = self.selu(self.fc3(out) + identity3)  # 残差连接3: (B, 32)
        out = self.fc4(out)                # (B, out_dim)
        return out

    def _forward_depth4(self, x):
        identity1 = self.proj1(x)          # (B, 256) 残差路径1
        identity2 = self.proj2(x)          # (B, 128) 残差路径2
        identity3 = self.proj3(x)          # (B, 64)  残差路径3
        identity4 = self.proj4(x)          # (B, 32)  残差路径4
        out = self.selu(self.fc1(x))       # (B, 256) 主路径1
        out = out + identity1              # 残差连接1
        out = self.dropout(out)
        out = self.selu(self.fc2(out))     # (B, 128) 主路径2
        out = out + identity2              # 残差连接2
        out = self.dropout(out)
        out = self.selu(self.fc3(out))     # (B, 64) 主路径3
        out = out + identity3              # 残差连接3
        out = self.dropout(out)
        out = self.selu(self.fc4(out) + identity4)  # (B, 32) 残差连接4
        out = self.fc5(out)                # (B, out_dim)
        return out


class DCTResponseBranch(nn.Module):
    """Low-frequency response-shape adapter for regression features."""
    def __init__(
        self,
        num_sensors=8,
        seq_len=100,
        k=8,
        feat_dim=64,
        gamma_init=0.0,
        dropout=0.1,
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.seq_len = int(seq_len)
        self.k = int(k)
        if self.k <= 0:
            raise ValueError(f"DCT k must be positive, got {k}")
        if self.k > self.seq_len:
            raise ValueError(f"DCT k={self.k} exceeds seq_len={self.seq_len}")

        basis = self._build_dct_basis(self.seq_len, self.k, torch.device("cpu"), torch.float32)
        self.register_buffer("basis", basis, persistent=False)
        input_dim = self.num_sensors * self.k
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, feat_dim)
        self.out_norm = nn.LayerNorm(feat_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    @staticmethod
    def _build_dct_basis(seq_len, k, device, dtype):
        n = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)
        freq = torch.arange(k, device=device, dtype=dtype).unsqueeze(1)
        basis = torch.cos(torch.pi / float(seq_len) * (n + 0.5) * freq)
        basis[0] = basis[0] * (1.0 / float(seq_len)) ** 0.5
        if k > 1:
            basis[1:] = basis[1:] * (2.0 / float(seq_len)) ** 0.5
        return basis

    def forward(self, x_seq):
        x_float = x_seq.float()
        seq_len = x_float.size(1)
        if seq_len == self.basis.size(1):
            basis = self.basis.to(device=x_float.device, dtype=x_float.dtype)
        else:
            if self.k > seq_len:
                raise ValueError(f"DCT k={self.k} exceeds runtime seq_len={seq_len}")
            basis = self._build_dct_basis(seq_len, self.k, x_float.device, x_float.dtype)
        coeff = torch.matmul(x_float.transpose(1, 2), basis.t())
        coeff = coeff.reshape(coeff.size(0), -1)
        feat = self.input_norm(coeff)
        feat = self.proj(feat)
        feat = self.out_norm(feat)
        feat = self.dropout(self.act(feat))
        return self.gamma.to(dtype=feat.dtype) * feat


class MultiScaleTemporalResponseBranch(nn.Module):
    """Multi-scale temporal adapter over raw sensor responses."""
    def __init__(
        self,
        num_sensors=8,
        feat_dim=64,
        hidden_channels=16,
        kernels=(3, 7, 15, 31),
        gamma_init=0.0,
        dropout=0.1,
    ):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.hidden_channels = int(hidden_channels)
        self.kernels = tuple(int(k) for k in kernels)
        if self.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if not self.kernels:
            raise ValueError("At least one temporal kernel is required")
        for kernel in self.kernels:
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError(f"Temporal kernels must be positive odd integers, got {kernel}")

        self.input_norm = nn.InstanceNorm1d(self.num_sensors, affine=True)
        self.branches = nn.ModuleList()
        for kernel in self.kernels:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(self.num_sensors, self.hidden_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.GELU(),
                    nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                )
            )
        branch_dim = self.hidden_channels * len(self.kernels)
        self.proj = nn.Sequential(
            nn.LayerNorm(branch_dim),
            nn.Linear(branch_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    def forward(self, x_seq):
        x = x_seq.float().transpose(1, 2)
        x = self.input_norm(x)
        feat = torch.cat([branch(x) for branch in self.branches], dim=1)
        feat = self.proj(feat)
        return self.gamma.to(dtype=feat.dtype) * feat


class RegTCNResidualAdapter(nn.Module):
    """Small regression-only residual adapter on encoded TCN features."""
    def __init__(self, channels=48, kernel_size=3, gamma_init=0.0, dropout=0.05):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.depthwise = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )
        self.pointwise = nn.Conv1d(self.channels, self.channels, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    def forward(self, x):
        delta = self.depthwise(x)
        delta = self.pointwise(delta)
        delta = self.dropout(self.act(delta))
        return x + self.gamma.to(dtype=delta.dtype) * delta


# ===============================
# 跨通道比率响应分支 (Cross-Channel Ratio Response Branch)
# 设计目标: 从原始传感器窗口提取每通道统计量和跨通道比率,
#           作为 per-class regression head 的增强特征输入。
#           不共享输出函数,不学习时序 pattern,
#           只做物理可解释的轻量特征工程。
#
# 输入: x_seq (B, 100, 8) 原始窗口
# 输出: ratio_feat (B, feat_dim) 残差特征, 与 reg_feat 相加
# ===============================
class CrossChannelRatioBranch(nn.Module):
    """跨通道比率响应分支

    从原始传感器窗口提取每通道统计量和跨通道比率,
    作为回归特征的低维增强信号。不做 Conv1d,
    只用可解释的统计量, 避免学出跨域不稳定的时序模式。

    参数:
        num_sensors: 传感器通道数 (默认 8)
        feat_dim: 回归特征维度 (默认 64)
        proj_dim: 内部投影维度 (默认 32)
        gamma_init: 残差缩放因子初始值 (默认 0.0, 安全初始化)
        dropout: Dropout 比例
    """
    def __init__(self, num_sensors: int = 8, feat_dim: int = 64,
                 proj_dim: int = 32, gamma_init: float = 0.0,
                 dropout: float = 0.05):
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.feat_dim = int(feat_dim)

        # 每通道 4 个统计量: mean, std, amp, slope
        per_channel_stats = self.num_sensors * 4

        # 跨通道比率: top-3 amp 比率 + top-2 slope 比率 = 3 + 1 = 4
        # 但实际提取时我们固定提取 amp_max1/max2、amp_max1/max3、slope_max1/max2
        # 以及 amp_max1/max4 作为补充 = 4 个比率
        num_ratios = 4

        input_dim = per_channel_stats + num_ratios  # 32 + 4 = 36

        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, proj_dim)
        self.out_norm = nn.LayerNorm(proj_dim)
        self.out_proj = nn.Linear(proj_dim, feat_dim)
        self.act = nn.SELU()
        self.dropout = nn.AlphaDropout(float(dropout))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, C)  e.g. (B, 100, 8)
        x = x_seq.float()  # (B, T, C)

        # --- 每通道统计量 (B, C*4) ---
        ch_mean = x.mean(dim=1)                                              # (B, C)
        ch_std = x.std(dim=1, unbiased=False)                                # (B, C)
        ch_amp = x.amax(dim=1) - x.amin(dim=1)                              # (B, C)
        # 线性斜率: 用差分均值近似
        ch_slope = (x[:, -1, :] - x[:, 0, :]) / max(x.size(1) - 1, 1)       # (B, C)
        per_channel = torch.cat([ch_mean, ch_std, ch_amp, ch_slope], dim=1)  # (B, C*4)

        # --- 跨通道比率 (B, 4) ---
        # 按 amp 排序取 top-4 通道索引
        _, amp_indices = ch_amp.sort(dim=1, descending=True)  # (B, C), 降序
        # 安全获取 top-k amp 值
        top1_amp = ch_amp.gather(1, amp_indices[:, 0:1])      # (B, 1)
        top2_amp = ch_amp.gather(1, amp_indices[:, 1:2])
        top3_amp = ch_amp.gather(1, amp_indices[:, 2:3])
        top4_amp = ch_amp.gather(1, amp_indices[:, 3:4])

        top1_slope = ch_slope.gather(1, amp_indices[:, 0:1])
        top2_slope = ch_slope.gather(1, amp_indices[:, 1:2])

        eps = 1e-6
        ratio1 = top1_amp / (top2_amp + eps)                                  # (B, 1)
        ratio2 = top1_amp / (top3_amp + eps)
        ratio3 = top1_amp / (top4_amp + eps)
        ratio4 = top1_slope / (top2_slope + eps)
        ratios = torch.cat([ratio1, ratio2, ratio3, ratio4], dim=1)           # (B, 4)

        # --- 拼接 + 投影 ---
        combined = torch.cat([per_channel, ratios], dim=1)                   # (B, 36)
        feat = self.input_norm(combined)
        feat = self.act(self.proj(feat))
        feat = self.out_norm(feat)
        feat = self.dropout(feat)
        feat = self.out_proj(feat)                                            # (B, feat_dim)
        return self.gamma.to(dtype=feat.dtype) * feat


class FedGasMultiTaskModel(FedGasBaseModel):
    def __init__(self, num_classes=4, num_sensors=8, feat_dim=64,
                 use_mixstyle=False, mixstyle_prob=0.5, mixstyle_alpha=0.1, noise_std=0.01,
                 num_phases=3, reg_head_depth=3, use_quantile=False, num_conc_buckets=0,
                 use_dual_proj=False, use_proto_reg=False,
                 encoder_type='tcn', transformer_d_model=48, transformer_nhead=4,
                 transformer_num_layers=2, transformer_ff_dim=96,
                 reg_grad_detach=False, tcn_norm='instance',
                 use_reg_window_stats=False, reg_window_stats_dim=8,
                 reg_window_stats_mode='global', reg_output_mode='sigmoid',
                 reg_response_branch='none', reg_dct_k=8,
                 reg_dct_gamma_init=0.0, reg_dct_dropout=0.1,
                 reg_msconv_channels=16, reg_msconv_kernels='3,7,15,31',
                 reg_msconv_gamma_init=0.0, reg_msconv_dropout=0.1,
                 use_reg_tcn_adapter=False, reg_tcn_adapter_kernel=3,
                 reg_tcn_adapter_gamma_init=0.0, reg_tcn_adapter_dropout=0.05,
                 use_reg_shared_trunk=False, reg_shared_trunk_dim=128,
                 reg_gas_emb_dim=16, reg_residual_head_depth=2,
                 use_reg_ratio_branch=False, reg_ratio_gamma_init=0.0,
                 reg_ratio_dropout=0.05):
        super().__init__(num_classes, num_sensors, feat_dim, use_mixstyle, mixstyle_prob, mixstyle_alpha, noise_std,
                         encoder_type=encoder_type, transformer_d_model=transformer_d_model,
                         transformer_nhead=transformer_nhead, transformer_num_layers=transformer_num_layers,
                         transformer_ff_dim=transformer_ff_dim, tcn_norm=tcn_norm)

        self.use_dual_proj = use_dual_proj
        self.use_reg_window_stats = bool(use_reg_window_stats)
        self.reg_window_stats_dim = int(reg_window_stats_dim) if self.use_reg_window_stats else 0
        self.reg_window_stats_mode = str(reg_window_stats_mode).lower()
        if self.reg_window_stats_mode not in ('global', 'per_channel'):
            raise ValueError(f'Unsupported reg_window_stats_mode: {reg_window_stats_mode}')
        self.reg_output_mode = str(reg_output_mode or 'sigmoid').lower()
        if self.reg_output_mode not in ('sigmoid', 'linear'):
            raise ValueError(f'Unsupported reg_output_mode: {reg_output_mode}')
        self.reg_response_branch = str(reg_response_branch or 'none').lower()
        if self.reg_response_branch not in ('none', 'dct', 'msconv'):
            raise ValueError(f'Unsupported reg_response_branch: {reg_response_branch}')
        self.reg_dct_k = int(reg_dct_k)
        self.reg_dct_gamma_init = float(reg_dct_gamma_init)
        self.reg_dct_dropout = float(reg_dct_dropout)
        self.reg_msconv_channels = int(reg_msconv_channels)
        if isinstance(reg_msconv_kernels, str):
            self.reg_msconv_kernels = tuple(
                int(k.strip()) for k in reg_msconv_kernels.split(',') if k.strip()
            )
        else:
            self.reg_msconv_kernels = tuple(int(k) for k in reg_msconv_kernels)
        self.reg_msconv_gamma_init = float(reg_msconv_gamma_init)
        self.reg_msconv_dropout = float(reg_msconv_dropout)
        self.use_reg_tcn_adapter = bool(use_reg_tcn_adapter)
        self.reg_tcn_adapter_kernel = int(reg_tcn_adapter_kernel)
        self.reg_tcn_adapter_gamma_init = float(reg_tcn_adapter_gamma_init)
        self.reg_tcn_adapter_dropout = float(reg_tcn_adapter_dropout)
        self.reg_grad_detach = reg_grad_detach  # 回归梯度阻断标志
        self.use_reg_shared_trunk = bool(use_reg_shared_trunk)
        self.reg_shared_trunk_dim = int(reg_shared_trunk_dim)
        self.reg_gas_emb_dim = int(reg_gas_emb_dim)
        self.reg_residual_head_depth = int(reg_residual_head_depth)
        branch_dim = transformer_d_model if encoder_type == 'transformer' else 48
        branch_heads = transformer_nhead if encoder_type == 'transformer' else 4
        branch_layers = transformer_num_layers if encoder_type == 'transformer' else 2
        branch_ff_dim = transformer_ff_dim if encoder_type == 'transformer' else 96

        if use_dual_proj:
            self.cls_proj = nn.Linear(branch_dim, feat_dim)
            self.reg_proj = nn.Linear(branch_dim, feat_dim)
            self.reg_transformer = TemporalTransformerEncoder(
                seq_len=100, d_model=branch_dim, nhead=branch_heads,
                num_layers=branch_layers, ff_dim=branch_ff_dim, dropout=0.1
            )
            self.reg_attn = nn.MultiheadAttention(embed_dim=branch_dim, num_heads=branch_heads, batch_first=True)
            self.reg_attn_linear = nn.Linear(branch_dim, 1)
        else:
            self.cls_proj = None
            self.reg_proj = None
            self.reg_transformer = None
            self.reg_attn = None
            self.reg_attn_linear = None

        if self.use_reg_window_stats:
            stats_input_dim = 4 if self.reg_window_stats_mode == 'global' else num_sensors * 4
            self.reg_stats_proj = nn.Sequential(
                nn.LayerNorm(stats_input_dim),
                nn.Linear(stats_input_dim, self.reg_window_stats_dim),
                nn.SELU()
            )
        else:
            self.reg_stats_proj = None
        if self.reg_response_branch == 'dct':
            self.reg_response_adapter = DCTResponseBranch(
                num_sensors=num_sensors,
                seq_len=100,
                k=self.reg_dct_k,
                feat_dim=feat_dim,
                gamma_init=self.reg_dct_gamma_init,
                dropout=self.reg_dct_dropout,
            )
        elif self.reg_response_branch == 'msconv':
            self.reg_response_adapter = MultiScaleTemporalResponseBranch(
                num_sensors=num_sensors,
                feat_dim=feat_dim,
                hidden_channels=self.reg_msconv_channels,
                kernels=self.reg_msconv_kernels,
                gamma_init=self.reg_msconv_gamma_init,
                dropout=self.reg_msconv_dropout,
            )
        else:
            self.reg_response_adapter = None
        self.use_reg_ratio_branch = bool(use_reg_ratio_branch)
        self.reg_ratio_gamma_init = float(reg_ratio_gamma_init)
        self.reg_ratio_dropout = float(reg_ratio_dropout)
        if self.use_reg_ratio_branch:
            self.reg_ratio_adapter = CrossChannelRatioBranch(
                num_sensors=num_sensors,
                feat_dim=feat_dim,
                proj_dim=32,
                gamma_init=self.reg_ratio_gamma_init,
                dropout=self.reg_ratio_dropout,
            )
        else:
            self.reg_ratio_adapter = None
        if self.use_reg_tcn_adapter and encoder_type == 'tcn':
            self.reg_tcn_adapter = RegTCNResidualAdapter(
                channels=branch_dim,
                kernel_size=self.reg_tcn_adapter_kernel,
                gamma_init=self.reg_tcn_adapter_gamma_init,
                dropout=self.reg_tcn_adapter_dropout,
            )
        else:
            self.reg_tcn_adapter = None
        reg_input_dim = feat_dim + self.reg_window_stats_dim

        self.use_proto_reg = use_proto_reg
        if use_proto_reg:
            # 浓度方向向量回归（ProtoReg）
            self.conc_directions = nn.Parameter(torch.randn(num_classes, reg_input_dim))
            self.conc_scale = nn.Parameter(torch.ones(num_classes, 1))
            self.conc_bias = nn.Parameter(torch.zeros(num_classes, 1))
            self.reg_heads = None
            self.proto_scale = None
            self.proto_bias = None
            self.use_quantile = False
            self.quantile_out_dim = 1
            self.gas_embedding = None
            self.reg_shared_trunk = None
            self.reg_residual_heads = None
        elif self.use_reg_shared_trunk:
            # 共享浓度主干 + 类别残差头
            #   reg_feat + gas_emb -> SharedTrunk -> shared_pred
            #   reg_feat -> ClassResidualHead_c -> delta_c
            #   final = shared_pred + delta_c
            self.reg_heads = None
            self.conc_directions = None
            self.conc_scale = None
            self.conc_bias = None
            self.use_quantile = False
            self.quantile_out_dim = 1
            self.gas_embedding = nn.Embedding(num_classes, self.reg_gas_emb_dim)
            self.reg_shared_trunk = SharedConcentrationTrunk(
                feat_dim=reg_input_dim,
                gas_emb_dim=self.reg_gas_emb_dim,
                shared_dim=self.reg_shared_trunk_dim,
                dropout=0.1,
            )
            self.reg_residual_heads = nn.ModuleList([
                ClassResidualHead(
                    feat_dim=reg_input_dim,
                    depth=self.reg_residual_head_depth,
                    dropout=0.1,
                )
                for _ in range(num_classes)
            ])
            # 原型条件偏置（每个类别一个 scale 和 bias）
            self.proto_scale = nn.Parameter(torch.ones(num_classes, 1))
            self.proto_bias = nn.Parameter(torch.zeros(num_classes, 1))
        else:
            self.reg_heads = nn.ModuleList(
                [RegHead(reg_input_dim, depth=reg_head_depth, quantile_mode=use_quantile) 
                 for _ in range(num_classes)]
            )
            self.use_quantile = use_quantile
            self.quantile_out_dim = 3 if use_quantile else 1
            # 原型条件偏置（每个类别一个 scale 和 bias）
            self.proto_scale = nn.Parameter(torch.ones(num_classes, 1))
            self.proto_bias = nn.Parameter(torch.zeros(num_classes, 1))
            self.gas_embedding = None
            self.reg_shared_trunk = None
            self.reg_residual_heads = None

        # 每个原型一个可学习的浓度中心（4类 × 3阶段）
        self.proto_conc = nn.Parameter(torch.zeros(num_classes, num_phases))
        
        # 浓度桶辅助分类器
        if num_conc_buckets > 0:
            self.conc_bucket_classifier = nn.Linear(reg_input_dim, num_classes * num_conc_buckets)
        else:
            self.conc_bucket_classifier = None
        self.num_conc_buckets = num_conc_buckets

    def _reg_base_output(self, raw_out):
        if self.reg_output_mode == 'linear':
            return raw_out
        return torch.sigmoid(raw_out)

    @staticmethod
    def _maybe_clamp_reg_output(pred, clamp_output=True):
        if clamp_output:
            return torch.clamp(pred, 0.0, 1.0)
        return pred

    def _append_reg_window_stats(self, reg_feat, x_seq):
        if self.reg_stats_proj is None:
            return reg_feat
        x_float = x_seq.float()
        ch_mean = x_float.mean(dim=1)
        ch_std = x_float.std(dim=1, unbiased=False)
        ch_amp = x_float.amax(dim=1) - x_float.amin(dim=1)
        slope = torch.diff(x_float, dim=1).abs().mean(dim=1)
        if self.reg_window_stats_mode == 'per_channel':
            stats = torch.cat([ch_mean, ch_std, ch_amp, slope], dim=1)
        else:
            stats = torch.stack([
                ch_mean.mean(dim=1),
                ch_mean.std(dim=1, unbiased=False),
                ch_amp.mean(dim=1),
                slope.mean(dim=1),
            ], dim=1)
        return torch.cat([reg_feat, self.reg_stats_proj(stats)], dim=1)

    def _apply_reg_response_branch(self, reg_feat, x_seq):
        if self.reg_response_adapter is not None:
            reg_feat = reg_feat + self.reg_response_adapter(x_seq)
        if self.reg_ratio_adapter is not None:
            reg_feat = reg_feat + self.reg_ratio_adapter(x_seq)
        return reg_feat

    def _apply_reg_tcn_adapter(self, temporal_feat):
        if self.reg_tcn_adapter is None:
            return temporal_feat
        return self.reg_tcn_adapter(temporal_feat)

    def forward(self, x):
        x_input = x
        if self.encoder_type == 'transformer':
            if self.use_dual_proj:
                temporal = self.transformer_encoder.encode_sequence(x)  # (B, T, D)

                cls_pooled = self.transformer_encoder.pool_sequence(temporal)
                raw_cls = self.cls_proj(cls_pooled)
                if self.training:
                    raw_cls = raw_cls + torch.randn_like(raw_cls) * self.noise_std
                raw_cls = self.dropout(raw_cls)
                cls_feat = F.normalize(raw_cls, dim=1, p=2)
                logits = self.classifier(cls_feat)

                temporal_for_reg = temporal.detach() if self.reg_grad_detach else temporal
                reg_temporal = self.reg_transformer(temporal_for_reg)
                reg_attn_out, _ = self.reg_attn(reg_temporal, reg_temporal, reg_temporal)
                reg_weights = self.reg_attn_linear(reg_attn_out)
                reg_weights = torch.softmax(reg_weights, dim=1)
                reg_pooled = (reg_weights * reg_temporal).sum(dim=1)
                reg_feat = self.reg_proj(reg_pooled)
                if self.training:
                    reg_feat = reg_feat + torch.randn_like(reg_feat) * self.noise_std
                reg_feat = self.dropout(reg_feat)
            else:
                logits, cls_feat, reg_feat = super().forward(x)
            reg_feat = self._apply_reg_response_branch(reg_feat, x_input)
            reg_feat = self._append_reg_window_stats(reg_feat, x_input)

            if self.conc_bucket_classifier is not None:
                self._conc_bucket_logits = self.conc_bucket_classifier(reg_feat)
            else:
                self._conc_bucket_logits = None
            return logits, cls_feat, reg_feat

        # TCN路径（原有逻辑）
        x = x.permute(0, 2, 1)          # (B, 8, 100)
        
        # 通道注意力
        ch_weights = self.channel_attn(x)        # (B, 8)
        ch_weights = ch_weights.unsqueeze(-1)    # (B, 8, 1)
        x = x * ch_weights

        # TCN 逐层传播（仅保留最后一层输出）
        out = x
        for layer in self.tcn_layers:
            out = layer(out)
        
        if self.use_mixstyle:
            out = self.mixstyle(out)
        
        # 原始注意力池化（用于分类）
        attn_pooled = self.attention_pooling(out.permute(0, 2, 1))  # (B, 48)

        if self.use_dual_proj:
            # 分类分支（独立投影，从注意力池化）
            raw_cls = self.cls_proj(attn_pooled)
            if self.training:
                raw_cls = raw_cls + torch.randn_like(raw_cls) * self.noise_std
            raw_cls = self.dropout(raw_cls)
            cls_feat = F.normalize(raw_cls, dim=1, p=2)
            logits = self.classifier(cls_feat)

            # 回归分支：Transformer + 独立池化
            # 梯度阻断：回归梯度止于此，不污染共享TCN
            out_for_reg = out.detach() if self.reg_grad_detach else out
            out_for_reg = self._apply_reg_tcn_adapter(out_for_reg)
            reg_temporal = self.reg_transformer(out_for_reg.permute(0, 2, 1))  # (B, 100, 48)
            reg_attn_out, _ = self.reg_attn(reg_temporal, reg_temporal, reg_temporal)
            reg_weights = self.reg_attn_linear(reg_attn_out)
            reg_weights = torch.softmax(reg_weights, dim=1)
            reg_pooled = (reg_weights * reg_temporal).sum(dim=1)       # (B, 48)
            reg_feat = self.reg_proj(reg_pooled)                      # (B, 64)
            if self.training:
                reg_feat = reg_feat + torch.randn_like(reg_feat) * self.noise_std
            reg_feat = self.dropout(reg_feat)
        else:
            # 共享投影：从 attn_pooled 映射（保持原有逻辑）
            raw = self.feat_proj(attn_pooled)
            if self.training:
                raw = raw + torch.randn_like(raw) * self.noise_std
            raw = self.dropout(raw)
            cls_feat = F.normalize(raw, dim=1, p=2)
            logits = self.classifier(cls_feat)
            reg_feat = raw
        reg_feat = self._apply_reg_response_branch(reg_feat, x_input)
        reg_feat = self._append_reg_window_stats(reg_feat, x_input)
        
        # 浓度桶分类（辅助任务）
        if self.conc_bucket_classifier is not None:
            self._conc_bucket_logits = self.conc_bucket_classifier(reg_feat)
        else:
            self._conc_bucket_logits = None

        return logits, cls_feat, reg_feat

    def forward_reg(self, feat, y_cls=None, y_reg=None, probs=None, y_phase=None, clamp_output=True):
        """回归前向传播：输出归一化浓度 pred ∈ [0, 1]
        
        重构说明: 统一输出线性空间 + sigmoid, 不再使用对数空间。
        与 normalize_concentration 的目标空间一致。
        """
        device = feat.device
        if self.use_proto_reg:
            if y_cls is not None:
                pred = torch.zeros(feat.size(0), 1, device=device)
                for c in range(len(self.conc_directions)):
                    mask = (y_cls == c)
                    if mask.sum() > 0:
                        d = F.normalize(self.conc_directions[c], dim=0)
                        proj = feat[mask] @ d
                        base_pred = self._reg_base_output(proj.unsqueeze(1))
                        base_pred = self.conc_scale[c] * base_pred + self.conc_bias[c]
                        if y_phase is not None:
                            phase_idx = y_phase[mask].long()
                            base_pred = base_pred + self.conc_scale[c] * self.proto_conc[c, phase_idx].unsqueeze(1)
                        pred[mask] = self._maybe_clamp_reg_output(base_pred, clamp_output)
                return pred
            else:
                all_preds = []
                for c in range(len(self.conc_directions)):
                    d = F.normalize(self.conc_directions[c], dim=0)
                    proj = feat @ d
                    base_pred = self._reg_base_output(proj.unsqueeze(1))
                    base_pred = self.conc_scale[c] * base_pred + self.conc_bias[c]
                    if y_phase is not None:
                        base_pred = base_pred + self.conc_scale[c] * self.proto_conc[c, y_phase.long()].unsqueeze(1)
                    all_preds.append(self._maybe_clamp_reg_output(base_pred, clamp_output))
                all_preds = torch.cat(all_preds, dim=1)
                return (probs * all_preds).sum(dim=1, keepdim=True)
        else:
            if y_cls is not None:
                # ---- 共享浓度主干路径: shared_pred + class_residual ----
                if self.use_reg_shared_trunk:
                    gas_ids = y_cls.long()                           # (B,)
                    gas_emb = self.gas_embedding(gas_ids)            # (B, gas_emb_dim)
                    shared_input = torch.cat([feat, gas_emb], dim=1) # (B, reg_input_dim + gas_emb_dim)
                    shared_pred = self.reg_shared_trunk(shared_input)# (B, 1)
                    pred = torch.zeros(feat.size(0), 1, device=device)
                    for c in range(len(self.reg_residual_heads)):
                        mask = (y_cls == c)
                        if mask.sum() > 0:
                            delta = self.reg_residual_heads[c](feat[mask])  # (N, 1)
                            raw_out = shared_pred[mask] + delta
                            pred_norm = self._reg_base_output(raw_out)
                            result = self.proto_scale[c] * pred_norm + self.proto_bias[c]
                            if y_phase is not None:
                                phase_idx = y_phase[mask].long()
                                result = result + self.proto_conc[c, phase_idx].unsqueeze(1)
                            pred[mask] = self._maybe_clamp_reg_output(result, clamp_output)
                    return pred
                # ---- 原始独立回归头路径 ----
                pred = torch.zeros(feat.size(0), 1, device=device)
                for c in range(len(self.reg_heads)):
                    mask = (y_cls == c)
                    if mask.sum() > 0:
                        raw_out = self.reg_heads[c](feat[mask])
                        pred_norm = self._reg_base_output(raw_out)
                        result = self.proto_scale[c] * pred_norm + self.proto_bias[c]
                        if y_phase is not None:
                            phase_idx = y_phase[mask].long()
                            result = result + self.proto_conc[c, phase_idx].unsqueeze(1)
                        pred[mask] = self._maybe_clamp_reg_output(result, clamp_output)
                return pred
            else:
                # y_cls=None: 使用 probs 对各气体预测加权混合
                if self.use_reg_shared_trunk:
                    # 对每个类别分别计算: gas_emb_c -> shared_pred + delta_c
                    reg_outputs = []
                    for c in range(len(self.reg_residual_heads)):
                        gas_ids = torch.full((feat.size(0),), c, dtype=torch.long, device=device)
                        gas_emb_c = self.gas_embedding(gas_ids)
                        shared_input_c = torch.cat([feat, gas_emb_c], dim=1)
                        shared_pred_c = self.reg_shared_trunk(shared_input_c)
                        delta_c = self.reg_residual_heads[c](feat)
                        raw_out = shared_pred_c + delta_c
                        pred_norm = self._reg_base_output(raw_out)
                        result = self.proto_scale[c] * pred_norm + self.proto_bias[c]
                        if y_phase is not None:
                            result = result + self.proto_conc[c, y_phase.long()].unsqueeze(1)
                        reg_outputs.append(self._maybe_clamp_reg_output(result, clamp_output))
                    reg_outputs = torch.cat(reg_outputs, dim=1)
                    return (probs * reg_outputs).sum(dim=1, keepdim=True)
                # 原始路径: y_cls=None 时 probs 加权混合
                reg_outputs = []
                for c in range(len(self.reg_heads)):
                    raw_out = self.reg_heads[c](feat)
                    pred_norm = self._reg_base_output(raw_out)
                    result = self.proto_scale[c] * pred_norm + self.proto_bias[c]
                    if y_phase is not None:
                        result = result + self.proto_conc[c, y_phase.long()].unsqueeze(1)
                    reg_outputs.append(self._maybe_clamp_reg_output(result, clamp_output))
                reg_outputs = torch.cat(reg_outputs, dim=1)
                return (probs * reg_outputs).sum(dim=1, keepdim=True)

    def get_regression_params(self):
        params = []
        if self.use_dual_proj:
            params.extend(self.reg_proj.parameters())
            if hasattr(self, 'reg_transformer') and self.reg_transformer is not None:
                params.extend(self.reg_transformer.parameters())
                params.extend(self.reg_attn.parameters())
                params.append(self.reg_attn_linear.weight)
                params.append(self.reg_attn_linear.bias)
        if getattr(self, 'reg_stats_proj', None) is not None:
            params.extend(self.reg_stats_proj.parameters())
        if getattr(self, 'reg_response_adapter', None) is not None:
            params.extend(self.reg_response_adapter.parameters())
        if getattr(self, 'reg_ratio_adapter', None) is not None:
            params.extend(self.reg_ratio_adapter.parameters())
        if getattr(self, 'reg_tcn_adapter', None) is not None:
            params.extend(self.reg_tcn_adapter.parameters())
        if self.use_proto_reg:
            params.append(self.conc_directions)
            params.append(self.conc_scale)
            params.append(self.conc_bias)
        elif self.use_reg_shared_trunk:
            params.extend(self.reg_shared_trunk.parameters())
            params.append(self.gas_embedding.weight)
            for head in self.reg_residual_heads:
                params.extend(head.parameters())
            if hasattr(self, 'proto_scale') and self.proto_scale is not None:
                params.append(self.proto_scale)
            if hasattr(self, 'proto_bias') and self.proto_bias is not None:
                params.append(self.proto_bias)
        elif hasattr(self, 'reg_heads') and self.reg_heads is not None:
            for head in self.reg_heads:
                params.extend(head.parameters())
            if hasattr(self, 'proto_scale') and self.proto_scale is not None:
                params.append(self.proto_scale)
            if hasattr(self, 'proto_bias') and self.proto_bias is not None:
                params.append(self.proto_bias)
        if hasattr(self, 'channel_attn') and self.channel_attn is not None:
            params.extend(self.channel_attn.parameters())
        return params


# 保持向后兼容
FedGasModel = FedGasBaseModel


class FedGasProtoRegMultiTaskModel(FedGasBaseModel):
    """联邦多任务模型：分类（softmax） + 浓度方向向量回归（ProtoReg）"""
    def __init__(self, num_classes=4, num_sensors=8, feat_dim=64,
                 use_mixstyle=False, mixstyle_prob=0.5, mixstyle_alpha=0.1,
                 noise_std=0.01, use_residual=False, num_conc_buckets=0,
                 encoder_type='tcn', transformer_d_model=48, transformer_nhead=4,
                 transformer_num_layers=2, transformer_ff_dim=96, tcn_norm='instance'):
        super().__init__(num_classes, num_sensors, feat_dim, use_mixstyle,
                         mixstyle_prob, mixstyle_alpha, noise_std,
                         encoder_type=encoder_type, transformer_d_model=transformer_d_model,
                         transformer_nhead=transformer_nhead,
                         transformer_num_layers=transformer_num_layers,
                         transformer_ff_dim=transformer_ff_dim, tcn_norm=tcn_norm)
        # 浓度方向向量（每个类别一个）
        self.conc_directions = nn.Parameter(torch.randn(num_classes, feat_dim))
        # 缩放与偏置（对数空间）
        self.conc_scale = nn.Parameter(torch.ones(num_classes, 1))
        self.conc_bias = nn.Parameter(torch.zeros(num_classes, 1))
        # 可选残差头（默认关闭）
        self.use_residual = use_residual
        if use_residual:
            self.residual_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(feat_dim, 16), nn.ReLU(), nn.Linear(16, 1))
                for _ in range(num_classes)
            ])
        # 浓度桶辅助分类器
        if num_conc_buckets > 0:
            self.conc_bucket_classifier = nn.Linear(feat_dim, num_classes * num_conc_buckets)
        else:
            self.conc_bucket_classifier = None
        self.num_conc_buckets = num_conc_buckets

    def forward(self, x):
        logits, cls_feat, reg_feat = super().forward(x)
        if self.conc_bucket_classifier is not None:
            self._conc_bucket_logits = self.conc_bucket_classifier(reg_feat)
        else:
            self._conc_bucket_logits = None
        return logits, cls_feat, reg_feat

    def forward_reg(self, feat, y_cls=None, probs=None, y_phase=None):
        """条件回归（重构版）：输出 sigmoid(方向投影) ∈ [0,1]"""
        if y_cls is not None:
            pred = torch.zeros(feat.size(0), 1, device=feat.device)
            for c in range(len(self.conc_directions)):
                mask = (y_cls == c)
                if mask.sum() > 0:
                    d = F.normalize(self.conc_directions[c], dim=0)
                    proj = feat[mask] @ d
                    base_pred = torch.sigmoid(proj.unsqueeze(1))
                    base_pred = self.conc_scale[c] * base_pred + self.conc_bias[c]
                    if self.use_residual:
                        residual = self.residual_heads[c](feat[mask])
                        base_pred = base_pred + residual
                    pred[mask] = torch.clamp(base_pred, 0.0, 1.0)
            return pred
        else:
            all_preds = []
            for c in range(len(self.conc_directions)):
                d = F.normalize(self.conc_directions[c], dim=0)
                proj = feat @ d
                base_pred = torch.sigmoid(proj.unsqueeze(1))
                base_pred = self.conc_scale[c] * base_pred + self.conc_bias[c]
                if self.use_residual:
                    residual = self.residual_heads[c](feat)
                    base_pred = base_pred + residual
                all_preds.append(torch.clamp(base_pred, 0.0, 1.0))
            all_preds = torch.cat(all_preds, dim=1)
            return (probs * all_preds).sum(dim=1, keepdim=True)


# ===============================
# 梯度反转层（用于域对抗训练）
# ===============================
class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层：前向传播不变，反向传播时将梯度乘以负系数
    
    用于域对抗训练：让特征提取器学习欺骗域判别器，
    从而产生域不变特征。与深度CORAL互补——
    CORAL对齐二阶统计量，对抗训练对齐高阶分布差异。
    """
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时将梯度乘以 -lambda，使特征提取器远离域判别器目标
        return grad_output.neg() * ctx.lambda_grl, None


class GradientReversalLayer(nn.Module):
    """梯度反转层模块包装器
    
    Args:
        lambda_grl: 梯度反转强度，越大则对抗越强。
                    训练初期通常从小值开始逐渐增大。
    """
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)


# ===============================
# Wasserstein域判别器（域对抗训练）
# ===============================
class DomainDiscriminator(nn.Module):
    """Wasserstein域判别器，带谱归一化保证Lipschitz连续
    
    用于WGAN风格的域对抗训练。判别器尝试区分源域和目标域特征，
    通过谱归一化保证1-Lipschitz连续性，使得Wasserstein距离估计有界。
    该模块完全在服务器端使用，不违反联邦隐私约束。
    
    Args:
        feat_dim: 特征维度
        hidden_dim: 隐藏层维度
    """
    def __init__(self, feat_dim=64, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))  # 输出标量Wasserstein距离估计
        )

    def forward(self, x):
        return self.net(x)


# ===============================
# 传感器交互Transformer编码器（替代TCN）
# ===============================
class SensorInteractionTransformerEncoder(nn.Module):
    """Transformer encoder for gas sensor sequences.

    Keeps the public forward contract ``(B, T, S) -> (B, feat_dim)`` while
    exposing sequence-level features for the multi-task dual-branch path.
    """
    def __init__(self, num_sensors=8, d_model=48, nhead=4, num_layers=2,
                 dim_feedforward=96, seq_len=100, dropout=0.1, feat_dim=64):
        super().__init__()
        self.sensor_attn = nn.Sequential(
            nn.Linear(num_sensors, num_sensors),
            nn.Tanh(),
            nn.Linear(num_sensors, num_sensors),
            nn.Sigmoid()
        )
        self.input_proj = nn.Linear(num_sensors, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        self.feat_proj = nn.Linear(d_model, feat_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_sequence(self, x):
        sensor_weights = self.sensor_attn(x.mean(dim=1)).unsqueeze(1)
        x = x * sensor_weights
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.transformer(x)

    def pool_sequence(self, x):
        attn_weights = torch.softmax(self.temporal_attn(x), dim=1)
        return (attn_weights * x).sum(dim=1)

    def forward(self, x):
        x = self.encode_sequence(x)
        x = self.pool_sequence(x)
        x = self.dropout(x)
        return self.feat_proj(x)
