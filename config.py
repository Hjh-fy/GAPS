"""
全局联邦学习配置文件
所有超参数集中管理，无硬编码，支持消融实验
"""
from dataclasses import dataclass, field
import torch
import os
import numpy as np
from typing import Dict, Optional, List

@dataclass
class FLConfig:
    # 基础设置
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据参数
    BATCH_SIZE: int = 32
    NUM_CLIENTS: int = 4
    NUM_CLASSES: int = 4
    SEQ_LEN: int = 100
    INPUT_DIM: int = 8
    NUM_PHASES: int = 3  # 0=E,1=M,2=L

    # 模型结构
    HIDDEN_DIM1: int = 128
    HIDDEN_DIM2: int = 64
    TCN_NORM: str = 'instance'             # instance, batch, or none; default preserves prior checkpoints
    REG_WINDOW_STATS: bool = False         # Append visible window amplitude/slope statistics to regression features
    REG_WINDOW_STATS_MODE: str = 'global'  # global or per_channel window statistics for regression features
    REG_WINDOW_STATS_DIM: int = 8
    FEATURE_NOISE_STD: float = 0.01        # 特征噪声标准差

    # 训练参数
    GLOBAL_ROUNDS: int = 10                 # 联邦训练轮次【实验设计：10轮，配合大K步数】
    LOCAL_EPOCHS: int = 5                   # 客户端本地训练轮次【保持5轮】
    LR_CLIENT: float = 5e-4                 # 客户端学习率

    # 可学习聚合 
    ETA_W: float = 0.01  # 增加学习率，加快权重更新速度
    TAU: float = 0.8      # 聚合温度
    WEIGHT_MIN: float = 1e-5

    # 差分隐私
    DP_SIGMA: float = 0.01

    # 路径
    RESULT_SAVE_DIR: str = "results/"          # 结果保存根目录
    LOG_DIR: str = "results/logs/"             # 日志目录
    MODEL_SAVE_DIR: str = "results/models/"    # 模型保存目录
    PLOT_SAVE_DIR: str = "results/plots/"      # 图表保存目录

    # 消融实验开关
    USE_LEARNABLE_AGG: bool = False  # 禁用可学习聚合，避免服务器和客户端模型结构不匹配
    USE_ALIGN: bool = True          # 是否启用原型对齐损失
    USE_DP: bool = False  # 关闭差分隐私
    USE_SERVER_OPT: bool = True  # 是否启用服务器联合优化

    LAMBDA_PROTO: float = 0.05            # 原型损失权重（从 0.1 降低至 0.05）
    LAMBDA_CONSISTENCY: float = 2.0      # 一致性损失权重
    LAMBDA_ALIGN: float = 0.05            # 对齐损失权重
    LAMBDA_SMOOTH: float = 0.01          # 阶段平滑损失权重
    SERVER_OPT_LR: float = 0.0005        # 服务器联合优化学习率
    PROTO_EMA_ALPHA: float = 0.8        # 原型 EMA 平滑系数
    SERVER_OPT_STEPS_BASE: int = 40     # 基础优化步数 K_base【最优：K=50实验验证】
    DRIFT_GAMMA: float = 2.0           # 漂移-步数缩放系数
    DOMAIN_ADAPT_K_BONUS: int = 60     # 域适应额外步数【最优K=100：40+drift+60≈100-110】
    EMA_ALPHA_F: float = 0.8           # 全局特征滑动平均系数
    PROTO_EXTRAP_ALPHA: float = 0.6    # 原型外推信任度

    # 特征蒸馏（抗遗忘）
    LAMBDA_REPLAY_DISTILL: float = 2   # 特征蒸馏损失权重（0.15）
    USE_REPLAY_DISTILL: bool = True      # 是否启用特征蒸馏
    
    # 对比损失配置
    USE_CONTRASTIVE_ALIGN: bool = True          # 客户端对齐是否用对比损失
    USE_CONTRASTIVE_CONSISTENCY: bool = True    # 服务器一致性是否用对比损失
    CONTRAST_TEMPERATURE: float = 0.1           # 温度系数
    
    # 学习率调度
    LR_SCHEDULER: str = "cosine"  # "step", "cosine", "none"
    LR_STEP_SIZE: int = 10
    LR_GAMMA: float = 0.5

    # ========= 原型解耦配置 =========
    USE_PROTO_DECOUPLING: bool = True          # 是否启用语义原型与设备残差解耦
    LAMBDA_RES: float = 0.1                    # 设备残差损失权重
    USE_SOFT_AGGREGATION: bool = True          # 是否启用软聚合推理
    SOFT_AGG_TEMPERATURE: float = 0.35         # 软聚合温度参数 (从 0.3 微调)
    SOFT_AGG_PRIOR_WEIGHT: float = 0.2         # 先验权重

    # ========= 自适应温度推理 (替代全链路马氏距离的务实方案) =========
    # 核心原理: 不同 (类,阶段) 原型的特征散布不同，统一温度忽略异方差性
    # 通过估计每原型的特征分散度，缩放其推理温度，宽散布→高温度→软决策
    # 参考文献: "Temperature Scaling meets Prototype Learning" 思路
    USE_ADAPTIVE_TEMPERATURE: bool = False     # 自适应温度【实验验证：关闭更稳定】
    ADAPTIVE_TEMP_BETA: float = 0.5            # 散布→温度缩放强度 (0=统一温度, 1=完全按散布缩放)
    ADAPTIVE_TEMP_MIN: float = 0.1             # 每原型温度下限 (防止过度锐化)
    ADAPTIVE_TEMP_MAX: float = 2.0             # 每原型温度上限 (防止过度软化)

    # ========= 马氏距离推理 (推理端可选, 不影响训练) =========
    # 使用 reg_feat (未归一化) + 对角方差矩阵计算马氏距离
    # 仅在推理时启用，训练阶段保持不变，可通过开关对比余弦相似度
    USE_MAHALANOBIS_INFERENCE: bool = False    # 推理时使用马氏距离 (需先提取校准集方差)
    MAHALANOBIS_EPS: float = 1e-4             # 方差除零保护
    MAHALANOBIS_MIN_VAR: float = 0.01         # 最小方差约束
    MAHALANOBIS_TEMP: float = 1.0             # 马氏距离后温度缩放 (可为1.0)

    # 域泛化 MMD 正则（语义原型和客户端原型对齐）
    USE_MMD_REG: bool = True
    LAMBDA_MMD: float = 0.5            # 原型对齐损失权重
    MMD_SIGMA: float = 1.0             # 高斯核带宽
    
    # 原型级 MMD 对齐（阶段2：训练阶段嵌入原型级域不变性约束：没有梯度）
    USE_PROTO_MMD: bool = False         # 是否启用原型间 MMD 对齐
    LAMBDA_PROTO_MMD: float = 0.2      # 原型级 MMD 损失权重
    
    # 深度CORAL配置（跨设备域自适应）
    USE_DEEP_CORAL: bool = True                   # 是否在服务器优化时启用深度CORAL对齐
    LAMBDA_DEEP_CORAL: float = 0.5                  # 深度CORAL损失权重
    DEEP_CORAL_CALIB_CLIENTS: List[int] = field(default_factory=lambda: [])  # 作为校准集的目标客户端ID
    DEEP_CORAL_CALIB_SIZE: int = 500                # 从每个客户端抽取的校准样本数
    CORAL_CLASS_CONDITIONAL: bool = True           # 是否使用类条件CORAL对齐（需要带标签校准数据）
    CALIB_PSEUDO_THRESHOLD: float = 0.9            # 伪标签置信度阈值 (仅 CALIB_USE_LABELS=False 时生效)
    
    CALIB_USE_LABELS: bool = True                  # 校准数据是否使用真实标签 (True=监督DA, False=无监督/伪标签)

    # ========= 域对抗训练配置（Wasserstein GAN + 梯度反转层）=========
    # 与深度CORAL互补：CORAL对齐二阶统计量，对抗训练对齐高阶分布差异
    # 该改动完全在服务器端完成，不违反联邦隐私约束
    USE_ADVERSARIAL_DOMAIN: bool = True            # 是否启用域对抗训练（服务器端）【已开启，核心域适应手段】
    LAMBDA_ADV_DOMAIN: float = 0.5                 # 域对抗损失权重【从0.1提升至0.5，与CORAL/MMD平衡】
    ADV_DOMAIN_LR: float = 0.001                   # 域判别器学习率（通常比特征提取器高）
    ADV_GRADIENT_PENALTY: float = 10.0             # WGAN-GP梯度惩罚系数【标准WGAN-GP值，强化Lipschitz约束】
    ADV_CRITIC_ITERS: int = 3                      # 判别器每步更新次数【标准n_critic=5, 这里折中取3】
    ADV_CLASS_CONDITIONAL: bool = True             # 是否使用类条件域对抗训练（强烈推荐，防止类别混淆）
    
    # ========= Transformer编码器配置（替代TCN特征提取）=========
    # 师兄模型在相同数据集上已验证有效，Transformer自注意力天然建立传感器通道全局关联
    USE_TRANSFORMER_ENCODER: bool = False          # True: Transformer编码器；False: TCN编码器
    TRANSFORMER_D_MODEL: int = 48                  # Transformer隐藏维度
    TRANSFORMER_NHEAD: int = 4                     # 多头注意力头数
    TRANSFORMER_NUM_LAYERS: int = 2                # Transformer编码器层数
    TRANSFORMER_FF_DIM: int = 96                   # 前馈网络维度
    
    # ========= 传感器特定数据增强（仅 warmup 阶段） =========
    USE_SENSOR_AUG: bool = True           # 是否启用设备级增强
    SENSOR_AUG_GAIN_STD: float = 0.04     # 设备增益标准差（降低）
    SENSOR_AUG_BIAS_STD: float = 0.015    # 设备偏置标准差（降低）
    SENSOR_AUG_CH_GAIN_STD: float = 0.02  # 通道独立增益标准差（降低）
    SENSOR_AUG_TIME_SCALE_RANGE: float = 0.08  # 时间缩放范围 ±8%
    SENSOR_AUG_TIME_PROB: float = 0.3     # 时间增强概率
    
    # 多任务回归配置
    USE_REG_LOSS: bool = True           # 启用客户端本地回归 (服务器端已移除)
    LAMBDA_REG: float = 1.0             # 回归损失权重
    USE_HUBER_LOSS: bool = True          # 核心回归损失: Huber (smooth_l1)
    HUBER_DELTA: float = 0.2             # Huber转折点 (20%)
    SEPARATE_REG_HUBER_DELTAS: str = ''  # 可选每类Huber beta, 例如 "1:0.1"；空字符串表示统一HUBER_DELTA
    SEPARATE_REG_ALLOW_ENCODER_BACKPROP: bool = True  # separate regression解冻encoder时是否允许回归梯度回传到encoder
    USE_LOG_REL_LOSS: bool = False       # 辅助: 对数相对误差 (默认关闭)
    LAMBDA_LOG_REL: float = 0.05         # 对数相对误差权重 (若启用)
    REG_TAIL_WEIGHT: float = 1.0         # 高浓度尾部样本权重，1.0 表示关闭
    REG_TAIL_THRESHOLD: float = 1.0      # 归一化浓度阈值，例如 0.8 表示每类高端 20%
    REG_TAIL_CLASSES: str = ""           # 为空表示所有类别；例如 "1" 表示仅 CO 高浓度加权
    REG_OUTPUT_MODE: str = "sigmoid"     # sigmoid: 旧版有界输出; linear: 线性归一化输出
    REG_RANGE_PENALTY: float = 0.0       # linear 模式下训练时的 [0,1] 越界惩罚权重
    REG_RESPONSE_BRANCH: str = "dct"     # mainline: fixed low-frequency response-shape adapter
    REG_DCT_K: int = 16                  # mainline R3aK16: DCT coefficients per sensor
    REG_DCT_GAMMA_INIT: float = 0.0      # initial residual scale for DCT response adapter
    REG_DCT_DROPOUT: float = 0.1         # dropout inside DCT response adapter
    REG_MSCONV_CHANNELS: int = 16        # hidden channels per temporal scale for REG_RESPONSE_BRANCH=msconv
    REG_MSCONV_KERNELS: str = "3,7,15,31"
    REG_MSCONV_GAMMA_INIT: float = 0.0
    REG_MSCONV_DROPOUT: float = 0.1
    REG_TCN_ADAPTER: bool = False        # regression-specific residual adapter on shared TCN features
    REG_TCN_ADAPTER_KERNEL: int = 3
    REG_TCN_ADAPTER_GAMMA_INIT: float = 0.0
    REG_TCN_ADAPTER_DROPOUT: float = 0.05
    # 共享浓度主干 (Shared Concentration Trunk)
    REG_USE_SHARED_TRUNK: bool = False    # 是否启用共享主干 + 类别残差头
    REG_SHARED_TRUNK_DIM: int = 128       # 共享主干隐藏层维度
    REG_GAS_EMB_DIM: int = 16             # 气体嵌入维度
    REG_RESIDUAL_HEAD_DEPTH: int = 2      # 类别残差头深度 (1/2/3)
    # 跨通道比率响应分支
    USE_REG_RATIO_BRANCH: bool = False    # 启用跨通道比率特征
    REG_RATIO_GAMMA_INIT: float = 0.0     # 比率分支残差缩放初始值
    REG_RATIO_DROPOUT: float = 0.05       # 比率分支 Dropout
    
    # 回归头配置
    PERSONALIZED_REG: bool = True            # True + SHARE_REG=False → 完全个性化
    SHARE_REG_HEAD: bool = True             # False: 回归头不上传 (服务器端无回归损失)
    REG_HEAD_DEPTH: int = 4                 # 回归头深度
    REG_GRAD_DETACH: bool = False           # 阻断回归梯度向共享TCN的回传（需USE_DUAL_PROJ=True配合）

    # 实验配置
    MMD_CACHE_INTERVAL: int = 10           # 更新源域特征的间隔（轮）
    MMD_MAX_SAMPLES: int = 500             # MMD 计算最大样本数
    CORAL_MAX_SAMPLES: int = 1000          # CORAL 源域特征提取样本数
    REGRESSION_TOLERANCE: float = 0.1      # 回归准确率容忍度
    FEATURE_EXTRACT_MAX_SAMPLES: int = 1000 # 特征提取默认上限
    
    # 模型选择配置
    MODEL_SELECTION_CLS_WEIGHT: float = 0.7  # 模型选择中分类准确率的权重
    MODEL_SELECTION_REG_WEIGHT: float = 0.3  # 模型选择中回归准确率的权重
    
    # RGPR 融合权重
    RGPR_SIM_WEIGHT: float = 0.7  # 特征相似度权重
    RGPR_CONC_WEIGHT: float = 0.3  # 浓度权重
    
    # 选择性聚合配置
    USE_SELECTIVE_AGG: bool = True
    SELECTIVE_AGG_WARMUP: int = 5          # 前5轮使用标准聚合
    SELECTIVE_AGG_MIN_SCALE: float = 0.3   # 相似度映射的最小缩放因子
    
    # 双分支特征解耦（路线B）
    # 启用后分类(cls_proj)与回归(reg_proj)拥有独立投影层，从结构上彻底杜绝特征空间冲突
    USE_DUAL_PROJ: bool = True              # True：分类和回归使用独立投影层；False：原有共享投影

    # ========= 分阶段训练 =========
    STAGEWISE_TRAINING: bool = False         # 是否启用分阶段训练
    PHASE1_END_ROUND: int = 15              # 第一阶段结束轮次（缩短至15轮）
    PHASE1_REG_WARMUP_WEIGHT: float = 0.01  # 第一阶段回归背景权重（软预热）
    PHASE2_REG_WEIGHT: float = 1.0          # 第二阶段回归损失目标权重
    PHASE2_REG_RAMPUP_ROUNDS: int = 5       # 第二阶段回归权重线性递增轮次
    PHASE2_LR_CLIENT: float = 5e-4          # 第二阶段回归分支学习率（提高到正常训练水平）
    PHASE2_CLS_PROJ_LR: float = 1e-5        # 第二阶段分类投影学习率（极低）
    PHASE2_SHARED_LR: float = 1e-4          # 第二阶段共享层学习率
    FREEZE_CLASSIFIER_IN_PHASE2: bool = True  # 第二阶段冻结分类头
    FREEZE_PROTOS_IN_PHASE2: bool = False   # 第二阶段继续更新原型（不再冻结）

    # ========= MMD对齐损失（新方案）=========
    USE_MMD_ALIGNMENT: bool = True               # 是否启用MMD对齐损失
    LAMBDA_GLOBAL_MMD: float = 0.5              # 全局MMD损失权重【从1.0降至0.5，与对抗/CORAL平衡】
    LAMBDA_CLASS_MMD: float = 0.5               # 类别条件MMD损失权重【从1.0降至0.5】
    LAMBDA_PROTO_ANCHOR: float = 0.3            # 原型锚定正则权重【从0.5降至0.3】
    LAMBDA_STAGE_MMD: float = 0.2              # 阶段间MMD损失权重【从0.3降至0.2】
    PSEUDO_LABEL_THRESHOLD: float = 0.9        # 伪标签置信度阈值

    # ========= 在线部署自适应（Stage 2）=========
    ADAPT_LR: float = 1e-4                # 适应学习率
    MT_EMA_DECAY: float = 0.99            # 教师模型EMA衰减率（0.99让教师能实际移动，旧版0.999几乎不动）
    MT_CONF_THRESH: float = 0.9           # 教师伪标签置信度阈值
    TENT_WEIGHT: float = 1.0              # TENT熵最小化损失权重（主要损失）
    ANCHOR_REG_WEIGHT: float = 0.3        # 软原型锚定正则权重（从0.05提升至0.3，强化马氏距离/余弦对比信号）
    PROJ_CONSISTENCY_WEIGHT: float = 0.01 # 投影一致性损失权重
    USE_PROJ_CONSISTENCY: bool = False    # 是否启用投影一致性约束
    MIN_PROTO_UPDATE_CONF: float = 0.8
    PROTO_EMA_ALPHA: float = 0.8
    OSR_ENABLE: bool = False
    OSR_WINDOW: int = 100
    OSR_UPDATE_FREQ: int = 50
    ONLINE_ADAPT_UPLOAD_FREQ: int = 5
    # 联邦校准参数
    DEPLOY_CONF_THRESH: float = 0.5  # 部署阶段上传置信度阈值
    TRIMMED_MEAN_FRACTION: float = 0.2  # 修剪均值的尾部比例
    # ========= 部署校准修剪 (马氏距离增强) =========
    USE_MAHALANOBIS_TRIM: bool = True        # True=马氏距离修剪, False=L2范数修剪
    USE_MAHALANOBIS_ANCHOR: bool = True      # 马氏距离对比实验组
    MAHALANOBIS_ANCHOR_TEMP: float = 1.0     # 马氏距离→相似度的温度缩放: sim = exp(-d_mahal / temp)
    MAHALANOBIS_MIN_VAR: float = 0.01        # 最小方差约束
    MAHALANOBIS_VAR_EMA: float = 0.9         # 方差EMA系数
    CALIB_CMAX: float = 0.5                  # 校正量上限

    # 回归头结构选择（阶段3A：ProtoReg）
    USE_PROTO_REG: bool = False              # True：ProtoReg回归头
    USE_REG_RESIDUAL: bool = False           # 在ProtoReg基础上是否加残差MLP（hybrid）
    USE_QUANTILE_LOSS: bool = False          # ProtoReg 不支持分位数，必须关闭
    
    # 浓度桶辅助分类
    NUM_CONC_BUCKETS: int = 0                     # 0 表示禁用（浓度桶是回归辅助任务，一并关闭）
    LAMBDA_CONC_BUCKET: float = 0.15               # 桶分类损失权重
    CONC_BUCKET_LOSS: str = 'hard'                 # hard=CE, soft=相邻浓度桶软标签
    CONC_BUCKET_SOFT_SIGMA: float = 1.0            # soft桶标签的桶距离标准差
    CONC_BUCKET_DETACH_FEAT: bool = False          # True时桶分类只训练探针头，不反向影响回归特征
    CONC_BUCKET_BOUNDARIES: Optional[Dict[int, List[float]]] = None   # 运行期填充

    # 目标感知聚合 (Target-Informed Aggregation)
    USE_TARGET_INFORMED_AGG: bool = False  # 临时关闭：方案A纯自适应温度
    TARGET_INFORMED_WEIGHT: float = 0.6    # 目标相似度权重 (0=纯原型, 1=纯目标)

    # MixStyle 本地增强
    USE_MIXSTYLE: bool = False        # 关闭作为默认配置
    MIXSTYLE_PROB: float = 0.5       # 应用 MixStyle 的概率
    MIXSTYLE_ALPHA: float = 0.5      # Beta 分布参数

def generate_config_signature(config, max_tags=8):
    """从配置中生成紧凑的实验签名，仅包含不同于默认值的字段

    生成的签名形如: _noRank.pers.dep2 （按字母排序）
    签名会截断到 max_tags 个标签以避免目录名过长。

    Args:
        config: FLConfig 实例（或任意包含同名属性的对象）
        max_tags: 最大标签数量，以防签名过长

    Returns:
        签名字符串，空字符串表示全默认
    """
    ref = FLConfig()

    tags = []

    # 布尔字段：仅当与默认值不同时添加标签
    # (字段名, 默认True时的否定标签, 默认False时的肯定标签)
    bool_fields = [
        ('PERSONALIZED_REG', 'noPrs', None),
        ('SHARE_REG_HEAD', None, 'shr'),
        ('USE_REG_LOSS', 'noReg', None),
        ('USE_MIXSTYLE', None, 'mix'),
        ('USE_SENSOR_AUG', 'noAug', None),
        ('USE_LEARNABLE_AGG', None, 'agg'),
        ('USE_ALIGN', 'noAln', None),
        ('USE_REPLAY_DISTILL', 'noDis', None),
        ('USE_SERVER_OPT', 'noOpt', None),
        ('USE_DP', None, 'dp'),
        ('USE_PROTO_DECOUPLING', 'noDec', None),
        ('USE_SOFT_AGGREGATION', 'noSft', None),
        ('USE_MMD_REG', 'noMmd', None),
        ('USE_SELECTIVE_AGG', 'noSlc', None),
        ('USE_CONTRASTIVE_ALIGN', 'noCtA', None),
        ('USE_CONTRASTIVE_CONSISTENCY', 'noCtC', None),
        ('USE_ADVERSARIAL_DOMAIN', None, 'adv'),
        ('ADV_CLASS_CONDITIONAL', 'advUc', None),
        ('USE_TRANSFORMER_ENCODER', None, 'tf'),
        ('USE_ADAPTIVE_TEMPERATURE', None, 'adp'),
        ('USE_TARGET_INFORMED_AGG', None, 'tia'),
    ]

    for field, tag_if_false, tag_if_true in bool_fields:
        if not hasattr(config, field) or not hasattr(ref, field):
            continue
        val = getattr(config, field)
        default_val = getattr(ref, field)
        if val != default_val:
            if default_val is True:
                tags.append(tag_if_false)
            else:
                tags.append(tag_if_true)

    # 数值字段：仅当与默认值不同时添加标签
    num_fields = [
        ('REG_HEAD_DEPTH', 'd{0}'),
        ('LAMBDA_REG', 'lR{0}'),
        ('LR_CLIENT', 'lr{0}'),
        ('LAMBDA_PROTO', 'lP{0}'),
        ('LAMBDA_ALIGN', 'lA{0}'),
        ('LAMBDA_CONSISTENCY', 'lC{0}'),
        ('LAMBDA_REPLAY_DISTILL', 'lD{0}'),
        ('LAMBDA_ADV_DOMAIN', 'aD{0}'),
        ('LAMBDA_DEEP_CORAL', 'dC{0}'),
    ]

    for field, fmt in num_fields:
        if not hasattr(config, field) or not hasattr(ref, field):
            continue
        val = getattr(config, field)
        default_val = getattr(ref, field)
        if val != default_val:
            tags.append(fmt.format(val))

    # 排序保证一致性
    tags = [t for t in tags if t is not None]
    tags.sort()

    # 截断到 max_tags
    if len(tags) > max_tags:
        tags = tags[:max_tags] + ['+']

    if not tags:
        return ''

    return '.' + '.'.join(tags)


# 自动创建目录
for path in [FLConfig.LOG_DIR, FLConfig.MODEL_SAVE_DIR, FLConfig.PLOT_SAVE_DIR]:
    os.makedirs(path, exist_ok=True)
