"""
GAPS 联邦持续学习实验脚本
严格遵循论文设计：可学习原型、修正聚合权重、服务器联合优化、漂移自适应触发器
支持多测试客户端、消融实验配置、模型选择、早停与 MMD 评估
"""

import torch
import copy
import json
import time
import argparse
import numpy as np
import csv
import os
import shutil
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence, Union
import torch.nn as nn

from model import FedGasModel
from client import Client
from server import Server
from config import FLConfig
from utils import (
    set_random_seed, setup_logging, evaluate_model, evaluate_model_with_phase, compute_mmd,
    save_results, plot_training_curves, plot_aggregation_weights, plot_phase_curves,
    create_experiment_dir, print_experiment_config, print_training_summary, format_time,plot_tsne_features, 
    plot_concentration_feature_correlation, plot_regression_scatter, extract_features_batch,plot_coral_tsne_comparison,
    plot_finetune_tsne_comparison,plot_classifier_weight_analysis,plot_class_separability_analysis,plot_regression_visualizations,
    save_accuracy_table,regression_contrastive_loss, few_shot_finetune_classification
)
from federated_dataset import (
    create_global_test_loader_with_phase,create_calibration_loader, 
    create_client_full_test_loader, create_client_test_only_loader,
    create_merged_test_loader, create_merged_calibration_loader
)
from data_scheduler import create_data_scheduler, GasSensorPhaseDataset
from split_dataset import create_federated_dataset
from torch.utils.data import DataLoader, Subset


# 全局源域特征缓存变量（模块级）
_source_feat_cache = {
    'feats': None,
    'labels': None,
    'round': -1,
    'phase_profile': None,
}


def get_source_features_for_coral(
    model, global_test_loader, device, current_round,
    use_time_drift=False, refresh_interval=5, max_samples=1000, logger=None
):
    """
    获取用于 CORAL 的源域特征，支持自动缓存与失效。

    Args:
        model: 当前全局模型（特征提取器）
        global_test_loader: 源域数据加载器
        device: 设备
        current_round: 当前通信轮次
        use_time_drift: 是否启用时序漂移（数据分布会变化）
        refresh_interval: 时序漂移模式下每多少轮强制刷新
        max_samples: 最大样本数
        logger: 日志器

    Returns:
        (feats, labels): 均为 CPU 上的 Tensor
    """
    global _source_feat_cache

    need_refresh = False

    if _source_feat_cache['feats'] is None:
        need_refresh = True
    elif use_time_drift:
        if current_round - _source_feat_cache['round'] >= refresh_interval:
            need_refresh = True

    if need_refresh:
        feats_np, labels_np = extract_features_batch(
            model, global_test_loader, device, max_samples=max_samples
        )
        _source_feat_cache['feats'] = torch.from_numpy(feats_np).float()
        _source_feat_cache['labels'] = torch.from_numpy(labels_np).long()
        _source_feat_cache['round'] = current_round
        if logger:
            logger.info(f"Refreshed source features at round {current_round}")
    else:
        if logger:
            logger.info(f"Reusing cached source features from round {_source_feat_cache['round']}")

    return _source_feat_cache['feats'], _source_feat_cache['labels']


def setup_config(args):
    """设置实验配置"""
    config = FLConfig()
    config.SEED = args.seed
    set_random_seed(config.SEED)

    config.GLOBAL_ROUNDS = args.rounds
    config.LOCAL_EPOCHS = args.local_epochs
    config.LAMBDA_ALIGN = args.lambda_align
    config.LAMBDA_REPLAY_DISTILL = args.lambda_distill
    config.LAMBDA_PROTO = args.lambda_proto
    config.LAMBDA_CONSISTENCY = args.lambda_consistency
    config.NUM_CONC_BUCKETS = args.num_conc_buckets
    config.LAMBDA_CONC_BUCKET = args.lambda_conc_bucket
    config.CONC_BUCKET_LOSS = args.conc_bucket_loss
    config.CONC_BUCKET_SOFT_SIGMA = args.conc_bucket_soft_sigma
    config.CONC_BUCKET_DETACH_FEAT = bool(args.conc_bucket_detach_feat)
    config.REG_TAIL_WEIGHT = args.reg_tail_weight
    config.REG_TAIL_THRESHOLD = args.reg_tail_threshold
    config.REG_TAIL_CLASSES = args.reg_tail_classes
    config.REG_WINDOW_STATS = bool(args.reg_window_stats)
    config.REG_WINDOW_STATS_MODE = args.reg_window_stats_mode
    config.REG_WINDOW_STATS_DIM = int(args.reg_window_stats_dim)
    config.REGRESSION_MODE = args.regression_mode
    config.EVAL_TEST_ONLY = args.eval_test_only
    config.SEPARATE_REG_HUBER_DELTAS = getattr(args, 'separate_reg_huber_deltas', '')
    config.SEPARATE_REG_ALLOW_ENCODER_BACKPROP = bool(
        getattr(args, 'separate_reg_allow_encoder_backprop', True)
    )

    if args.use_reg_loss is not None:
        config.USE_REG_LOSS = args.use_reg_loss
    if args.regression_mode == 'separate':
        config.USE_REG_LOSS = False #回归损失


    # 消融实验配置
    ablation_configs = {
        'gaps_full': {
            'USE_LEARNABLE_AGG': True, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': True, 'USE_SERVER_OPT': True, 'USE_DP': False,
            # 以下保持默认 True（全功能）
        },
        'gaps_no_server_opt': {
            'USE_LEARNABLE_AGG': True, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': True, 'USE_SERVER_OPT': False, 'USE_DP': False
        },
        'gaps_no_align': {
            'USE_LEARNABLE_AGG': True, 'USE_ALIGN': False,
            'USE_REPLAY_DISTILL': True, 'USE_SERVER_OPT': True, 'USE_DP': False
        },
        'gaps_no_distill': {
            'USE_LEARNABLE_AGG': True, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': False, 'USE_SERVER_OPT': True, 'USE_DP': False
        },
        'fedavg': {
            'USE_LEARNABLE_AGG': False, 'USE_ALIGN': False,
            'USE_REPLAY_DISTILL': False, 'USE_SERVER_OPT': False, 'USE_DP': False,
            'USE_SELECTIVE_AGG': False,          # 关闭选择性聚合
            'USE_PROTO_DECOUPLING': False,       # 关闭原型解耦
            'USE_SOFT_AGGREGATION': False,       # 关闭软聚合推理
            'USE_MMD_REG': False                 # 关闭域泛化正则
        },
        'fedavg_align': {
            'USE_LEARNABLE_AGG': False, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': False, 'USE_SERVER_OPT': False, 'USE_DP': False,
            'USE_SELECTIVE_AGG': False,
            'USE_PROTO_DECOUPLING': False,
            'USE_SOFT_AGGREGATION': True,        # 对齐需要原型，推理时仍可使用软聚合（可控制）
            'USE_MMD_REG': False
        },
        'gaps_no_decoupling': {
            'USE_LEARNABLE_AGG': True, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': True, 'USE_SERVER_OPT': True, 'USE_DP': False,
            'USE_PROTO_DECOUPLING': False
        },
        'gaps_no_soft_agg': {
            'USE_LEARNABLE_AGG': True, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': True, 'USE_SERVER_OPT': True, 'USE_DP': False,
            'USE_SOFT_AGGREGATION': False
        },
        'gaps_fedavg': {
            'USE_LEARNABLE_AGG': False, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': True, 'USE_SERVER_OPT': True, 'USE_DP': False
        },
        # 新增：回归相关消融
        'gaps_no_rank_loss': {         # 关闭排序损失，仅使用MSE+相对误差
            'USE_LEARNABLE_AGG': True, 'USE_ALIGN': True,
            'USE_REPLAY_DISTILL': True, 'USE_SERVER_OPT': True, 'USE_DP': False,
            'USE_RANKING_LOSS': False
        },
        'gaps_no_fewshot_reg': {      # 关闭少样本回归微调（在 final_evaluation 中控制）
            # 此组合通过命令行 --few_shot_regression False 实现，无需在这里设置
            # 但我们可以保留一个占位，或通过代码识别
        }
    }

    if args.combination not in ablation_configs:
        raise ValueError(f"Unknown combination: {args.combination}")
    for key, val in ablation_configs[args.combination].items():
        setattr(config, key, val)

    # 固定其他超参数
    config.USE_DP = False  # 本实验不启用差分隐私

    # 分阶段训练配置覆盖
    if args.stagewise:
        config.STAGEWISE_TRAINING = True
        config.PHASE1_END_ROUND = args.phase1_rounds
        config.PHASE2_REG_WEIGHT = args.phase2_reg_weight
    else:
        config.STAGEWISE_TRAINING = False

    # 回归头共享配置覆盖
    if args.no_share_reg_head:
        config.SHARE_REG_HEAD = False
    elif args.share_reg_head:
        config.SHARE_REG_HEAD = True

    if args.no_personalized_reg:
        config.PERSONALIZED_REG = False
    elif args.personalized_reg:
        config.PERSONALIZED_REG = True

    # 回归梯度阻断配置覆盖
    config.REG_GRAD_DETACH = args.reg_grad_detach

    # 服务器优化步数覆盖
    if args.server_opt_steps is not None:
        config.SERVER_OPT_STEPS_BASE = args.server_opt_steps

    calib_client_ids = [int(x) for x in args.coral_calib_clients.split(',') if x.strip()]

    # 深度CORAL配置
    if args.no_use_deep_coral:
        config.USE_DEEP_CORAL = False
    elif args.use_deep_coral:
        config.USE_DEEP_CORAL = True
        config.DEEP_CORAL_CALIB_SIZE = args.coral_calib_size

    # 域对抗训练配置（服务器端Wasserstein GAN）
    if args.no_use_adversarial_domain:
        config.USE_ADVERSARIAL_DOMAIN = False
    elif args.use_adversarial_domain:
        config.USE_ADVERSARIAL_DOMAIN = True
        config.LAMBDA_ADV_DOMAIN = args.lambda_adv_domain
        config.ADV_DOMAIN_LR = args.adv_domain_lr
        if args.adv_no_class_conditional:
            config.ADV_CLASS_CONDITIONAL = False

    # Transformer编码器配置
    if config.USE_DEEP_CORAL or config.USE_ADVERSARIAL_DOMAIN:
        config.DEEP_CORAL_CALIB_CLIENTS = calib_client_ids
        config.DEEP_CORAL_CALIB_SIZE = args.coral_calib_size

    if args.encoder == 'transformer':
        config.USE_TRANSFORMER_ENCODER = True
        config.TRANSFORMER_D_MODEL = args.transformer_d_model
        config.TRANSFORMER_NHEAD = args.transformer_nhead
        config.TRANSFORMER_NUM_LAYERS = args.transformer_num_layers
        config.TRANSFORMER_FF_DIM = args.transformer_ff_dim
    config.TCN_NORM = args.tcn_norm

    # MMD对齐消融开关
    if args.no_use_mmd_alignment:
        config.USE_MMD_ALIGNMENT = False
        config.USE_PROTO_MMD = False

    # 校准集标签控制开关（监督 vs 无监督域适应）
    if getattr(args, 'calib_use_labels', False):
        config.CALIB_USE_LABELS = True
    if args.no_calib_use_labels:
        config.CALIB_USE_LABELS = False

    # 马氏距离推理开关（影响推理分类决策 + 选择性聚合 + 部署阶段分配）
    if args.use_mahalanobis_inference:
        config.USE_MAHALANOBIS_INFERENCE = True

    # 自适应温度推理开关
    if args.use_adaptive_temperature:
        config.USE_ADAPTIVE_TEMPERATURE = True

    # TENT部署适应配置
    config.TENT_WEIGHT = args.tent_weight
    config.ANCHOR_REG_WEIGHT = args.anchor_reg_weight
    config.MT_EMA_DECAY = args.ema_decay

    return config

def setup_experiment(args, config):
    """设置实验环境"""
    from config import generate_config_signature
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    train_clients_str = args.train_clients.replace(',', '')
    test_clients_str = args.test_clients.replace(',', '')
    config_sig = generate_config_signature(config)
    short_combo = args.combination.replace('gaps_', '')
    experiment_name = f"exp_{short_combo}_s{config.SEED}_r{config.GLOBAL_ROUNDS}_le{config.LOCAL_EPOCHS}_t{train_clients_str}_te{test_clients_str}{config_sig}_{timestamp}"
    if args.output_dir:
        exp_dir = Path(args.output_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = create_experiment_dir(config.RESULT_SAVE_DIR, experiment_name)
    config.PLOT_SAVE_DIR = str(exp_dir / "plots")
    config.MODEL_SAVE_DIR = str(exp_dir / "checkpoints")
    results_dir = str(exp_dir / "results")
    for path in [results_dir, config.MODEL_SAVE_DIR, config.PLOT_SAVE_DIR]:
        os.makedirs(path, exist_ok=True)

    logger = setup_logging(results_dir, log_filename="federated.log")
    logger.info(f"=== GAPS Experiment: {args.combination} ===")
    logger.info(f"Config: { {k: v for k, v in config.__dict__.items() if not k.startswith('_')} }")
    print_experiment_config(config)

    return exp_dir, logger

def copy_client_data_to_temp(train_client_ids, federated_dir, logger):
    """将客户端数据复制到临时目录
    
    Args:
        train_client_ids: 训练客户端ID列表
        federated_dir: 联邦数据目录
        logger: 日志记录器
    
    Returns:
        temp_federated_dir: 临时目录路径
    """
    import tempfile
    temp_federated_dir = tempfile.mkdtemp(prefix="client_data_federated_temp_")
    logger.info(f"Created temporary training directory: {temp_federated_dir}")

    for cid in train_client_ids:
        src = os.path.join(federated_dir, f'client_{cid}')
        dst = os.path.join(temp_federated_dir, f'client_{cid}')
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
    
    # 复制norm_stats.npz到临时目录
    norm_stats_src = os.path.join(federated_dir, 'norm_stats.npz')
    norm_stats_dst = os.path.join(temp_federated_dir, 'norm_stats.npz')
    if os.path.exists(norm_stats_src) and not os.path.exists(norm_stats_dst):
        shutil.copy2(norm_stats_src, norm_stats_dst)
        logger.info(f"Copied norm_stats.npz to temporary directory")
    
    return temp_federated_dir

def build_client_loaders(train_client_ids, temp_federated_dir, config, logger):
    """构建客户端数据加载器
    
    Args:
        train_client_ids: 训练客户端ID列表
        temp_federated_dir: 临时目录路径
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        client_loaders: 客户端数据加载器列表
    """
    logger.info("Loading all data without time drift simulation")
    client_loaders = []
    import numpy as np
    
    for cid in train_client_ids:
        client_dir = os.path.join(temp_federated_dir, f'client_{cid}')
        if os.path.exists(client_dir):
            features = np.load(os.path.join(client_dir, 'train_features.npy'))
            cls_labels = np.load(os.path.join(client_dir, 'train_classification_labels.npy'))
            reg_labels = np.load(os.path.join(client_dir, 'train_regression_labels.npy'))
            phase_labels = np.load(os.path.join(client_dir, 'train_phase_labels.npy'), allow_pickle=True)
            dataset = GasSensorPhaseDataset(
                features, cls_labels, reg_labels, phase_labels, phase=None
            )
            loader = DataLoader(
                dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
            )
            client_loaders.append(loader)
            logger.info(f"Client {cid} loaded {len(dataset)} samples")
        else:
                logger.error(f"Client {cid} directory not found: {client_dir}")
                # 跳过不存在的客户端目录
                continue
    
    return client_loaders

def build_test_loaders(test_client_ids, federated_dir, config, logger, eval_test_only=False):
    """Build target-client evaluation loaders.

    eval_test_only=True reads only test_*.npy; False keeps legacy full split behavior.
    """
    test_client_loaders = {}
    loader_fn = create_client_test_only_loader if eval_test_only else create_client_full_test_loader
    split_name = "test-only" if eval_test_only else "full(train+test+calibration)"
    for cid in test_client_ids:
        client_dir = os.path.join(federated_dir, f'client_{cid}')
        loader = loader_fn(client_dir, batch_size=config.BATCH_SIZE)
        test_client_loaders[cid] = loader
        try:
            if hasattr(loader, 'dataset') and hasattr(loader.dataset, '__len__'):
                dataset_size = len(loader.dataset) # type: ignore
                logger.info(f"Test client {cid} {split_name} dataset size: {dataset_size}")
            else:
                logger.info(f"Test client {cid} {split_name} dataset size: unknown")
        except (AttributeError, TypeError):
            logger.info(f"Test client {cid} {split_name} dataset size: unknown")
    
    return test_client_loaders

def _split_allows_file_overlap(federated_dir):
    info_path = Path(federated_dir) / 'split_info.json'
    if not info_path.exists():
        return False
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return bool(info.get('allows_file_overlap', False)) or info.get('split_level') == 'window'
    except Exception:
        return False


def check_data_leakage(train_client_ids, temp_federated_dir, logger, allows_file_overlap=False):
    """检查数据泄露
    
    Args:
        train_client_ids: 训练客户端ID列表
        temp_federated_dir: 临时目录路径
        logger: 日志记录器
    """
    import json
    train_files = set()
    for cid in train_client_ids:
        train_info_path = os.path.join(temp_federated_dir, f'client_{cid}', 'train_experiment_info.json')
        if os.path.exists(train_info_path):
            with open(train_info_path, 'r') as f:
                try:
                    train_info = json.load(f)
                    for exp in train_info:
                        if 'filename' in exp:
                            train_files.add(exp['filename'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to load train_experiment_info.json for client {cid}")
    
    test_files = set()
    for cid in train_client_ids:
        test_info_path = os.path.join(temp_federated_dir, f'client_{cid}', 'test_experiment_info.json')
        if os.path.exists(test_info_path):
            with open(test_info_path, 'r') as f:
                try:
                    test_info = json.load(f)
                    for exp in test_info:
                        if 'filename' in exp:
                            test_files.add(exp['filename'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to load test_experiment_info.json for client {cid}")
    
    # 计算交集
    overlap_files = train_files & test_files
    logger.info(f"=== Data Leakage Check ===")
    logger.info(f"Training files: {len(train_files)}")
    logger.info(f"Test files: {len(test_files)}")
    logger.info(f"Overlap files: {len(overlap_files)}")
    if overlap_files and allows_file_overlap:
        logger.info("File overlap is expected for this window-level split; windows remain split units.")
    elif overlap_files:
        logger.warning(f"Data leakage detected! Overlap files: {overlap_files}")
    else:
        logger.info("No data leakage detected. Train and test sets use different files.")

def check_phase_and_class_distribution(loader, logger, set_name):
    """一次遍历同时获取阶段分布和阶段-类别分布
    
    Args:
        loader: 数据加载器
        logger: 日志记录器
        set_name: 数据集名称
    """
    from collections import Counter
    import numpy as np
    phase_counts = np.zeros(3, dtype=int)
    phase_cls = {0: Counter(), 1: Counter(), 2: Counter()}
    phase_names = {0: 'early', 1: 'middle', 2: 'late'}
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    
    for batch in loader:
        _, cls, _, phases = batch
        phases_np = phases.numpy()
        cls_np = cls.numpy()
        phase_counts += np.bincount(phases_np, minlength=3)
        for p, c in zip(phases_np, cls_np):
            phase_cls[int(p)][int(c)] += 1
    
    total = sum(phase_counts)
    if total == 0:
        logger.warning(f"{set_name} is empty")
        return
    
    logger.info(f"{set_name} phase distribution: early={phase_counts[0]}, middle={phase_counts[1]}, late={phase_counts[2]}")
    logger.info(f"=== Phase-wise Class Distribution in {set_name} ===")
    for p in [0, 1, 2]:
        if phase_counts[p] == 0:
            continue
        logger.info(f"Phase {phase_names[p]} (total={phase_counts[p]}):")
        for c in range(4):
            count = phase_cls[p].get(c, 0)
            logger.info(f"  {gas_names[c]}: {count} ({count/phase_counts[p]*100:.1f}%)")

def build_global_test_loader(train_client_dirs, federated_dir, config, logger):
    """构建全局测试加载器
    
    Args:
        train_client_dirs: 训练客户端目录列表
        federated_dir: 联邦数据目录
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        global_test_loader: 全局测试加载器
        val_loader: 验证加载器
    """
    try:
        # 转换为Path对象以匹配函数参数类型
        train_client_paths: Sequence[Union[str, Path]] = [Path(d) for d in train_client_dirs]
        global_test_loader = create_merged_test_loader(train_client_paths, batch_size=config.BATCH_SIZE)
        val_loader = create_merged_calibration_loader(train_client_paths, batch_size=config.BATCH_SIZE)
        logger.info(f"Built global test/calibration from {len(train_client_paths)} training clients.")
        
        # 检查全局测试集的阶段分布和阶段-类别分布
        check_phase_and_class_distribution(global_test_loader, logger, "Global test set")
        
        # 检查数据泄露
        check_data_leakage(
            [int(d.split('_')[-1]) for d in train_client_dirs],
            os.path.dirname(train_client_dirs[0]), logger,
            allows_file_overlap=_split_allows_file_overlap(federated_dir)
        )
    except Exception as e:
        logger.error(f"Failed to build merged test/calibration: {e}, falling back to pre-built global sets (may contain leakage)")
        # 兜底：仍使用原始目录下的 global_test/ 和 calibration/
        global_test_loader = create_global_test_loader_with_phase(federated_dir, batch_size=config.BATCH_SIZE)
        val_loader = create_calibration_loader(federated_dir, batch_size=config.BATCH_SIZE)
        
        # 检查全局测试集的阶段分布和阶段-类别分布
        check_phase_and_class_distribution(global_test_loader, logger, "Fallback global test set")
    
    return global_test_loader, val_loader


def _split_audit_for_client(client_dir: Path, num_classes: int = 4) -> Dict[str, Any]:
    """Summarize class/concentration/phase coverage for one client split."""
    result: Dict[str, Any] = {}
    for split in ["train", "calibration", "test"]:
        feat_path = client_dir / f"{split}_features.npy"
        cls_path = client_dir / f"{split}_classification_labels.npy"
        reg_path = client_dir / f"{split}_regression_labels.npy"
        phase_path = client_dir / f"{split}_phase_labels.npy"
        if not (feat_path.exists() and cls_path.exists() and reg_path.exists()):
            continue
        features = np.load(feat_path, mmap_mode='r')
        cls_labels = np.load(cls_path)
        reg_labels = np.load(reg_path)
        split_info: Dict[str, Any] = {
            "n_samples": int(len(features)),
            "classes": {},
        }
        for cls_id in range(num_classes):
            mask = cls_labels == cls_id
            if np.any(mask):
                concentrations = np.unique(np.round(reg_labels[mask, cls_id].astype(float), 6))
                split_info["classes"][str(cls_id)] = {
                    "n_samples": int(mask.sum()),
                    "n_concentration_levels": int(len(concentrations)),
                    "concentrations": [float(v) for v in concentrations.tolist()],
                }
            else:
                split_info["classes"][str(cls_id)] = {
                    "n_samples": 0,
                    "n_concentration_levels": 0,
                    "concentrations": [],
                }
        if phase_path.exists():
            phases = np.load(phase_path, allow_pickle=True)
            unique, counts = np.unique(phases, return_counts=True)
            split_info["phase_counts"] = {str(k): int(v) for k, v in zip(unique, counts)}
        result[split] = split_info
    return result


def _experiment_files(client_dir: Path, split: str) -> set:
    info_path = client_dir / f"{split}_experiment_info.json"
    if not info_path.exists():
        return set()
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except Exception:
        return set()
    return {item.get('filename') for item in info if isinstance(item, dict) and item.get('filename')}


def audit_federated_split(federated_dir: Union[str, Path], client_ids: Sequence[int],
                          logger, save_dir: Optional[Union[str, Path]] = None,
                          num_classes: int = 4) -> Dict[str, Any]:
    """Audit split coverage and leakage for selected federated clients."""
    root = Path(federated_dir)
    audit: Dict[str, Any] = {
        "data_dir": str(root),
        "clients": {},
    }
    allows_file_overlap = _split_allows_file_overlap(root)
    audit["allows_file_overlap"] = bool(allows_file_overlap)
    logger.info("=== Federated Data Split Audit ===")
    logger.info(f"Data directory: {root}")
    if allows_file_overlap:
        logger.info("Split protocol allows original-file overlap because split_level=window.")
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    for cid in client_ids:
        client_dir = root / f"client_{cid}"
        if not client_dir.exists():
            logger.warning(f"Data audit: client directory missing: {client_dir}")
            continue
        client_audit = _split_audit_for_client(client_dir, num_classes=num_classes)
        overlap = {}
        split_files = {split: _experiment_files(client_dir, split) for split in ["train", "calibration", "test"]}
        for left, right in [("train", "calibration"), ("train", "test"), ("calibration", "test")]:
            overlap[f"{left}_vs_{right}"] = len(split_files[left] & split_files[right])
        client_audit["file_overlap"] = overlap
        audit["clients"][str(cid)] = client_audit
        overlap_total = sum(overlap.values())
        if overlap_total > 0 and allows_file_overlap:
            logger.info(f"Client {cid}: expected file overlap under window split {overlap}")
        elif overlap_total > 0:
            logger.warning(f"Client {cid}: unexpected file overlap {overlap}")
        else:
            logger.info(f"Client {cid}: file overlap {overlap}")
        for split in ["train", "calibration", "test"]:
            split_info = client_audit.get(split)
            if not split_info:
                continue
            parts = []
            for cls_id in range(num_classes):
                cls_info = split_info["classes"].get(str(cls_id), {})
                parts.append(
                    f"{gas_names[cls_id]} n={cls_info.get('n_samples', 0)} "
                    f"levels={cls_info.get('n_concentration_levels', 0)}"
                )
            logger.info(f"Client {cid} {split}: N={split_info['n_samples']} | " + " | ".join(parts))
    if save_dir is not None:
        save_path = Path(save_dir) / "data_split_audit.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        logger.info(f"Data split audit saved to {save_path}")
    return audit

def setup_data(args, config, logger):
    """准备数据"""
    processed_dir = args.processed_dir
    federated_dir = args.data_dir
    if not Path(federated_dir).exists():
        logger.info("Creating federated dataset...")
        create_federated_dataset(processed_dir, federated_dir)

    train_client_ids = [int(x.strip()) for x in args.train_clients.split(',')]
    test_client_ids = [int(x.strip()) for x in args.test_clients.split(',') if x.strip()]
    logger.info(f"Using federated data directory: {federated_dir}")
    audit_save_dir = Path(config.MODEL_SAVE_DIR).parent / "results"
    audit_client_ids = sorted(set(train_client_ids + test_client_ids + getattr(config, 'DEEP_CORAL_CALIB_CLIENTS', [])))
    audit_federated_split(
        federated_dir, audit_client_ids, logger,
        save_dir=audit_save_dir, num_classes=config.NUM_CLASSES
    )


    # 创建唯一临时目录（避免并行运行冲突）
    temp_federated_dir = copy_client_data_to_temp(train_client_ids, federated_dir, logger)

    # 数据加载逻辑
    if args.no_time_drift:
        client_loaders = build_client_loaders(train_client_ids, temp_federated_dir, config, logger)
        scheduler = None
    else:
        phase_ratios = [float(x) for x in args.phase_ratios.split(',')]
        use_cumulative = not args.no_use_cumulative
        scheduler = create_data_scheduler(
            temp_federated_dir, batch_size=config.BATCH_SIZE,
            phase_ratios=phase_ratios, use_cumulative=use_cumulative, device=config.DEVICE
        )
        client_loaders = None

    # 测试客户端加载器（全量数据，用于零样本评估）
    test_client_loaders = build_test_loaders(test_client_ids, federated_dir, config, logger, eval_test_only=args.eval_test_only)

    # 全局测试集与校准集：仅从训练客户端的测试/校准数据动态合并
    train_client_dirs = [os.path.join(temp_federated_dir, f'client_{cid}') for cid in train_client_ids]
    global_test_loader, val_loader = build_global_test_loader(train_client_dirs, federated_dir, config, logger)

    # === 生成浓度桶边界 ===
    if config.NUM_CONC_BUCKETS > 0:
        from utils import compute_conc_bucket_boundaries
        # 从训练客户端收集浓度数据
        conc_data = []
        cls_data = []
        for cid in train_client_ids:
            # 从原始 federated_dir 读取数据（更可靠）
            client_dir = os.path.join(federated_dir, f'client_{cid}')
            y_cls_path = os.path.join(client_dir, 'train_classification_labels.npy')
            y_reg_path = os.path.join(client_dir, 'train_regression_labels.npy')
            if os.path.exists(y_cls_path) and os.path.exists(y_reg_path):
                y_cls = np.load(y_cls_path)
                y_reg = np.load(y_reg_path)
                conc_data.append(y_reg)
                cls_data.append(y_cls)
        
        if len(conc_data) > 0:
            conc_data = np.concatenate(conc_data)
            cls_data = np.concatenate(cls_data)
            
            config.CONC_BUCKET_BOUNDARIES = compute_conc_bucket_boundaries(
                conc_data, cls_data, config.NUM_CLASSES, config.NUM_CONC_BUCKETS
            )
            logger.info(f"Generated concentration bucket boundaries: {config.CONC_BUCKET_BOUNDARIES}")
        else:
            logger.warning("No training data found for concentration bucket boundary generation")

    # === 构建深度CORAL校准集 / 域对抗训练目标域数据 ===
    # 数据来源: 测试客户端的 calibration_features.npy (10% 预分割)
    # 标签模式: CALIB_USE_LABELS=True → 加载真实标签; False → 仅特征(无标签)
    calib_loader = None
    need_calib = config.USE_DEEP_CORAL or config.USE_ADVERSARIAL_DOMAIN or config.USE_MMD_ALIGNMENT
    if need_calib and config.DEEP_CORAL_CALIB_CLIENTS:
        all_calib_features = []
        all_calib_cls_labels = []
        use_labels = getattr(config, 'CALIB_USE_LABELS', True)
        
        for cid in config.DEEP_CORAL_CALIB_CLIENTS:
            client_dir = os.path.join(federated_dir, f'client_{cid}')
            feat_path = os.path.join(client_dir, 'calibration_features.npy')
            cls_path = os.path.join(client_dir, 'calibration_classification_labels.npy')
            
            if os.path.exists(feat_path):
                features = np.load(feat_path)
                n_total = len(features)
                n_select = min(config.DEEP_CORAL_CALIB_SIZE, n_total)
                indices = np.random.choice(n_total, n_select, replace=False)
                all_calib_features.append(features[indices])
                
                if use_labels and os.path.exists(cls_path):
                    cls_labels = np.load(cls_path)
                    all_calib_cls_labels.append(cls_labels[indices])
                    logger.info(f"Calib: loaded {n_select} labeled samples from Client {cid} calibration set")
                else:
                    logger.info(f"Calib: loaded {n_select} unlabeled samples from Client {cid} calibration set")
            else:
                logger.warning(f"Calib: Client {cid} calibration features not found.")
        
        if all_calib_features:
            calib_features = np.concatenate(all_calib_features)
            from torch.utils.data import TensorDataset, DataLoader
            if all_calib_cls_labels:
                calib_cls_labels = np.concatenate(all_calib_cls_labels)
                calib_dataset = TensorDataset(
                    torch.from_numpy(calib_features).float(),
                    torch.from_numpy(calib_cls_labels).long()
                )
                logger.info(f"Calib loader: {len(calib_features)} samples WITH labels")
            else:
                calib_dataset = TensorDataset(
                    torch.from_numpy(calib_features).float()
                )
                logger.info(f"Calib loader: {len(calib_features)} samples WITHOUT labels")
            calib_loader = DataLoader(calib_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 注意：临时目录 temp_federated_dir 需要在整个训练结束后清理
    # 我们将清理工作放到 main 函数的 finally 块，或通过返回值传递
    return train_client_ids, test_client_ids, temp_federated_dir, scheduler, client_loaders, test_client_loaders, global_test_loader, val_loader, federated_dir, calib_loader

def setup_model(config, global_test_loader, test_client_loaders, test_client_ids, train_client_ids, val_loader, logger, calib_loader=None):
    """初始化模型、服务器、客户端"""
    from utils import create_model_by_config, load_shared_weights
    # 服务器全局模型不含回归头（回归参数为客户端私有，不参与聚合）
    global_model = create_model_by_config(config, with_reg_head=False).to(config.DEVICE)
    first_test_id = test_client_ids[0] if test_client_ids else None
    server = Server(global_model, val_loader, global_test_loader,
                    test_client_loaders.get(first_test_id) if first_test_id is not None else None, config, logger, calib_loader=calib_loader)

    clients: List[Client] = []
    for cid in train_client_ids:
        # 根据 USE_REG_LOSS 决定是否使用多任务模型（含回归头）
        client_model = create_model_by_config(config, with_reg_head=config.USE_REG_LOSS).to(config.DEVICE)
        # 加载服务器下发的共享参数（strict=False 允许缺少回归头）
        load_shared_weights(client_model, global_model.state_dict(), strict=False)
        client = Client(client_id=cid, config=config)
        client.set_model(client_model)
        clients.append(client)
        logger.info(f"Initialized training client {cid}")

    return global_model, server, clients


def warmup_and_init_prototypes(clients, scheduler, client_loaders, server, config, logger, args):
    """预热训练和初始化原型"""
    logger.info("Pre-training warmup...")
    if args.no_time_drift:
        if client_loaders is None:
            logger.error("client_loaders is None when no_time_drift is True")
            return
        # 列表转字典，按顺序匹配
        loaders_dict = {cl.client_id: ld for cl, ld in zip(clients, client_loaders)}
    else:
        if scheduler is None:
            logger.error("scheduler is None when no_time_drift is False")
            return
        loaders_dict = scheduler.get_client_loaders(current_round=0, total_rounds=config.GLOBAL_ROUNDS)
        # 直接得到字典

    for client in clients:
        cid = client.client_id
        # 尝试两种格式的客户端ID
        if cid not in loaders_dict:
            # 尝试带前缀的格式
            cid_str = f"client_{cid}"
            if cid_str not in loaders_dict:
                logger.warning(f"Client {cid} has no data for warmup, skip")
                continue
            client.update_dataloader(loaders_dict[cid_str])
        else:
            client.update_dataloader(loaders_dict[cid])
        client.train_warmup(epochs=2)

    # 聚合参数，注意只针对训练过的客户端
    trained_clients = []
    for cl in clients:
        cid = cl.client_id
        # 检查两种格式的客户端ID
        if cid in loaders_dict or f"client_{cid}" in loaders_dict:
            trained_clients.append(cl)
    if not trained_clients:
        logger.error("No clients trained during warmup")
        return
    skip_reg = False
    if hasattr(config, 'PERSONALIZED_REG') and hasattr(config, 'SHARE_REG_HEAD'):
        skip_reg = config.PERSONALIZED_REG and not config.SHARE_REG_HEAD
    elif hasattr(config, 'PERSONALIZED_REG'):
        skip_reg = config.PERSONALIZED_REG
    partial_personalize = bool(
        getattr(config, 'PERSONALIZED_REG', False) and
        getattr(config, 'SHARE_REG_HEAD', False)
    )
    client_params = [cl.get_parameters(skip_reg=skip_reg, partial_personalize=partial_personalize)
                     for cl in trained_clients]
    warmup_weights = torch.ones(len(trained_clients), device=config.DEVICE) / len(trained_clients)
    fedavg_params = server._aggregate_params(client_params, warmup_weights)
    server.global_model.load_state_dict(fedavg_params, strict=False)
    if getattr(config, 'USE_REG_LOSS', False) and getattr(config, 'SHARE_REG_HEAD', False):
        server.aggregate_shared_regression_state(client_params, warmup_weights)
    warmup_state = dict(fedavg_params)
    if getattr(server, 'shared_reg_state', None):
        warmup_state.update(server.shared_reg_state)
    for cl in clients:
        cid = cl.client_id
        if cid in loaders_dict or f"client_{cid}" in loaders_dict:
            cl.set_parameters(warmup_state, skip_reg=skip_reg,
                              partial_personalize=partial_personalize)

    all_mus, all_counts = [], []
    for cl in trained_clients:
        mu, cnt = cl.compute_means_only()
        if mu:
            all_mus.append(mu)
            all_counts.append(cnt)
    
    if not all_mus:
        logger.error("No prototypes computed during warmup! Using random initialization.")
        # 回退：随机初始化原型
        for c in range(config.NUM_CLASSES):
            for p in range(config.NUM_PHASES):
                str_key = f"({c},{p})"
                server.semantic_protos[str_key] = nn.Parameter(
                    torch.randn(config.HIDDEN_DIM2, device=config.DEVICE) * 0.01
                )
    else:
        server._init_protos_from_clients(all_mus, all_counts, server.client_weights)
    logger.info(f"Initialized {len(server.semantic_protos)} semantic prototypes.")


def get_client_loaders(round_idx, config, args, scheduler, clients, client_loaders, logger):
    """获取客户端数据加载器
    
    Args:
        round_idx: 当前轮次
        config: 配置对象
        args: 命令行参数
        scheduler: 数据调度器
        clients: 客户端列表
        client_loaders: 预加载的客户端加载器
        logger: 日志记录器
    
    Returns:
        client_loaders_dict: 客户端数据加载器字典，键为客户端ID
    """
    if args.no_time_drift:
        if client_loaders is None:
            logger.error("client_loaders is None when no_time_drift is True")
            return None
        return {cl.client_id: ld for cl, ld in zip(clients, client_loaders)}
    else:
        if scheduler is None:
            logger.error("scheduler is None when no_time_drift is False")
            return None
        return scheduler.get_client_loaders(current_round=round_idx, total_rounds=config.GLOBAL_ROUNDS)


def prepare_clients(clients, client_loaders_dict, current_state, prev_state, logger):
    """准备客户端进行训练
    
    Args:
        clients: 客户端列表
        client_loaders_dict: 客户端数据加载器字典，键为客户端ID
        current_state: 当前全局模型状态
        prev_state: 上一轮模型状态
        logger: 日志记录器
    
    Returns:
        active_client_loader_pairs: 活跃客户端和其数据集大小的列表
    """
    active_client_loader_pairs = []
    for client in clients:
        cid = client.client_id
        # 尝试两种格式的客户端ID
        loader = client_loaders_dict.get(cid)
        if loader is None:
            # 尝试带前缀的格式
            cid_str = f"client_{cid}"
            loader = client_loaders_dict.get(cid_str)
        if loader is None or len(loader.dataset) == 0:
            logger.info(f"Client {cid} has no data, skipping.")
            continue
        client.update_dataloader(loader)
        skip_reg = False
        partial_personalize = False
        if hasattr(client.config, 'PERSONALIZED_REG') and hasattr(client.config, 'SHARE_REG_HEAD'):
            skip_reg = client.config.PERSONALIZED_REG and not client.config.SHARE_REG_HEAD
            partial_personalize = client.config.PERSONALIZED_REG and client.config.SHARE_REG_HEAD
        elif hasattr(client.config, 'PERSONALIZED_REG'):
            skip_reg = client.config.PERSONALIZED_REG
        client.set_parameters(current_state, skip_reg=skip_reg,
                              partial_personalize=partial_personalize)
        if prev_state is not None:
            client.set_prev_parameters(prev_state)
        else:
            client.set_prev_parameters(current_state)
        active_client_loader_pairs.append((client, len(loader.dataset)))
    
    if not active_client_loader_pairs:
        logger.warning("No active clients, skipping round.")
        return None
    
    return active_client_loader_pairs


def train_clients(active_client_loader_pairs, server, round_idx, logger):
    """串行训练客户端
    
    Args:
        active_client_loader_pairs: 活跃客户端和其数据集大小的列表
        server: 服务器对象
        round_idx: 当前轮次
        logger: 日志记录器
    
    Returns:
        client_params_list: 客户端参数列表
        all_mus: 客户端原型列表
        all_counts: 客户端样本计数列表
        client_Fs: 客户端特征列表
        client_residuals: 客户端残差列表
        all_vars: 客户端方差列表 (P2-1)
    """
    client_params_list, all_mus, all_counts, client_Fs, client_residuals, all_vars = [], [], [], [], [], []
    
    for cli, _ in active_client_loader_pairs:
        semantic_protos_detached = {k: v.detach().cpu() for k, v in server.semantic_protos.items()}
        full_protos_detached = server.get_global_protos_detached(client_id=cli.client_id)
        params, mu, cnt, F_i, dev_res, var = cli.train_one_round(round_idx, global_protos=full_protos_detached,
                                   semantic_protos=semantic_protos_detached)
        client_params_list.append(params)
        all_mus.append(mu)
        all_counts.append(cnt)
        client_Fs.append(F_i)
        client_residuals.append(dev_res)
        all_vars.append(var)
    
    return client_params_list, all_mus, all_counts, client_Fs, client_residuals, all_vars


def aggregate_model(config, server, client_params_list, active_client_loader_pairs, all_mus, round_idx, logger, client_Fs=None):
    """聚合客户端模型
    
    Args:
        config: 配置对象
        server: 服务器对象
        client_params_list: 客户端参数列表
        active_client_loader_pairs: 活跃客户端和其数据集大小的列表
        all_mus: 客户端原型列表
        round_idx: 当前轮次
        logger: 日志记录器
        client_Fs: 客户端特征均值列表 (可选, 用于目标感知聚合)
    
    Returns:
        w_new: 聚合权重
        final_params: 聚合后的参数
    """
    num_active = len(client_params_list)
    if config.USE_LEARNABLE_AGG and config.USE_REG_LOSS and getattr(config, 'FORCE_LEARNABLE_AGG_WITH_REG', False):
        if len(server.client_weights) != num_active:
            server.client_weights = torch.ones(num_active, device=config.DEVICE) / num_active
        base_weights = server.learnable_aggregate(client_params_list, server.client_weights)
        w_new = server.compute_selective_weights(all_mus, base_weights, round_idx, client_Fs=client_Fs)
    else:
        sample_sizes = [sz for _, sz in active_client_loader_pairs]
        total = sum(sample_sizes)
        base_weights = torch.tensor([s / total for s in sample_sizes], device=config.DEVICE)
        w_new = server.compute_selective_weights(all_mus, base_weights, round_idx, client_Fs=client_Fs)

    # 聚合模型
    final_params = server._aggregate_params(client_params_list, w_new)
    server.global_model.load_state_dict(final_params, strict=False)
    if getattr(config, 'USE_REG_LOSS', False) and getattr(config, 'SHARE_REG_HEAD', False):
        server.aggregate_shared_regression_state(client_params_list, w_new)
    
    return w_new, final_params


def server_optimization(config, server, all_mus, all_counts, w_new, client_ids, client_Fs, client_residuals, logger, current_round, all_vars=None):
    """服务器优化
    
    Args:
        config: 配置对象
        server: 服务器对象
        all_mus: 客户端原型列表
        all_counts: 客户端样本计数列表
        w_new: 聚合权重
        client_ids: 客户端ID列表
        client_Fs: 客户端特征列表
        client_residuals: 客户端残差列表
        logger: 日志记录器
        current_round: 当前训练轮次
    
    Returns:
        align_loss: 对齐损失
    """
    # 初始化/扩展原型（传入客户端ID和残差）
    server._init_protos_from_clients(all_mus, all_counts, w_new, client_ids, all_vars=all_vars)

    # 漂移估计与服务器优化（传入残差）
    K_t = server.compute_drift_and_K(client_Fs)
    align_loss = 0.0
    if config.USE_SERVER_OPT:
        align_loss = server.server_representation_learning(
            all_mus, all_counts, w_new, K_t, client_ids, client_residuals, current_round=current_round
        )

    # 生成超晚期虚拟原型 - 暂时注释掉，因为 Server 类没有这个方法
    # server.generate_super_late_protos()
    
    return align_loss


def evaluate_clients(config, server, test_client_ids, test_client_loaders, global_test_loader, args, tta_client_ids, logger, proto_temperatures=None):
    """评估客户端模型
    
    Args:
        config: 配置对象
        server: 服务器对象
        test_client_ids: 测试客户端ID列表
        test_client_loaders: 测试客户端加载器字典
        global_test_loader: 全局测试加载器
        args: 命令行参数
        tta_client_ids: 需要执行TTA的客户端ID列表
        logger: 日志记录器
        proto_temperatures: 每原型自适应温度 (K,) 或 None
    
    Returns:
        acc_early: 早期阶段准确率
        acc_middle: 中期阶段准确率
        acc_late: 晚期阶段准确率
        acc_test: 全局测试准确率
        test_accs: 各测试客户端准确率字典
    """
    import copy
    
    # 使用服务器端原型缓存机制加速推理
    proto_matrix, proto_classes = server.get_cached_protos()
    proto_matrix = proto_matrix.to(config.DEVICE) if proto_matrix is not None else None
    proto_classes = proto_classes.to(config.DEVICE) if proto_classes is not None else None
    
    # 准备软聚合所需参数（保留字典形式用于向后兼容）
    semantic_protos_for_eval = {k: v.detach().cpu() for k, v in server.semantic_protos.items()}
    # 马氏距离推理：准备各原型的对角方差
    use_mahalanobis_eval = getattr(config, 'USE_MAHALANOBIS_INFERENCE', False)
    proto_vars_for_eval = {
        k: v.detach().cpu() for k, v in server.semantic_proto_vars.items()
    } if (use_mahalanobis_eval and server.semantic_proto_vars) else None

    # 评估（使用缓存的原型矩阵）
    eval_results = evaluate_model_with_phase(
        server.global_model, global_test_loader, config.DEVICE,
        semantic_protos=semantic_protos_for_eval,
        device_residuals=None,  # 零样本，残差置零
        use_soft_agg=config.USE_SOFT_AGGREGATION,
        soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
        num_classes=config.NUM_CLASSES
    )
    acc_test = eval_results['global']
    acc_early = eval_results['early']
    acc_middle = eval_results['middle']
    acc_late = eval_results['late']

    hard_client_ids = [int(x.strip()) for x in args.hard_clients.split(',')] if args.hard_clients else []
    # 预提取源域特征（只要存在困难客户端且启用了CORAL）
    coral_source_feats = None
    if args.use_coral and hard_client_ids:
        from utils import extract_features_batch
        source_feats_np, _ = extract_features_batch(server.global_model, global_test_loader, config.DEVICE, max_samples=1000)
        coral_source_feats = torch.from_numpy(source_feats_np).float()
        logger.info(f"Extracted {len(coral_source_feats)} source features for CORAL")

    test_accs = {}
    for cid in test_client_ids:
        # 估计设备残差
        from utils import get_device_residual
        device_res = get_device_residual(args, server, cid, test_client_loaders, semantic_protos_for_eval, config.DEVICE)

        # 决定该客户端的CORAL参数
        client_coral_feats = None
        use_cond_coral = False
        if args.use_coral and cid in hard_client_ids:
            client_coral_feats = coral_source_feats
            use_cond_coral = (args.coral_strategy == 'class_conditional')

        # 对所有客户端进行分阶段评估（使用缓存的原型矩阵）
        from utils import evaluate_model_with_phase_and_soft_agg
        phase_result = evaluate_model_with_phase_and_soft_agg(
            server.global_model, test_client_loaders[cid], config.DEVICE,
            semantic_protos=semantic_protos_for_eval,
            device_residuals=device_res,  # 使用估计的残差或通用残差
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
            num_classes=config.NUM_CLASSES,
            coral_source_feats=client_coral_feats,
            use_class_conditional_coral=use_cond_coral,
            global_test_loader=global_test_loader if use_cond_coral else None,
            source_model=server.global_model if use_cond_coral else None,
            proto_matrix=proto_matrix,
            proto_classes=proto_classes,
            proto_temperatures=proto_temperatures,
            use_mahalanobis_inference=use_mahalanobis_eval,
            semantic_proto_vars=proto_vars_for_eval
        )
        test_accs[cid] = phase_result['global']
        logger.info(f"Client {cid} class accuracy: Class0={phase_result['class_0']:.4f}, Class1={phase_result['class_1']:.4f}, Class2={phase_result['class_2']:.4f}, Class3={phase_result['class_3']:.4f}")
    
    reg_metrics = {}
    if config.USE_REG_LOSS:
        from utils import evaluate_regression_metrics, create_model_by_config, load_shared_weights
        # 构建带回归头的模型副本，避免 server.global_model (FedGasModel) 没有 forward_reg
        eval_model_for_reg = create_model_by_config(config, with_reg_head=True).to(config.DEVICE)
        load_shared_weights(eval_model_for_reg, server.global_model.state_dict(), strict=False)
        if getattr(server, 'shared_reg_state', None):
            load_shared_weights(eval_model_for_reg, server.shared_reg_state, strict=False)
        for cid in test_client_ids:
            try:
                reg_per_class, reg_overall = evaluate_regression_metrics(
                    eval_model_for_reg, test_client_loaders[cid], config.DEVICE
                )
                reg_metrics[cid] = {'per_class': reg_per_class, 'overall': reg_overall}
            except Exception as e:
                logger.warning(f"Regression eval failed for client {cid}: {e}")
    
    return acc_early, acc_middle, acc_late, acc_test, test_accs, reg_metrics


# 全局变量，用于缓存源域特征
_cached_source_feats = None
_cached_round = -1


def compute_mmd_metrics(args, round_idx, server, global_test_loader, test_client_ids, test_client_loaders, config, logger):
    """计算MMD指标
    
    Args:
        args: 命令行参数
        round_idx: 当前轮次
        server: 服务器对象
        global_test_loader: 全局测试加载器
        test_client_ids: 测试客户端ID列表
        test_client_loaders: 测试客户端加载器字典
        config: 配置对象
        logger: 日志记录器
    
    Returns:
        current_mmds: MMD值字典
    """
    current_mmds = {}
    if args.compute_mmd and round_idx % args.mmd_interval == 0:
        from utils import extract_features_batch
        
        # 缓存源域特征，每10轮更新一次
        global _cached_source_feats, _cached_round
        if _cached_source_feats is None or round_idx - _cached_round >= 10:
            train_feats_np, _ = extract_features_batch(server.global_model, global_test_loader, config.DEVICE, max_samples=500)  # 减少样本数
            _cached_source_feats = torch.from_numpy(train_feats_np).float()
            _cached_round = round_idx
            logger.info(f"Cached source features at round {round_idx}")
        train_feats = _cached_source_feats
        
        for cid in test_client_ids:
            test_feats_np, _ = extract_features_batch(server.global_model, test_client_loaders[cid], config.DEVICE, max_samples=500)  # 减少样本数
            test_feats = torch.from_numpy(test_feats_np).float()
            mmd_val = compute_mmd(train_feats, test_feats).item()
            current_mmds[cid] = mmd_val
            logger.info(f"MMD client {cid}: {mmd_val:.4f}")
    
    return current_mmds


def run_round(round_idx, config, server, clients, scheduler, client_loaders, test_client_ids, test_client_loaders, global_test_loader, val_loader, args, logger, tta_client_ids):
    """执行一轮联邦学习训练
    
    Args:
        round_idx: 当前轮次
        config: 配置对象
        server: 服务器对象
        clients: 客户端列表
        scheduler: 数据调度器
        client_loaders: 预加载的客户端加载器
        test_client_ids: 测试客户端ID列表
        test_client_loaders: 测试客户端加载器字典
        global_test_loader: 全局测试加载器
        val_loader: 验证加载器
        args: 命令行参数
        logger: 日志记录器
        tta_client_ids: 需要执行TTA的客户端ID列表
    
    Returns:
        acc_early: 早期阶段准确率
        acc_middle: 中期阶段准确率
        acc_late: 晚期阶段准确率
        acc_test: 全局测试准确率
        test_accs: 各测试客户端准确率字典
        weight_dict: 聚合权重字典
        align_loss: 对齐损失
        current_mmds: MMD值字典
    """
    import copy
    logger.info(f"\n===== Round {round_idx}/{config.GLOBAL_ROUNDS} =====")

    # 获取本轮数据加载器
    client_loaders_dict = get_client_loaders(round_idx, config, args, scheduler, clients, client_loaders, logger)
    if client_loaders_dict is None:
        return None, None, None, None, None, None, None, None, None

    current_state = dict(server.global_model.state_dict())
    if getattr(config, 'USE_REG_LOSS', False) and getattr(config, 'SHARE_REG_HEAD', False):
        shared_reg_state = getattr(server, 'shared_reg_state', None)
        if shared_reg_state:
            current_state.update(shared_reg_state)
    prev_state = server.get_prev_model_state()  # 用于蒸馏

    # 准备客户端进行训练
    active_client_loader_pairs = prepare_clients(clients, client_loaders_dict, current_state, prev_state, logger)
    if active_client_loader_pairs is None:
        return None, None, None, None, None, None, None, None, None

    # 并行训练客户端
    client_params_list, all_mus, all_counts, client_Fs, client_residuals, all_vars = train_clients(active_client_loader_pairs, server, round_idx, logger)

    # 计算校准集特征均值 (方案1: 目标感知聚合)
    server._compute_calib_feature_mean()

    # 聚合模型 (传入client_Fs用于目标感知聚合)
    w_new, final_params = aggregate_model(config, server, client_params_list, active_client_loader_pairs, all_mus, round_idx, logger, client_Fs=client_Fs)

    # 服务器优化
    client_ids = [cli.client_id for cli, _ in active_client_loader_pairs]
    align_loss = server_optimization(config, server, all_mus, all_counts, w_new, client_ids, client_Fs, client_residuals, logger, current_round=round_idx, all_vars=all_vars)

    # 保存上一轮模型
    server.prev_model = copy.deepcopy(server.global_model)
    server.prev_model.eval()
    server.client_weights = w_new

    # 记录权重
    # 使用实际的客户端ID而不是索引
    weight_dict = {f"client_{cli.client_id}": w.item() for (cli, _), w in zip(active_client_loader_pairs, w_new)}

    # 评估客户端
    # 计算原型自适应温度 (仅当校准集可用时)
    proto_temps = None
    if getattr(config, 'USE_ADAPTIVE_TEMPERATURE', False):
        if server._compute_proto_spreads_from_calib():
            proto_keys = list(server.semantic_protos.keys())
            proto_temps = server.get_adaptive_temperatures(proto_keys)
    acc_early, acc_middle, acc_late, acc_test, test_accs, reg_metrics = evaluate_clients(
        config, server, test_client_ids, test_client_loaders, global_test_loader, args, tta_client_ids, logger,
        proto_temperatures=proto_temps
    )

    # 计算MMD指标
    current_mmds = compute_mmd_metrics(args, round_idx, server, global_test_loader, test_client_ids, test_client_loaders, config, logger)

    # 日志
    acc_str = ", ".join([f"C{cid}={acc:.4f}" for cid, acc in test_accs.items()])
    mmd_str = ", ".join([f"{cid}={v:.4f}" for cid, v in current_mmds.items()]) if current_mmds else "N/A"
    log_msg = f"Round {round_idx}: Early={acc_early:.4f} Mid={acc_middle:.4f} Late={acc_late:.4f} | Test={acc_test:.4f} | {acc_str} | MMD={mmd_str}"
    if reg_metrics:
        reg_strs = []
        for cid, rm in reg_metrics.items():
            ro = rm['overall']
            if ro and ro.get('n_samples', 0) > 0:
                reg_strs.append(f"C{cid}: R²={ro['R2']:.3f} RMSE={ro['RMSE']:.1f}")
        if reg_strs:
            log_msg += f" | REG {', '.join(reg_strs)}"
    logger.info(log_msg)
    logger.info(f"Weights: {weight_dict}")

    return acc_early, acc_middle, acc_late, acc_test, test_accs, weight_dict, align_loss, current_mmds, reg_metrics


def build_few_shot_and_test_loaders(full_loader, num_per_class, batch_size):
    """
    从目标客户端全量数据加载器中按类别抽样少量样本，返回微调用的DataLoader和排除微调样本后的测试DataLoader
    Args:
        full_loader: 包含全量数据的DataLoader (dataset 是 GasSensorWindowDataset)
        num_per_class: 每类采样数量
        batch_size: 批次大小
    Returns:
        tuple: (train_loader, test_loader)
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    
    dataset = full_loader.dataset
    cls_labels = dataset.classification_labels
    # 确保是 numpy 数组
    if torch.is_tensor(cls_labels):
        cls_labels = cls_labels.numpy()
    
    train_indices = []
    for c in range(4):
        c_idx = np.where(cls_labels == c)[0]
        if len(c_idx) == 0:
            continue
        n_sample = min(num_per_class, len(c_idx))
        chosen = np.random.choice(c_idx, size=n_sample, replace=False)
        train_indices.extend(chosen.tolist())
    
    train_indices_set = set(train_indices)
    test_indices = [i for i in range(len(dataset)) if i not in train_indices_set]
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=min(len(train_indices), batch_size), shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def build_client_calibration_loader(cid: int, federated_dir: str, batch_size: int):
    """
    从目标客户端的校准集文件 (calibration_features.npy 等) 构建 DataLoader。
    校准集是云端维护的目标域 10% 数据分片，包含分类标签和回归标签。
    用于替代从全量数据抽取少样本/主动学习查询样本。
    
    Args:
        cid: 客户端 ID
        federated_dir: 联邦数据目录路径
        batch_size: DataLoader 的批次大小
    Returns:
        DataLoader: 包含完整标签的校准集加载器
    """
    import numpy as np
    from pathlib import Path
    from torch.utils.data import DataLoader
    from federated_dataset import GasSensorWindowDataset
    
    client_path = Path(federated_dir) / f'client_{cid}'
    feat_path = client_path / 'calibration_features.npy'
    
    if not feat_path.exists():
        return None
    
    features = np.load(feat_path)
    reg_labels = np.load(client_path / 'calibration_regression_labels.npy')
    cls_labels = np.load(client_path / 'calibration_classification_labels.npy')
    phase_path = client_path / 'calibration_phase_labels.npy'
    if phase_path.exists():
        phase_labels = np.load(phase_path, allow_pickle=True)
    else:
        phase_labels = np.full(len(features), -1, dtype=np.int64)
    
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=reg_labels,
        classification_labels=cls_labels,
        phase_labels=phase_labels,
        normalize=False
    )
    info_path = client_path / 'calibration_experiment_info.json'
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                experiment_info = json.load(f)
            if len(experiment_info) == len(features):
                dataset.experiment_info = experiment_info
                dataset.sample_filenames = [
                    item.get('filename', f'unknown_{idx}') if isinstance(item, dict) else f'unknown_{idx}'
                    for idx, item in enumerate(experiment_info)
                ]
            else:
                dataset.experiment_info = []
                dataset.sample_filenames = []
        except Exception:
            dataset.experiment_info = []
            dataset.sample_filenames = []
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



def evaluate_logits_with_phase_and_class(model, data_loader, device, num_classes=4):
    """Evaluate classifier logits with global/phase/class accuracy."""
    from collections import defaultdict
    model.eval()
    total = 0
    correct = 0
    phase_correct = defaultdict(int)
    phase_total = defaultdict(int)
    class_correct = {c: 0 for c in range(num_classes)}
    class_total = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
        for x, y_cls, _, y_phase in data_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            logits, _, _ = model(x)
            preds = logits.argmax(dim=1)
            total += int(y_cls.numel())
            correct += int((preds == y_cls).sum().item())
            for idx in range(y_cls.size(0)):
                cls_id = int(y_cls[idx].item())
                phase_id = int(y_phase[idx].item())
                class_total[cls_id] += 1
                phase_total[phase_id] += 1
                if preds[idx].item() == y_cls[idx].item():
                    class_correct[cls_id] += 1
                    phase_correct[phase_id] += 1

    phase_names = {0: 'early', 1: 'middle', 2: 'late'}
    result = {
        'global': correct / total if total else 0.0,
    }
    for phase_id, phase_name in phase_names.items():
        result[phase_name] = phase_correct[phase_id] / phase_total[phase_id] if phase_total[phase_id] else 0.0
    for cls_id in range(num_classes):
        result[f'class_{cls_id}'] = class_correct[cls_id] / class_total[cls_id] if class_total[cls_id] else 0.0
    return result


def _collect_logits_classification_predictions(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) < 2:
                continue
            x = batch[0].to(device)
            labels = batch[1].to(device)
            logits, _, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def _classification_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix


def _per_class_accuracy_from_confusion(matrix):
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    return np.divide(correct, totals, out=np.zeros_like(correct, dtype=np.float64), where=totals > 0)


def _plot_classification_confusion_matrix(matrix, save_path, class_names, title):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_yticklabels(class_names)
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = 'white' if matrix[row, col] > threshold else 'black'
            ax.text(col, row, str(int(matrix[row, col])), ha='center', va='center', color=color, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(save_path)


def _plot_classification_per_class_comparison(before_acc, after_acc, save_path, class_names, title):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(class_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, before_acc, width, label='Before calibration', color='#ff7f0e', alpha=0.85)
    ax.bar(x + width / 2, after_acc, width, label='After calibration', color='#2ca02c', alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for idx, value in enumerate(before_acc):
        ax.text(idx - width / 2, min(1.02, value + 0.02), f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    for idx, value in enumerate(after_acc):
        ax.text(idx + width / 2, min(1.02, value + 0.02), f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(save_path)


def plot_target_classification_calibration_diagnostics(cid, before_true, before_pred, after_true, after_pred,
                                                       save_dir, num_classes):
    class_names = ['Ethanol', 'CO', 'Ethylene', 'Methane'][:num_classes]
    save_dir = Path(save_dir)
    before_cm = _classification_confusion_matrix(before_true, before_pred, num_classes)
    after_cm = _classification_confusion_matrix(after_true, after_pred, num_classes)
    before_acc = _per_class_accuracy_from_confusion(before_cm)
    after_acc = _per_class_accuracy_from_confusion(after_cm)
    paths = {
        'before_confusion': _plot_classification_confusion_matrix(
            before_cm, save_dir / f'target_client{cid}_confusion_before_calibration.png',
            class_names, f'Target Client {cid} Confusion Before Calibration'
        ),
        'after_confusion': _plot_classification_confusion_matrix(
            after_cm, save_dir / f'target_client{cid}_confusion_after_calibration.png',
            class_names, f'Target Client {cid} Confusion After Calibration'
        ),
        'per_class_accuracy': _plot_classification_per_class_comparison(
            before_acc, after_acc, save_dir / f'target_client{cid}_per_class_acc_before_after.png',
            class_names, f'Target Client {cid} Per-Class Accuracy'
        ),
    }
    diagnostics = {
        'before_confusion': before_cm.tolist(),
        'after_confusion': after_cm.tolist(),
        'before_per_class_accuracy': before_acc.tolist(),
        'after_per_class_accuracy': after_acc.tolist(),
        'plots': paths,
    }
    return paths, diagnostics


def run_target_classification_calibration(eval_model, cid, calibration_loaders, eval_loader, args, config, logger):
    """Supervised target-domain classification calibration on the reserved calibration split."""
    if not getattr(args, 'target_cls_calibration', False):
        return False, {}
    calib_clients = _parse_client_id_set(getattr(args, 'target_cls_calib_clients', ''))
    if calib_clients is not None and int(cid) not in calib_clients:
        logger.info(f"Client {cid}: target classification calibration skipped by client filter")
        return False, {}
    calib_loader = calibration_loaders.get(cid) if calibration_loaders else None
    if calib_loader is None:
        logger.warning(f"Client {cid}: target classification calibration skipped (no calibration loader)")
        return False, {}

    from utils import few_shot_finetune_classification

    epochs_by_client = _parse_client_value_string(getattr(args, 'target_cls_client_epochs', ''), int)
    lr_by_client = _parse_client_value_string(getattr(args, 'target_cls_client_lr', ''), float)
    feat_lr_by_client = _parse_client_value_string(getattr(args, 'target_cls_client_feat_lr', ''), float)
    focal_by_client = _parse_client_value_string(getattr(args, 'target_cls_client_focal_gamma', ''), float)
    aug_prob_by_client = _parse_client_value_string(getattr(args, 'target_cls_client_aug_prob', ''), float)
    cost_weight_by_client = _parse_client_value_string(getattr(args, 'target_cls_client_cost_weight', ''), float)
    calib_epochs = int(epochs_by_client.get(int(cid), args.target_cls_calib_epochs))
    calib_lr = float(lr_by_client.get(int(cid), args.target_cls_lr))
    calib_feat_lr = float(feat_lr_by_client.get(int(cid), args.target_cls_feat_lr))
    focal_gamma = float(focal_by_client.get(int(cid), getattr(args, 'target_cls_focal_gamma', 0.0)))
    aug_prob = float(aug_prob_by_client.get(int(cid), getattr(args, 'target_cls_aug_prob', 0.0)))
    cost_weight = float(cost_weight_by_client.get(int(cid), getattr(args, 'target_cls_cost_weight', 0.0)))
    class_weight_vector, active_class_weights = _build_class_weight_vector(
        getattr(args, 'target_cls_class_weights', ''),
        getattr(args, 'target_cls_client_class_weights', ''),
        cid, config.NUM_CLASSES
    )
    cost_matrix = _build_cost_matrix(
        getattr(args, 'target_cls_cost_matrix', ''),
        getattr(args, 'target_cls_client_cost_matrix', ''),
        cid, config.NUM_CLASSES
    )
    aug_config = None
    if aug_prob > 0:
        aug_config = copy.deepcopy(config)
        aug_config.SENSOR_AUG_GAIN_STD = float(getattr(args, 'target_cls_aug_gain_std', config.SENSOR_AUG_GAIN_STD))
        aug_config.SENSOR_AUG_BIAS_STD = float(getattr(args, 'target_cls_aug_bias_std', config.SENSOR_AUG_BIAS_STD))
        aug_config.SENSOR_AUG_CH_GAIN_STD = float(getattr(args, 'target_cls_aug_ch_gain_std', config.SENSOR_AUG_CH_GAIN_STD))
        aug_config.SENSOR_AUG_TIME_SCALE_RANGE = float(getattr(args, 'target_cls_aug_time_scale_range', config.SENSOR_AUG_TIME_SCALE_RANGE))
        aug_config.SENSOR_AUG_TIME_PROB = float(getattr(args, 'target_cls_aug_time_prob', config.SENSOR_AUG_TIME_PROB))

    before_logits = evaluate_logits_with_phase_and_class(eval_model, eval_loader, config.DEVICE, config.NUM_CLASSES)
    before_true, before_pred = _collect_logits_classification_predictions(eval_model, eval_loader, config.DEVICE)
    few_shot_finetune_classification(
        eval_model, calib_loader, config.DEVICE,
        epochs=calib_epochs,
        lr=calib_lr,
        finetune_feat_lr=calib_feat_lr,
        class_weights=class_weight_vector,
        focal_gamma=focal_gamma,
        aug_config=aug_config,
        aug_prob=aug_prob,
        cost_matrix=cost_matrix,
        cost_weight=cost_weight
    )
    after_logits = evaluate_logits_with_phase_and_class(eval_model, eval_loader, config.DEVICE, config.NUM_CLASSES)
    after_true, after_pred = _collect_logits_classification_predictions(eval_model, eval_loader, config.DEVICE)

    diagnostic_plot_dir = Path(config.PLOT_SAVE_DIR) / "classification"
    plot_paths, diagnostic_metrics = plot_target_classification_calibration_diagnostics(
        cid, before_true, before_pred, after_true, after_pred, diagnostic_plot_dir, config.NUM_CLASSES
    )

    save_dir = Path(config.MODEL_SAVE_DIR) / "target_classifiers"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"target_cls_client{cid}.pth"
    torch.save(eval_model.state_dict(), ckpt_path)

    metrics = {
        'calibration_samples': len(calib_loader.dataset) if hasattr(calib_loader, 'dataset') else None,
        'test_samples': len(eval_loader.dataset) if hasattr(eval_loader, 'dataset') else None,
        'epochs': calib_epochs,
        'lr': calib_lr,
        'feat_lr': calib_feat_lr,
        'focal_gamma': focal_gamma,
        'class_weights': active_class_weights,
        'cost_weight': cost_weight,
        'cost_matrix': cost_matrix,
        'aug_prob': aug_prob,
        'aug_gain_std': getattr(aug_config, 'SENSOR_AUG_GAIN_STD', 0.0) if aug_config is not None else 0.0,
        'aug_bias_std': getattr(aug_config, 'SENSOR_AUG_BIAS_STD', 0.0) if aug_config is not None else 0.0,
        'aug_ch_gain_std': getattr(aug_config, 'SENSOR_AUG_CH_GAIN_STD', 0.0) if aug_config is not None else 0.0,
        'aug_time_scale_range': getattr(aug_config, 'SENSOR_AUG_TIME_SCALE_RANGE', 0.0) if aug_config is not None else 0.0,
        'aug_time_prob': getattr(aug_config, 'SENSOR_AUG_TIME_PROB', 0.0) if aug_config is not None else 0.0,
        'before_logits': before_logits,
        'after_logits': after_logits,
        'diagnostics': diagnostic_metrics,
        'plots': plot_paths,
        'checkpoint': str(ckpt_path),
    }
    logger.info(
        f"Client {cid}: target classification calibrated on {metrics['calibration_samples']} samples; "
        f"logits acc {before_logits['global']:.4f} -> {after_logits['global']:.4f}; "
        f"epochs={calib_epochs}, lr={calib_lr:g}, feat_lr={calib_feat_lr:g}, "
        f"focal_gamma={focal_gamma:g}, class_weights={active_class_weights or 'none'}, "
        f"cost_weight={cost_weight:g}, cost_matrix={'yes' if cost_matrix is not None else 'none'}, "
        f"aug_prob={aug_prob:g}"
    )
    return True, metrics

def final_evaluation(args, config, server, test_client_ids, test_client_loaders, best_model_states, best_rounds, logger, global_test_loader=None, tta_client_ids=None, best_model_state_for_all=None, best_round_for_all=None, calibration_loaders=None):
    """最终评估与模型选择（使用测试客户端准确率选择的最佳模型）"""
    import torch
    from pathlib import Path
    from utils import (get_device_residual, extract_features_batch, coral_transform_class_conditional,
                       few_shot_finetune_classification, evaluate_model_with_phase_and_soft_agg,
                       create_model_by_config, load_shared_weights)
    import numpy as np
    import os
    # 调试日志：打印分类微调参数
    logger.info(f"DEBUG: few_shot_classification={args.few_shot_classification}")
    final_accs = {}
    target_cls_calibration_metrics = {}
    target_classifier_states = {}
    # 存储每个客户端的回归指标
    regression_metrics_all = {}
    classification_metrics_all = {'target_final': {}, 'target_calibration': {}}

    # 准备软聚合所需参数
    semantic_protos_for_eval = {k: v.detach().cpu() for k, v in server.semantic_protos.items()}
    # 马氏距离推理：准备各原型的对角方差
    use_mahalanobis_eval = getattr(config, 'USE_MAHALANOBIS_INFERENCE', False)
    proto_vars_for_eval = {
        k: v.detach().cpu() for k, v in server.semantic_proto_vars.items()
    } if (use_mahalanobis_eval and server.semantic_proto_vars) else None
    hard_client_ids = [int(x.strip()) for x in args.hard_clients.split(',')] if args.hard_clients else []

    # === 统一提取源域特征（带缓存） ===
    coral_source_feats = None
    coral_source_labels = None
    if args.use_coral and hard_client_ids:
        use_time_drift = not args.no_time_drift
        round_for_cache = best_round_for_all if best_round_for_all else config.GLOBAL_ROUNDS
        coral_source_feats, coral_source_labels = get_source_features_for_coral(
            model=server.global_model,
            global_test_loader=global_test_loader,
            device=config.DEVICE,
            current_round=round_for_cache,
            use_time_drift=use_time_drift,
            refresh_interval=5,
            max_samples=1000,
            logger=logger
        )
    # ===============================

    if not test_client_ids:
        # 没有指定测试客户端，直接在全局测试集上评估
        eval_model = copy.deepcopy(server.global_model).to(config.DEVICE)

        phase_result = evaluate_model_with_phase_and_soft_agg(
            eval_model, global_test_loader, config.DEVICE,
            semantic_protos=semantic_protos_for_eval,
            device_residuals=None,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
            num_classes=config.NUM_CLASSES,
            use_mahalanobis_inference=use_mahalanobis_eval,
            semantic_proto_vars=proto_vars_for_eval
        )
        final_accs['global'] = phase_result['global']
        classification_metrics_all['global'] = phase_result
        logger.info(f"Global test accuracy: {phase_result['global']:.4f}")

        if config.USE_REG_LOSS:
            from utils import evaluate_regression_metrics
            reg_metrics, reg_overall = evaluate_regression_metrics(
                eval_model, global_test_loader, config.DEVICE, tolerance=0.1
            )
            regression_metrics_all['global'] = {'per_class': reg_metrics, 'overall': reg_overall}
            logger.info(f"Global regression: R²={reg_overall['R2']:.4f}, RMSE={reg_overall['RMSE']:.2f}, MAE={reg_overall['MAE']:.2f}")

        # 保存最终模型
        torch.save(server.global_model.state_dict(), Path(config.MODEL_SAVE_DIR) / "final_model.pth")
        return final_accs, regression_metrics_all, classification_metrics_all

    for cid in test_client_ids:
        # 估计设备残差
        device_res = get_device_residual(args, server, cid, test_client_loaders, semantic_protos_for_eval, config.DEVICE)

        # 统一使用最后轮次的全局模型进行评估（与轮次评估保持一致）
        eval_model = copy.deepcopy(server.global_model).to(config.DEVICE)
        eval_loader = test_client_loaders[cid]

        if args.model_selection and best_model_state_for_all is not None:
            logger.info(f"\n=== Using final round model for Client {cid} (best model was at round {best_round_for_all}) ===")

        # ----- 分类校准微调（使用最后轮次模型） -----
        calibrated_cls, calib_cls_metrics = run_target_classification_calibration(
            eval_model, cid, calibration_loaders or {}, eval_loader, args, config, logger
        )
        if calibrated_cls:
            target_cls_calibration_metrics[cid] = calib_cls_metrics
            target_classifier_states[cid] = copy.deepcopy(eval_model.state_dict())

        # ---------- CORAL 特征提取与保存（仅困难客户端） ----------
        use_cond_coral = False
        source_feats = None
        source_labels = None
        if args.use_coral and cid in hard_client_ids and coral_source_feats is not None:
            source_feats = coral_source_feats
            source_labels = coral_source_labels
            raw_feats, raw_labels = extract_features_batch(eval_model, test_client_loaders[cid], config.DEVICE, max_samples=1500)

            if args.coral_strategy == 'class_conditional':
                coral_feats = coral_transform_class_conditional(
                    source_feats, source_labels,
                    torch.from_numpy(raw_feats).float(),
                    torch.from_numpy(raw_labels).long()
                )
                save_dir = os.path.join(config.PLOT_SAVE_DIR, "tsne_features")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"tsne_c{cid}_coral_before_after.npz")
                np.savez(save_path,
                         raw=raw_feats, coral=coral_feats.numpy(), labels=raw_labels)
                logger.info(f"Saved CORAL features to {save_path}")
                use_cond_coral = True
            else:
                logger.info(f"Client {cid}: Global CORAL applied, no feature file saved for visualization.")

        # ---------- 少样本分类微调及权重/特征保存（困难客户端） ----------
        if args.few_shot_classification and cid not in target_cls_calibration_metrics:
            fs_loader, eval_loader = build_few_shot_and_test_loaders(
                test_client_loaders[cid], args.few_shot_samples, config.BATCH_SIZE
            )
            if cid in hard_client_ids:
                before_classifier_weight = eval_model.classifier.weight.detach().cpu().numpy()
                weights_dir = os.path.join(config.PLOT_SAVE_DIR, "classifier_weights")
                os.makedirs(weights_dir, exist_ok=True)
                np.savez(os.path.join(weights_dir, f"classifier_weight_before_finetune_c{cid}.npz"),
                         weight=before_classifier_weight)

                before_feats, before_labels = extract_features_batch(eval_model, test_client_loaders[cid],
                                                                     config.DEVICE, max_samples=2000)
                tsne_dir = os.path.join(config.PLOT_SAVE_DIR, "tsne_features")
                os.makedirs(tsne_dir, exist_ok=True)
                np.savez(os.path.join(tsne_dir, f"tsne_c{cid}_before_finetune.npz"),
                         feats=before_feats, labels=before_labels)

            few_shot_finetune_classification(
                eval_model, fs_loader, config.DEVICE,
                epochs=5, lr=1e-3, finetune_feat_lr=args.few_shot_cls_feat_lr
            )
            logger.info(f"Client {cid}: Few-shot classification fine-tuned with {len(fs_loader.dataset)} samples")

            if cid in hard_client_ids:
                after_classifier_weight = eval_model.classifier.weight.detach().cpu().numpy()
                np.savez(os.path.join(weights_dir, f"classifier_weight_after_finetune_c{cid}.npz"),
                         weight=after_classifier_weight)

                after_feats, after_labels = extract_features_batch(eval_model, eval_loader,
                                                                   config.DEVICE, max_samples=2000)
                np.savez(os.path.join(tsne_dir, f"tsne_c{cid}_after_finetune.npz"),
                         feats=after_feats, labels=after_labels)

        # ---------- 分类评估 ----------
        if cid in target_cls_calibration_metrics:
            phase_result = evaluate_logits_with_phase_and_class(
                eval_model, eval_loader, config.DEVICE, config.NUM_CLASSES
            )
            logger.info(f"Client {cid}: final classification uses calibrated logits head")
        else:
            phase_result = evaluate_model_with_phase_and_soft_agg(
                eval_model, eval_loader, config.DEVICE,
                semantic_protos=semantic_protos_for_eval,
                device_residuals=device_res,
                soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
                num_classes=config.NUM_CLASSES,
                coral_source_feats=coral_source_feats if cid in hard_client_ids else None,
                use_class_conditional_coral=use_cond_coral if cid in hard_client_ids else False,
                global_test_loader=global_test_loader if cid in hard_client_ids and use_cond_coral else None,
                source_model=server.global_model if cid in hard_client_ids and use_cond_coral else None,
                source_feats_provided=source_feats if cid in hard_client_ids and use_cond_coral else None,
                source_labels_provided=source_labels if cid in hard_client_ids and use_cond_coral else None,
                use_mahalanobis_inference=use_mahalanobis_eval,
                semantic_proto_vars=proto_vars_for_eval
            )
        acc = phase_result['global']
        logger.info(f"Client {cid} class accuracy: "
                    f"Class0={phase_result['class_0']:.4f}, Class1={phase_result['class_1']:.4f}, "
                    f"Class2={phase_result['class_2']:.4f}, Class3={phase_result['class_3']:.4f}")
        final_accs[cid] = acc
        logger.info(f"Client {cid} final accuracy: {acc:.4f}")
        classification_metrics_all['target_final'][str(cid)] = {
            'accuracy': float(acc),
            'phase_and_class': phase_result,
            'calibrated': bool(cid in target_cls_calibration_metrics),
        }
        
        # === 新增：计算该客户端的回归指标 ===
        if config.USE_REG_LOSS:
            from utils import (evaluate_regression_metrics, plot_regression_scatter,
                               plot_concentration_feature_correlation)
            # 统一使用最后轮次的全局模型进行回归评估（与分类评估保持一致）
            eval_model_for_reg = create_model_by_config(config, with_reg_head=True).to(config.DEVICE)
            load_shared_weights(eval_model_for_reg, server.global_model.state_dict(), strict=False)
            if getattr(server, 'shared_reg_state', None):
                load_shared_weights(eval_model_for_reg, server.shared_reg_state, strict=False)
            
            # ----- 新增少量样本微调（从云端维护的校准集中抽取） -----
            reg_eval_loader = test_client_loaders[cid]  # 评估仍使用全量测试数据

            before_reg_metrics = None
            before_reg_overall = None
            if args.few_shot_regression:
                before_reg_metrics, before_reg_overall = evaluate_regression_metrics(
                    eval_model_for_reg, reg_eval_loader, config.DEVICE,
                    tolerance=0.1, enable_calibration=False, calib_params=None
                )
                logger.info(
                    f"Client {cid}: regression before fine-tune "
                    f"R2={before_reg_overall.get('R2', float('nan')):.4f}, "
                    f"RMSE={before_reg_overall.get('RMSE', float('nan')):.2f}, "
                    f"MAE={before_reg_overall.get('MAE', float('nan')):.2f}"
                )
                try:
                    plot_regression_scatter(
                        eval_model_for_reg, reg_eval_loader, config.DEVICE, config.PLOT_SAVE_DIR,
                        filename=f"reg_scatter_client{cid}_before_regft.png", tolerance=0.1
                    )
                except Exception as e:
                    logger.warning(f"Client {cid}: failed to plot regression before fine-tune: {e}")
            if args.few_shot_regression:
                # 优先使用校准集；若不存在则回退到全量数据
                calib_loader = calibration_loaders.get(cid) if calibration_loaders else None
                source_loader = calib_loader if calib_loader is not None else test_client_loaders.get(cid)
                if source_loader is not None:
                    fs_loader, _ = build_few_shot_and_test_loaders(
                        source_loader, args.few_shot_samples, config.BATCH_SIZE
                    )
                    from utils import few_shot_finetune_regression
                    few_shot_finetune_regression(
                        eval_model_for_reg, fs_loader, config.DEVICE, config,
                        num_steps=args.few_shot_reg_steps, lr=args.few_shot_reg_lr,
                        finetune_feat_lr=args.few_shot_feat_lr,
                        weight_decay=args.few_shot_reg_weight_decay,
                        force_aug=True
                    )
                    logger.info(f"Client {cid}: Few-shot regression fine-tuned with " 
                                f"{len(fs_loader.dataset)} samples from calibration set")
                else:
                    logger.warning(f"Client {cid}: No calibration loader for few-shot regression, skipped.")
            
            # ----- 困难客户端：个性化回归偏置校准 -----
            if cid in hard_client_ids and hasattr(config, 'USE_REG_LOSS') and config.USE_REG_LOSS:
                from utils import calibrate_regression_bias
                calib_for_bias = calibration_loaders.get(cid) if calibration_loaders else None
                cal_loader = calib_for_bias if calib_for_bias is not None else (test_client_loaders.get(cid) if not args.few_shot_regression else None)
                if cal_loader is not None:
                    eval_model_for_reg = calibrate_regression_bias(
                        eval_model_for_reg, cal_loader, config.DEVICE, config,
                        num_steps=10, lr=1e-3
                    )
                    logger.info(f"Client {cid}: hard-client regression bias calibrated (from calibration set)")
            
            calib_params = None
            if calibration_loaders and cid in calibration_loaders:
                from utils import compute_calibration_params
                calib_params = compute_calibration_params(eval_model_for_reg, calibration_loaders[cid], config.DEVICE)
            reg_metrics, reg_overall = evaluate_regression_metrics(
                eval_model_for_reg, reg_eval_loader, config.DEVICE,
                tolerance=0.1, enable_calibration=(calib_params is not None), calib_params=calib_params
            )
            regression_metrics_all[cid] = {
                'per_class': reg_metrics,
                'overall': reg_overall,
                'before_finetune': {
                    'per_class': before_reg_metrics,
                    'overall': before_reg_overall
                } if before_reg_overall is not None else None,
                'calibration_applied': bool(calib_params is not None)
            }
            plot_suffix = "after_regft" if args.few_shot_regression else "evaluated"
            try:
                plot_regression_scatter(
                    eval_model_for_reg, reg_eval_loader, config.DEVICE, config.PLOT_SAVE_DIR,
                    filename=f"reg_scatter_client{cid}_{plot_suffix}.png", tolerance=0.1
                )
                plot_concentration_feature_correlation(
                    eval_model_for_reg, reg_eval_loader, config.DEVICE, config.PLOT_SAVE_DIR,
                    filename=f"conc_feat_corr_client{cid}_{plot_suffix}.png"
                )
            except Exception as e:
                logger.warning(f"Client {cid}: failed to plot evaluated regression model: {e}")
            if before_reg_overall is not None:
                logger.info(
                    f"Client {cid}: regression after fine-tune/calibration "
                    f"R2={reg_overall.get('R2', float('nan')):.4f}, "
                    f"RMSE={reg_overall.get('RMSE', float('nan')):.2f}, "
                    f"MAE={reg_overall.get('MAE', float('nan')):.2f}"
                )
            logger.info(f"Client {cid} regression metrics calculated.")

    # 保存最终模型
    save_dir = Path(config.MODEL_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(server.global_model.state_dict(), save_dir / "final_model.pth")
    if args.model_selection:
        for cid, state in best_model_states.items():
            if state:
                torch.save(state, Path(config.MODEL_SAVE_DIR) / f"best_model_client{cid}.pth")

    if target_cls_calibration_metrics:
        classification_metrics_all['target_calibration'] = target_cls_calibration_metrics
        server.target_classifier_states = target_classifier_states

    return final_accs, regression_metrics_all, classification_metrics_all


def make_json_serializable(obj):
    """Convert nested experiment metrics to JSON-safe Python objects."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _module_trainable_params(module):
    if module is None:
        return []
    return [p for p in module.parameters()]


def _unique_params(params):
    seen = set()
    unique = []
    for param in params:
        if param is None:
            continue
        pid = id(param)
        if pid not in seen:
            seen.add(pid)
            unique.append(param)
    return unique


def _get_separate_regression_params(model):
    params = []
    for name in ['reg_proj', 'reg_transformer', 'reg_attn', 'reg_attn_linear', 'reg_stats_proj', 'conc_bucket_classifier']:
        params.extend(_module_trainable_params(getattr(model, name, None)))
    if getattr(model, 'reg_heads', None) is not None:
        params.extend(model.reg_heads.parameters())
    for name in ['proto_scale', 'proto_bias', 'proto_conc', 'conc_directions', 'conc_scale', 'conc_bias']:
        param = getattr(model, name, None)
        if param is not None:
            params.append(param)
    return _unique_params(params)


def _get_encoder_unfreeze_params(model, unfreeze_policy):
    if unfreeze_policy == 'none':
        return []
    params = []
    if unfreeze_policy == 'last_tcn':
        if getattr(model, 'encoder_type', 'tcn') == 'tcn':
            params.extend(_module_trainable_params(getattr(model, 'tcn3', None)))
        elif getattr(model, 'transformer_encoder', None) is not None:
            encoder = model.transformer_encoder
            layers = getattr(getattr(encoder, 'encoder', None), 'layers', None)
            if layers is not None and len(layers) > 0:
                params.extend(layers[-1].parameters())
    elif unfreeze_policy == 'all_encoder':
        for name in ['tcn', 'channel_attn', 'self_attn', 'attn_linear', 'transformer_encoder']:
            params.extend(_module_trainable_params(getattr(model, name, None)))
    return _unique_params(params)


def _init_regression_branch_from_classifier(model, logger=None):
    """Initialize regression projection from classification projection when shapes match."""
    if getattr(model, 'cls_proj', None) is not None and getattr(model, 'reg_proj', None) is not None:
        if model.cls_proj.weight.shape == model.reg_proj.weight.shape:
            with torch.no_grad():
                model.reg_proj.weight.copy_(model.cls_proj.weight)
                model.reg_proj.bias.copy_(model.cls_proj.bias)
            if logger:
                logger.info("Separate regression: initialized reg_proj from cls_proj")


def _build_source_regression_loaders(train_client_ids, temp_federated_dir, batch_size, logger):
    """Build one labeled regression DataLoader per source client.

    This keeps source regression pretraining federated: the server never mixes
    raw Client4/Client5 data into a centralized dataset. Each loader is used
    only by its corresponding local client update before FedAvg aggregation.
    """
    loaders = {}
    sample_counts = {}
    for cid in train_client_ids:
        client_dir = Path(temp_federated_dir) / f'client_{cid}'
        feat_path = client_dir / 'train_features.npy'
        if not feat_path.exists():
            logger.warning(f"Separate regression: source client {cid} train split not found")
            continue
        features = np.load(feat_path)
        cls_labels = np.load(client_dir / 'train_classification_labels.npy')
        reg_labels = np.load(client_dir / 'train_regression_labels.npy')
        phase_path = client_dir / 'train_phase_labels.npy'
        phase_labels = np.load(phase_path, allow_pickle=True) if phase_path.exists() else np.full(len(features), -1, dtype=np.int64)
        dataset = GasSensorPhaseDataset(features, cls_labels, reg_labels, phase_labels, phase=None)
        loaders[cid] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        sample_counts[cid] = len(features)
        logger.info(f"Separate regression: source Client {cid} local train samples={len(features)}")
    if not loaders:
        raise RuntimeError("Separate regression: no source training data found")
    return loaders, sample_counts


def _build_source_regression_test_loaders(train_client_ids, temp_federated_dir, batch_size, logger):
    """为源域回归预训练构建测试集DataLoader（诊断指标用）

    加载源客户端（Client4/5）的 test_*.npy，在源域 FedAvg 预训练后评估模型
    回答 Phase 0 的第一个问题：B_source 在源域 test 上 R²/MAE 是多少
    """
    loaders = {}
    sample_counts = {}
    for cid in train_client_ids:
        client_dir = Path(temp_federated_dir) / f'client_{cid}'
        feat_path = client_dir / 'test_features.npy'
        if not feat_path.exists():
            logger.warning(f"Separate regression: source client {cid} test split not found, skipping source eval")
            continue
        features = np.load(feat_path)
        cls_labels = np.load(client_dir / 'test_classification_labels.npy')
        reg_labels = np.load(client_dir / 'test_regression_labels.npy')
        phase_path = client_dir / 'test_phase_labels.npy'
        phase_labels = np.load(phase_path, allow_pickle=True) if phase_path.exists() else np.full(len(features), -1, dtype=np.int64)
        dataset = GasSensorPhaseDataset(features, cls_labels, reg_labels, phase_labels, phase=None)
        loaders[cid] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        sample_counts[cid] = len(features)
        logger.info(f"Separate regression: source Client {cid} test samples={len(features)}")
    return loaders, sample_counts


def _get_regression_state_keys(model):
    prefixes = ('reg_proj.', 'reg_transformer.', 'reg_attn.', 'reg_attn_linear.', 'reg_stats_proj.', 'reg_heads.')
    exact_names = {'proto_scale', 'proto_bias', 'proto_conc', 'conc_directions', 'conc_scale', 'conc_bias'}
    return [
        key for key in model.state_dict().keys()
        if key.startswith(prefixes) or key in exact_names
    ]


def _filter_regression_state_keys_for_scope(state_keys, agg_scope):
    agg_scope = str(agg_scope or 'all').lower()
    state_keys = list(state_keys)
    if agg_scope == 'all':
        return state_keys
    if agg_scope == 'no_calib':
        calib_keys = {'proto_scale', 'proto_bias', 'proto_conc', 'conc_scale', 'conc_bias'}
        return [key for key in state_keys if key not in calib_keys]
    raise ValueError(f"Unknown separate regression source aggregation scope: {agg_scope}")


def _fedavg_regression_states(local_states, sample_counts, state_keys, device):
    total = float(sum(sample_counts.values()))
    if total <= 0:
        raise RuntimeError("Separate regression FedAvg: no source samples to aggregate")
    aggregated = {}
    for key in state_keys:
        weighted = None
        for cid, state in local_states.items():
            weight = float(sample_counts[cid]) / total
            value = state[key].to(device)
            contrib = value * weight
            weighted = contrib if weighted is None else weighted + contrib
        aggregated[key] = weighted
    return aggregated


def _load_source_local_states_from_checkpoints(checkpoint_dir, source_client_ids, logger=None):
    checkpoint_dir = Path(checkpoint_dir)
    local_states = {}
    local_checkpoint_paths = {}
    for cid in source_client_ids:
        ckpt_path = checkpoint_dir / f"separate_regression_source_client{cid}_local.pth"
        if not ckpt_path.exists():
            continue
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        local_states[int(cid)] = _extract_model_state_from_checkpoint(ckpt_obj)
        local_checkpoint_paths[str(cid)] = str(ckpt_path)
    if logger:
        if local_states:
            logger.info(
                f"Separate regression: loaded source local checkpoints for clients "
                f"{sorted(local_states.keys())}"
            )
        else:
            logger.info(f"Separate regression: no source local checkpoints found in {checkpoint_dir}")
    return local_states, local_checkpoint_paths


def _select_source_initialization_for_target(base_model, source_local_states, classifier_model, calib_loader,
                                             device, config, metric='r2', scope='overall', logger=None,
                                             client_id=None, min_score_gain=0.0):
    """Select FedAvg or a source-local regression initialization on target calibration data."""
    diagnostics = {
        'mode': 'fedavg',
        'selected': 'fedavg',
        'selected_source_client': '',
        'metric': metric,
        'scope': scope,
        'min_score_gain': float(min_score_gain),
        'candidates': {},
    }
    if calib_loader is None or not source_local_states:
        return base_model, diagnostics

    best_model = base_model
    best_score = -float('inf')
    base_metrics = evaluate_separate_regression(
        base_model, classifier_model, calib_loader, device, config.NUM_CLASSES
    )
    best_score = _calibration_score(base_metrics, metric=metric, scope=scope, num_classes=config.NUM_CLASSES)
    base_score = best_score
    diagnostics['candidates']['fedavg'] = {
        'score': float(best_score),
        'metrics': base_metrics,
    }
    diagnostics['fedavg_score'] = float(base_score)
    diagnostics['best_candidate'] = 'fedavg'
    diagnostics['best_candidate_score'] = float(base_score)
    diagnostics['best_candidate_gain'] = 0.0

    for source_cid, local_state in source_local_states.items():
        cand_model = copy.deepcopy(base_model).to(device)
        cand_model.load_state_dict(local_state, strict=True)
        cand_model.eval()
        cand_metrics = evaluate_separate_regression(
            cand_model, classifier_model, calib_loader, device, config.NUM_CLASSES
        )
        cand_score = _calibration_score(cand_metrics, metric=metric, scope=scope, num_classes=config.NUM_CLASSES)
        diagnostics['candidates'][f'source_client{source_cid}'] = {
            'score': float(cand_score),
            'source_client': int(source_cid),
            'metrics': cand_metrics,
        }
        if cand_score > diagnostics['best_candidate_score']:
            diagnostics['best_candidate'] = f'source_client{source_cid}'
            diagnostics['best_candidate_score'] = float(cand_score)
            diagnostics['best_candidate_gain'] = float(cand_score - base_score)
        if cand_score > best_score and (cand_score - base_score) >= float(min_score_gain):
            best_score = cand_score
            best_model = cand_model
            diagnostics['selected'] = f'source_client{source_cid}'
            diagnostics['selected_source_client'] = int(source_cid)
        else:
            del cand_model

    diagnostics['selected_score'] = float(best_score)
    diagnostics['selected_gain'] = float(best_score - base_score)
    if logger:
        prefix = f"Separate regression Client {client_id}: " if client_id is not None else "Separate regression: "
        candidate_scores = ', '.join(
            f"{name}={info['score']:.4f}" for name, info in diagnostics['candidates'].items()
        )
        logger.info(
            f"{prefix}source init selected {diagnostics['selected']} "
            f"({scope} {metric}; {candidate_scores})"
        )
    return best_model, diagnostics


def _select_source_models_per_class_for_target(base_model, source_local_states, classifier_model, calib_loader,
                                               device, config, min_mae_gain=5.0, logger=None, client_id=None):
    """Route predicted gas heads to source-local regression models when calibration proves useful."""
    diagnostics = {
        'mode': 'per_class_calib_select',
        'fallback': 'fedavg',
        'min_mae_gain': float(min_mae_gain),
        'selected_by_class': {},
        'candidates': {},
    }
    routing_config = {'selected_modes': {}, 'affine_params': {}, 'phase_affine_params': {}}
    specialist_models = {}
    if calib_loader is None or not source_local_states:
        return None, None, diagnostics

    candidate_models = {'fedavg': base_model}
    for source_cid, local_state in source_local_states.items():
        cand_model = copy.deepcopy(base_model).to(device)
        cand_model.load_state_dict(local_state, strict=True)
        cand_model.eval()
        candidate_models[f'source_client{source_cid}'] = cand_model

    candidate_metrics = {}
    for name, cand_model in candidate_models.items():
        candidate_metrics[name] = evaluate_separate_regression(
            cand_model, classifier_model, calib_loader, device, config.NUM_CLASSES
        )
        diagnostics['candidates'][name] = candidate_metrics[name]

    for cls_id in range(config.NUM_CLASSES):
        per_class_mae = {}
        for name, metrics in candidate_metrics.items():
            cls_metrics = metrics.get('pipeline', {}).get('per_class', {}).get(cls_id)
            if cls_metrics is None:
                cls_metrics = metrics.get('pipeline', {}).get('per_class', {}).get(str(cls_id), {})
            per_class_mae[name] = float(cls_metrics.get('MAE', 1e9))
        fallback_mae = per_class_mae.get('fedavg', 1e9)
        raw_selected = min(per_class_mae, key=per_class_mae.get)
        mae_gain = fallback_mae - per_class_mae[raw_selected]
        selected = raw_selected if (raw_selected == 'fedavg' or mae_gain >= float(min_mae_gain)) else 'fedavg'
        diagnostics['selected_by_class'][cls_id] = {
            'raw_selected': raw_selected,
            'selected': selected,
            'mae_by_candidate': per_class_mae,
            'mae_gain_vs_fedavg': float(mae_gain),
            'guard_passed': bool(raw_selected == 'fedavg' or selected == raw_selected),
        }
        if selected != 'fedavg':
            routing_config['selected_modes'][cls_id] = 'specialist_full'
            specialist_models[cls_id] = candidate_models[selected]
        if logger:
            prefix = f"Separate regression Client {client_id}: " if client_id is not None else "Separate regression: "
            score_text = ', '.join(f'{name}=MAE{mae:.2f}' for name, mae in per_class_mae.items())
            logger.info(
                f"{prefix}source per-class route class {cls_id}: selected={selected}, "
                f"raw={raw_selected}, gain={mae_gain:.2f}, {score_text}"
            )

    if not specialist_models:
        return None, None, diagnostics
    return routing_config, specialist_models, diagnostics


def _train_federated_source_regression(model, source_loaders, sample_counts, device, config,
                                       total_steps_per_client, source_rounds, lr, logger=None,
                                       save_dir=None):
    """Federated source-domain regression pretraining with FedAvg.

    The model is copied to each source client, trained locally on that client's
    private train split, and only regression-branch parameters are aggregated.
    """
    if total_steps_per_client <= 0:
        if logger:
            logger.warning("Separate regression FedAvg: source pretraining skipped because steps <= 0")
        return {}, {}
    source_rounds = max(1, int(source_rounds))
    base_steps = total_steps_per_client // source_rounds
    extra_steps = total_steps_per_client % source_rounds
    all_state_keys = _get_regression_state_keys(model)
    agg_scope = getattr(config, 'SEPARATE_REG_SOURCE_AGG_SCOPE', 'all')
    state_keys = _filter_regression_state_keys_for_scope(all_state_keys, agg_scope)
    if not state_keys:
        raise RuntimeError("Separate regression FedAvg: no regression parameters selected for aggregation")
    skipped_keys = [key for key in all_state_keys if key not in set(state_keys)]

    if logger:
        logger.info(
            f"Separate regression FedAvg: source clients={list(source_loaders.keys())}, "
            f"rounds={source_rounds}, total local steps/client={total_steps_per_client}, "
            f"agg_scope={agg_scope}, aggregated tensors={len(state_keys)}, "
            f"non-aggregated tensors={len(skipped_keys)}"
        )

    final_local_states = {}
    local_checkpoint_paths = {}
    for round_idx in range(1, source_rounds + 1):
        local_steps = base_steps + (1 if round_idx <= extra_steps else 0)
        if local_steps <= 0:
            continue
        global_state = copy.deepcopy(model.state_dict())
        local_states = {}
        for cid, loader in source_loaders.items():
            local_model = copy.deepcopy(model).to(device)
            local_model.load_state_dict(global_state, strict=True)
            _train_separate_regression(
                local_model, loader, device, config,
                steps=local_steps, lr=lr, feat_lr=0.0, unfreeze_policy='none',
                logger=logger, stage_name=f'source_client{cid}_fed_round{round_idx}'
            )
            local_state_cpu = {key: value.detach().cpu() for key, value in local_model.state_dict().items()}
            local_states[cid] = local_state_cpu
            if round_idx == source_rounds:
                final_local_states[cid] = copy.deepcopy(local_state_cpu)
                if save_dir is not None:
                    ckpt_path = Path(save_dir) / f"separate_regression_source_client{cid}_local.pth"
                    torch.save({'model_state': local_state_cpu, 'config': config, 'client_id': cid}, ckpt_path)
                    local_checkpoint_paths[str(cid)] = str(ckpt_path)
            del local_model
        averaged = _fedavg_regression_states(local_states, sample_counts, state_keys, device)
        new_state = copy.deepcopy(global_state)
        for key, value in averaged.items():
            new_state[key] = value.to(new_state[key].device).type_as(new_state[key])
        model.load_state_dict(new_state, strict=True)
        if logger:
            logger.info(f"Separate regression FedAvg: completed source round {round_idx}/{source_rounds}")
    return final_local_states, local_checkpoint_paths


def _evaluate_source_regression_matrix(global_model, local_states, classifier_model, source_test_loaders,
                                       device, config, semantic_protos, save_dir, logger):
    diagnostics = {
        'fedavg': {},
        'local': {},
        'cross': {},
        'local_checkpoints': {},
    }
    if not source_test_loaders:
        logger.warning("Separate regression: no source test data available for local/cross evaluation")
        return diagnostics

    logger.info("\n=== Separate Regression: Source Local/Cross/FedAvg Evaluation ===")
    for test_cid, loader in source_test_loaders.items():
        fedavg_metrics = evaluate_separate_regression(
            global_model, classifier_model, loader, device, config.NUM_CLASSES,
            semantic_protos=semantic_protos,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT
        )
        diagnostics['fedavg'][str(test_cid)] = fedavg_metrics
        scatter_path = plot_separate_regression_scatter(
            global_model, classifier_model, loader, device, Path(save_dir) / 'regression_plots',
            filename=f"source_fedavg_to_client{test_cid}_oracle.png",
            mode='oracle', title_prefix=f"Source FedAvg -> Client {test_cid} (oracle)",
            semantic_protos=semantic_protos,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT
        )
        diagnostics['fedavg'][str(test_cid)]['oracle_scatter'] = scatter_path
        logger.info(
            f"Source FedAvg -> Client {test_cid}: "
            f"oracle R2={fedavg_metrics['oracle']['overall']['R2']:.4f}, "
            f"MAE={fedavg_metrics['oracle']['overall']['MAE']:.2f}"
        )

    for train_cid, local_state in local_states.items():
        local_model = copy.deepcopy(global_model).to(device)
        local_model.load_state_dict(local_state, strict=True)
        local_model.eval()
        train_key = str(train_cid)
        diagnostics['local'].setdefault(train_key, {})
        diagnostics['cross'].setdefault(train_key, {})
        for test_cid, loader in source_test_loaders.items():
            metrics = evaluate_separate_regression(
                local_model, classifier_model, loader, device, config.NUM_CLASSES,
                semantic_protos=semantic_protos,
                soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                prior_weight=config.SOFT_AGG_PRIOR_WEIGHT
            )
            bucket = diagnostics['local'] if train_cid == test_cid else diagnostics['cross']
            bucket[train_key][str(test_cid)] = metrics
            relation = 'local' if train_cid == test_cid else 'cross'
            scatter_path = plot_separate_regression_scatter(
                local_model, classifier_model, loader, device, Path(save_dir) / 'regression_plots',
                filename=f"source_{relation}_B{train_cid}_to_client{test_cid}_oracle.png",
                mode='oracle', title_prefix=f"Source {relation} B{train_cid} -> Client {test_cid} (oracle)",
                semantic_protos=semantic_protos,
                soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                prior_weight=config.SOFT_AGG_PRIOR_WEIGHT
            )
            bucket[train_key][str(test_cid)]['oracle_scatter'] = scatter_path
            logger.info(
                f"Source {relation} B{train_cid} -> Client {test_cid}: "
                f"oracle R2={metrics['oracle']['overall']['R2']:.4f}, "
                f"MAE={metrics['oracle']['overall']['MAE']:.2f}"
            )
        del local_model
    return diagnostics


def _reshuffle_loader(loader, batch_size):
    return DataLoader(loader.dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def _dataset_filenames(dataset):
    if isinstance(dataset, Subset):
        base_filenames = _dataset_filenames(dataset.dataset)
        if base_filenames is None:
            return None
        return [base_filenames[int(i)] for i in dataset.indices]
    filenames = getattr(dataset, 'sample_filenames', None)
    if filenames is not None and len(filenames) == len(dataset):
        return [str(v) for v in filenames]
    experiment_info = getattr(dataset, 'experiment_info', None)
    if experiment_info is not None and len(experiment_info) == len(dataset):
        return [
            item.get('filename', f'unknown_{idx}') if isinstance(item, dict) else f'unknown_{idx}'
            for idx, item in enumerate(experiment_info)
        ]
    return None


def _dataset_class_labels(dataset):
    if isinstance(dataset, Subset):
        base_labels = _dataset_class_labels(dataset.dataset)
        if base_labels is None:
            return None
        return np.asarray(base_labels)[list(dataset.indices)]
    labels = getattr(dataset, 'classification_labels', None)
    if labels is not None and len(labels) == len(dataset):
        return np.asarray(labels)
    cls_labels = getattr(dataset, 'cls_labels', None)
    if cls_labels is not None and len(cls_labels) == len(dataset):
        return np.asarray(cls_labels)
    return None


def _split_calibration_loader(loader, batch_size, val_ratio=0.3, seed=42, split_by='window', logger=None):
    """Split a target calibration loader into train/validation loaders."""
    if loader is None or not hasattr(loader, 'dataset'):
        return loader, None
    dataset = loader.dataset
    n = len(dataset)
    if n < 4 or val_ratio <= 0:
        return _reshuffle_loader(loader, batch_size), loader

    val_size = max(1, int(round(n * val_ratio)))
    train_size = n - val_size
    if train_size < 1:
        return _reshuffle_loader(loader, batch_size), loader

    split_by = str(split_by or 'window').lower()
    if split_by in ('file', 'filename'):
        filenames = _dataset_filenames(dataset)
        if filenames is not None and len(set(filenames)) > 1:
            file_groups = {}
            for idx, filename in enumerate(filenames):
                file_groups.setdefault(str(filename), []).append(idx)
            class_labels = _dataset_class_labels(dataset)
            rng = np.random.RandomState(int(seed))
            val_files = set()
            if class_labels is not None and len(class_labels) == n:
                files_by_class = {}
                for filename, indices in file_groups.items():
                    labels = np.asarray(class_labels)[indices].astype(int)
                    values, counts = np.unique(labels, return_counts=True)
                    cls_id = int(values[np.argmax(counts)])
                    files_by_class.setdefault(cls_id, []).append(filename)
                for _, class_files in sorted(files_by_class.items()):
                    class_files = list(class_files)
                    rng.shuffle(class_files)
                    cls_val = max(1, int(round(len(class_files) * val_ratio)))
                    cls_val = min(cls_val, max(0, len(class_files) - 1))
                    val_files.update(class_files[:cls_val])
            else:
                all_files = list(file_groups.keys())
                rng.shuffle(all_files)
                file_val = max(1, int(round(len(all_files) * val_ratio)))
                file_val = min(file_val, max(0, len(all_files) - 1))
                val_files.update(all_files[:file_val])

            val_indices = []
            train_indices = []
            for filename, indices in file_groups.items():
                if filename in val_files:
                    val_indices.extend(indices)
                else:
                    train_indices.extend(indices)
            if train_indices and val_indices:
                if logger is not None:
                    logger.info(
                        f"Calibration split by file: train_files={len(file_groups) - len(val_files)}, "
                        f"val_files={len(val_files)}, train_samples={len(train_indices)}, "
                        f"val_samples={len(val_indices)}"
                    )
                train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=0)
                return train_loader, val_loader
        if logger is not None:
            logger.warning("Calibration file split requested but filename metadata is unavailable; fallback to window split")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _parse_class_weight_string(spec):
    """Parse strings such as '2:2.0,1:1.3' into {2: 2.0, 1: 1.3}."""
    weights = {}
    if not spec:
        return weights
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Invalid class weight item: {item}. Expected class_id:weight")
        cls_id, weight = item.split(':', 1)
        weights[int(cls_id.strip())] = float(weight.strip())
    return weights


def _parse_class_float_string(spec, label='class value'):
    """Parse strings such as '1:0.1,2:0.2' into {1: 0.1, 2: 0.2}."""
    values = {}
    if not spec:
        return values
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Invalid {label} item: {item}. Expected class_id:value")
        cls_id, value = item.split(':', 1)
        values[int(cls_id.strip())] = float(value.strip())
    return values


def _smooth_l1_loss_with_optional_class_beta(pred, target, y_cls, base_beta, class_betas=None):
    """SmoothL1 with optional per-class beta in normalized concentration space."""
    if not class_betas:
        return F.smooth_l1_loss(pred, target, beta=float(base_beta), reduction='none')
    diff = torch.abs(pred - target)
    beta = torch.full_like(diff, float(base_beta))
    y_cls_flat = y_cls.view(-1)
    beta_flat = beta.view(-1)
    for cls_id, cls_beta in class_betas.items():
        beta_flat[y_cls_flat == int(cls_id)] = float(cls_beta)
    beta = torch.clamp(beta, min=1e-8)
    return torch.where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)


def _parse_client_id_set(spec):
    if not spec:
        return None
    ids = {int(x.strip()) for x in str(spec).split(',') if x.strip()}
    return ids if ids else None


def _parse_client_value_string(spec, value_type=float):
    """Parse '5:80,3:40' into {5: 80, 3: 40}."""
    values = {}
    if not spec:
        return values
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Invalid client value item: {item}. Expected client_id:value")
        client_id, value = item.split(':', 1)
        values[int(client_id.strip())] = value_type(value.strip())
    return values


def _parse_client_class_weight_string(spec):
    """Parse '5:1:2.0,5:2:2.0' into {5: {1: 2.0, 2: 2.0}}."""
    weights = {}
    if not spec:
        return weights
    for item in str(spec).split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid client class weight item: {item}. Expected client_id:class_id:weight")
        client_id, class_id, weight = parts
        weights.setdefault(int(client_id.strip()), {})[int(class_id.strip())] = float(weight.strip())
    return weights


def _build_class_weight_vector(global_spec, client_spec, client_id, num_classes):
    class_weights = {int(k): float(v) for k, v in _parse_class_weight_string(global_spec).items()}
    client_weights = _parse_client_class_weight_string(client_spec).get(int(client_id), {})
    class_weights.update({int(k): float(v) for k, v in client_weights.items()})
    if not class_weights:
        return None, {}
    vector = [1.0] * int(num_classes)
    for cls_id, weight in class_weights.items():
        if 0 <= int(cls_id) < int(num_classes):
            vector[int(cls_id)] = float(weight)
    return vector, class_weights


def _parse_cost_matrix_string(spec, num_classes=None):
    """Parse a square class-routing cost matrix.

    Accepted row separators: ';' or '/'. Example:
    '0,1,2,1;3,0,2,3;2,5,0,2;1,1,2,0'
    """
    if not spec:
        return None
    text = str(spec).strip()
    row_sep = ';' if ';' in text else '/'
    rows = []
    for row in text.split(row_sep):
        row = row.strip()
        if not row:
            continue
        values = [float(v.strip()) for v in row.split(',') if v.strip()]
        rows.append(values)
    if not rows:
        return None
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Invalid cost matrix: rows have different lengths: {spec}")
    if len(rows) != width:
        raise ValueError(f"Invalid cost matrix: expected square matrix, got {len(rows)}x{width}")
    if num_classes is not None and int(num_classes) > 0 and len(rows) != int(num_classes):
        raise ValueError(f"Invalid cost matrix: expected {num_classes} classes, got {len(rows)}")
    return rows


def _parse_client_cost_matrix_string(spec, num_classes=None):
    """Parse client-specific cost matrices.

    Use '|' between clients and '/' between matrix rows, e.g.
    '5=0,1,2,1/3,0,2,3/2,5,0,2/1,1,2,0'.
    """
    matrices = {}
    if not spec:
        return matrices
    for item in str(spec).split('|'):
        item = item.strip()
        if not item:
            continue
        if '=' in item:
            client_id, matrix_spec = item.split('=', 1)
        elif ':' in item:
            client_id, matrix_spec = item.split(':', 1)
        else:
            raise ValueError("Invalid client cost matrix item. Expected client_id=matrix")
        matrices[int(client_id.strip())] = _parse_cost_matrix_string(matrix_spec.strip(), num_classes)
    return matrices


def _build_cost_matrix(global_spec, client_spec, client_id, num_classes):
    matrix = _parse_cost_matrix_string(global_spec, num_classes)
    client_matrices = _parse_client_cost_matrix_string(client_spec, num_classes)
    if int(client_id) in client_matrices:
        matrix = client_matrices[int(client_id)]
    return matrix


def _parse_client_class_spec(spec):
    mapping = {}
    if not spec:
        return mapping
    for item in str(spec).split(';'):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError('Invalid client-class item. Expected client_id:class_id[,class_id]')
        client_id, class_ids = item.split(':', 1)
        classes = {int(x.strip()) for x in class_ids.split(',') if x.strip()}
        if classes:
            mapping[int(client_id.strip())] = classes
    return mapping


def _pairwise_rank_loss(pred_norm, target_norm, margin=0.02, max_pairs=256):
    """Small monotonicity loss: larger target concentration should rank higher."""
    pred = pred_norm.view(-1)
    target = target_norm.view(-1)
    n = pred.numel()
    if n < 2:
        return pred.new_tensor(0.0)
    diff_y = target.unsqueeze(0) - target.unsqueeze(1)
    valid = diff_y.abs() > 1e-6
    idx = valid.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return pred.new_tensor(0.0)
    if idx.size(0) > max_pairs:
        idx = idx[torch.randperm(idx.size(0), device=idx.device)[:max_pairs]]
    i, j = idx[:, 0], idx[:, 1]
    direction = torch.sign(target[i] - target[j])
    pred_diff = pred[i] - pred[j]
    return torch.relu(float(margin) - direction * pred_diff).mean()


def _calibration_score(metrics, metric='r2', scope='overall', num_classes=4):
    """Score calibration candidates on held-out calibration data."""
    oracle = metrics.get('oracle', {})
    if scope == 'min_class':
        values = []
        for c in range(num_classes):
            cls_metrics = oracle.get('per_class', {}).get(c) or oracle.get('per_class', {}).get(str(c))
            if cls_metrics and cls_metrics.get('n_samples', 0) >= 2:
                values.append(float(cls_metrics.get('R2', -999.0)))
        return min(values) if values else -999.0
    overall = oracle.get('overall', {})
    if metric == 'neg_mae':
        return -float(overall.get('MAE', 1e9))
    return float(overall.get('R2', -999.0))


def _select_per_class_affine_modes(base_model, classifier_model, train_loader, val_loader, device,
                                   num_classes, logger=None):
    """Choose none/bias/affine per class using held-out calibration validation R2."""
    bias_params = _fit_per_class_affine_params(train_loader, base_model, classifier_model, device,
                                               num_classes=num_classes, mode='bias_only')
    affine_params = _fit_per_class_affine_params(train_loader, base_model, classifier_model, device,
                                                 num_classes=num_classes, mode='affine_only')
    candidates = {
        'none': (None, evaluate_separate_regression(base_model, classifier_model, val_loader, device, num_classes)),
        'bias_only': (bias_params, evaluate_separate_regression(base_model, classifier_model, val_loader, device, num_classes,
                                                               affine_params=bias_params)),
        'affine_only': (affine_params, evaluate_separate_regression(base_model, classifier_model, val_loader, device, num_classes,
                                                                   affine_params=affine_params)),
    }

    selected_params = {}
    selected_modes = {}
    diagnostics = {}
    for c in range(num_classes):
        best_mode = 'none'
        best_r2 = -999.0
        diagnostics[c] = {}
        for mode, (params, metrics) in candidates.items():
            cls_metrics = metrics['oracle']['per_class'].get(c) or metrics['oracle']['per_class'].get(str(c))
            r2 = float(cls_metrics.get('R2', -999.0)) if cls_metrics else -999.0
            diagnostics[c][mode] = r2
            if r2 > best_r2:
                best_mode = mode
                best_r2 = r2
        selected_modes[c] = best_mode
        if best_mode == 'bias_only':
            selected_params[c] = bias_params[c]
        elif best_mode == 'affine_only':
            selected_params[c] = affine_params[c]
        if logger:
            logger.info(f"  Class {c}: per-class affine gate selected {best_mode} (val R2={best_r2:.4f})")
    return selected_params if selected_params else None, selected_modes, diagnostics


def _run_gated_target_calibration(base_model, classifier_model, calib_loader, device, config, args,
                                  reg_config, cid, reg_seed, logger):
    """Run guarded target calibration and return the selected model/affine params/diagnostics."""
    train_loader, val_loader = _split_calibration_loader(
        calib_loader, config.BATCH_SIZE,
        val_ratio=getattr(args, 'separate_reg_val_ratio', 0.3),
        seed=int(reg_seed + 20000 + cid),
        split_by=getattr(args, 'separate_reg_val_split', 'window'),
        logger=logger
    )
    if val_loader is None:
        val_loader = calib_loader

    gate_metric = getattr(args, 'separate_reg_gate_metric', 'r2')
    gate_scope = getattr(args, 'separate_reg_gate_scope', 'overall')
    min_delta = float(getattr(args, 'separate_reg_gate_min_delta', 0.0))
    fallback = getattr(args, 'separate_reg_gate_fallback', 'affine_only')

    before_metrics = evaluate_separate_regression(
        base_model, classifier_model, val_loader, device, config.NUM_CLASSES
    )
    before_score = _calibration_score(before_metrics, metric=gate_metric, scope=gate_scope,
                                      num_classes=config.NUM_CLASSES)

    full_model = copy.deepcopy(base_model).to(device)
    client_seed = reg_seed + 10000 + cid
    set_random_seed(client_seed)
    logger.info(f"Separate regression Client {cid}: gated full calibration seed={client_seed}")
    _train_separate_regression(
        full_model, train_loader, device, reg_config,
        steps=args.separate_reg_target_steps, lr=args.separate_reg_lr,
        feat_lr=args.separate_reg_feat_lr,
        unfreeze_policy=args.separate_reg_unfreeze,
        logger=logger, stage_name=f'target_client{cid}_gated_full'
    )
    full_metrics = evaluate_separate_regression(
        full_model, classifier_model, val_loader, device, config.NUM_CLASSES
    )
    full_score = _calibration_score(full_metrics, metric=gate_metric, scope=gate_scope,
                                    num_classes=config.NUM_CLASSES)

    selected_model = base_model
    selected_mode = 'none'
    affine_params = None
    selected_score = before_score
    fallback_info = None

    if full_score >= before_score + min_delta:
        selected_model = full_model
        selected_mode = 'full'
        selected_score = full_score
        logger.info(
            f"Separate regression Client {cid}: gated accepted full calibration "
            f"({gate_scope} {gate_metric}: {before_score:.4f} -> {full_score:.4f})"
        )
    else:
        logger.info(
            f"Separate regression Client {cid}: gated rejected full calibration "
            f"({gate_scope} {gate_metric}: {before_score:.4f} -> {full_score:.4f}); fallback={fallback}"
        )
        if fallback in ('bias_only', 'affine_only'):
            affine_params = _fit_per_class_affine_params(
                train_loader, base_model, classifier_model, device,
                num_classes=config.NUM_CLASSES, mode=fallback
            )
            fallback_metrics = evaluate_separate_regression(
                base_model, classifier_model, val_loader, device, config.NUM_CLASSES,
                affine_params=affine_params
            )
            fallback_score = _calibration_score(fallback_metrics, metric=gate_metric, scope=gate_scope,
                                                num_classes=config.NUM_CLASSES)
            fallback_info = {'mode': fallback, 'val_metrics': fallback_metrics, 'score': fallback_score}
            if fallback_score >= before_score + min_delta:
                selected_mode = fallback
                selected_score = fallback_score
                logger.info(
                    f"Separate regression Client {cid}: gated accepted fallback {fallback} "
                    f"({before_score:.4f} -> {fallback_score:.4f})"
                )
            else:
                affine_params = None
                logger.info(
                    f"Separate regression Client {cid}: gated rejected fallback {fallback} "
                    f"({before_score:.4f} -> {fallback_score:.4f}); keeping source model"
                )

    diagnostics = {
        'mode': 'gated',
        'selected_mode': selected_mode,
        'gate_metric': gate_metric,
        'gate_scope': gate_scope,
        'min_delta': min_delta,
        'before_score': before_score,
        'full_score': full_score,
        'selected_score': selected_score,
        'before_val_metrics': before_metrics,
        'full_val_metrics': full_metrics,
        'fallback': fallback_info,
    }
    return selected_model, affine_params, diagnostics


def _run_auto_target_calibration(base_model, classifier_model, calib_loader, device, config, args,
                                 reg_config, cid, reg_seed, logger):
    """Select none/full/bias/affine per client or per class using calibration validation data."""
    train_loader, val_loader = _split_calibration_loader(
        calib_loader, config.BATCH_SIZE,
        val_ratio=getattr(args, 'separate_reg_val_ratio', 0.3),
        seed=int(reg_seed + 30000 + cid),
        split_by=getattr(args, 'separate_reg_val_split', 'window'),
        logger=logger
    )
    if val_loader is None:
        val_loader = calib_loader

    gate_metric = getattr(args, 'separate_reg_gate_metric', 'r2')
    gate_scope = getattr(args, 'separate_reg_gate_scope', 'overall')
    min_delta = float(getattr(args, 'separate_reg_gate_min_delta', 0.0))
    selection_scope = getattr(args, 'separate_reg_auto_scope', 'per_class')

    before_metrics = evaluate_separate_regression(base_model, classifier_model, val_loader, device, config.NUM_CLASSES)
    before_score = _calibration_score(before_metrics, metric=gate_metric, scope=gate_scope,
                                      num_classes=config.NUM_CLASSES)

    full_model = copy.deepcopy(base_model).to(device)
    client_seed = reg_seed + 11000 + cid
    set_random_seed(client_seed)
    logger.info(f"Separate regression Client {cid}: auto full calibration seed={client_seed}")
    _train_separate_regression(
        full_model, train_loader, device, reg_config,
        steps=args.separate_reg_target_steps, lr=args.separate_reg_lr,
        feat_lr=args.separate_reg_feat_lr,
        unfreeze_policy=args.separate_reg_unfreeze,
        logger=logger, stage_name=f'target_client{cid}_auto_full'
    )
    full_metrics = evaluate_separate_regression(full_model, classifier_model, val_loader, device, config.NUM_CLASSES)
    full_score = _calibration_score(full_metrics, metric=gate_metric, scope=gate_scope,
                                    num_classes=config.NUM_CLASSES)

    selected_model = base_model
    selected_mode = 'none'
    affine_params = None
    selected_score = before_score
    per_class_info = None

    if selection_scope == 'per_class':
        base_affine, base_modes, base_diag = _select_per_class_affine_modes(
            base_model, classifier_model, train_loader, val_loader, device, config.NUM_CLASSES, logger=logger
        )
        full_eligible = full_score >= before_score + min_delta
        if not full_eligible:
            logger.info(
                f"Separate regression Client {cid}: auto will not use full per-class because "
                f"overall gate rejected it ({before_score:.4f} -> {full_score:.4f})"
            )
        selected_modes = {}
        selected_params = {}
        use_full_classes = set()
        for c in range(config.NUM_CLASSES):
            before_cls = before_metrics['oracle']['per_class'].get(c) or before_metrics['oracle']['per_class'].get(str(c))
            full_cls = full_metrics['oracle']['per_class'].get(c) or full_metrics['oracle']['per_class'].get(str(c))
            before_r2 = float(before_cls.get('R2', -999.0)) if before_cls else -999.0
            full_r2 = float(full_cls.get('R2', -999.0)) if full_cls else -999.0
            best_mode = base_modes.get(c, 'none')
            best_r2 = base_diag.get(c, {}).get(best_mode, before_r2)
            if full_eligible and full_r2 >= best_r2 + min_delta:
                best_mode = 'full'
                best_r2 = full_r2
                use_full_classes.add(c)
            selected_modes[c] = best_mode
            if best_mode in ('bias_only', 'affine_only') and base_affine and c in base_affine:
                selected_params[c] = base_affine[c]
            logger.info(f"  Class {c}: auto selected {best_mode} (before={before_r2:.4f}, full={full_r2:.4f}, selected={best_r2:.4f})")

        if use_full_classes:
            selected_model = full_model
        affine_params = selected_params if selected_params else None
        selected_mode = 'auto_per_class'
        selected_eval = evaluate_separate_regression(
            selected_model, classifier_model, val_loader, device, config.NUM_CLASSES,
            affine_params=affine_params
        )
        selected_score = _calibration_score(selected_eval, metric=gate_metric, scope=gate_scope,
                                            num_classes=config.NUM_CLASSES)
        per_class_info = {
            'selected_modes': selected_modes,
            'affine_candidate_r2': base_diag,
            'uses_full_classes': sorted(use_full_classes),
            'selected_val_metrics': selected_eval,
        }
    else:
        candidates = {'none': (base_model, None, before_metrics, before_score)}
        for mode in ('bias_only', 'affine_only'):
            params = _fit_per_class_affine_params(train_loader, base_model, classifier_model, device,
                                                  num_classes=config.NUM_CLASSES, mode=mode)
            cand_metrics = evaluate_separate_regression(base_model, classifier_model, val_loader, device,
                                                        config.NUM_CLASSES, affine_params=params)
            cand_score = _calibration_score(cand_metrics, metric=gate_metric, scope=gate_scope,
                                            num_classes=config.NUM_CLASSES)
            candidates[mode] = (base_model, params, cand_metrics, cand_score)
        candidates['full'] = (full_model, None, full_metrics, full_score)
        for mode, (cand_model, params, _, score) in candidates.items():
            if score >= selected_score + min_delta:
                selected_model = cand_model
                affine_params = params
                selected_mode = mode
                selected_score = score
        logger.info(f"Separate regression Client {cid}: auto selected {selected_mode} score={selected_score:.4f}")

    diagnostics = {
        'mode': 'auto',
        'auto_scope': selection_scope,
        'selected_mode': selected_mode,
        'gate_metric': gate_metric,
        'gate_scope': gate_scope,
        'min_delta': min_delta,
        'before_score': before_score,
        'full_score': full_score,
        'selected_score': selected_score,
        'before_val_metrics': before_metrics,
        'full_val_metrics': full_metrics,
        'per_class': per_class_info,
    }
    return selected_model, affine_params, diagnostics


def _per_class_oracle_r2(metrics, class_id):
    cls_metrics = metrics.get('oracle', {}).get('per_class', {}).get(class_id)
    if cls_metrics is None:
        cls_metrics = metrics.get('oracle', {}).get('per_class', {}).get(str(class_id))
    return float(cls_metrics.get('R2', -999.0)) if cls_metrics else -999.0


def _run_auto_v2_target_calibration(base_model, classifier_model, calib_loader, device, config, args,
                                    reg_config, cid, reg_seed, logger, train_loader=None, val_loader=None):
    """Strict class-wise selector over none/full/bias/affine/phase-affine candidates."""
    if train_loader is None or val_loader is None:
        train_loader, val_loader = _split_calibration_loader(
            calib_loader, config.BATCH_SIZE,
            val_ratio=getattr(args, 'separate_reg_val_ratio', 0.3),
            seed=int(reg_seed + 40000 + cid),
            split_by=getattr(args, 'separate_reg_val_split', 'window'),
            logger=logger
        )
        if val_loader is None:
            val_loader = calib_loader

    gate_metric = getattr(args, 'separate_reg_gate_metric', 'r2')
    gate_scope = getattr(args, 'separate_reg_gate_scope', 'overall')
    min_delta = float(getattr(args, 'separate_reg_gate_min_delta', 0.0))
    selection_scope = getattr(args, 'separate_reg_auto_scope', 'per_class')

    before_metrics = evaluate_separate_regression(base_model, classifier_model, val_loader, device, config.NUM_CLASSES)

    full_model = copy.deepcopy(base_model).to(device)
    client_seed = reg_seed + 12000 + cid
    set_random_seed(client_seed)
    logger.info(f"Separate regression Client {cid}: auto_v2 full calibration seed={client_seed}")
    _train_separate_regression(
        full_model, train_loader, device, reg_config,
        steps=args.separate_reg_target_steps, lr=args.separate_reg_lr,
        feat_lr=args.separate_reg_feat_lr,
        unfreeze_policy=args.separate_reg_unfreeze,
        logger=logger, stage_name=f'target_client{cid}_auto_v2_full'
    )
    full_metrics = evaluate_separate_regression(full_model, classifier_model, val_loader, device, config.NUM_CLASSES)

    bias_params = _fit_per_class_affine_params(
        train_loader, base_model, classifier_model, device, num_classes=config.NUM_CLASSES, mode='bias_only'
    )
    bias_metrics = evaluate_separate_regression(
        base_model, classifier_model, val_loader, device, config.NUM_CLASSES, affine_params=bias_params
    )

    affine_params_all = _fit_per_class_affine_params(
        train_loader, base_model, classifier_model, device, num_classes=config.NUM_CLASSES, mode='affine_only'
    )
    affine_metrics = evaluate_separate_regression(
        base_model, classifier_model, val_loader, device, config.NUM_CLASSES, affine_params=affine_params_all
    )

    phase_affine_params_all = _fit_per_class_phase_affine_params(
        train_loader, base_model, classifier_model, device,
        num_classes=config.NUM_CLASSES, num_phases=getattr(config, 'NUM_PHASES', 3)
    )
    phase_metrics = evaluate_separate_regression(
        base_model, classifier_model, val_loader, device, config.NUM_CLASSES,
        phase_affine_params=phase_affine_params_all
    )

    candidates = {
        'none': {'metrics': before_metrics, 'model': 'base', 'params': None},
        'bias_only': {'metrics': bias_metrics, 'model': 'base', 'params': bias_params},
        'affine_only': {'metrics': affine_metrics, 'model': 'base', 'params': affine_params_all},
        'phase_affine_only': {'metrics': phase_metrics, 'model': 'base', 'params': phase_affine_params_all},
        'full': {'metrics': full_metrics, 'model': 'full', 'params': None},
    }

    if selection_scope == 'client':
        selected_mode = 'none'
        selected_score = _calibration_score(before_metrics, metric=gate_metric, scope=gate_scope,
                                            num_classes=config.NUM_CLASSES)
        for mode, info in candidates.items():
            score = _calibration_score(info['metrics'], metric=gate_metric, scope=gate_scope,
                                       num_classes=config.NUM_CLASSES)
            if score >= selected_score + min_delta:
                selected_mode = mode
                selected_score = score
        logger.info(f"Separate regression Client {cid}: auto_v2 client selected {selected_mode} score={selected_score:.4f}")
        if selected_mode == 'full':
            diagnostics = {
                'mode': 'auto_v2', 'auto_scope': selection_scope, 'selected_mode': selected_mode,
                'selected_score': selected_score, 'candidate_val_metrics': {k: v['metrics'] for k, v in candidates.items()}
            }
            return full_model, None, None, None, None, diagnostics
        selected_affine = candidates[selected_mode]['params'] if selected_mode in ('bias_only', 'affine_only') else None
        selected_phase = candidates[selected_mode]['params'] if selected_mode == 'phase_affine_only' else None
        diagnostics = {
            'mode': 'auto_v2', 'auto_scope': selection_scope, 'selected_mode': selected_mode,
            'selected_score': selected_score, 'candidate_val_metrics': {k: v['metrics'] for k, v in candidates.items()}
        }
        return base_model, selected_affine, selected_phase, None, None, diagnostics

    selected_modes = {}
    routed_affine = {}
    routed_phase_affine = {}
    class_candidate_r2 = {}
    for c in range(config.NUM_CLASSES):
        best_mode = 'none'
        best_r2 = _per_class_oracle_r2(before_metrics, c)
        class_candidate_r2[c] = {}
        for mode, info in candidates.items():
            r2 = _per_class_oracle_r2(info['metrics'], c)
            class_candidate_r2[c][mode] = r2
            if r2 >= best_r2 + min_delta:
                best_mode = mode
                best_r2 = r2
        selected_modes[c] = best_mode
        if best_mode == 'bias_only':
            routed_affine[c] = bias_params[c]
        elif best_mode == 'affine_only':
            routed_affine[c] = affine_params_all[c]
        elif best_mode == 'phase_affine_only':
            routed_phase_affine[c] = phase_affine_params_all[c]
        logger.info(
            f"  Class {c}: auto_v2 selected {best_mode} "
            f"(none={class_candidate_r2[c]['none']:.4f}, full={class_candidate_r2[c]['full']:.4f}, "
            f"bias={class_candidate_r2[c]['bias_only']:.4f}, affine={class_candidate_r2[c]['affine_only']:.4f}, "
            f"phase={class_candidate_r2[c]['phase_affine_only']:.4f})"
        )

    routing_config = {
        'selected_modes': selected_modes,
        'affine_params': routed_affine,
        'phase_affine_params': routed_phase_affine,
    }
    selected_eval = evaluate_separate_regression(
        base_model, classifier_model, val_loader, device, config.NUM_CLASSES,
        routing_config=routing_config, full_model=full_model
    )
    selected_score = _calibration_score(selected_eval, metric=gate_metric, scope=gate_scope,
                                        num_classes=config.NUM_CLASSES)

    diagnostics = {
        'mode': 'auto_v2',
        'auto_scope': selection_scope,
        'selected_mode': 'auto_v2_per_class',
        'gate_metric': gate_metric,
        'gate_scope': gate_scope,
        'min_delta': min_delta,
        'selected_score': selected_score,
        'selected_modes': selected_modes,
        'class_candidate_r2': class_candidate_r2,
        'candidate_val_metrics': {k: v['metrics'] for k, v in candidates.items()},
        'selected_val_metrics': selected_eval,
    }
    return base_model, None, None, routing_config, full_model, diagnostics


def _run_auto_v2_specialist_target_calibration(base_model, classifier_model, calib_loader, device, config, args,
                                               reg_config, cid, reg_seed, logger):
    """Use general auto_v2 routing, then override selected classes with specialist full models."""
    train_loader, val_loader = _split_calibration_loader(
        calib_loader, config.BATCH_SIZE,
        val_ratio=getattr(args, 'separate_reg_val_ratio', 0.3),
        seed=int(reg_seed + 40000 + cid),
        split_by=getattr(args, 'separate_reg_val_split', 'window'),
        logger=logger
    )
    if val_loader is None:
        val_loader = calib_loader

    general_model, affine_params, phase_affine_params, routing_config, full_model, general_diag = _run_auto_v2_target_calibration(
        base_model, classifier_model, calib_loader, device, config, args, reg_config, cid, reg_seed, logger,
        train_loader=train_loader, val_loader=val_loader
    )
    if routing_config is None:
        routing_config = {'selected_modes': {}, 'affine_params': {}, 'phase_affine_params': {}}

    specialist_classes = [
        int(x.strip()) for x in str(getattr(args, 'separate_reg_specialist_classes', '2')).split(',') if x.strip()
    ]
    specialist_weight = float(getattr(args, 'separate_reg_specialist_weight', 2.0))
    specialist_steps = int(getattr(args, 'separate_reg_specialist_steps', args.separate_reg_target_steps))
    specialist_refit_full_calib = bool(getattr(args, 'separate_reg_specialist_refit_full_calib', False))
    specialist_refit_steps_arg = getattr(args, 'separate_reg_specialist_refit_steps', None)
    specialist_refit_steps = specialist_steps if specialist_refit_steps_arg is None else int(specialist_refit_steps_arg)
    specialist_refit_map = _parse_client_class_spec(getattr(args, 'separate_reg_specialist_refit_client_classes', ''))
    specialist_models = {}
    specialist_val_metrics = {}
    specialist_settings = {}
    specialist_gate = bool(getattr(args, 'separate_reg_specialist_gate', False))
    specialist_gate_min_delta = float(getattr(args, 'separate_reg_specialist_gate_min_delta', 0.0))
    selected_modes = {int(k): v for k, v in dict(routing_config.get('selected_modes', {})).items()}
    general_selected_metrics = None
    gate_decisions = {}

    if specialist_gate:
        general_selected_metrics = general_diag.get('selected_val_metrics') if isinstance(general_diag, dict) else None
        if general_selected_metrics is None:
            general_selected_metrics = evaluate_separate_regression(
                general_model, classifier_model, val_loader, device, config.NUM_CLASSES,
                routing_config=routing_config, full_model=full_model
            )

    for cls_id in specialist_classes:
        specialist_model = copy.deepcopy(base_model).to(device)
        specialist_config = copy.deepcopy(reg_config)
        specialist_config.SEPARATE_REG_CLASS_WEIGHTS = f"{cls_id}:{specialist_weight}"
        specialist_config.SEPARATE_REG_RANK_CLASSES = []
        specialist_config.SEPARATE_REG_RANK_WEIGHT = 0.0
        client_seed = reg_seed + 13000 + cid * 10 + cls_id
        set_random_seed(client_seed)
        logger.info(
            f"Separate regression Client {cid}: specialist class {cls_id} seed={client_seed}, "
            f"weight={specialist_weight}, steps={specialist_steps}"
        )
        _train_separate_regression(
            specialist_model, train_loader, device, specialist_config,
            steps=specialist_steps, lr=args.separate_reg_lr,
            feat_lr=args.separate_reg_feat_lr,
            unfreeze_policy=args.separate_reg_unfreeze,
            logger=logger, stage_name=f'target_client{cid}_specialist_class{cls_id}'
        )
        specialist_models[cls_id] = specialist_model
        specialist_val_metrics[cls_id] = evaluate_separate_regression(
            specialist_model, classifier_model, val_loader, device, config.NUM_CLASSES
        )
        specialist_settings[cls_id] = {
            'class_weight': specialist_weight,
            'steps': specialist_steps,
            'seed': client_seed,
            'refit_full_calib': False,
        }
        general_r2 = _per_class_oracle_r2(general_selected_metrics, cls_id) if general_selected_metrics else -999.0
        specialist_r2 = _per_class_oracle_r2(specialist_val_metrics[cls_id], cls_id)
        use_specialist = (not specialist_gate) or (specialist_r2 >= general_r2 + specialist_gate_min_delta)
        if use_specialist:
            selected_modes[cls_id] = 'specialist_full'
        else:
            specialist_models.pop(cls_id, None)
        selected_label = selected_modes.get(cls_id, 'general')
        gate_decisions[cls_id] = {
            'general_r2': general_r2,
            'specialist_r2': specialist_r2,
            'min_delta': specialist_gate_min_delta,
            'accepted': bool(use_specialist),
            'final_mode': selected_label,
        }
        logger.info(
            f"  Class {cls_id}: specialist full val R2={specialist_r2:.4f}, "
            f"general val R2={general_r2:.4f}, selected={selected_label}"
        )

    if specialist_refit_full_calib:
        for cls_id in list(specialist_models.keys()):
            if specialist_refit_map and cls_id not in specialist_refit_map.get(int(cid), set()):
                continue
            refit_model = copy.deepcopy(base_model).to(device)
            refit_config = copy.deepcopy(reg_config)
            refit_config.SEPARATE_REG_CLASS_WEIGHTS = f'{cls_id}:{specialist_weight}'
            refit_config.SEPARATE_REG_RANK_CLASSES = []
            refit_config.SEPARATE_REG_RANK_WEIGHT = 0.0
            refit_seed = reg_seed + 14000 + cid * 10 + cls_id
            set_random_seed(refit_seed)
            logger.info(
                f'Separate regression Client {cid}: refit specialist class {cls_id} on full calibration '
                f'seed={refit_seed}, weight={specialist_weight}, steps={specialist_refit_steps}'
            )
            _train_separate_regression(
                refit_model, calib_loader, device, refit_config,
                steps=specialist_refit_steps, lr=args.separate_reg_lr,
                feat_lr=args.separate_reg_feat_lr,
                unfreeze_policy=args.separate_reg_unfreeze,
                logger=logger, stage_name=f'target_client{cid}_specialist_class{cls_id}_full_refit'
            )
            specialist_models[cls_id] = refit_model
            specialist_settings[cls_id]['refit_full_calib'] = True
            specialist_settings[cls_id]['refit_steps'] = specialist_refit_steps
            specialist_settings[cls_id]['refit_seed'] = refit_seed

    routing_config['selected_modes'] = selected_modes

    selected_eval = evaluate_separate_regression(
        general_model, classifier_model, val_loader, device, config.NUM_CLASSES,
        routing_config=routing_config, full_model=full_model,
        specialist_models=specialist_models
    )
    diagnostics = {
        'mode': 'auto_v2_specialist',
        'general_auto_v2': general_diag,
        'specialist_classes': specialist_classes,
        'specialist_settings': specialist_settings,
        'specialist_gate': specialist_gate,
        'specialist_gate_min_delta': specialist_gate_min_delta,
        'specialist_gate_decisions': gate_decisions,
        'general_selected_val_metrics': general_selected_metrics,
        'specialist_val_metrics': specialist_val_metrics,
        'selected_val_metrics': selected_eval,
        'selected_modes': routing_config.get('selected_modes', {}),
    }
    return general_model, affine_params, phase_affine_params, routing_config, full_model, specialist_models, diagnostics


def _train_separate_regression(model, loader, device, config, steps, lr, feat_lr=0.0,
                               unfreeze_policy='none', logger=None, stage_name='source'):
    if loader is None or steps <= 0:
        return
    for param in model.parameters():
        param.requires_grad = False

    reg_params = _get_separate_regression_params(model)
    encoder_params = _get_encoder_unfreeze_params(model, unfreeze_policy)
    if encoder_params and feat_lr > 0 and hasattr(model, 'reg_grad_detach'):
        model.reg_grad_detach = not bool(getattr(config, 'SEPARATE_REG_ALLOW_ENCODER_BACKPROP', True))
    for param in reg_params + encoder_params:
        param.requires_grad = True

    param_groups = []
    if reg_params:
        param_groups.append({'params': reg_params, 'lr': lr})
    if encoder_params and feat_lr > 0:
        param_groups.append({'params': encoder_params, 'lr': feat_lr})
    if not param_groups:
        raise RuntimeError("Separate regression: no trainable parameters selected")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=getattr(config, 'SEPARATE_REG_WEIGHT_DECAY', 1e-3))
    model.train()
    iterator = iter(loader)
    running_loss = 0.0
    log_every = max(1, steps // 5)

    from utils import normalize_concentration, get_conc_bucket_labels
    import torch.nn.functional as F
    class_weights = _parse_class_weight_string(getattr(config, 'SEPARATE_REG_CLASS_WEIGHTS', ''))
    rank_classes = set(getattr(config, 'SEPARATE_REG_RANK_CLASSES', []))
    rank_weight = float(getattr(config, 'SEPARATE_REG_RANK_WEIGHT', 0.0))
    rank_margin = float(getattr(config, 'SEPARATE_REG_RANK_MARGIN', 0.02))
    tail_weight = float(getattr(config, 'REG_TAIL_WEIGHT', 1.0))
    tail_threshold = float(getattr(config, 'REG_TAIL_THRESHOLD', 1.0))
    tail_classes_spec = str(getattr(config, 'REG_TAIL_CLASSES', '') or '').strip()
    tail_classes = {int(x.strip()) for x in tail_classes_spec.split(',') if x.strip()}
    num_conc_buckets = int(getattr(config, 'NUM_CONC_BUCKETS', 0) or 0)
    lambda_conc_bucket = float(getattr(config, 'LAMBDA_CONC_BUCKET', 0.0) or 0.0)
    conc_bucket_loss = str(getattr(config, 'CONC_BUCKET_LOSS', 'hard')).lower()
    conc_bucket_soft_sigma = max(float(getattr(config, 'CONC_BUCKET_SOFT_SIGMA', 1.0) or 1.0), 1e-6)
    conc_bucket_detach_feat = bool(getattr(config, 'CONC_BUCKET_DETACH_FEAT', False))
    conc_bucket_boundaries = getattr(config, 'CONC_BUCKET_BOUNDARIES', None)
    huber_delta = float(getattr(config, 'HUBER_DELTA', 0.2))
    class_huber_deltas = _parse_class_float_string(
        getattr(config, 'SEPARATE_REG_HUBER_DELTAS', ''), label='Huber delta'
    )
    if logger and class_huber_deltas:
        logger.info(
            f"Separate regression {stage_name}: Huber beta base={huber_delta}, "
            f"per_class={class_huber_deltas}"
        )

    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        x, y_cls, y_reg_full, y_phase = batch
        x = x.to(device)
        y_cls = y_cls.to(device)
        y_reg_full = y_reg_full.to(device)
        y_phase = y_phase.to(device)

        y_reg = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls].unsqueeze(1)
        y_norm = normalize_concentration(y_reg, y_cls)

        optimizer.zero_grad()
        _, _, reg_feat = model(x)
        pred_norm = model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
        per_sample_loss = _smooth_l1_loss_with_optional_class_beta(
            pred_norm, y_norm, y_cls, huber_delta, class_huber_deltas
        ).view(-1)
        sample_weights = torch.ones_like(per_sample_loss)
        if tail_weight > 1.0 and tail_threshold < 1.0:
            tail_mask = y_norm.view(-1) >= tail_threshold
            if tail_classes:
                class_mask = torch.zeros_like(tail_mask, dtype=torch.bool)
                for cls_id in tail_classes:
                    class_mask = class_mask | (y_cls == int(cls_id))
                tail_mask = tail_mask & class_mask
            sample_weights = torch.where(tail_mask, sample_weights.new_tensor(tail_weight), sample_weights)
        if class_weights:
            for cls_id, weight in class_weights.items():
                sample_weights = torch.where(
                    y_cls == int(cls_id),
                    sample_weights * sample_weights.new_tensor(float(weight)),
                    sample_weights,
                )
        if torch.any(sample_weights != 1.0):
            loss = (per_sample_loss * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1.0)
        else:
            loss = per_sample_loss.mean()
        if rank_weight > 0 and rank_classes:
            rank_loss = pred_norm.new_tensor(0.0)
            active_classes = 0
            for cls_id in rank_classes:
                mask = y_cls == int(cls_id)
                if mask.sum() >= 2:
                    rank_loss = rank_loss + _pairwise_rank_loss(pred_norm[mask], y_norm[mask], margin=rank_margin)
                    active_classes += 1
            if active_classes > 0:
                loss = loss + rank_weight * rank_loss / active_classes
        bucket_logits = getattr(model, '_conc_bucket_logits', None)
        if (
            num_conc_buckets > 0 and lambda_conc_bucket > 0.0
            and conc_bucket_boundaries is not None
            and bucket_logits is not None
        ):
            if conc_bucket_detach_feat:
                bucket_logits = model.conc_bucket_classifier(reg_feat.detach())
            bucket_logits = bucket_logits.view(y_cls.size(0), int(config.NUM_CLASSES), num_conc_buckets)
            class_bucket_logits = bucket_logits[torch.arange(y_cls.size(0), device=device), y_cls.long()]
            bucket_targets = get_conc_bucket_labels(y_reg, y_cls, conc_bucket_boundaries)
            if conc_bucket_loss == 'soft':
                bucket_ids = torch.arange(num_conc_buckets, device=device, dtype=class_bucket_logits.dtype).unsqueeze(0)
                center = bucket_targets.to(class_bucket_logits.dtype).unsqueeze(1)
                soft_targets = torch.exp(-0.5 * ((bucket_ids - center) / conc_bucket_soft_sigma).pow(2))
                soft_targets = soft_targets / torch.clamp(soft_targets.sum(dim=1, keepdim=True), min=1e-8)
                log_probs = F.log_softmax(class_bucket_logits, dim=1)
                bucket_loss = -(soft_targets * log_probs).sum(dim=1).mean()
            else:
                bucket_loss = F.cross_entropy(class_bucket_logits, bucket_targets)
            loss = loss + lambda_conc_bucket * bucket_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for group in param_groups for p in group['params']], max_norm=5.0)
        optimizer.step()
        running_loss += float(loss.item())

        if logger and (step == 1 or step % log_every == 0 or step == steps):
            logger.info(f"Separate regression {stage_name}: step {step}/{steps}, loss={running_loss / step:.6f}")


def _denormalize_norm_by_class(pred_norm, class_ids):
    pred_norm = pred_norm.view(-1)
    class_ids = class_ids.view(-1).long()
    ppm = torch.zeros_like(pred_norm)
    from utils import CONC_STATS
    for cls_id in sorted(CONC_STATS.keys()):
        mask = class_ids == cls_id
        if mask.any():
            min_c = CONC_STATS[cls_id]['min']
            max_c = CONC_STATS[cls_id]['max']
            ppm[mask] = pred_norm[mask] * (max_c - min_c) + min_c
    return ppm


def _metric_dict(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return {
            'R2': -999.0, 'RMSE': -1.0, 'MAE': -1.0, 'MedAE': -1.0,
            'P90AE': -1.0, 'P95AE': -1.0, 'NRMSE': -1.0, 'nMAE': -1.0,
            'Bias': 0.0, 'n_samples': int(valid.sum())
        }
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    abs_err = np.abs(y_pred - y_true)
    value_range = float(np.max(y_true) - np.min(y_true))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'RMSE': rmse,
        'MAE': mae,
        'MedAE': float(np.median(abs_err)),
        'P90AE': float(np.percentile(abs_err, 90)),
        'P95AE': float(np.percentile(abs_err, 95)),
        'NRMSE': float(rmse / value_range) if value_range > 1e-12 else -1.0,
        'nMAE': float(mae / value_range) if value_range > 1e-12 else -1.0,
        'Bias': float(np.mean(y_pred - y_true)),
        'n_samples': int(len(y_true)),
    }


def _fit_affine_from_arrays(y_true, y_pred, mode='affine_only'):
    """Fit y_true = a * y_pred + b in ppm space."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    n = int(len(y_true))

    if n < 2:
        return {'a': 1.0, 'b': 0.0, 'n_samples': n, 'calib_r2': 0.0, 'calib_mae': 0.0}

    residual = y_true - y_pred
    if mode == 'bias_only':
        a = 1.0
        b = float(np.mean(residual))
    else:
        y_pred_var = np.var(y_pred)
        if y_pred_var < 1e-12:
            a = 1.0
            b = float(np.mean(residual))
        else:
            A = np.column_stack([y_pred, np.ones_like(y_pred)])
            coeffs, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
            a, b = float(coeffs[0]), float(coeffs[1])

    y_adj = a * y_pred + b
    ss_res = np.sum((y_true - y_adj) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    calib_r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
    calib_mae = float(np.mean(np.abs(y_true - y_adj)))
    return {'a': a, 'b': b, 'n_samples': n, 'calib_r2': calib_r2, 'calib_mae': calib_mae}


def _fit_per_class_affine_params(data_loader, reg_model, classifier_model, device, num_classes=4,
                                  mode='affine_only', semantic_protos=None,
                                  soft_agg_temp=0.35, prior_weight=0.2):
    """
    在 ppm 原始浓度空间，为每个类别拟合 affine/bias 校准参数。

    affine_only: 对每个类别 c, 拟合 y_true = a_c * y_pred + b_c
    bias_only: 对每个类别 c, 拟合 y_true = y_pred + b_c

    参数校准在 oracle 模式下进行（使用真实类别选择回归头）。

    返回: {class_id: {'a': a, 'b': b, 'n_samples': n, 'calib_r2': r2, 'calib_mae': mae}}
    """
    reg_model.eval()
    classifier_model.eval()
    stores = {c: {'true': [], 'pred': []} for c in range(num_classes)}

    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in data_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device)

            _, _, reg_feat = reg_model(x)
            # oracle 模式：使用真实类别选择回归头
            pred_norm = reg_model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            pred_ppm = _denormalize_norm_by_class(pred_norm, y_cls)

            for c in range(num_classes):
                mask = y_cls == c
                if mask.any():
                    stores[c]['true'].extend(y_true[mask].detach().cpu().numpy().tolist())
                    stores[c]['pred'].extend(pred_ppm[mask].detach().cpu().numpy().tolist())

    params = {}
    for c in range(num_classes):
        y_true = np.asarray(stores[c]['true'], dtype=np.float64)
        y_pred = np.asarray(stores[c]['pred'], dtype=np.float64)
        n = len(y_true)

        if n < 2:
            params[c] = {'a': 1.0, 'b': 0.0, 'n_samples': n, 'calib_r2': 0.0, 'calib_mae': 0.0}
            continue

        residual = y_true - y_pred

        if mode == 'bias_only':
            # 只拟合偏置: y_true = y_pred + b
            b = float(np.mean(residual))
            a = 1.0
        else:
            # affine: 拟合 y_true = a * y_pred + b
            y_pred_var = np.var(y_pred)
            if y_pred_var < 1e-12:
                # 预测方差接近零，退化到 bias_only
                b = float(np.mean(residual))
                a = 1.0
            else:
                A = np.column_stack([y_pred, np.ones_like(y_pred)])
                coeffs, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
                a, b = float(coeffs[0]), float(coeffs[1])

        y_adj = a * y_pred + b
        ss_res = np.sum((y_true - y_adj) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        calib_r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
        calib_mae = float(np.mean(np.abs(y_true - y_adj)))

        params[c] = {'a': a, 'b': b, 'n_samples': n, 'calib_r2': calib_r2, 'calib_mae': calib_mae}

    return params


def _fit_per_class_phase_affine_params(data_loader, reg_model, classifier_model, device,
                                       num_classes=4, num_phases=3):
    """Fit class+phase affine calibration in ppm space using oracle class labels."""
    reg_model.eval()
    classifier_model.eval()
    class_stores = {c: {'true': [], 'pred': []} for c in range(num_classes)}
    phase_stores = {
        c: {p: {'true': [], 'pred': []} for p in range(num_phases)}
        for c in range(num_classes)
    }

    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in data_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device)

            _, _, reg_feat = reg_model(x)
            pred_norm = reg_model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            pred_ppm = _denormalize_norm_by_class(pred_norm, y_cls)

            for c in range(num_classes):
                class_mask = y_cls == c
                if class_mask.any():
                    class_stores[c]['true'].extend(y_true[class_mask].detach().cpu().numpy().tolist())
                    class_stores[c]['pred'].extend(pred_ppm[class_mask].detach().cpu().numpy().tolist())
                for p in range(num_phases):
                    mask = (y_cls == c) & (y_phase == p)
                    if mask.any():
                        phase_stores[c][p]['true'].extend(y_true[mask].detach().cpu().numpy().tolist())
                        phase_stores[c][p]['pred'].extend(pred_ppm[mask].detach().cpu().numpy().tolist())

    params = {}
    for c in range(num_classes):
        class_fallback = _fit_affine_from_arrays(class_stores[c]['true'], class_stores[c]['pred'], mode='affine_only')
        phase_params = {}
        for p in range(num_phases):
            fitted = _fit_affine_from_arrays(
                phase_stores[c][p]['true'], phase_stores[c][p]['pred'], mode='affine_only'
            )
            if fitted['n_samples'] < 2:
                fitted = dict(class_fallback)
                fitted['fallback'] = 'class_affine'
            phase_params[p] = fitted
        params[c] = {'class_fallback': class_fallback, 'phases': phase_params}
    return params


def _apply_affine_to_predictions(pred_ppm, class_ids, affine_params):
    """
    将 per-class affine 校正应用于 ppm 空间预测。

    输入:
        pred_ppm: [batch_size], 去归一化后的 ppm 预测值
        class_ids: [batch_size], 每个样本对应的类别（oracle 用真实类，pipeline 用预测类）
        affine_params: dict, {class_id: {'a': a, 'b': b, ...}}
    返回:
        [batch_size], 校正后的 ppm 预测值
    """
    class_ids = class_ids.view(-1).long()
    pred_ppm = pred_ppm.view(-1)
    result = pred_ppm.clone()
    for c in sorted(affine_params.keys()):
        mask = class_ids == int(c)
        if mask.any():
            a = affine_params[c]['a']
            b = affine_params[c]['b']
            result[mask] = pred_ppm[mask] * a + b
    return result


def _apply_phase_affine_to_predictions(pred_ppm, class_ids, phase_ids, phase_affine_params):
    """Apply class+phase affine calibration in ppm space."""
    class_ids = class_ids.view(-1).long()
    phase_ids = phase_ids.view(-1).long()
    pred_ppm = pred_ppm.view(-1)
    result = pred_ppm.clone()
    for c, class_params in phase_affine_params.items():
        class_key = int(c)
        phase_params = class_params.get('phases', {})
        fallback = class_params.get('class_fallback', {'a': 1.0, 'b': 0.0})
        for p in torch.unique(phase_ids[class_ids == class_key]).detach().cpu().numpy().tolist():
            if int(p) < 0:
                params = fallback
            else:
                params = phase_params.get(int(p)) or phase_params.get(str(int(p))) or fallback
            mask = (class_ids == class_key) & (phase_ids == int(p))
            if mask.any():
                result[mask] = pred_ppm[mask] * float(params.get('a', 1.0)) + float(params.get('b', 0.0))
    return result


def _select_routed_params(routing_config, class_id, model_key='base'):
    if not routing_config:
        return model_key, None, None
    modes = routing_config.get('selected_modes', {})
    mode_name = modes.get(int(class_id), modes.get(str(int(class_id)), 'none'))
    if mode_name in ('specialist_full', 'specialist'):
        return 'specialist', None, None
    if mode_name == 'full':
        return 'full', None, None
    if mode_name in ('affine_only', 'bias_only'):
        params = routing_config.get('affine_params', {})
        return 'base', params.get(int(class_id)) or params.get(str(int(class_id))), None
    if mode_name == 'phase_affine_only':
        params = routing_config.get('phase_affine_params', {})
        return 'base', None, params.get(int(class_id)) or params.get(str(int(class_id)))
    return 'base', None, None


def _predict_routed_ppm(base_model, full_model, x, class_for_head, y_phase, num_classes,
                        routing_config=None, affine_params=None, phase_affine_params=None,
                        specialist_models=None):
    """Predict ppm with optional per-class model/calibration routing."""
    class_for_head = class_for_head.view(-1).long()
    y_phase = y_phase.view(-1).long()
    base_model.eval()
    if full_model is not None:
        full_model.eval()
    if specialist_models:
        for model in specialist_models.values():
            model.eval()
    _, _, base_feat = base_model(x)
    full_feat = None
    specialist_feat_cache = {}
    pred_ppm = torch.zeros(class_for_head.size(0), device=x.device)

    if routing_config is None:
        pred_norm = base_model.forward_reg(base_feat, y_cls=class_for_head, y_phase=y_phase)
        pred_ppm = _denormalize_norm_by_class(pred_norm, class_for_head)
        if affine_params is not None:
            pred_ppm = _apply_affine_to_predictions(pred_ppm, class_for_head, affine_params)
        if phase_affine_params is not None:
            pred_ppm = _apply_phase_affine_to_predictions(pred_ppm, class_for_head, y_phase, phase_affine_params)
        return pred_ppm

    for c in range(num_classes):
        mask = class_for_head == c
        if not mask.any():
            continue
        model_key, class_affine, class_phase_affine = _select_routed_params(routing_config, c)
        if model_key == 'specialist' and specialist_models and c in specialist_models:
            specialist_model = specialist_models[c]
            if c not in specialist_feat_cache:
                _, _, specialist_feat_cache[c] = specialist_model(x)
            pred_norm = specialist_model.forward_reg(
                specialist_feat_cache[c][mask], y_cls=class_for_head[mask], y_phase=y_phase[mask]
            )
        elif model_key == 'full' and full_model is not None:
            if full_feat is None:
                _, _, full_feat = full_model(x)
            pred_norm = full_model.forward_reg(full_feat[mask], y_cls=class_for_head[mask], y_phase=y_phase[mask])
        else:
            pred_norm = base_model.forward_reg(base_feat[mask], y_cls=class_for_head[mask], y_phase=y_phase[mask])
        ppm = _denormalize_norm_by_class(pred_norm, class_for_head[mask])
        if class_affine is not None:
            ppm = _apply_affine_to_predictions(ppm, class_for_head[mask], {c: class_affine})
        if class_phase_affine is not None:
            ppm = _apply_phase_affine_to_predictions(ppm, class_for_head[mask], y_phase[mask], {c: class_phase_affine})
        pred_ppm[mask] = ppm
    return pred_ppm


def _predict_soft_routed_ppm(base_model, full_model, x, pipeline_scores, y_phase, num_classes,
                             routing_config=None, affine_params=None, phase_affine_params=None,
                             specialist_models=None, top_k=2, temperature=1.0,
                             min_confidence=1.01, max_margin=-1.0):
    import torch.nn.functional as F
    scores = pipeline_scores.float()
    if temperature > 0 and abs(float(temperature) - 1.0) > 1e-8:
        scores = F.softmax(torch.log(torch.clamp(scores, min=1e-8)) / float(temperature), dim=1)
    top_k = max(1, min(int(top_k), int(num_classes)))
    top_vals, top_idx = torch.topk(scores, k=min(2, num_classes), dim=1)
    top_conf = top_vals[:, 0]
    margin = top_vals[:, 0] - top_vals[:, 1] if top_vals.size(1) > 1 else torch.ones_like(top_conf)
    use_soft = (top_conf < float(min_confidence)) | (margin < float(max_margin))
    hard_cls = scores.argmax(dim=1)
    hard_ppm = _predict_routed_ppm(
        base_model, full_model, x, hard_cls, y_phase, num_classes,
        routing_config=routing_config, affine_params=affine_params,
        phase_affine_params=phase_affine_params, specialist_models=specialist_models
    )
    if not use_soft.any():
        return hard_ppm
    top_vals, top_idx = torch.topk(scores, k=top_k, dim=1)
    norm_weights = top_vals / torch.clamp(top_vals.sum(dim=1, keepdim=True), min=1e-8)
    soft_ppm = torch.zeros_like(hard_ppm)
    for rank in range(top_k):
        cls_rank = top_idx[:, rank].long()
        ppm_rank = _predict_routed_ppm(
            base_model, full_model, x, cls_rank, y_phase, num_classes,
            routing_config=routing_config, affine_params=affine_params,
            phase_affine_params=phase_affine_params, specialist_models=specialist_models
        )
        soft_ppm = soft_ppm + norm_weights[:, rank] * ppm_rank
    return torch.where(use_soft, soft_ppm, hard_ppm)


def _collect_separate_regression_predictions(reg_model, classifier_model, data_loader, device, num_classes=4,
                                             mode='oracle', semantic_protos=None,
                                             soft_agg_temp=0.35, prior_weight=0.2, affine_params=None,
                                             phase_affine_params=None, routing_config=None, full_model=None,
                                             specialist_models=None, pipeline_regression_route='hard',
                                             pipeline_soft_topk=2, pipeline_soft_temperature=1.0,
                                             pipeline_soft_min_confidence=1.01, pipeline_soft_max_margin=-1.0,
                                             target_route_bank=None, class_route='logits',
                                             route_temperature=0.1, route_knn_k=7, route_mix_alpha=0.5):
    reg_model.eval()
    classifier_model.eval()
    stores = {c: {'true': [], 'pred': []} for c in range(num_classes)}
    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in data_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            if mode in ('pipeline', 'conditional_pipeline', 'wrong_pipeline'):
                if class_route == 'target_reg_router':
                    pipeline_scores = _predict_pipeline_scores_with_reg_router(
                        classifier_model, reg_model, full_model, x, y_phase, num_classes,
                        routing_config=routing_config, affine_params=affine_params,
                        phase_affine_params=phase_affine_params, specialist_models=specialist_models,
                        target_route_bank=target_route_bank
                    )
                    if pipeline_scores is None:
                        pipeline_scores = _predict_pipeline_scores(
                            classifier_model, x, num_classes, semantic_protos=semantic_protos,
                            soft_agg_temp=soft_agg_temp, prior_weight=prior_weight
                        )
                else:
                    pipeline_scores = _predict_pipeline_scores(
                        classifier_model, x, num_classes, semantic_protos=semantic_protos,
                        soft_agg_temp=soft_agg_temp, prior_weight=prior_weight,
                        target_route_bank=target_route_bank, class_route=class_route,
                        route_temperature=route_temperature, route_knn_k=route_knn_k,
                        route_mix_alpha=route_mix_alpha, y_phase=y_phase
                    )
                class_for_head = pipeline_scores.argmax(dim=1)
                if pipeline_regression_route == 'soft_topk':
                    pred_ppm = _predict_soft_routed_ppm(
                        reg_model, full_model, x, pipeline_scores, y_phase, num_classes,
                        routing_config=routing_config, affine_params=affine_params,
                        phase_affine_params=phase_affine_params, specialist_models=specialist_models,
                        top_k=pipeline_soft_topk, temperature=pipeline_soft_temperature,
                        min_confidence=pipeline_soft_min_confidence, max_margin=pipeline_soft_max_margin
                    )
                else:
                    pred_ppm = _predict_routed_ppm(
                        reg_model, full_model, x, class_for_head, y_phase, num_classes,
                        routing_config=routing_config, affine_params=affine_params,
                        phase_affine_params=phase_affine_params,
                        specialist_models=specialist_models
                    )
            else:
                class_for_head = y_cls
                pred_ppm = _predict_routed_ppm(
                    reg_model, full_model, x, class_for_head, y_phase, num_classes,
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params,
                    specialist_models=specialist_models
                )
            for c in range(num_classes):
                mask = y_cls == c
                if mode == 'conditional_pipeline':
                    mask = mask & (class_for_head == y_cls)
                elif mode == 'wrong_pipeline':
                    mask = mask & (class_for_head != y_cls)
                if mask.any():
                    stores[c]['true'].extend(y_true[mask].detach().cpu().numpy().tolist())
                    stores[c]['pred'].extend(pred_ppm[mask].detach().cpu().numpy().tolist())
    return stores


def _plot_prediction_stores(stores, save_dir, filename, title_prefix='Regression'):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for c in range(4):
        ax = axes[c]
        y_true = np.asarray(stores.get(c, {}).get('true', []), dtype=np.float64)
        y_pred = np.asarray(stores.get(c, {}).get('pred', []), dtype=np.float64)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if len(y_true) < 2:
            ax.text(0.5, 0.5, 'Insufficient samples', ha='center', va='center')
            ax.set_title(gas_names[c])
            continue
        metrics = _metric_dict(y_true, y_pred)
        ax.scatter(y_true, y_pred, c=colors[c], alpha=0.55, s=12)
        lo = min(float(y_true.min()), float(y_pred.min()))
        hi = max(float(y_true.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('True Concentration (ppm)')
        ax.set_ylabel('Predicted Concentration (ppm)')
        ax.set_title(
            f"{gas_names[c]}\nR2={metrics['R2']:.3f}, MAE={metrics['MAE']:.1f}, n={metrics['n_samples']}"
        )
        ax.grid(True, alpha=0.3)
    fig.suptitle(title_prefix)
    plt.tight_layout()
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(save_path)


def plot_separate_regression_scatter(reg_model, classifier_model, data_loader, device, save_dir, filename,
                                     mode='oracle', title_prefix='Regression', semantic_protos=None,
                                     soft_agg_temp=0.35, prior_weight=0.2, affine_params=None,
                                     phase_affine_params=None, routing_config=None, full_model=None,
                                     specialist_models=None, pipeline_regression_route='hard',
                                     pipeline_soft_topk=2, pipeline_soft_temperature=1.0,
                                     pipeline_soft_min_confidence=1.01, pipeline_soft_max_margin=-1.0,
                                     target_route_bank=None, class_route='logits',
                                     route_temperature=0.1, route_knn_k=7, route_mix_alpha=0.5):
    stores = _collect_separate_regression_predictions(
        reg_model, classifier_model, data_loader, device, mode=mode,
        semantic_protos=semantic_protos, soft_agg_temp=soft_agg_temp, prior_weight=prior_weight,
        affine_params=affine_params, phase_affine_params=phase_affine_params,
        routing_config=routing_config, full_model=full_model,
        specialist_models=specialist_models,
        pipeline_regression_route=pipeline_regression_route,
        pipeline_soft_topk=pipeline_soft_topk,
        pipeline_soft_temperature=pipeline_soft_temperature,
        pipeline_soft_min_confidence=pipeline_soft_min_confidence,
        pipeline_soft_max_margin=pipeline_soft_max_margin,
        target_route_bank=target_route_bank,
        class_route=class_route,
        route_temperature=route_temperature,
        route_knn_k=route_knn_k,
        route_mix_alpha=route_mix_alpha
    )
    return _plot_prediction_stores(stores, save_dir, filename, title_prefix=title_prefix)


def _make_router_features(logits, cls_feat, reg_feat, y_phase, feature_set='all', num_phases=3):
    parts = []
    feature_set = str(feature_set or 'all').lower()
    if feature_set in ('logits', 'logits_cls', 'all') and logits is not None:
        parts.append(logits.float())
    if feature_set in ('cls', 'logits_cls', 'all') and cls_feat is not None:
        parts.append(cls_feat.float())
    if feature_set in ('reg', 'all') and reg_feat is not None:
        parts.append(reg_feat.float())
    if feature_set in ('all', 'phase') and y_phase is not None:
        phase = y_phase.view(-1).long().clamp(min=0, max=int(num_phases) - 1)
        phase_oh = torch.zeros(phase.size(0), int(num_phases), device=phase.device, dtype=torch.float32)
        phase_oh.scatter_(1, phase.unsqueeze(1), 1.0)
        parts.append(phase_oh)
    if not parts:
        raise ValueError(f"No route features selected for feature_set={feature_set}")
    return torch.cat(parts, dim=1)


def _phase_one_hot(y_phase, num_phases):
    phase = y_phase.view(-1).long().clamp(min=0, max=int(num_phases) - 1)
    phase_oh = torch.zeros(phase.size(0), int(num_phases), device=phase.device, dtype=torch.float32)
    phase_oh.scatter_(1, phase.unsqueeze(1), 1.0)
    return phase_oh


def _candidate_regression_ppm(reg_model, full_model, x, y_phase, num_classes,
                              routing_config=None, affine_params=None,
                              phase_affine_params=None, specialist_models=None):
    cols = []
    for cls_id in range(num_classes):
        cls_tensor = torch.full((x.size(0),), int(cls_id), dtype=torch.long, device=x.device)
        ppm = _predict_routed_ppm(
            reg_model, full_model, x, cls_tensor, y_phase, num_classes,
            routing_config=routing_config, affine_params=affine_params,
            phase_affine_params=phase_affine_params, specialist_models=specialist_models
        )
        cols.append(ppm.view(-1, 1))
    return torch.cat(cols, dim=1)


def _normalize_candidate_ppm(candidate_ppm, num_classes):
    from utils import CONC_STATS
    cols = []
    for cls_id in range(num_classes):
        stats = CONC_STATS[int(cls_id)]
        cols.append((candidate_ppm[:, cls_id:cls_id + 1] - stats['min']) / (stats['max'] - stats['min'] + 1e-6))
    return torch.cat(cols, dim=1)


def _make_reg_router_features(logits, candidate_ppm, y_phase, num_classes, num_phases=3):
    probs = torch.softmax(logits.float(), dim=1)
    cand_norm = _normalize_candidate_ppm(candidate_ppm.float(), num_classes)
    return torch.cat([logits.float(), probs, cand_norm, _phase_one_hot(y_phase, num_phases)], dim=1)


def _scores_from_target_route_bank(cls_feat, route_bank, num_classes, mode='target_proto',
                                   temperature=0.1, knn_k=7, logits=None, reg_feat=None, y_phase=None):
    import torch.nn.functional as F
    if route_bank is None:
        return None
    feats_norm = F.normalize(cls_feat.float(), dim=-1)
    temperature = max(float(temperature), 1e-6)

    if mode in ('target_router', 'logits_target_router_mix'):
        router = route_bank.get('router')
        if router is None:
            return None
        router = router.to(cls_feat.device)
        router.eval()
        feature_set = route_bank.get('router_feature_set', 'all')
        num_phases = int(route_bank.get('num_phases', 3))
        route_x = _make_router_features(logits, cls_feat, reg_feat, y_phase, feature_set, num_phases)
        mean = route_bank.get('router_mean')
        std = route_bank.get('router_std')
        if mean is not None and std is not None:
            mean = mean.to(route_x.device)
            std = std.to(route_x.device)
            route_x = (route_x - mean) / torch.clamp(std, min=1e-6)
        return F.softmax(router(route_x), dim=1)

    if mode in ('target_proto', 'logits_target_proto_mix'):
        proto_matrix = route_bank.get('prototypes')
        if proto_matrix is None:
            return None
        proto_matrix = proto_matrix.to(feats_norm.device).float()
        valid_mask = route_bank.get('proto_valid')
        if valid_mask is not None:
            valid_mask = valid_mask.to(feats_norm.device).bool()
        else:
            valid_mask = torch.isfinite(proto_matrix).all(dim=1)
        if not valid_mask.any():
            return None
        sim = torch.matmul(feats_norm, F.normalize(proto_matrix, dim=-1).T) / temperature
        sim[:, ~valid_mask] = -1e9
        return F.softmax(sim, dim=1)

    if mode in ('target_knn', 'logits_target_knn_mix'):
        bank_feats = route_bank.get('features')
        bank_labels = route_bank.get('labels')
        if bank_feats is None or bank_labels is None or bank_feats.numel() == 0:
            return None
        bank_feats = F.normalize(bank_feats.to(feats_norm.device).float(), dim=-1)
        bank_labels = bank_labels.to(feats_norm.device).long()
        k = max(1, min(int(knn_k), int(bank_feats.size(0))))
        sim = torch.matmul(feats_norm, bank_feats.T)
        top_vals, top_idx = torch.topk(sim, k=k, dim=1)
        weights = F.softmax(top_vals / temperature, dim=1)
        top_labels = bank_labels[top_idx]
        scores = torch.zeros(feats_norm.size(0), num_classes, device=feats_norm.device)
        scores.scatter_add_(1, top_labels, weights)
        return scores / torch.clamp(scores.sum(dim=1, keepdim=True), min=1e-8)
    return None


def _predict_pipeline_scores(classifier_model, x, num_classes, semantic_protos=None,
                             soft_agg_temp=0.35, prior_weight=0.2,
                             target_route_bank=None, class_route='logits',
                             route_temperature=0.1, route_knn_k=7, route_mix_alpha=0.5,
                             y_phase=None):
    import torch.nn.functional as F
    logits, cls_feat, reg_feat = classifier_model(x)
    logits_scores = F.softmax(logits, dim=1)
    class_route = str(class_route or 'logits')
    if target_route_bank is not None and class_route != 'logits':
        route_scores = _scores_from_target_route_bank(
            cls_feat, target_route_bank, num_classes, mode=class_route,
            temperature=route_temperature, knn_k=route_knn_k,
            logits=logits, reg_feat=reg_feat, y_phase=y_phase
        )
        if route_scores is not None:
            if class_route.startswith('logits_'):
                alpha = min(max(float(route_mix_alpha), 0.0), 1.0)
                mixed = (1.0 - alpha) * logits_scores + alpha * route_scores
                return mixed / torch.clamp(mixed.sum(dim=1, keepdim=True), min=1e-8)
            return route_scores
    if semantic_protos:
        proto_keys = list(semantic_protos.keys())
        proto_matrix = torch.stack([semantic_protos[k].to(cls_feat.device) for k in proto_keys])
        proto_classes = torch.tensor(
            [int(str(k).strip('()').split(',')[0]) for k in proto_keys],
            device=cls_feat.device
        )
        feats_norm = F.normalize(cls_feat, dim=-1)
        proto_norm = F.normalize(proto_matrix, dim=-1)
        sim = torch.matmul(feats_norm, proto_norm.T) / soft_agg_temp
        weights = F.softmax(sim, dim=-1)
        scores = torch.zeros(cls_feat.size(0), num_classes, device=cls_feat.device)
        scores.scatter_add_(1, proto_classes.unsqueeze(0).expand(cls_feat.size(0), -1), weights)
        class_counts = torch.bincount(proto_classes, minlength=num_classes).float().to(cls_feat.device)
        scores = scores / torch.clamp(class_counts, min=1.0).unsqueeze(0)
        prior = torch.ones(num_classes, device=cls_feat.device) / num_classes
        scores = (1 - prior_weight) * scores + prior_weight * prior.unsqueeze(0)
        return scores / torch.clamp(scores.sum(dim=1, keepdim=True), min=1e-8)
    return logits_scores


def _predict_pipeline_scores_with_reg_router(classifier_model, reg_model, full_model, x, y_phase, num_classes,
                                             routing_config=None, affine_params=None,
                                             phase_affine_params=None, specialist_models=None,
                                             target_route_bank=None):
    if target_route_bank is None or target_route_bank.get('reg_router') is None:
        return None
    logits, _cls_feat, _reg_feat = classifier_model(x)
    candidate_ppm = _candidate_regression_ppm(
        reg_model, full_model, x, y_phase, num_classes,
        routing_config=routing_config, affine_params=affine_params,
        phase_affine_params=phase_affine_params, specialist_models=specialist_models
    )
    route_x = _make_reg_router_features(
        logits, candidate_ppm, y_phase, num_classes, int(target_route_bank.get('num_phases', 3))
    )
    mean = target_route_bank.get('reg_router_mean')
    std = target_route_bank.get('reg_router_std')
    if mean is not None and std is not None:
        mean = mean.to(route_x.device)
        std = std.to(route_x.device)
        route_x = (route_x - mean) / torch.clamp(std, min=1e-6)
    router = target_route_bank['reg_router'].to(route_x.device)
    router.eval()
    return torch.softmax(router(route_x), dim=1)


def _predict_pipeline_class(classifier_model, x, num_classes, semantic_protos=None,
                            soft_agg_temp=0.35, prior_weight=0.2):
    scores = _predict_pipeline_scores(
        classifier_model, x, num_classes, semantic_protos=semantic_protos,
        soft_agg_temp=soft_agg_temp, prior_weight=prior_weight
    )
    return scores.argmax(dim=1)


def _build_target_route_bank(classifier_model, calib_loader, device, num_classes=4,
                             max_samples=0, logger=None, client_id=None):
    if calib_loader is None:
        return None
    classifier_model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y_cls, *_rest in calib_loader:
            x = x.to(device)
            _, cls_feat, _ = classifier_model(x)
            feats.append(cls_feat.detach().cpu())
            labels.append(y_cls.detach().cpu().long())
    if not feats:
        return None
    feat_tensor = torch.cat(feats, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    if max_samples and max_samples > 0 and feat_tensor.size(0) > max_samples:
        # Keep deterministic coverage by taking evenly spaced calibration samples.
        idx = torch.linspace(0, feat_tensor.size(0) - 1, steps=int(max_samples)).long()
        feat_tensor = feat_tensor[idx]
        label_tensor = label_tensor[idx]
    proto_list = []
    valid = []
    for cls_id in range(num_classes):
        mask = label_tensor == cls_id
        if mask.any():
            proto_list.append(feat_tensor[mask].mean(dim=0))
            valid.append(True)
        else:
            proto_list.append(torch.zeros_like(feat_tensor[0]))
            valid.append(False)
    bank = {
        'features': feat_tensor,
        'labels': label_tensor,
        'prototypes': torch.stack(proto_list, dim=0),
        'proto_valid': torch.tensor(valid, dtype=torch.bool),
        'n_samples': int(feat_tensor.size(0)),
    }
    if logger:
        cid_text = f" Client {client_id}" if client_id is not None else ""
        logger.info(f"Target route bank{cid_text}: {bank['n_samples']} calibration features")
    return bank


def _train_target_router(classifier_model, calib_loader, device, num_classes=4, num_phases=3,
                         feature_set='all', epochs=80, lr=1e-3, weight_decay=1e-3,
                         hidden_dim=32, class_weights=None, focal_gamma=0.0,
                         logger=None, client_id=None):
    if calib_loader is None:
        return None, {}
    classifier_model.eval()
    x_parts = []
    y_parts = []
    with torch.no_grad():
        for x, y_cls, _y_reg_full, y_phase in calib_loader:
            x = x.to(device)
            y_phase = y_phase.to(device)
            logits, cls_feat, reg_feat = classifier_model(x)
            route_x = _make_router_features(logits, cls_feat, reg_feat, y_phase, feature_set, num_phases)
            x_parts.append(route_x.detach().cpu())
            y_parts.append(y_cls.detach().cpu().long())
    if not x_parts:
        return None, {}
    train_x = torch.cat(x_parts, dim=0).float().to(device)
    train_y = torch.cat(y_parts, dim=0).long().to(device)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_xn = (train_x - mean) / std

    input_dim = int(train_xn.size(1))
    hidden_dim = int(hidden_dim)
    if hidden_dim > 0:
        router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)
    else:
        router = nn.Linear(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    focal_gamma = float(focal_gamma or 0.0)
    batch_size = min(64, int(train_xn.size(0)))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(2025 + int(client_id or 0))
    for _epoch in range(int(epochs)):
        perm = torch.randperm(train_xn.size(0), generator=generator, device='cpu').to(device)
        for start in range(0, train_xn.size(0), batch_size):
            idx = perm[start:start + batch_size]
            logits = router(train_xn[idx])
            ce = torch.nn.functional.cross_entropy(
                logits, train_y[idx], weight=class_weights_tensor, reduction='none'
            )
            if focal_gamma > 0:
                probs = torch.softmax(logits, dim=1)
                pt = probs.gather(1, train_y[idx].view(-1, 1)).squeeze(1).clamp_min(1e-6)
                loss = (((1.0 - pt) ** focal_gamma) * ce).mean()
            else:
                loss = ce.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        train_pred = router(train_xn).argmax(dim=1)
        train_acc = float((train_pred == train_y).float().mean().item())
        cm = _classification_confusion_matrix(
            train_y.detach().cpu().numpy().astype(int).tolist(),
            train_pred.detach().cpu().numpy().astype(int).tolist(),
            num_classes
        ).tolist()
    metrics = {
        'router_train_acc': train_acc,
        'router_train_confusion': cm,
        'router_feature_set': str(feature_set),
        'router_epochs': int(epochs),
        'router_lr': float(lr),
        'router_weight_decay': float(weight_decay),
        'router_hidden_dim': int(hidden_dim),
        'router_focal_gamma': focal_gamma,
        'router_n_samples': int(train_xn.size(0)),
        'router_input_dim': input_dim,
    }
    if logger:
        cid_text = f" Client {client_id}" if client_id is not None else ""
        logger.info(
            f"Target router{cid_text}: feature_set={feature_set}, n={train_xn.size(0)}, "
            f"input_dim={input_dim}, train_acc={train_acc:.4f}"
        )
    return {
        'router': router.cpu(),
        'router_mean': mean.detach().cpu(),
        'router_std': std.detach().cpu(),
        'router_feature_set': str(feature_set),
        'num_phases': int(num_phases),
    }, metrics


def _train_target_reg_router(classifier_model, reg_model, full_model, calib_loader, device,
                             num_classes=4, num_phases=3, routing_config=None,
                             affine_params=None, phase_affine_params=None, specialist_models=None,
                             epochs=120, lr=1e-3, weight_decay=1e-3, hidden_dim=32,
                             class_weights=None, focal_gamma=0.0, logger=None, client_id=None):
    if calib_loader is None:
        return None, {}
    classifier_model.eval()
    reg_model.eval()
    if full_model is not None:
        full_model.eval()
    x_parts = []
    y_parts = []
    with torch.no_grad():
        for x, y_cls, _y_reg_full, y_phase in calib_loader:
            x = x.to(device)
            y_phase = y_phase.to(device)
            logits, _cls_feat, _reg_feat = classifier_model(x)
            candidate_ppm = _candidate_regression_ppm(
                reg_model, full_model, x, y_phase, num_classes,
                routing_config=routing_config, affine_params=affine_params,
                phase_affine_params=phase_affine_params, specialist_models=specialist_models
            )
            route_x = _make_reg_router_features(logits, candidate_ppm, y_phase, num_classes, num_phases)
            x_parts.append(route_x.detach().cpu())
            y_parts.append(y_cls.detach().cpu().long())
    if not x_parts:
        return None, {}
    train_x = torch.cat(x_parts, dim=0).float().to(device)
    train_y = torch.cat(y_parts, dim=0).long().to(device)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_xn = (train_x - mean) / std

    input_dim = int(train_xn.size(1))
    hidden_dim = int(hidden_dim)
    if hidden_dim > 0:
        router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)
    else:
        router = nn.Linear(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    focal_gamma = float(focal_gamma or 0.0)
    batch_size = min(64, int(train_xn.size(0)))
    generator = torch.Generator(device='cpu')
    generator.manual_seed(3025 + int(client_id or 0))
    for _epoch in range(int(epochs)):
        perm = torch.randperm(train_xn.size(0), generator=generator, device='cpu').to(device)
        for start in range(0, train_xn.size(0), batch_size):
            idx = perm[start:start + batch_size]
            logits_r = router(train_xn[idx])
            ce = torch.nn.functional.cross_entropy(
                logits_r, train_y[idx], weight=class_weights_tensor, reduction='none'
            )
            if focal_gamma > 0:
                probs = torch.softmax(logits_r, dim=1)
                pt = probs.gather(1, train_y[idx].view(-1, 1)).squeeze(1).clamp_min(1e-6)
                loss = (((1.0 - pt) ** focal_gamma) * ce).mean()
            else:
                loss = ce.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        train_pred = router(train_xn).argmax(dim=1)
        train_acc = float((train_pred == train_y).float().mean().item())
        cm = _classification_confusion_matrix(
            train_y.detach().cpu().numpy().astype(int).tolist(),
            train_pred.detach().cpu().numpy().astype(int).tolist(),
            num_classes
        ).tolist()
    metrics = {
        'reg_router_train_acc': train_acc,
        'reg_router_train_confusion': cm,
        'reg_router_epochs': int(epochs),
        'reg_router_lr': float(lr),
        'reg_router_weight_decay': float(weight_decay),
        'reg_router_hidden_dim': int(hidden_dim),
        'reg_router_focal_gamma': focal_gamma,
        'reg_router_n_samples': int(train_xn.size(0)),
        'reg_router_input_dim': input_dim,
    }
    if logger:
        cid_text = f" Client {client_id}" if client_id is not None else ""
        logger.info(
            f"Target reg-router{cid_text}: n={train_xn.size(0)}, input_dim={input_dim}, "
            f"train_acc={train_acc:.4f}"
        )
    return {
        'reg_router': router.cpu(),
        'reg_router_mean': mean.detach().cpu(),
        'reg_router_std': std.detach().cpu(),
        'num_phases': int(num_phases),
    }, metrics


def export_separate_regression_predictions(reg_model, classifier_model, data_loader, device, save_path,
                                           client_id, num_classes=4, semantic_protos=None,
                                           soft_agg_temp=0.35, prior_weight=0.2,
                                           affine_params=None, phase_affine_params=None,
                                           routing_config=None, full_model=None, specialist_models=None,
                                           pipeline_regression_route='hard', pipeline_soft_topk=2,
                                           pipeline_soft_temperature=1.0, pipeline_soft_min_confidence=1.01,
                                           pipeline_soft_max_margin=-1.0, target_route_bank=None,
                                           class_route='logits', route_temperature=0.1,
                                           route_knn_k=7, route_mix_alpha=0.5):
    reg_model.eval()
    classifier_model.eval()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    gas_names = ['Ethanol', 'CO', 'Ethylene', 'Methane']
    rows = []
    sample_idx = 0
    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in data_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device)
            if class_route == 'target_reg_router':
                pipeline_scores = _predict_pipeline_scores_with_reg_router(
                    classifier_model, reg_model, full_model, x, y_phase, num_classes,
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params, specialist_models=specialist_models,
                    target_route_bank=target_route_bank
                )
                if pipeline_scores is None:
                    pipeline_scores = _predict_pipeline_scores(
                        classifier_model, x, num_classes, semantic_protos=semantic_protos,
                        soft_agg_temp=soft_agg_temp, prior_weight=prior_weight
                    )
            else:
                pipeline_scores = _predict_pipeline_scores(
                    classifier_model, x, num_classes, semantic_protos=semantic_protos,
                    soft_agg_temp=soft_agg_temp, prior_weight=prior_weight,
                    target_route_bank=target_route_bank, class_route=class_route,
                    route_temperature=route_temperature, route_knn_k=route_knn_k,
                    route_mix_alpha=route_mix_alpha, y_phase=y_phase
                )
            pred_cls = pipeline_scores.argmax(dim=1)
            top_vals2, _ = torch.topk(pipeline_scores, k=min(2, num_classes), dim=1)
            top1_conf = top_vals2[:, 0]
            top2_conf = top_vals2[:, 1] if top_vals2.size(1) > 1 else torch.zeros_like(top1_conf)
            confidence_margin = top1_conf - top2_conf
            candidate_ppm = _candidate_regression_ppm(
                reg_model, full_model, x, y_phase, num_classes,
                routing_config=routing_config, affine_params=affine_params,
                phase_affine_params=phase_affine_params, specialist_models=specialist_models
            )
            soft_used = (
                (top1_conf < float(pipeline_soft_min_confidence))
                | (confidence_margin < float(pipeline_soft_max_margin))
            ) if pipeline_regression_route == 'soft_topk' else torch.zeros_like(top1_conf, dtype=torch.bool)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            oracle_ppm = _predict_routed_ppm(
                reg_model, full_model, x, y_cls, y_phase, num_classes,
                routing_config=routing_config, affine_params=affine_params,
                phase_affine_params=phase_affine_params, specialist_models=specialist_models
            )
            if pipeline_regression_route == 'soft_topk':
                pipeline_ppm = _predict_soft_routed_ppm(
                    reg_model, full_model, x, pipeline_scores, y_phase, num_classes,
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params, specialist_models=specialist_models,
                    top_k=pipeline_soft_topk, temperature=pipeline_soft_temperature,
                    min_confidence=pipeline_soft_min_confidence, max_margin=pipeline_soft_max_margin
                )
            else:
                pipeline_ppm = _predict_routed_ppm(
                    reg_model, full_model, x, pred_cls, y_phase, num_classes,
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params, specialist_models=specialist_models
                )
            for i in range(y_cls.size(0)):
                true_class = int(y_cls[i].item())
                pred_class = int(pred_cls[i].item())
                phase_id = int(y_phase[i].item())
                true_ppm = float(y_true[i].item())
                for mode_name, pred_value in [('oracle', oracle_ppm[i]), ('pipeline', pipeline_ppm[i])]:
                    pred_ppm = float(pred_value.item())
                    err = pred_ppm - true_ppm
                    rows.append({
                        'client_id': int(client_id),
                        'sample_index': sample_idx,
                        'mode': mode_name,
                        'true_class': true_class,
                        'true_gas': gas_names[true_class] if true_class < len(gas_names) else str(true_class),
                        'pred_class': pred_class,
                        'pred_gas': gas_names[pred_class] if pred_class < len(gas_names) else str(pred_class),
                        'phase': phase_id,
                        'true_ppm': true_ppm,
                        'pred_ppm': pred_ppm,
                        'error_ppm': err,
                        'abs_error_ppm': abs(err),
                        'class_correct': int(pred_class == true_class),
                        'top1_confidence': float(top1_conf[i].item()),
                        'top2_confidence': float(top2_conf[i].item()),
                        'confidence_margin': float(confidence_margin[i].item()),
                        'soft_route_used': int(bool(soft_used[i].item())),
                        **{f'route_score_{c}': float(pipeline_scores[i, c].item()) for c in range(num_classes)},
                        **{f'candidate_ppm_{c}': float(candidate_ppm[i, c].item()) for c in range(num_classes)},
                    })
                sample_idx += 1
    fieldnames = [
        'client_id', 'sample_index', 'mode', 'true_class', 'true_gas', 'pred_class', 'pred_gas',
        'phase', 'true_ppm', 'pred_ppm', 'error_ppm', 'abs_error_ppm', 'class_correct',
        'top1_confidence', 'top2_confidence', 'confidence_margin', 'soft_route_used'
    ]
    fieldnames.extend([f'route_score_{c}' for c in range(num_classes)])
    fieldnames.extend([f'candidate_ppm_{c}' for c in range(num_classes)])
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(save_path)

def evaluate_separate_regression(reg_model, classifier_model, data_loader, device, num_classes=4,
                                 semantic_protos=None, soft_agg_temp=0.35, prior_weight=0.2,
                                 affine_params=None, phase_affine_params=None,
                                 routing_config=None, full_model=None, specialist_models=None,
                                 pipeline_regression_route='hard', pipeline_soft_topk=2,
                                 pipeline_soft_temperature=1.0, pipeline_soft_min_confidence=1.01,
                                 pipeline_soft_max_margin=-1.0, target_route_bank=None,
                                 class_route='logits', route_temperature=0.1,
                                 route_knn_k=7, route_mix_alpha=0.5):
    """Report oracle-class and pipeline-class regression metrics on a labeled test loader."""
    reg_model.eval()
    classifier_model.eval()
    stores = {
        'oracle': {c: {'true': [], 'pred': []} for c in range(num_classes)},
        'pipeline': {c: {'true': [], 'pred': []} for c in range(num_classes)},
        'conditional_pipeline': {c: {'true': [], 'pred': []} for c in range(num_classes)},
        'wrong_pipeline': {c: {'true': [], 'pred': []} for c in range(num_classes)},
    }
    total_cls = 0
    correct_cls = 0
    cls_true_all = []
    cls_pred_all = []
    soft_route_count = 0

    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in data_loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device)

            if class_route == 'target_reg_router':
                pipeline_scores = _predict_pipeline_scores_with_reg_router(
                    classifier_model, reg_model, full_model, x, y_phase, num_classes,
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params, specialist_models=specialist_models,
                    target_route_bank=target_route_bank
                )
                if pipeline_scores is None:
                    pipeline_scores = _predict_pipeline_scores(
                        classifier_model, x, num_classes, semantic_protos=semantic_protos,
                        soft_agg_temp=soft_agg_temp, prior_weight=prior_weight
                    )
            else:
                pipeline_scores = _predict_pipeline_scores(
                    classifier_model, x, num_classes, semantic_protos=semantic_protos,
                    soft_agg_temp=soft_agg_temp, prior_weight=prior_weight,
                    target_route_bank=target_route_bank, class_route=class_route,
                    route_temperature=route_temperature, route_knn_k=route_knn_k,
                    route_mix_alpha=route_mix_alpha, y_phase=y_phase
                )
            pred_cls = pipeline_scores.argmax(dim=1)
            top_vals2, _ = torch.topk(pipeline_scores, k=min(2, num_classes), dim=1)
            top1_conf = top_vals2[:, 0]
            top2_conf = top_vals2[:, 1] if top_vals2.size(1) > 1 else torch.zeros_like(top1_conf)
            confidence_margin = top1_conf - top2_conf
            if pipeline_regression_route == 'soft_topk':
                soft_mask = (top1_conf < float(pipeline_soft_min_confidence)) | (
                    confidence_margin < float(pipeline_soft_max_margin)
                )
                soft_route_count += int(soft_mask.sum().item())

            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            oracle_ppm = _predict_routed_ppm(
                reg_model, full_model, x, y_cls, y_phase, num_classes,
                routing_config=routing_config, affine_params=affine_params,
                phase_affine_params=phase_affine_params,
                specialist_models=specialist_models
            )
            if pipeline_regression_route == 'soft_topk':
                pipeline_ppm = _predict_soft_routed_ppm(
                    reg_model, full_model, x, pipeline_scores, y_phase, num_classes,
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params,
                    specialist_models=specialist_models,
                    top_k=pipeline_soft_topk, temperature=pipeline_soft_temperature,
                    min_confidence=pipeline_soft_min_confidence, max_margin=pipeline_soft_max_margin
                )
            else:
                pipeline_ppm = _predict_routed_ppm(
                    reg_model, full_model, x, pred_cls, y_phase, num_classes,
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params,
                    specialist_models=specialist_models
                )

            total_cls += int(y_cls.numel())
            correct_cls += int((pred_cls == y_cls).sum().item())
            cls_true_all.extend(y_cls.detach().cpu().numpy().astype(int).tolist())
            cls_pred_all.extend(pred_cls.detach().cpu().numpy().astype(int).tolist())

            for c in range(num_classes):
                mask = y_cls == c
                if mask.any():
                    stores['oracle'][c]['true'].extend(y_true[mask].detach().cpu().numpy().tolist())
                    stores['oracle'][c]['pred'].extend(oracle_ppm[mask].detach().cpu().numpy().tolist())
                    stores['pipeline'][c]['true'].extend(y_true[mask].detach().cpu().numpy().tolist())
                    stores['pipeline'][c]['pred'].extend(pipeline_ppm[mask].detach().cpu().numpy().tolist())
                    correct_mask = mask & (pred_cls == y_cls)
                    wrong_mask = mask & (pred_cls != y_cls)
                    if correct_mask.any():
                        stores['conditional_pipeline'][c]['true'].extend(y_true[correct_mask].detach().cpu().numpy().tolist())
                        stores['conditional_pipeline'][c]['pred'].extend(pipeline_ppm[correct_mask].detach().cpu().numpy().tolist())
                    if wrong_mask.any():
                        stores['wrong_pipeline'][c]['true'].extend(y_true[wrong_mask].detach().cpu().numpy().tolist())
                        stores['wrong_pipeline'][c]['pred'].extend(pipeline_ppm[wrong_mask].detach().cpu().numpy().tolist())

    metrics = {}
    for mode in ['oracle', 'pipeline', 'conditional_pipeline', 'wrong_pipeline']:
        per_class = {}
        overall_true, overall_pred = [], []
        for c in range(num_classes):
            true_c = stores[mode][c]['true']
            pred_c = stores[mode][c]['pred']
            per_class[c] = _metric_dict(true_c, pred_c)
            overall_true.extend(true_c)
            overall_pred.extend(pred_c)
        metrics[mode] = {
            'per_class': per_class,
            'overall': _metric_dict(overall_true, overall_pred),
        }
    gap_summary = {}
    for cls_id in range(num_classes):
        oracle_r2 = metrics['oracle']['per_class'][cls_id].get('R2', -999.0)
        pipeline_r2 = metrics['pipeline']['per_class'][cls_id].get('R2', -999.0)
        conditional_r2 = metrics['conditional_pipeline']['per_class'][cls_id].get('R2', -999.0)
        gap_summary[cls_id] = {
            'oracle_minus_pipeline_R2': float(oracle_r2 - pipeline_r2),
            'conditional_minus_pipeline_R2': float(conditional_r2 - pipeline_r2),
            'oracle_minus_conditional_R2': float(oracle_r2 - conditional_r2),
        }
    gap_summary['overall'] = {
        'oracle_minus_pipeline_R2': float(metrics['oracle']['overall'].get('R2', -999.0) - metrics['pipeline']['overall'].get('R2', -999.0)),
        'conditional_minus_pipeline_R2': float(metrics['conditional_pipeline']['overall'].get('R2', -999.0) - metrics['pipeline']['overall'].get('R2', -999.0)),
        'oracle_minus_conditional_R2': float(metrics['oracle']['overall'].get('R2', -999.0) - metrics['conditional_pipeline']['overall'].get('R2', -999.0)),
    }
    metrics['gap_summary'] = gap_summary
    metrics['pipeline']['classification_accuracy'] = float(correct_cls / max(1, total_cls))
    metrics['pipeline']['soft_route_fraction'] = float(soft_route_count / max(1, total_cls))
    metrics['pipeline']['soft_route_count'] = int(soft_route_count)
    if total_cls > 0:
        cls_cm = _classification_confusion_matrix(cls_true_all, cls_pred_all, num_classes)
        metrics['pipeline']['classification_confusion'] = cls_cm.tolist()
        metrics['pipeline']['per_class_classification_accuracy'] = {
            c: float(v) for c, v in enumerate(_per_class_accuracy_from_confusion(cls_cm))
        }
    metrics['n_samples'] = int(total_cls)
    return metrics



def _extract_model_state_from_checkpoint(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ['model_state', 'model_state_dict', 'state_dict']:
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError('Checkpoint does not contain a recognizable model state dict')


def load_classifier_state_from_checkpoint(checkpoint_path, device, logger=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Classifier checkpoint not found: {checkpoint_path}')
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state = _extract_model_state_from_checkpoint(checkpoint_obj)
    if logger:
        logger.info(f"Loaded classifier checkpoint from {checkpoint_path} ({len(state)} tensors)")
    return state


def load_target_classifier_states_from_dir(target_classifier_dir, test_client_ids, device, logger=None):
    states = {}
    if not target_classifier_dir:
        return states
    target_classifier_dir = Path(target_classifier_dir)
    if not target_classifier_dir.exists():
        if logger:
            logger.warning(f"Target classifier directory not found: {target_classifier_dir}")
        return states
    for cid in test_client_ids:
        ckpt_path = target_classifier_dir / f"target_cls_client{cid}.pth"
        if not ckpt_path.exists():
            if logger:
                logger.warning(f"Target classifier checkpoint missing for Client {cid}: {ckpt_path}")
            continue
        ckpt_obj = torch.load(ckpt_path, map_location=device)
        try:
            states[cid] = _extract_model_state_from_checkpoint(ckpt_obj)
        except ValueError:
            if isinstance(ckpt_obj, dict):
                states[cid] = ckpt_obj
            else:
                raise
        if logger:
            logger.info(f"Loaded calibrated target classifier for Client {cid}: {ckpt_path}")
    return states


def build_regression_only_semantic_protos(config, classifier_state, train_client_ids, temp_federated_dir, logger=None):
    """Initialize semantic prototypes from the frozen classification checkpoint and source train splits."""
    from utils import create_model_by_config, load_shared_weights

    model = create_model_by_config(config, with_reg_head=False).to(config.DEVICE)
    load_shared_weights(model, classifier_state, strict=False)
    model.eval()
    proto_sums = {}
    proto_counts = {}
    for cid in train_client_ids:
        client_dir = Path(temp_federated_dir) / f'client_{cid}'
        feature_path = client_dir / 'train_features.npy'
        cls_path = client_dir / 'train_classification_labels.npy'
        phase_path = client_dir / 'train_phase_labels.npy'
        reg_path = client_dir / 'train_regression_labels.npy'
        if not (feature_path.exists() and cls_path.exists() and phase_path.exists() and reg_path.exists()):
            if logger:
                logger.warning(f"Regression-only prototype init skipped Client {cid}: missing train split")
            continue
        dataset = GasSensorPhaseDataset(
            np.load(feature_path), np.load(cls_path), np.load(reg_path),
            np.load(phase_path, allow_pickle=True)
        )
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        with torch.no_grad():
            for x, y_cls, _, y_phase in loader:
                x = x.to(config.DEVICE)
                _, cls_feat, _ = model(x)
                cls_feat = cls_feat.detach().cpu()
                y_cls = y_cls.detach().cpu()
                y_phase = y_phase.detach().cpu()
                for row in range(cls_feat.size(0)):
                    key = (int(y_cls[row].item()), int(y_phase[row].item()))
                    if key not in proto_sums:
                        proto_sums[key] = torch.zeros_like(cls_feat[row])
                        proto_counts[key] = 0
                    proto_sums[key] += cls_feat[row]
                    proto_counts[key] += 1
    protos = {key: proto_sums[key] / max(1, proto_counts[key]) for key in proto_sums}
    if logger:
        logger.info(f"Regression-only: initialized {len(protos)} semantic prototypes from source train splits")
    return protos


def recalibrate_target_classifiers_for_regression_only(args, config, classifier_state, initial_target_classifier_states,
                                                       test_client_ids, calibration_loaders, test_client_loaders, logger):
    """Rebuild selected target classifiers from the saved classification base for fast routing experiments."""
    from utils import create_model_by_config, load_shared_weights

    selected_clients = _parse_client_id_set(getattr(args, 'target_cls_calib_clients', ''))
    if selected_clients is None:
        selected_clients = set(int(cid) for cid in test_client_ids)
    else:
        selected_clients = set(int(cid) for cid in selected_clients)

    target_classifier_states = dict(initial_target_classifier_states or {})
    recalibration_metrics = {}
    previous_flag = getattr(args, 'target_cls_calibration', False)
    args.target_cls_calibration = True
    try:
        for cid in test_client_ids:
            if int(cid) not in selected_clients:
                continue
            eval_loader = test_client_loaders.get(cid)
            if eval_loader is None:
                logger.warning(f"Client {cid}: target classifier recalibration skipped (no test loader)")
                continue
            eval_model = create_model_by_config(config, with_reg_head=False).to(config.DEVICE)
            start_state = classifier_state
            if getattr(args, 'regression_only_recalibrate_from_existing_target', False) and cid in target_classifier_states:
                start_state = target_classifier_states[cid]
                logger.info(f"Client {cid}: target classifier recalibration starts from existing target classifier")
            else:
                logger.info(f"Client {cid}: target classifier recalibration starts from global classification base")
            load_shared_weights(eval_model, start_state, strict=False)
            calibrated, metrics = run_target_classification_calibration(
                eval_model, cid, calibration_loaders or {}, eval_loader, args, config, logger
            )
            if calibrated:
                target_classifier_states[cid] = copy.deepcopy(eval_model.state_dict())
                recalibration_metrics[cid] = metrics
    finally:
        args.target_cls_calibration = previous_flag
    return target_classifier_states, recalibration_metrics


def run_regression_only_pipeline(args, config, train_client_ids, test_client_ids, temp_federated_dir,
                                 calibration_loaders, test_client_loaders, logger):
    if args.regression_mode != 'separate':
        raise ValueError('--regression_only requires --regression_mode separate')
    classifier_state = load_classifier_state_from_checkpoint(args.classifier_checkpoint, config.DEVICE, logger)
    target_classifier_states = load_target_classifier_states_from_dir(
        args.target_classifier_dir, test_client_ids, config.DEVICE, logger
    )
    target_cls_recalibration_metrics = {}
    if getattr(args, 'regression_only_recalibrate_target_classifiers', False):
        target_classifier_states, target_cls_recalibration_metrics = recalibrate_target_classifiers_for_regression_only(
            args=args, config=config, classifier_state=classifier_state,
            initial_target_classifier_states=target_classifier_states,
            test_client_ids=test_client_ids, calibration_loaders=calibration_loaders,
            test_client_loaders=test_client_loaders, logger=logger
        )
    semantic_protos = build_regression_only_semantic_protos(
        config, classifier_state, train_client_ids, temp_federated_dir, logger
    )
    metrics = run_separate_regression_pipeline(
        args=args, config=config, classifier_state=classifier_state,
        train_client_ids=train_client_ids, temp_federated_dir=temp_federated_dir,
        calibration_loaders=calibration_loaders, test_client_loaders=test_client_loaders,
        logger=logger, semantic_protos=semantic_protos,
        target_classifier_states=target_classifier_states
    )
    results_dir = Path(config.MODEL_SAVE_DIR).parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'mode': 'regression_only',
        'classifier_checkpoint': str(args.classifier_checkpoint),
        'target_classifier_dir': str(args.target_classifier_dir) if args.target_classifier_dir else '',
        'train_clients': train_client_ids,
        'test_clients': test_client_ids,
        'target_cls_recalibration': target_cls_recalibration_metrics,
        'separate_regression': metrics,
    }
    with open(results_dir / 'regression_only_summary.json', 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable(summary), f, indent=2, ensure_ascii=False)
    logger.info(f"Regression-only summary saved to {results_dir / 'regression_only_summary.json'}")
    return metrics

def run_separate_regression_pipeline(args, config, classifier_state, train_client_ids, temp_federated_dir,
                                     calibration_loaders, test_client_loaders, logger, semantic_protos=None,
                                     target_classifier_states=None):
    """Two-model route: supervised regression model initialized from classification model."""
    from utils import create_model_by_config, load_shared_weights

    reg_seed = getattr(args, 'separate_reg_seed', None)
    if reg_seed is not None and reg_seed >= 0:
        set_random_seed(reg_seed)
        logger.info(f"Separate regression: reset random seed to {reg_seed}")

    device = config.DEVICE
    reg_config = copy.deepcopy(config)
    reg_config.USE_REG_LOSS = True
    reg_config.USE_DUAL_PROJ = True
    reg_config.REG_GRAD_DETACH = True
    reg_config.NUM_CONC_BUCKETS = int(getattr(args, 'num_conc_buckets', getattr(config, 'NUM_CONC_BUCKETS', 0)) or 0)
    reg_config.LAMBDA_CONC_BUCKET = float(getattr(args, 'lambda_conc_bucket', getattr(config, 'LAMBDA_CONC_BUCKET', 0.0)) or 0.0)
    reg_config.CONC_BUCKET_LOSS = str(getattr(args, 'conc_bucket_loss', getattr(config, 'CONC_BUCKET_LOSS', 'hard')))
    reg_config.CONC_BUCKET_SOFT_SIGMA = float(getattr(args, 'conc_bucket_soft_sigma', getattr(config, 'CONC_BUCKET_SOFT_SIGMA', 1.0)) or 1.0)
    reg_config.CONC_BUCKET_DETACH_FEAT = bool(getattr(args, 'conc_bucket_detach_feat', getattr(config, 'CONC_BUCKET_DETACH_FEAT', False)))
    reg_config.PERSONALIZED_REG = False
    reg_config.SHARE_REG_HEAD = True
    reg_config.SEPARATE_REG_SOURCE_AGG_SCOPE = getattr(args, 'separate_reg_source_agg_scope', 'all')
    reg_config.SEPARATE_REG_CLASS_WEIGHTS = getattr(args, 'separate_reg_class_weights', '')
    reg_config.SEPARATE_REG_HUBER_DELTAS = getattr(args, 'separate_reg_huber_deltas', '')
    reg_config.SEPARATE_REG_ALLOW_ENCODER_BACKPROP = bool(getattr(args, 'separate_reg_allow_encoder_backprop', True))
    reg_config.SEPARATE_REG_RANK_WEIGHT = getattr(args, 'separate_reg_rank_weight', 0.0)
    reg_config.SEPARATE_REG_RANK_MARGIN = getattr(args, 'separate_reg_rank_margin', 0.02)
    rank_classes_arg = getattr(args, 'separate_reg_rank_classes', '')
    reg_config.SEPARATE_REG_RANK_CLASSES = [
        int(x.strip()) for x in str(rank_classes_arg).split(',') if x.strip()
    ]

    if reg_config.NUM_CONC_BUCKETS > 0 and getattr(reg_config, 'CONC_BUCKET_BOUNDARIES', None) is None:
        from utils import compute_conc_bucket_boundaries
        conc_data = []
        cls_data = []
        for cid in train_client_ids:
            client_dir = Path(temp_federated_dir) / f'client_{cid}'
            y_cls_path = client_dir / 'train_classification_labels.npy'
            y_reg_path = client_dir / 'train_regression_labels.npy'
            if y_cls_path.exists() and y_reg_path.exists():
                cls_arr = np.load(y_cls_path)
                reg_arr = np.load(y_reg_path)
                if reg_arr.ndim > 1:
                    reg_arr = reg_arr[np.arange(len(cls_arr)), cls_arr.astype(int)]
                conc_data.append(reg_arr.reshape(-1))
                cls_data.append(cls_arr.reshape(-1))
        if conc_data:
            reg_config.CONC_BUCKET_BOUNDARIES = compute_conc_bucket_boundaries(
                np.concatenate(conc_data), np.concatenate(cls_data),
                reg_config.NUM_CLASSES, reg_config.NUM_CONC_BUCKETS
            )
            logger.info(f"Separate regression: concentration bucket boundaries={reg_config.CONC_BUCKET_BOUNDARIES}")
        else:
            logger.warning("Separate regression: concentration buckets requested but no source train labels were found")

    classifier_model = create_model_by_config(config, with_reg_head=False).to(device)
    load_shared_weights(classifier_model, classifier_state, strict=False)
    classifier_model.eval()

    reg_model = create_model_by_config(reg_config, with_reg_head=True).to(device)
    load_shared_weights(reg_model, classifier_state, strict=False)
    _init_regression_branch_from_classifier(reg_model, logger)

    save_dir = Path(config.MODEL_SAVE_DIR) / "separate_regression"
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config.MODEL_SAVE_DIR).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint_arg = getattr(args, 'separate_reg_source_checkpoint', '')
    source_local_states = {}
    source_local_ckpts = {}
    if source_checkpoint_arg:
        source_ckpt = Path(source_checkpoint_arg)
        if not source_ckpt.exists():
            raise FileNotFoundError(f"Separate regression source checkpoint not found: {source_ckpt}")
        ckpt_obj = torch.load(source_ckpt, map_location=device)
        source_state = _extract_model_state_from_checkpoint(ckpt_obj)
        reg_model.load_state_dict(source_state, strict=True)
        logger.info(f"Separate regression: loaded source checkpoint from {source_ckpt}")
        source_local_states, source_local_ckpts = _load_source_local_states_from_checkpoints(
            source_ckpt.parent, train_client_ids, logger=logger
        )
    else:
        source_loaders, source_sample_counts = _build_source_regression_loaders(
            train_client_ids, temp_federated_dir, config.BATCH_SIZE, logger
        )
        source_local_states, source_local_ckpts = _train_federated_source_regression(
            reg_model, source_loaders, source_sample_counts, device, reg_config,
            total_steps_per_client=args.separate_reg_source_steps,
            source_rounds=args.separate_reg_source_rounds,
            lr=args.separate_reg_lr,
            logger=logger,
            save_dir=save_dir
        )
        source_ckpt = save_dir / "separate_regression_source.pth"
        torch.save({'model_state': reg_model.state_dict(), 'config': reg_config}, source_ckpt)
        logger.info(f"Separate regression: source checkpoint saved to {source_ckpt}")

    # ===== Phase 0: ?????? =====
    # ?? B_source(FedAvg)??? B4/B5???? B4->C5/B5->C4?
    if getattr(args, 'separate_reg_skip_source_diagnostics', False):
        source_diagnostics = {'skipped': True}
        logger.info("Separate regression: skipped source diagnostics")
    else:
        source_test_loaders, _ = _build_source_regression_test_loaders(
            train_client_ids, temp_federated_dir, config.BATCH_SIZE, logger
        )
        source_diagnostics = _evaluate_source_regression_matrix(
            reg_model, source_local_states, classifier_model, source_test_loaders,
            device, reg_config, semantic_protos, save_dir, logger
        )
    if source_local_ckpts:
        source_diagnostics['local_checkpoints'] = source_local_ckpts

    calib_mode = getattr(args, 'separate_reg_calib_mode', 'full')
    logger.info(f"Separate regression: calib_mode={calib_mode}")

    metrics = {
        'mode': 'separate_federated_source_supervised_calibration',
        'source_checkpoint': str(source_ckpt),
        'settings': {
            'source_pretrain': 'federated_fedavg',
            'source_checkpoint_loaded': str(source_checkpoint_arg) if source_checkpoint_arg else '',
            'source_agg_scope': reg_config.SEPARATE_REG_SOURCE_AGG_SCOPE,
            'skip_source_diagnostics': bool(getattr(args, 'separate_reg_skip_source_diagnostics', False)),
            'source_steps_per_client': args.separate_reg_source_steps,
            'source_rounds': args.separate_reg_source_rounds,
            'target_steps': args.separate_reg_target_steps,
            'target_clients': args.separate_reg_target_clients if getattr(args, 'separate_reg_target_clients', '') else 'all',
            'calib_mode': calib_mode,
            'reg_lr': args.separate_reg_lr,
            'feat_lr': args.separate_reg_feat_lr,
            'target_unfreeze': args.separate_reg_unfreeze,
            'reg_window_stats': bool(getattr(config, 'REG_WINDOW_STATS', False)),
            'reg_window_stats_mode': getattr(config, 'REG_WINDOW_STATS_MODE', 'global'),
            'reg_window_stats_dim': getattr(config, 'REG_WINDOW_STATS_DIM', 8),
            'eval_test_only': bool(args.eval_test_only),
            'calib_use_labels': bool(getattr(config, 'CALIB_USE_LABELS', True)),
            'gate_metric': getattr(args, 'separate_reg_gate_metric', 'r2'),
            'gate_scope': getattr(args, 'separate_reg_gate_scope', 'overall'),
            'gate_min_delta': getattr(args, 'separate_reg_gate_min_delta', 0.0),
            'gate_fallback': getattr(args, 'separate_reg_gate_fallback', 'affine_only'),
            'val_ratio': getattr(args, 'separate_reg_val_ratio', 0.3),
            'val_split': getattr(args, 'separate_reg_val_split', 'window'),
            'auto_scope': getattr(args, 'separate_reg_auto_scope', 'per_class'),
            'class_weights': reg_config.SEPARATE_REG_CLASS_WEIGHTS,
            'huber_delta': float(getattr(reg_config, 'HUBER_DELTA', 0.2)),
            'class_huber_deltas': getattr(reg_config, 'SEPARATE_REG_HUBER_DELTAS', ''),
            'allow_encoder_backprop': bool(getattr(reg_config, 'SEPARATE_REG_ALLOW_ENCODER_BACKPROP', True)),
            'rank_classes': reg_config.SEPARATE_REG_RANK_CLASSES,
            'rank_weight': reg_config.SEPARATE_REG_RANK_WEIGHT,
            'rank_margin': reg_config.SEPARATE_REG_RANK_MARGIN,
        },
        'source_diagnostics': source_diagnostics,
        'clients': {}
    }

    soft_client_arg = str(getattr(args, 'separate_reg_pipeline_soft_clients', '') or '').strip()
    pipeline_soft_clients = None
    if soft_client_arg:
        pipeline_soft_clients = {int(x.strip()) for x in soft_client_arg.split(',') if x.strip()}
        logger.info(f"Separate regression: soft pipeline routing enabled only for clients {sorted(pipeline_soft_clients)}")
    metrics['settings']['pipeline_regression_route'] = getattr(args, 'separate_reg_pipeline_route', 'hard')
    metrics['settings']['pipeline_soft_topk'] = int(getattr(args, 'separate_reg_pipeline_soft_topk', 2))
    metrics['settings']['pipeline_soft_temperature'] = float(getattr(args, 'separate_reg_pipeline_soft_temperature', 1.0))
    metrics['settings']['pipeline_soft_min_confidence'] = float(getattr(args, 'separate_reg_pipeline_soft_min_confidence', 1.01))
    metrics['settings']['pipeline_soft_max_margin'] = float(getattr(args, 'separate_reg_pipeline_soft_max_margin', -1.0))
    metrics['settings']['pipeline_soft_clients'] = sorted(pipeline_soft_clients) if pipeline_soft_clients is not None else 'all'

    route_client_arg = str(getattr(args, 'separate_reg_route_clients', '') or '').strip()
    class_route_clients = None
    if route_client_arg:
        class_route_clients = {int(x.strip()) for x in route_client_arg.split(',') if x.strip()}
        logger.info(f"Separate regression: target class route enabled only for clients {sorted(class_route_clients)}")
    metrics['settings']['class_route'] = getattr(args, 'separate_reg_class_route', 'logits')
    metrics['settings']['route_temperature'] = float(getattr(args, 'separate_reg_route_temperature', 0.1))
    metrics['settings']['route_knn_k'] = int(getattr(args, 'separate_reg_route_knn_k', 7))
    metrics['settings']['route_mix_alpha'] = float(getattr(args, 'separate_reg_route_mix_alpha', 0.5))
    metrics['settings']['route_clients'] = sorted(class_route_clients) if class_route_clients is not None else 'all'
    metrics['settings']['route_router_feature_set'] = getattr(args, 'separate_reg_route_router_feature_set', 'all')
    metrics['settings']['route_router_epochs'] = int(getattr(args, 'separate_reg_route_router_epochs', 80))
    metrics['settings']['route_router_hidden_dim'] = int(getattr(args, 'separate_reg_route_router_hidden_dim', 32))
    metrics['settings']['source_init'] = getattr(args, 'separate_reg_source_init', 'fedavg')
    metrics['settings']['source_init_metric'] = getattr(args, 'separate_reg_source_init_metric', 'r2')
    metrics['settings']['source_init_scope'] = getattr(args, 'separate_reg_source_init_scope', 'overall')
    metrics['settings']['source_init_min_score_gain'] = float(getattr(args, 'separate_reg_source_init_min_score_gain', 0.0))
    metrics['settings']['source_route'] = getattr(args, 'separate_reg_source_route', 'none')
    metrics['settings']['source_route_min_mae_gain'] = float(getattr(args, 'separate_reg_source_route_min_mae_gain', 5.0))
    metrics['settings']['source_init_available_clients'] = sorted(int(cid) for cid in source_local_states.keys())
    if target_classifier_states:
        metrics['settings']['target_classifier_clients'] = sorted(int(cid) for cid in target_classifier_states.keys())

    source_init_mode = getattr(args, 'separate_reg_source_init', 'fedavg')
    source_init_metric = getattr(args, 'separate_reg_source_init_metric', 'r2')
    source_init_scope = getattr(args, 'separate_reg_source_init_scope', 'overall')
    source_init_min_score_gain = float(getattr(args, 'separate_reg_source_init_min_score_gain', 0.0))
    source_route_mode = getattr(args, 'separate_reg_source_route', 'none')
    source_route_min_mae_gain = float(getattr(args, 'separate_reg_source_route_min_mae_gain', 5.0))

    for cid, test_loader in test_client_loaders.items():
        client_model = copy.deepcopy(reg_model).to(device)
        calib_loader = calibration_loaders.get(cid) if calibration_loaders else None
        calib_samples = len(calib_loader.dataset) if calib_loader is not None and hasattr(calib_loader, 'dataset') else 0
        test_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0
        logger.info(f"Separate regression Client {cid}: calibration samples={calib_samples}, final test samples={test_samples}")

        source_init_diagnostics = {'mode': source_init_mode, 'selected': 'fedavg'}
        if source_init_mode == 'calib_select':
            client_model, source_init_diagnostics = _select_source_initialization_for_target(
                client_model, source_local_states, classifier_model, calib_loader,
                device, reg_config, metric=source_init_metric, scope=source_init_scope,
                logger=logger, client_id=cid, min_score_gain=source_init_min_score_gain
            )

        # 解析需要校准的目标客户端列表
        reg_target_clients = None
        if getattr(args, 'separate_reg_target_clients', ''):
            reg_target_clients = set(int(x.strip()) for x in args.separate_reg_target_clients.split(',') if x.strip())
            logger.info(f"Selective target calibration: only clients {reg_target_clients} will be calibrated")

        # 判断是否应该跳过校准（选择性和校准模式逻辑）
        should_skip_calib = False
        if calib_mode == 'none':
            should_skip_calib = True
            logger.info(f"Separate regression Client {cid}: calib_mode=none, no calibration")
        elif calib_loader is None:
            should_skip_calib = True
        elif calib_mode == 'full' and args.separate_reg_target_steps == 0:
            should_skip_calib = True
        elif reg_target_clients is not None and cid not in reg_target_clients:
            should_skip_calib = True
            logger.info(f"Separate regression Client {cid}: skipped calibration (not in target_clients={reg_target_clients})")

        # 不同类型校准的参数
        affine_params = None
        phase_affine_params = None
        routing_config = None
        routed_full_model = None
        specialist_models = None
        calibration_diagnostics = None

        if should_skip_calib:
            logger.info(f"Separate regression Client {cid}: no calibration performed")
            if source_route_mode == 'per_class_calib_select':
                source_route_classifier = classifier_model
                if target_classifier_states and cid in target_classifier_states:
                    source_route_classifier = create_model_by_config(config, with_reg_head=False).to(device)
                    load_shared_weights(source_route_classifier, target_classifier_states[cid], strict=False)
                    source_route_classifier.eval()
                routing_config, specialist_models, source_route_diagnostics = _select_source_models_per_class_for_target(
                    client_model, source_local_states, source_route_classifier, calib_loader,
                    device, reg_config, min_mae_gain=source_route_min_mae_gain,
                    logger=logger, client_id=cid
                )
                calibration_diagnostics = {'source_model_routing': source_route_diagnostics}
        elif calib_mode in ('affine_only', 'bias_only'):
            # 在 ppm 空间拟合 per-class affine/bias 参数，不更新神经网络
            logger.info(f"Separate regression Client {cid}: fitting {calib_mode} params on calibration set")
            affine_params = _fit_per_class_affine_params(
                calib_loader, client_model, classifier_model, device,
                num_classes=config.NUM_CLASSES, mode=calib_mode,
                semantic_protos=None, soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                prior_weight=config.SOFT_AGG_PRIOR_WEIGHT
            )
            for cls_id, p in affine_params.items():
                logger.info(f"  Class {cls_id}: a={p['a']:.4f}, b={p['b']:.2f}, "
                           f"n={p['n_samples']}, calib_R2={p['calib_r2']:.4f}, calib_MAE={p['calib_mae']:.2f}")
        elif calib_mode == 'phase_affine_only':
            logger.info(f"Separate regression Client {cid}: fitting class+phase affine params on calibration set")
            phase_affine_params = _fit_per_class_phase_affine_params(
                calib_loader, client_model, classifier_model, device,
                num_classes=config.NUM_CLASSES, num_phases=getattr(config, 'NUM_PHASES', 3)
            )
            for cls_id, class_params in phase_affine_params.items():
                fallback = class_params.get('class_fallback', {})
                logger.info(
                    f"  Class {cls_id}: fallback a={fallback.get('a', 1.0):.4f}, "
                    f"b={fallback.get('b', 0.0):.2f}, n={fallback.get('n_samples', 0)}"
                )
                for phase_id, p in class_params.get('phases', {}).items():
                    logger.info(
                        f"    Phase {phase_id}: a={p.get('a', 1.0):.4f}, b={p.get('b', 0.0):.2f}, "
                        f"n={p.get('n_samples', 0)}, calib_R2={p.get('calib_r2', 0.0):.4f}"
                    )
        elif calib_mode == 'full':
            # 原有的神经网络微调校准
            train_calib_loader = _reshuffle_loader(calib_loader, config.BATCH_SIZE)
            client_seed = reg_seed + 10000 + cid
            set_random_seed(client_seed)
            logger.info(f"Separate regression Client {cid}: reset random seed to {client_seed}")
            _train_separate_regression(
                client_model, train_calib_loader, device, reg_config,
                steps=args.separate_reg_target_steps, lr=args.separate_reg_lr,
                feat_lr=args.separate_reg_feat_lr,
                unfreeze_policy=args.separate_reg_unfreeze,
                logger=logger, stage_name=f'target_client{cid}'
            )
        elif calib_mode == 'gated':
            client_model, affine_params, calibration_diagnostics = _run_gated_target_calibration(
                client_model, classifier_model, calib_loader, device, config, args,
                reg_config, cid, reg_seed, logger
            )
        elif calib_mode == 'auto':
            client_model, affine_params, calibration_diagnostics = _run_auto_target_calibration(
                client_model, classifier_model, calib_loader, device, config, args,
                reg_config, cid, reg_seed, logger
            )
        elif calib_mode == 'auto_v2':
            client_model, affine_params, phase_affine_params, routing_config, routed_full_model, calibration_diagnostics = _run_auto_v2_target_calibration(
                client_model, classifier_model, calib_loader, device, config, args,
                reg_config, cid, reg_seed, logger
            )
        elif calib_mode == 'auto_v2_specialist':
            client_model, affine_params, phase_affine_params, routing_config, routed_full_model, specialist_models, calibration_diagnostics = _run_auto_v2_specialist_target_calibration(
                client_model, classifier_model, calib_loader, device, config, args,
                reg_config, cid, reg_seed, logger
            )

        pipeline_classifier = classifier_model
        if target_classifier_states and cid in target_classifier_states:
            pipeline_classifier = create_model_by_config(config, with_reg_head=False).to(device)
            load_shared_weights(pipeline_classifier, target_classifier_states[cid], strict=False)
            pipeline_classifier.eval()
            logger.info(f"Separate regression Client {cid}: pipeline uses calibrated target classifier")
        eval_semantic_protos = None if target_classifier_states and cid in target_classifier_states else semantic_protos
        client_pipeline_route = args.separate_reg_pipeline_route
        if client_pipeline_route == 'soft_topk' and pipeline_soft_clients is not None and cid not in pipeline_soft_clients:
            client_pipeline_route = 'hard'
        if client_pipeline_route != args.separate_reg_pipeline_route:
            logger.info(f"Separate regression Client {cid}: pipeline route hard (soft_topk disabled by client filter)")
        else:
            logger.info(f"Separate regression Client {cid}: pipeline route {client_pipeline_route}")
        client_class_route = getattr(args, 'separate_reg_class_route', 'logits')
        if class_route_clients is not None and cid not in class_route_clients:
            client_class_route = 'logits'
        target_route_bank = None
        if client_class_route != 'logits':
            if client_class_route == 'target_reg_router':
                class_weight_vector, active_class_weights = _build_class_weight_vector(
                    getattr(args, 'target_cls_class_weights', ''),
                    getattr(args, 'target_cls_client_class_weights', ''),
                    cid, config.NUM_CLASSES
                )
                target_route_bank, router_metrics = _train_target_reg_router(
                    pipeline_classifier, client_model, routed_full_model, calib_loader, device,
                    num_classes=config.NUM_CLASSES, num_phases=getattr(config, 'NUM_PHASES', 3),
                    routing_config=routing_config, affine_params=affine_params,
                    phase_affine_params=phase_affine_params, specialist_models=specialist_models,
                    epochs=getattr(args, 'separate_reg_route_router_epochs', 80),
                    lr=getattr(args, 'separate_reg_route_router_lr', 1e-3),
                    weight_decay=getattr(args, 'separate_reg_route_router_weight_decay', 1e-3),
                    hidden_dim=getattr(args, 'separate_reg_route_router_hidden_dim', 32),
                    class_weights=class_weight_vector,
                    focal_gamma=getattr(args, 'separate_reg_route_router_focal_gamma', 0.0),
                    logger=logger, client_id=cid
                )
                if calibration_diagnostics is None:
                    calibration_diagnostics = {}
                calibration_diagnostics['target_reg_router'] = router_metrics
                if active_class_weights:
                    calibration_diagnostics['target_reg_router']['class_weights'] = active_class_weights
            elif 'target_router' in client_class_route:
                class_weight_vector, active_class_weights = _build_class_weight_vector(
                    getattr(args, 'target_cls_class_weights', ''),
                    getattr(args, 'target_cls_client_class_weights', ''),
                    cid, config.NUM_CLASSES
                )
                target_route_bank, router_metrics = _train_target_router(
                    pipeline_classifier, calib_loader, device, num_classes=config.NUM_CLASSES,
                    num_phases=getattr(config, 'NUM_PHASES', 3),
                    feature_set=getattr(args, 'separate_reg_route_router_feature_set', 'all'),
                    epochs=getattr(args, 'separate_reg_route_router_epochs', 80),
                    lr=getattr(args, 'separate_reg_route_router_lr', 1e-3),
                    weight_decay=getattr(args, 'separate_reg_route_router_weight_decay', 1e-3),
                    hidden_dim=getattr(args, 'separate_reg_route_router_hidden_dim', 32),
                    class_weights=class_weight_vector,
                    focal_gamma=getattr(args, 'separate_reg_route_router_focal_gamma', 0.0),
                    logger=logger, client_id=cid
                )
                if calibration_diagnostics is None:
                    calibration_diagnostics = {}
                calibration_diagnostics['target_router'] = router_metrics
                if active_class_weights:
                    calibration_diagnostics['target_router']['class_weights'] = active_class_weights
            else:
                target_route_bank = _build_target_route_bank(
                    pipeline_classifier, calib_loader, device, num_classes=config.NUM_CLASSES,
                    logger=logger, client_id=cid
                )
            if target_route_bank is None:
                logger.warning(f"Separate regression Client {cid}: target class route fallback to logits (no route bank)")
                client_class_route = 'logits'
        logger.info(f"Separate regression Client {cid}: class route {client_class_route}")
        client_metrics = evaluate_separate_regression(
            client_model, pipeline_classifier, test_loader, device, config.NUM_CLASSES,
            semantic_protos=eval_semantic_protos,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
            affine_params=affine_params,
            phase_affine_params=phase_affine_params,
            routing_config=routing_config,
            full_model=routed_full_model,
            specialist_models=specialist_models,
            pipeline_regression_route=client_pipeline_route,
            pipeline_soft_topk=args.separate_reg_pipeline_soft_topk,
            pipeline_soft_temperature=args.separate_reg_pipeline_soft_temperature,
            pipeline_soft_min_confidence=args.separate_reg_pipeline_soft_min_confidence,
            pipeline_soft_max_margin=args.separate_reg_pipeline_soft_max_margin,
            target_route_bank=target_route_bank,
            class_route=client_class_route,
            route_temperature=args.separate_reg_route_temperature,
            route_knn_k=args.separate_reg_route_knn_k,
            route_mix_alpha=args.separate_reg_route_mix_alpha
        )
        client_metrics['source_initialization'] = source_init_diagnostics
        client_metrics['pipeline_regression_route'] = {
            'requested_mode': args.separate_reg_pipeline_route,
            'mode': client_pipeline_route,
            'soft_topk': int(args.separate_reg_pipeline_soft_topk),
            'soft_temperature': float(args.separate_reg_pipeline_soft_temperature),
            'soft_min_confidence': float(args.separate_reg_pipeline_soft_min_confidence),
            'soft_max_margin': float(args.separate_reg_pipeline_soft_max_margin),
            'soft_clients': sorted(pipeline_soft_clients) if pipeline_soft_clients is not None else 'all',
            'class_route': client_class_route,
            'route_temperature': float(args.separate_reg_route_temperature),
            'route_knn_k': int(args.separate_reg_route_knn_k),
            'route_mix_alpha': float(args.separate_reg_route_mix_alpha),
            'route_clients': sorted(class_route_clients) if class_route_clients is not None else 'all',
            'route_bank_samples': int(target_route_bank.get('n_samples', 0)) if target_route_bank is not None else 0,
        }
        plot_dir = Path(save_dir) / 'regression_plots'
        client_metrics['oracle_scatter'] = plot_separate_regression_scatter(
            client_model, pipeline_classifier, test_loader, device, plot_dir,
            filename=f"target_client{cid}_oracle.png",
            mode='oracle', title_prefix=f"Target Client {cid} (oracle)",
            semantic_protos=eval_semantic_protos,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
            affine_params=affine_params,
            phase_affine_params=phase_affine_params,
            routing_config=routing_config,
            full_model=routed_full_model,
            specialist_models=specialist_models
        )
        client_metrics['pipeline_scatter'] = plot_separate_regression_scatter(
            client_model, pipeline_classifier, test_loader, device, plot_dir,
            filename=f"target_client{cid}_pipeline.png",
            mode='pipeline', title_prefix=f"Target Client {cid} (pipeline)",
            semantic_protos=eval_semantic_protos,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
            affine_params=affine_params,
            phase_affine_params=phase_affine_params,
            routing_config=routing_config,
            full_model=routed_full_model,
            specialist_models=specialist_models,
            pipeline_regression_route=client_pipeline_route,
            pipeline_soft_topk=args.separate_reg_pipeline_soft_topk,
            pipeline_soft_temperature=args.separate_reg_pipeline_soft_temperature,
            pipeline_soft_min_confidence=args.separate_reg_pipeline_soft_min_confidence,
            pipeline_soft_max_margin=args.separate_reg_pipeline_soft_max_margin,
            target_route_bank=target_route_bank,
            class_route=client_class_route,
            route_temperature=args.separate_reg_route_temperature,
            route_knn_k=args.separate_reg_route_knn_k,
            route_mix_alpha=args.separate_reg_route_mix_alpha
        )
        client_metrics['conditional_pipeline_scatter'] = plot_separate_regression_scatter(
            client_model, pipeline_classifier, test_loader, device, plot_dir,
            filename=f"target_client{cid}_conditional_pipeline.png",
            mode='conditional_pipeline', title_prefix=f"Target Client {cid} (pipeline, correct class only)",
            semantic_protos=eval_semantic_protos,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
            affine_params=affine_params,
            phase_affine_params=phase_affine_params,
            routing_config=routing_config,
            full_model=routed_full_model,
            specialist_models=specialist_models,
            pipeline_regression_route=client_pipeline_route,
            pipeline_soft_topk=args.separate_reg_pipeline_soft_topk,
            pipeline_soft_temperature=args.separate_reg_pipeline_soft_temperature,
            pipeline_soft_min_confidence=args.separate_reg_pipeline_soft_min_confidence,
            pipeline_soft_max_margin=args.separate_reg_pipeline_soft_max_margin,
            target_route_bank=target_route_bank,
            class_route=client_class_route,
            route_temperature=args.separate_reg_route_temperature,
            route_knn_k=args.separate_reg_route_knn_k,
            route_mix_alpha=args.separate_reg_route_mix_alpha
        )
        client_metrics['wrong_pipeline_scatter'] = plot_separate_regression_scatter(
            client_model, pipeline_classifier, test_loader, device, plot_dir,
            filename=f"target_client{cid}_wrong_pipeline.png",
            mode='wrong_pipeline', title_prefix=f"Target Client {cid} (pipeline, wrong class only)",
            semantic_protos=eval_semantic_protos,
            soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
            prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
            affine_params=affine_params,
            phase_affine_params=phase_affine_params,
            routing_config=routing_config,
            full_model=routed_full_model,
            specialist_models=specialist_models,
            pipeline_regression_route=client_pipeline_route,
            pipeline_soft_topk=args.separate_reg_pipeline_soft_topk,
            pipeline_soft_temperature=args.separate_reg_pipeline_soft_temperature,
            pipeline_soft_min_confidence=args.separate_reg_pipeline_soft_min_confidence,
            pipeline_soft_max_margin=args.separate_reg_pipeline_soft_max_margin,
            target_route_bank=target_route_bank,
            class_route=client_class_route,
            route_temperature=args.separate_reg_route_temperature,
            route_knn_k=args.separate_reg_route_knn_k,
            route_mix_alpha=args.separate_reg_route_mix_alpha
        )
        if getattr(args, 'separate_reg_export_predictions', False):
            pred_dir = Path(save_dir) / 'prediction_exports'
            client_metrics['prediction_export_csv'] = export_separate_regression_predictions(
                client_model, pipeline_classifier, test_loader, device,
                pred_dir / f'target_client{cid}_predictions.csv',
                client_id=cid, num_classes=config.NUM_CLASSES,
                semantic_protos=eval_semantic_protos,
                soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
                affine_params=affine_params,
                phase_affine_params=phase_affine_params,
                routing_config=routing_config,
                full_model=routed_full_model,
                specialist_models=specialist_models,
                pipeline_regression_route=client_pipeline_route,
                pipeline_soft_topk=args.separate_reg_pipeline_soft_topk,
                pipeline_soft_temperature=args.separate_reg_pipeline_soft_temperature,
                pipeline_soft_min_confidence=args.separate_reg_pipeline_soft_min_confidence,
                pipeline_soft_max_margin=args.separate_reg_pipeline_soft_max_margin,
                target_route_bank=target_route_bank,
                class_route=client_class_route,
                route_temperature=args.separate_reg_route_temperature,
                route_knn_k=args.separate_reg_route_knn_k,
                route_mix_alpha=args.separate_reg_route_mix_alpha
            )
            if getattr(args, 'separate_reg_export_calibration_predictions', False):
                client_metrics['calibration_prediction_export_csv'] = export_separate_regression_predictions(
                    client_model, pipeline_classifier, calib_loader, device,
                    pred_dir / f'target_client{cid}_calibration_predictions.csv',
                    client_id=cid, num_classes=config.NUM_CLASSES,
                    semantic_protos=eval_semantic_protos,
                    soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                    prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
                    affine_params=affine_params,
                    phase_affine_params=phase_affine_params,
                    routing_config=routing_config,
                    full_model=routed_full_model,
                    specialist_models=specialist_models,
                    pipeline_regression_route=client_pipeline_route,
                    pipeline_soft_topk=args.separate_reg_pipeline_soft_topk,
                    pipeline_soft_temperature=args.separate_reg_pipeline_soft_temperature,
                    pipeline_soft_min_confidence=args.separate_reg_pipeline_soft_min_confidence,
                    pipeline_soft_max_margin=args.separate_reg_pipeline_soft_max_margin,
                    target_route_bank=target_route_bank,
                    class_route=client_class_route,
                    route_temperature=args.separate_reg_route_temperature,
                    route_knn_k=args.separate_reg_route_knn_k,
                    route_mix_alpha=args.separate_reg_route_mix_alpha
                )
        client_metrics['calibration_samples'] = int(calib_samples)
        client_metrics['test_samples'] = int(test_samples)
        if affine_params is not None:
            client_metrics['calib_affine_params'] = affine_params
        if phase_affine_params is not None:
            client_metrics['calib_phase_affine_params'] = phase_affine_params
        if routing_config is not None:
            client_metrics['routing_config'] = routing_config
        if calibration_diagnostics is not None:
            client_metrics['calibration_diagnostics'] = calibration_diagnostics
        metrics['clients'][cid] = client_metrics
        ckpt_path = save_dir / f"separate_regression_client{cid}.pth"
        torch.save({'model_state': client_model.state_dict(), 'classifier_state': classifier_state, 'metrics': make_json_serializable(client_metrics)}, ckpt_path)
        logger.info(
            f"Separate regression Client {cid}: "
            f"oracle R2={client_metrics['oracle']['overall']['R2']:.4f}, "
            f"pipeline R2={client_metrics['pipeline']['overall']['R2']:.4f}, "
            f"pipeline cls acc={client_metrics['pipeline']['classification_accuracy']:.4f}"
        )

    metrics_json = make_json_serializable(metrics)
    for out_path in [save_dir / "separate_regression_metrics.json", results_dir / "separate_regression_metrics.json"]:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)
        logger.info(f"Separate regression metrics saved to {out_path}")
    return metrics

def save_and_visualize(args, config, exp_dir, history, weights_history, mmd_history, final_accs, 
                       peak_acc, peak_round, training_time, test_client_ids, temp_federated_dir, 
                       logger, best_test_accs, best_rounds, 
                       server=None, test_client_loaders=None, global_test_loader=None, 
                       regression_metrics_all=None, tta_client_ids=None, source_cls_accs=None,
                       classification_metrics_all=None):  # 新增参数
    """保存结果与可视化（增强版）"""
    # 1. 整理history为轮次聚合格式
    rounds = sorted(set(h['round'] for h in history))
    aggregated_history = []
    for r in rounds:
        round_entries = [h for h in history if h['round'] == r]
        entry = round_entries[0].copy()
        client_acc_dict = {}
        for e in round_entries:
            cid = e.get('test_client_id')
            if cid is not None:
                client_acc_dict[f"client_{cid}"] = e.get('test_client_acc', 0.0)
        entry['client_accuracies'] = client_acc_dict
        entry.pop('test_client_id', None)
        entry.pop('test_client_acc', None)
        aggregated_history.append(entry)

    # 2. 计算各阶段最终准确率和遗忘率
    if aggregated_history:
        final_early = aggregated_history[-1]['acc_early']
        final_middle = aggregated_history[-1]['acc_middle']
        final_late = aggregated_history[-1]['acc_late']
        final_overall = aggregated_history[-1]['test_acc']
    else:
        final_early = final_middle = final_late = final_overall = 0.0

    forgetting_early = peak_acc['early'] - final_early
    forgetting_middle = peak_acc['middle'] - final_middle
    # Late 峰值：从所有轮次中取最大值
    late_accs = [h['acc_late'] for h in aggregated_history]
    peak_late = max(late_accs) if late_accs else final_late
    peak_round_late = late_accs.index(peak_late) + 1 if late_accs else 0
    forgetting_late = peak_late - final_late

    # 3. 构建结果字典
    results = {
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        "training_time_sec": training_time,
        "history": aggregated_history,
        "raw_history": history,   # 保留原始数据备用
        "final_metrics": {
            "global": {
                "early": final_early,
                "middle": final_middle,
                "late": final_late,
                "overall": final_overall
            },
            "per_client": {cid: {"accuracy": acc} for cid, acc in final_accs.items()}
        },
        "peak_metrics": {
            "early": {"accuracy": peak_acc['early'], "round": peak_round['early']},
            "middle": {"accuracy": peak_acc['middle'], "round": peak_round['middle']},
            "late": {"accuracy": peak_late, "round": peak_round_late}
        },
        "forgetting": {
            "early": forgetting_early,
            "middle": forgetting_middle,
            "late": forgetting_late
        },
        "best_per_client": {
            cid: {"accuracy": best_test_accs[cid], "round": best_rounds[cid]}
            for cid in test_client_ids
        },
        "weights_history": weights_history,
        "mmd_history": mmd_history,
        "test_clients": test_client_ids
    }
    
    # === 新增：添加源域分类准确率 ===
    if source_cls_accs is not None:
        results['source_classification_acc'] = {str(cid): acc for cid, acc in source_cls_accs.items()}

    classification_metrics = {}
    if classification_metrics_all is not None:
        classification_metrics.update(make_json_serializable(classification_metrics_all))
    if source_cls_accs is not None:
        classification_metrics['source_accuracy'] = {str(cid): acc for cid, acc in source_cls_accs.items()}
    if classification_metrics:
        results['classification_metrics'] = make_json_serializable(classification_metrics)
    
    # === Regression metrics ===
    if regression_metrics_all is not None:
        results['regression_metrics'] = make_json_serializable(regression_metrics_all)
    
    save_results(results, str(exp_dir / "results"), "training_results.json")

    # 4. 可视化
    plot_training_curves(history, config.PLOT_SAVE_DIR)
    plot_aggregation_weights(weights_history, config.PLOT_SAVE_DIR)
    forgetting_dict = {
        'early': forgetting_early, 'middle': forgetting_middle, 'late': forgetting_late,
        'peak_early': peak_acc['early'], 'peak_middle': peak_acc['middle'], 'peak_late': peak_late
    }
    plot_phase_curves(history, forgetting_dict, config.PLOT_SAVE_DIR)

    if args.compute_mmd and mmd_history:
        plt.figure(figsize=(10, 6))
        for cid in test_client_ids:
            cid_mmd = [m for m in mmd_history if m.get('test_client_id') == cid]
            if cid_mmd:
                rounds = [m['round'] for m in cid_mmd]
                mmds = [m['mmd'] for m in cid_mmd]
                plt.plot(rounds, mmds, marker='o', label=f'Client {cid}')
        plt.xlabel('Round'); plt.ylabel('MMD'); plt.title('Distribution Shift (MMD)')
        plt.legend(); plt.grid(True)
        plt.savefig(Path(config.PLOT_SAVE_DIR) / "mmd_curve.png", dpi=150)
        plt.close()

    summary = {
        "Total rounds": len(aggregated_history),
        "Best global accuracy": max(h['test_acc'] for h in aggregated_history) if aggregated_history else 0.0,
        "Final global accuracy": final_overall,
        "Training time": format_time(training_time)
    }
    for cid, acc in final_accs.items():
        summary[f"Final Client{cid}"] = acc
    print_training_summary(summary)

    # === 新增：绘制源域分类准确率柱状图 ===
    if source_cls_accs is not None:
        from utils import plot_source_classification_acc
        plot_source_classification_acc(source_cls_accs, config.PLOT_SAVE_DIR)

    if args.plot_tsne and server and test_client_loaders and global_test_loader:
        try:
            from utils import plot_tsne_features
            plot_tsne_features(
                model=server.global_model,
                train_loader=global_test_loader,
                test_loaders=test_client_loaders,
                device=config.DEVICE,
                save_dir=config.PLOT_SAVE_DIR,
                filename="tsne_features.png",
                max_samples_per_set=1000
            )
            logger.info("t-SNE feature visualization saved successfully.")
        except Exception as e:
            logger.warning(f"t-SNE plotting failed: {e}")

    if args.plot_umap and server and test_client_loaders and global_test_loader:
        try:
            from utils import plot_umap_features
            plot_umap_features(
                model=server.global_model,
                train_loader=global_test_loader,
                test_loaders=test_client_loaders,
                device=config.DEVICE,
                save_dir=config.PLOT_SAVE_DIR,
                filename="umap_features.png",
                max_samples_per_set=1000
            )
            logger.info("UMAP feature visualization saved successfully.")
        except Exception as e:
            logger.warning(f"UMAP plotting failed: {e}")
    
    # ---- 通用可视化（调用 utils 中函数） ----
    hard_ids = [int(x.strip()) for x in (args.hard_clients.split(',') if args.hard_clients else [])]
    # CORAL 前后 t-SNE
    plot_coral_tsne_comparison(hard_ids, config.PLOT_SAVE_DIR, args.use_coral, logger)
    # 微调前后 t-SNE（仅当启用分类微调时）
    if args.few_shot_classification:
        plot_finetune_tsne_comparison(hard_ids, config.PLOT_SAVE_DIR, logger)
        plot_classifier_weight_analysis(hard_ids, config.PLOT_SAVE_DIR, logger)
    # 特征类别分离度
    plot_class_separability_analysis(hard_ids, config.PLOT_SAVE_DIR, logger)
    # 回归可视化
    if config.USE_REG_LOSS and server is not None and test_client_loaders is not None and not args.few_shot_regression:
        plot_regression_visualizations(server, test_client_loaders, test_client_ids,
                                       config, best_rounds, best_test_accs,
                                       args.model_selection, logger)
    # 保存准确率表格
    save_accuracy_table(history, str(exp_dir / "results"), logger)

    logger.info("Experiment finished.")


def deployment_phase(server, train_clients, test_client_ids, test_client_loaders,
                     config, args, logger, deploy_scheduler, phase_ratios,
                     active_query_freq=0, active_query_samples=2,
                     calibration_loaders=None):
    """
    部署在线适应阶段
    在联邦训练结束后，对测试客户端进行无标签在线自适应。
    使用 CumulativeDataScheduler 按"早期→中期→晚期"顺序逐步注入无标签数据流，
    每轮更新 online_loader，客户端仅使用 batch[0] 进行适应，标签被完全忽略。
    
    Args:
        calibration_loaders: dict[cid → DataLoader]，目标客户端校准集加载器。
            用于主动学习查询时的少样本标注数据抽取，避免使用全量数据。
    """
    import copy
    from utils import create_model_by_config, evaluate_model_with_phase_and_soft_agg

    deploy_history = []

    # 1. 为每个测试客户端创建部署客户端（online_loader 将在循环中动态更新）
    deploy_clients = []
    for cid in test_client_ids:
        deploy_model = create_model_by_config(config, with_reg_head=config.USE_REG_LOSS).to(config.DEVICE)
        deploy_model.load_state_dict(server.global_model.state_dict(), strict=False)
        client_obj = Client(client_id=cid, config=config)
        client_obj.set_model(deploy_model)
        # 使用纯语义原型，保持训练与评估时原型空间一致
        global_protos = {k: v.detach().cpu() for k, v in server.semantic_protos.items()}
        # 传递各原型的对角方差，用于马氏距离锚定 + 推理阶段分配
        # 条件：USE_MAHALANOBIS_ANCHOR（锚定损失）或 USE_MAHALANOBIS_INFERENCE（推理/阶段分配）
        need_proto_vars = (
            getattr(config, 'USE_MAHALANOBIS_ANCHOR', False)
            or getattr(config, 'USE_MAHALANOBIS_INFERENCE', False)
        )
        proto_vars_for_deploy = {
            k: v.detach().cpu() for k, v in server.semantic_proto_vars.items()
        } if (need_proto_vars and server.semantic_proto_vars) else None
        client_obj.init_online_adaptation(
            global_protos, unfreeze_level=args.adapt_unfreeze_level,
            proto_vars=proto_vars_for_deploy
        )
        client_obj.online_loader = None
        deploy_clients.append(client_obj)
        logger.info(f"Deploy client {cid} ready.")

    # 初始化 warmup：部署初期使用较低置信度阈值，避免适应停滞
    for client in deploy_clients:
        client.adaptation_engine.conf_thresh = 0.7

    # 3. 部署主循环
    for d_round in range(1, args.deploy_rounds + 1):
        logger.info(f"Deploy Round {d_round}/{args.deploy_rounds}")

        # 3.0 按漂移阶段获取当前轮次的无标签数据加载器
        loaders_dict = deploy_scheduler.get_client_loaders(d_round, total_rounds=args.deploy_rounds)
        for client in deploy_clients:
            cid = client.client_id
            loader = loaders_dict.get(cid) or loaders_dict.get(f'client_{cid}')
            if loader is not None:
                client.online_loader = loader
            else:
                logger.warning(f"Client {cid} has no data for deploy round {d_round}, skipping.")

        # 3.1 主动学习查询（每隔 active_query_freq 轮触发一次）
        if active_query_freq > 0 and d_round % active_query_freq == 0:
            for client in deploy_clients:
                cid = client.client_id
                logger.info(f"  Active Learning Query for Client {cid} at round {d_round}")
                # 从云端维护的校准集中抽取少量有标签样本（而非全量测试数据）
                calib_loader = calibration_loaders.get(cid) if calibration_loaders else None
                source_loader = calib_loader if calib_loader is not None else test_client_loaders.get(cid)
                if source_loader is None:
                    continue
                fs_loader, _ = build_few_shot_and_test_loaders(
                    source_loader, active_query_samples, config.BATCH_SIZE
                )
                # 保存当前模型中除分类头外的所有参数
                import copy
                saved_params = {}
                for name, param in client.model.named_parameters():
                    if 'classifier' not in name:
                        saved_params[name] = param.clone().detach()
                
                # 在原模型上进行微调（仅微调分类头）
                few_shot_finetune_classification(
                    client.model, fs_loader, config.DEVICE,
                    epochs=5, lr=1e-3, finetune_feat_lr=0.0
                )
                
                # 恢复非分类头的参数
                with torch.no_grad():
                    for name, param in client.model.named_parameters():
                        if 'classifier' not in name and name in saved_params:
                            param.copy_(saved_params[name])
                
                # 只更新分类头部分，保留特征提取器的适应成果
                with torch.no_grad():
                    for name, param in client.model.named_parameters():
                        if 'classifier' in name:
                            # 更新学生模型的分类头
                            student_param = client.adaptation_engine.student.state_dict()[name]
                            student_param.copy_(param)
                            # 更新教师模型的分类头
                            teacher_param = client.adaptation_engine.teacher.state_dict()[name]
                            teacher_param.copy_(param)
                logger.info(f"    Query: {len(fs_loader.dataset)} labeled samples used, "
                            f"client model classifier updated (only classifier head).")

        # 3.2 本地在线适应（仅使用 batch[0]，完全无标签）
        for client in deploy_clients:
            if client.online_loader is None or len(client.online_loader.dataset) == 0:
                continue
            loader_iter = iter(client.online_loader)
            total_loss = 0.0
            total_entropy = 0.0
            total_anchor = 0.0
            steps = 0
            for _ in range(args.adapt_steps_per_round):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(client.online_loader)
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        break
                loss, loss_ent, loss_anch = client.adaptation_engine.adapt_step(batch)
                total_loss += loss
                total_entropy += loss_ent
                total_anchor += loss_anch
                steps += 1
            if steps > 0:
                logger.info(f"  Client {client.client_id} adapt: total={total_loss/steps:.4f}, "
                            f"entropy={total_entropy/steps:.4f}, anchor={total_anchor/steps:.4f}")

        # 逐步提高置信度阈值（warmup策略：前5轮从0.7线性增加到目标值）
        target_thresh = config.MT_CONF_THRESH
        current_thresh = 0.7 + (target_thresh - 0.7) * min(1.0, d_round / 5)
        for client in deploy_clients:
            client.adaptation_engine.conf_thresh = current_thresh

        # 3.3 周期性上传与服务器聚合
        if d_round % args.upload_freq == 0:
            uploads = []
            for client in deploy_clients:
                if client.online_loader is None:
                    continue
                stat = client.upload_online_statistics()
                if stat is not None and stat['confidence'] > config.MIN_PROTO_UPDATE_CONF:
                    uploads.append(stat)
            if uploads:
                logger.info(f"  Upload & Aggregation from {len(uploads)} clients")
                server.deployment_aggregation_round(uploads)
                updated_protos = {k: v.detach().cpu() for k, v in server.semantic_protos.items()}
                for client in deploy_clients:
                    # 下发全局原型（原有逻辑）
                    client.adaptation_engine.protos = updated_protos
                    # 新增：下发个性化校正量
                    cid = client.client_id
                    if hasattr(server, '_latest_corrections') and cid in server._latest_corrections:
                        client.adaptation_engine.set_corrections(server._latest_corrections[cid])
                        logger.info(f"  Client {cid}: applied federated calibration corrections.")

        # 3.4 评估（使用 test_client_loaders 的有标签全量数据）
        if d_round % args.eval_freq == 0 or d_round == 1:
            acc_dict = {}
            early_sum, mid_sum, late_sum = 0.0, 0.0, 0.0
            sem_protos_eval = {k: v.detach().cpu() for k, v in server.semantic_protos.items()}
            # 马氏距离推理方差
            use_mahalanobis_eval_dep = getattr(config, 'USE_MAHALANOBIS_INFERENCE', False)
            proto_vars_eval_dep = {
                k: v.detach().cpu() for k, v in server.semantic_proto_vars.items()
            } if (use_mahalanobis_eval_dep and server.semantic_proto_vars) else None
            for cid in test_client_ids:
                loader = test_client_loaders[cid]
                deploy_client = next((c for c in deploy_clients if c.client_id == cid), None)
                eval_model = deploy_client.model if deploy_client else server.global_model

                eval_results = evaluate_model_with_phase_and_soft_agg(
                    eval_model, loader, config.DEVICE,
                    semantic_protos=sem_protos_eval,
                    soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                    prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
                    num_classes=config.NUM_CLASSES,
                    use_mahalanobis_inference=use_mahalanobis_eval_dep,
                    semantic_proto_vars=proto_vars_eval_dep
                )
                acc_dict[cid] = eval_results['global']
                early_sum += eval_results.get('early', 0)
                mid_sum += eval_results.get('middle', 0)
                late_sum += eval_results.get('late', 0)

            num_clients = len(test_client_ids)
            overall = sum(acc_dict.values()) / num_clients if num_clients > 0 else 0.0
            entry = {
                'round': config.GLOBAL_ROUNDS + d_round,
                'test_acc': overall,
                'acc_early': early_sum / num_clients,
                'acc_middle': mid_sum / num_clients,
                'acc_late': late_sum / num_clients,
                'test_client_id': None,
                'test_client_acc': None,
                'align_loss': 0,
                'weights': {},
                'type': 'deploy',
            }
            for cid, acc in acc_dict.items():
                e = entry.copy()
                e['test_client_id'] = cid
                e['test_client_acc'] = acc
                deploy_history.append(e)
            logger.info(f"Deploy Round {d_round}: Accuracies: {acc_dict}")

    return deploy_history


def main():
    # 导入配置类以使用其默认值
    from config import FLConfig
    default_config = FLConfig()
    
    # 固定全局随机种子，确保实验可重复性
    from utils import set_global_seed
    set_global_seed(default_config.SEED)
    
    parser = argparse.ArgumentParser(description='GAPS Federated Continual Learning')
    parser.add_argument('--combination', type=str, default='gaps_full',
                        choices=['gaps_full', 'gaps_no_server_opt', 'gaps_no_align', 'gaps_no_distill',
                                 'gaps_no_decoupling', 'gaps_no_soft_agg', 'gaps_fedavg', 'fedavg', 'fedavg_align'],
                        help='Ablation study combination')
    parser.add_argument('--rounds', type=int, default=default_config.GLOBAL_ROUNDS, help='Number of global rounds')
    parser.add_argument('--data_dir', type=str, default='./dataset/client_data_federated_window_fullgrid_src45_tgt123',
                        help='Federated dataset directory containing client_*/ split npy files')
    parser.add_argument('--processed_dir', type=str, default='./dataset/processed',
                        help='Processed dataset directory used only when --data_dir does not exist')
    parser.add_argument('--local_epochs', type=int, default=default_config.LOCAL_EPOCHS, help='Local epochs per round')
    parser.add_argument('--use_reg_loss', dest='use_reg_loss', action='store_true', default=None,
                        help='Enable joint regression loss during federated client training')
    parser.add_argument('--no_use_reg_loss', dest='use_reg_loss', action='store_false',
                        help='Disable joint regression loss during federated client training')
    parser.add_argument('--eval_test_only', action='store_true',
                        help='Use only test_*.npy for final target evaluation, excluding calibration/train split')
    parser.add_argument('--regression_mode', type=str, default='joint', choices=['joint', 'separate'],
                        help='joint: existing multitask route; separate: classification model + supervised calibrated regression model')
    parser.add_argument('--regression_only', action='store_true',
                        help='Skip federated classification training and run only separate regression B from a saved classification checkpoint')
    parser.add_argument('--classifier_checkpoint', type=str, default='',
                        help='Path to saved classification model checkpoint for --regression_only')
    parser.add_argument('--target_classifier_dir', type=str, default='',
                        help='Directory containing target_cls_client*.pth calibrated classifiers for pipeline evaluation')
    parser.add_argument('--separate_reg_source_steps', type=int, default=600,
                        help='Local source training steps per source client for federated separate regression pretraining')
    parser.add_argument('--separate_reg_source_rounds', type=int, default=5,
                        help='FedAvg rounds for source-domain separate regression pretraining')
    parser.add_argument('--separate_reg_source_agg_scope', type=str, default='all', choices=['all', 'no_calib'],
                        help='Source FedAvg aggregation scope for separate regression: all=legacy behavior; no_calib excludes per-client calibration/prototype scale/bias tensors')
    parser.add_argument('--separate_reg_source_init', type=str, default='fedavg', choices=['fedavg', 'calib_select'],
                        help='Target-client regression initialization: fedavg uses the source FedAvg checkpoint; calib_select chooses FedAvg or a source-local checkpoint by target calibration score')
    parser.add_argument('--separate_reg_source_init_metric', type=str, default='r2', choices=['r2', 'neg_mae'],
                        help='Metric used by calib_select source initialization')
    parser.add_argument('--separate_reg_source_init_scope', type=str, default='overall', choices=['overall', 'min_class'],
                        help='Calibration scoring scope used by calib_select source initialization')
    parser.add_argument('--separate_reg_source_init_min_score_gain', type=float, default=0.0,
                        help='Minimum calibration score gain over FedAvg required before selecting a source-local regression initialization')
    parser.add_argument('--separate_reg_source_route', type=str, default='none', choices=['none', 'per_class_calib_select'],
                        help='Optional source-local model routing per predicted gas, selected by target calibration MAE')
    parser.add_argument('--separate_reg_source_route_min_mae_gain', type=float, default=5.0,
                        help='Minimum calibration MAE gain over FedAvg required before routing a gas to a source-local model')
    parser.add_argument('--separate_reg_source_checkpoint', type=str, default='',
                        help='Reuse a saved separate_regression_source.pth and skip source-domain regression pretraining')
    parser.add_argument('--separate_reg_skip_source_diagnostics', action='store_true',
                        help='Skip source local/cross/FedAvg diagnostics for faster regression calibration experiments')
    parser.add_argument('--separate_reg_export_predictions', action='store_true',
                        help='Export per-sample oracle and pipeline regression predictions to CSV')
    parser.add_argument('--separate_reg_export_calibration_predictions', action='store_true',
                        help='When exporting predictions, also export calibration split predictions for post-calibration diagnostics')
    parser.add_argument('--separate_reg_pipeline_route', type=str, default='hard', choices=['hard', 'soft_topk'],
                        help='Pipeline regression routing: hard argmax class or top-k soft weighted heads')
    parser.add_argument('--separate_reg_pipeline_soft_topk', type=int, default=2,
                        help='Top-k classes used by soft_topk pipeline regression routing')
    parser.add_argument('--separate_reg_pipeline_soft_temperature', type=float, default=1.0,
                        help='Temperature applied to classifier probabilities for soft_topk routing')
    parser.add_argument('--separate_reg_pipeline_soft_min_confidence', type=float, default=1.01,
                        help='Use soft routing when top-1 confidence is below this threshold')
    parser.add_argument('--separate_reg_pipeline_soft_max_margin', type=float, default=-1.0,
                        help='Use soft routing when top1-top2 confidence margin is below this threshold')
    parser.add_argument('--separate_reg_pipeline_soft_clients', type=str, default='',
                        help='Comma-separated client IDs allowed to use soft_topk pipeline routing. Empty means all target clients')
    parser.add_argument('--separate_reg_class_route', type=str, default='logits',
                        choices=['logits', 'target_proto', 'target_knn', 'target_router',
                                 'target_reg_router', 'logits_target_proto_mix',
                                 'logits_target_knn_mix', 'logits_target_router_mix'],
                        help='Class route used before pipeline regression: logits or labeled target-calibration feature routing')
    parser.add_argument('--separate_reg_route_temperature', type=float, default=0.1,
                        help='Temperature for target prototype/kNN class route')
    parser.add_argument('--separate_reg_route_knn_k', type=int, default=7,
                        help='k for target_knn class route')
    parser.add_argument('--separate_reg_route_mix_alpha', type=float, default=0.5,
                        help='Weight for target route scores in logits_target_*_mix routes')
    parser.add_argument('--separate_reg_route_clients', type=str, default='',
                        help='Comma-separated client IDs using target calibration class route. Empty means all target clients')
    parser.add_argument('--separate_reg_route_router_feature_set', type=str, default='all',
                        choices=['logits', 'cls', 'reg', 'logits_cls', 'phase', 'all'],
                        help='Feature set used by target_router class route')
    parser.add_argument('--separate_reg_route_router_epochs', type=int, default=80,
                        help='Epochs for target_router training on calibration split')
    parser.add_argument('--separate_reg_route_router_lr', type=float, default=1e-3,
                        help='Learning rate for target_router training')
    parser.add_argument('--separate_reg_route_router_weight_decay', type=float, default=1e-3,
                        help='Weight decay for target_router training')
    parser.add_argument('--separate_reg_route_router_hidden_dim', type=int, default=32,
                        help='Hidden dimension for target_router MLP; 0 means linear router')
    parser.add_argument('--separate_reg_route_router_focal_gamma', type=float, default=0.0,
                        help='Optional focal gamma for target_router training')
    parser.add_argument('--separate_reg_target_steps', type=int, default=300,
                        help='Target calibration fine-tuning steps per client for separate regression model')
    parser.add_argument('--separate_reg_target_clients', type=str, default='',
                        help='Comma-separated target client IDs for calibration (default: all test clients)')
    parser.add_argument('--separate_reg_lr', type=float, default=5e-4,
                        help='Regression branch learning rate for separate regression training')
    parser.add_argument('--separate_reg_feat_lr', type=float, default=5e-5,
                        help='Encoder learning rate when separate regression unfreezes encoder layers')
    parser.add_argument('--separate_reg_unfreeze', type=str, default='none', choices=['none', 'last_tcn', 'all_encoder'],
                        help='Encoder unfreeze policy during target regression calibration')
    parser.add_argument('--separate_reg_allow_encoder_backprop', action=argparse.BooleanOptionalAction,
                        default=default_config.SEPARATE_REG_ALLOW_ENCODER_BACKPROP,
                        help='When encoder layers are unfrozen, allow regression loss to backpropagate into them')
    parser.add_argument('--separate_reg_calib_mode', type=str, default='full',
                        choices=['full', 'affine_only', 'bias_only', 'phase_affine_only', 'none', 'gated', 'auto', 'auto_v2', 'auto_v2_specialist'],
                        help='Target calibration mode: full=neural fine-tuning, '
                             'affine_only=fit per-class a,b in ppm space, '
                             'bias_only=fit per-class b in ppm space, '
                             'phase_affine_only=fit per-class-per-phase a,b in ppm space, '
                             'gated=accept full/fallback only if held-out calibration improves, '
                             'auto=legacy select none/full/bias/affine by held-out calibration, '
                             'auto_v2=strict class-wise select none/full/bias/affine/phase-affine, '
                             'auto_v2_specialist=auto_v2 plus specialist full model for selected classes, '
                             'none=skip calibration')
    parser.add_argument('--separate_reg_val_ratio', type=float, default=0.3,
                        help='Held-out fraction of target calibration data for gated/auto regression calibration')
    parser.add_argument('--separate_reg_val_split', type=str, default='window', choices=['window', 'file'],
                        help='Held-out split unit for gated/auto regression calibration routing')
    parser.add_argument('--separate_reg_gate_metric', type=str, default='r2', choices=['r2', 'neg_mae'],
                        help='Metric used by gated/auto separate regression calibration')
    parser.add_argument('--separate_reg_gate_scope', type=str, default='overall', choices=['overall', 'min_class'],
                        help='Scope used by gated/auto calibration score')
    parser.add_argument('--separate_reg_gate_min_delta', type=float, default=0.0,
                        help='Minimum held-out score improvement required to accept calibration')
    parser.add_argument('--separate_reg_gate_fallback', type=str, default='affine_only',
                        choices=['none', 'bias_only', 'affine_only'],
                        help='Fallback tried by gated mode when full neural calibration is rejected')
    parser.add_argument('--separate_reg_auto_scope', type=str, default='per_class', choices=['client', 'per_class'],
                        help='Auto calibration selection granularity')
    parser.add_argument('--separate_reg_class_weights', type=str, default='',
                        help='Optional regression loss class weights, e.g. "2:2.0" to emphasize ethylene')
    parser.add_argument('--separate_reg_huber_deltas', type=str, default=default_config.SEPARATE_REG_HUBER_DELTAS,
                        help='Optional per-class SmoothL1 beta in normalized space, e.g. "1:0.1" for CO')
    parser.add_argument('--separate_reg_specialist_classes', type=str, default='2',
                        help='Comma-separated classes routed to specialist full models in auto_v2_specialist mode')
    parser.add_argument('--separate_reg_specialist_weight', type=float, default=2.0,
                        help='Per-class training weight used for specialist full models')
    parser.add_argument('--separate_reg_specialist_steps', type=int, default=80,
                        help='Target calibration steps for specialist full models')
    parser.add_argument('--separate_reg_specialist_gate', action='store_true',
                        help='Use calibration validation R2 to route each specialist class only when it improves over general auto_v2')
    parser.add_argument('--separate_reg_specialist_gate_min_delta', type=float, default=0.0,
                        help='Minimum per-class validation R2 improvement required to accept a specialist route')
    parser.add_argument('--separate_reg_specialist_refit_full_calib', action='store_true',
                        help='After gate selection, refit accepted specialist models on the full calibration split')
    parser.add_argument('--separate_reg_specialist_refit_steps', type=int, default=None,
                        help='Steps for full-calibration specialist refit; defaults to specialist_steps')
    parser.add_argument('--separate_reg_specialist_refit_client_classes', type=str, default='',
                        help='Optional selective refit map, e.g. 2:1,2;3:2. Empty means refit all accepted specialists')
    parser.add_argument('--separate_reg_rank_classes', type=str, default='',
                        help='Comma-separated classes for monotonic ranking loss, e.g. "2" for ethylene')
    parser.add_argument('--separate_reg_rank_weight', type=float, default=0.0,
                        help='Weight of optional per-class monotonic ranking loss')
    parser.add_argument('--separate_reg_rank_margin', type=float, default=0.02,
                        help='Margin for optional monotonic ranking loss in normalized concentration space')
    parser.add_argument('--separate_reg_seed', type=int, default=default_config.SEED,
                        help='Random seed reset before separate regression B training for reproducible regression-only experiments')
    parser.add_argument('--seed', type=int, default=default_config.SEED, help='Random seed')
    parser.add_argument('--lambda_align', type=float, default=default_config.LAMBDA_ALIGN, help='Alignment loss weight')
    parser.add_argument('--lambda_distill', type=float, default=default_config.LAMBDA_REPLAY_DISTILL, help='Feature distillation weight')
    parser.add_argument('--lambda_proto', type=float, default=default_config.LAMBDA_PROTO, help='Prototype learning weight')
    parser.add_argument('--lambda_consistency', type=float, default=default_config.LAMBDA_CONSISTENCY, help='Consistency loss weight')
    parser.add_argument('--no_use_cumulative', action='store_true', help='Disable cumulative data scheduler (use non-cumulative instead)')
    parser.add_argument('--no_time_drift', action='store_true', help='Disable time drift simulation, load all data directly')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output directory')
    parser.add_argument('--model_selection', action='store_true', help='Enable model selection based on test clients')
    parser.add_argument('--compute_mmd', action='store_true', help='Compute MMD during training')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Early stopping patience (0=disabled)')
    parser.add_argument('--mmd_interval', type=int, default=5, help='MMD computation interval')
    parser.add_argument('--train_clients', type=str, default='1,2', help='Comma-separated training client IDs')
    parser.add_argument('--test_clients', type=str, default='3', help='Comma-separated test client IDs')
    parser.add_argument('--phase_ratios', type=str, default='0.3,0.4,0.3', help='Phase ratios for scheduler')
    parser.add_argument('--plot_tsne', action='store_true', help='Enable t-SNE visualization of features')
    parser.add_argument('--plot_umap', action='store_true', help='Enable UMAP visualization of features')
    parser.add_argument('--few_shot_residual', action='store_true', help='Estimate device residual from few-shot samples')
    parser.add_argument('--use_coral', action='store_true', help='Enable CORAL feature alignment for hard clients')
    parser.add_argument('--tta_clients', type=str, default='5',
                        help='Comma-separated client IDs to apply TTA (default: 5)')
    parser.add_argument('--few_shot_regression', action='store_true',
                     help='Enable few-shot finetuning of regression heads on target clients')
    parser.add_argument('--few_shot_classification', action='store_true',
                     help='Enable few-shot finetuning of classification heads on target clients')
    # 
    parser.add_argument('--target_cls_calibration', action='store_true',
                     help='Use full labeled target calibration split to calibrate the classification head')
    parser.add_argument('--no_target_cls_calibration', action='store_true',
                     help='Disable target labeled classification calibration')
    parser.add_argument('--target_cls_calib_epochs', type=int, default=30,
                     help='Epochs for supervised target classification calibration')
    parser.add_argument('--target_cls_lr', type=float, default=1e-3,
                     help='Classifier-head LR for target classification calibration')
    parser.add_argument('--target_cls_feat_lr', type=float, default=5e-5,
                     help='Feature/projection LR for target classification calibration')
    parser.add_argument('--target_cls_calib_clients', type=str, default='',
                     help='Comma-separated target clients to recalibrate. Empty means all target clients')
    parser.add_argument('--target_cls_client_epochs', type=str, default='',
                     help='Client-specific target classification epochs, e.g. "5:80"')
    parser.add_argument('--target_cls_client_lr', type=str, default='',
                     help='Client-specific target classification head LR, e.g. "5:5e-4"')
    parser.add_argument('--target_cls_client_feat_lr', type=str, default='',
                     help='Client-specific target classification feature LR, e.g. "5:1e-4"')
    parser.add_argument('--target_cls_class_weights', type=str, default='',
                     help='Global target classification class weights, e.g. "1:1.5,2:2.0"')
    parser.add_argument('--target_cls_client_class_weights', type=str, default='',
                     help='Client-specific target classification weights, e.g. "5:1:2.0,5:2:2.0"')
    parser.add_argument('--target_cls_focal_gamma', type=float, default=0.0,
                     help='Focal loss gamma for target classification calibration (0 disables focal loss)')
    parser.add_argument('--target_cls_client_focal_gamma', type=str, default='',
                     help='Client-specific focal gamma, e.g. "5:1.5"')
    parser.add_argument('--target_cls_cost_matrix', type=str, default='',
                     help='Global class-routing cost matrix, rows true class and columns predicted class, e.g. "0,1,2,1;3,0,2,3;2,5,0,2;1,1,2,0"')
    parser.add_argument('--target_cls_client_cost_matrix', type=str, default='',
                     help='Client-specific cost matrices, e.g. "5=0,1,2,1/3,0,2,3/2,5,0,2/1,1,2,0"')
    parser.add_argument('--target_cls_cost_weight', type=float, default=0.0,
                     help='Weight for cost-sensitive target classification calibration loss')
    parser.add_argument('--target_cls_client_cost_weight', type=str, default='',
                     help='Client-specific target classification cost weights, e.g. "5:0.1"')
    parser.add_argument('--target_cls_aug_prob', type=float, default=0.0,
                     help='Probability of sensor augmentation during target classification calibration')
    parser.add_argument('--target_cls_client_aug_prob', type=str, default='',
                     help='Client-specific target classification augmentation probability, e.g. "5:0.8"')
    parser.add_argument('--target_cls_aug_gain_std', type=float, default=None,
                     help='Global gain std for target classification augmentation')
    parser.add_argument('--target_cls_aug_bias_std', type=float, default=None,
                     help='Global bias std for target classification augmentation')
    parser.add_argument('--target_cls_aug_ch_gain_std', type=float, default=None,
                     help='Channel gain std for target classification augmentation')
    parser.add_argument('--target_cls_aug_time_scale_range', type=float, default=None,
                     help='Time-scale range for target classification augmentation')
    parser.add_argument('--target_cls_aug_time_prob', type=float, default=None,
                     help='Probability of time-scale augmentation inside target classification augmentation')
    parser.add_argument('--regression_only_recalibrate_target_classifiers', action='store_true',
                     help='In --regression_only mode, rebuild selected target classifiers before regression evaluation')
    parser.add_argument('--regression_only_recalibrate_from_existing_target', action='store_true',
                     help='Start regression-only target classifier recalibration from --target_classifier_dir states when available')

    parser.add_argument('--few_shot_samples', type=int, default=21,
                     help='Number of labeled samples per class for few-shot regression')
    parser.add_argument('--few_shot_reg_steps', type=int, default=100, help='Few-shot regression fine-tune steps')
    parser.add_argument('--few_shot_reg_lr', type=float, default=5e-4, help='Learning rate for regression head in few-shot')
    parser.add_argument('--few_shot_feat_lr', type=float, default=1e-4, help='Learning rate for feature layers in few-shot regression')
    parser.add_argument('--few_shot_cls_feat_lr', type=float, default=0.0, help='Feature LR for classification few-shot')
    parser.add_argument('--few_shot_reg_weight_decay', type=float, default=1e-3, help='Weight decay for few-shot regression')
    parser.add_argument('--hard_clients', type=str, default='5',
                        help='Comma-separated client IDs that require enhanced strategy (CORAL alignment)')
    parser.add_argument('--coral_strategy', type=str, default='class_conditional',
                        choices=['none', 'global', 'class_conditional'],
                        help='CORAL alignment strategy for hard clients (if use_coral is enabled)')
    parser.add_argument('--num_conc_buckets', type=int, default=default_config.NUM_CONC_BUCKETS,
                        help='Number of concentration buckets for auxiliary classification (0 to disable)')
    parser.add_argument('--lambda_conc_bucket', type=float, default=default_config.LAMBDA_CONC_BUCKET,
                        help='Loss weight for concentration bucket auxiliary classification')
    parser.add_argument('--conc_bucket_loss', type=str, default=default_config.CONC_BUCKET_LOSS,
                        choices=['hard', 'soft'],
                        help='Concentration bucket auxiliary loss: hard CE or soft neighboring-bin targets')
    parser.add_argument('--conc_bucket_soft_sigma', type=float, default=default_config.CONC_BUCKET_SOFT_SIGMA,
                        help='Bucket-distance sigma for --conc_bucket_loss soft')
    parser.add_argument('--conc_bucket_detach_feat', action='store_true',
                        help='Train the concentration bucket head as a detached risk probe without changing regression features')
    parser.add_argument('--reg_tail_weight', type=float, default=default_config.REG_TAIL_WEIGHT,
                        help='Regression sample weight for high normalized concentrations; 1 disables')
    parser.add_argument('--reg_tail_threshold', type=float, default=default_config.REG_TAIL_THRESHOLD,
                        help='Normalized concentration threshold for tail weighting, e.g. 0.8')
    parser.add_argument('--reg_tail_classes', type=str, default=default_config.REG_TAIL_CLASSES,
                        help='Comma-separated class IDs for tail weighting; empty means all classes')
    parser.add_argument('--reg_window_stats', action='store_true',
                        help='Append visible window amplitude/slope statistics to separate regression features')
    parser.add_argument('--reg_window_stats_mode', type=str, default=default_config.REG_WINDOW_STATS_MODE,
                        choices=['global', 'per_channel'],
                        help='Window-statistics feature set for regression: global or per_channel')
    parser.add_argument('--reg_window_stats_dim', type=int, default=default_config.REG_WINDOW_STATS_DIM,
                        help='Hidden dimension for the regression window-statistics projection')
    parser.add_argument('--stagewise', action='store_true', help='Enable stagewise training (phase1 classification only)')
    parser.add_argument('--phase1_rounds', type=int, default=15, help='Phase 1 end round (1-based)')
    parser.add_argument('--phase2_reg_weight', type=float, default=1.0, help='Regression loss weight in phase 2')
    parser.add_argument('--share_reg_head', action='store_true', default=False,
                        help='Share regression head across clients (default follows config)')
    parser.add_argument('--no_share_reg_head', action='store_true',
                        help='Disable regression head sharing (keep personalized)')
    parser.add_argument('--personalized_reg', action='store_true', default=False,
                        help='Force personalized regression calibration parameters')
    parser.add_argument('--no_personalized_reg', action='store_true', default=False,
                        help='Disable personalized regression so all regression parameters are shared')
    # 回归梯度阻断开关（用于验证梯度污染假说的关键实验）
    parser.add_argument('--reg_grad_detach', action='store_true', default=False,
                        help='Enable gradient detach for regression branch to prevent TCN contamination')
    parser.add_argument('--server_opt_steps', type=int, default=None,
                        help='Override SERVER_OPT_STEPS_BASE (K) in server optimization')
    # 部署在线自适应参数
    parser.add_argument('--deploy_rounds', type=int, default=0,
                        help='Number of deployment adaptation rounds (0 to disable)')
    parser.add_argument('--adapt_steps_per_round', type=int, default=10,
                        help='Adaptation steps per deployment round')
    parser.add_argument('--upload_freq', type=int, default=5,
                        help='Upload and aggregation frequency in deployment')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Evaluation frequency in deployment')
    parser.add_argument('--active_query_freq', type=int, default=5,
                        help='Active learning query frequency in deployment rounds (0 to disable)')
    parser.add_argument('--active_query_samples', type=int, default=2,
                        help='Number of labeled samples per class for active learning query')
    parser.add_argument('--dynamic_anchor', action='store_true', default=True,
                        help='Enable dynamic prototype anchoring (default: True)')
    parser.add_argument('--adapt_unfreeze_level', type=str, default='basic',
                        choices=['basic', 'medium', 'full'],
                        help='Unfreeze level for online adaptation (basic: classifier only, medium: +TCN last layer, full: all)')
    # TENT 部署适应参数
    parser.add_argument('--tent_weight', type=float, default=1.0,
                        help='TENT entropy minimization loss weight in deployment adaptation')
    parser.add_argument('--anchor_reg_weight', type=float, default=0.05,
                        help='Soft anchor regularization weight in deployment adaptation')
    parser.add_argument('--ema_decay', type=float, default=0.99,
                        help='Teacher EMA decay rate in deployment adaptation')
    # 深度CORAL配置参数
    parser.add_argument('--use_deep_coral', action='store_true',
                        help='Enable deep CORAL alignment in server optimization')
    parser.add_argument('--no_use_deep_coral', action='store_true',
                        help='Disable deep CORAL alignment (消融实验用)')
    parser.add_argument('--coral_calib_clients', type=str, default='5',
                        help='Comma-separated target client IDs for deep CORAL calibration')
    parser.add_argument('--coral_calib_size', type=int, default=500,
                        help='Number of unlabeled samples per client for deep CORAL')
    # 域对抗训练参数
    parser.add_argument('--use_adversarial_domain', action='store_true',
                        help='Enable Wasserstein GAN adversarial domain training on server side')
    parser.add_argument('--no_use_adversarial_domain', action='store_true',
                        help='Disable adversarial domain training (消融实验用)')
    parser.add_argument('--lambda_adv_domain', type=float, default=default_config.LAMBDA_ADV_DOMAIN,
                        help='Adversarial domain loss weight')
    parser.add_argument('--adv_domain_lr', type=float, default=0.001,
                        help='Learning rate for domain discriminator')
    parser.add_argument('--adv_no_class_conditional', action='store_true',
                        help='Disable class-conditional adversarial domain (use unconditional variant)')
    # Transformer编码器参数
    parser.add_argument('--encoder', type=str, default='tcn', choices=['tcn', 'transformer'],
                        help='Feature encoder type: tcn or transformer')
    parser.add_argument('--transformer_d_model', type=int, default=48,
                        help='Transformer d_model dimension')
    parser.add_argument('--transformer_nhead', type=int, default=4,
                        help='Transformer attention heads')
    parser.add_argument('--transformer_num_layers', type=int, default=2,
                        help='Transformer encoder layers')
    parser.add_argument('--transformer_ff_dim', type=int, default=96,
                        help='Transformer feed-forward dimension')
    parser.add_argument('--tcn_norm', type=str, default=default_config.TCN_NORM,
                        choices=['instance', 'batch', 'none'],
                        help='Normalization inside TCN blocks')
    # MMD对齐与标签控制开关（消融实验用）
    parser.add_argument('--no_use_mmd_alignment', action='store_true',
                        help='Disable MMD alignment loss in server optimization')
    parser.add_argument('--calib_use_labels', action='store_true',
                        help='Explicitly use labeled calibration data (CALIB_USE_LABELS=True)')
    parser.add_argument('--no_calib_use_labels', action='store_true',
                        help='Use unlabeled calibration data (CALIB_USE_LABELS=False)')
    parser.add_argument('--use_adaptive_temperature', action='store_true',
                        help='Enable adaptive prototype temperatures for soft aggregation')
    # 马氏距离推理开关（P0核心：控制推理分类、选择性聚合、部署阶段分配使用马氏距离替代余弦相似度）
    parser.add_argument('--use_mahalanobis_inference', action='store_true',
                        help='Enable Mahalanobis distance for inference classification, selective aggregation, and deployment phase assignment')
    
    args = parser.parse_args()

    if args.regression_mode == 'separate':
        args.eval_test_only = True
        if not args.no_calib_use_labels:
            args.calib_use_labels = True
        if not args.no_target_cls_calibration:
            args.target_cls_calibration = True

    # 参数验证
    if args.regression_only:
        assert args.regression_mode == 'separate', "--regression_only requires --regression_mode separate"
        assert args.classifier_checkpoint, "--regression_only requires --classifier_checkpoint"
    else:
        assert args.rounds > 0, "Number of global rounds must be positive"
        assert args.local_epochs > 0, "Number of local epochs must be positive"
    assert args.lambda_align >= 0, "Lambda align must be non-negative"
    assert args.lambda_distill >= 0, "Lambda distill must be non-negative"
    assert args.lambda_proto >= 0, "Lambda proto must be non-negative"
    assert args.lambda_consistency >= 0, "Lambda consistency must be non-negative"
    assert args.early_stop_patience >= 0, "Early stop patience must be non-negative"
    assert args.mmd_interval > 0, "MMD interval must be positive"
    
    # 验证训练和测试客户端
    train_client_ids = [int(x.strip()) for x in args.train_clients.split(',')]
    test_client_ids = [int(x.strip()) for x in args.test_clients.split(',') if x.strip()]
    assert len(train_client_ids) > 0, "At least one training client must be specified"
    # 阶段4允许test_client_ids为空
    assert len(set(train_client_ids).intersection(test_client_ids)) == 0, "Training and test clients must be disjoint"
    
    # 解析 TTA 客户端（不强制要求在测试客户端列表中，TTA可应用于训练或测试客户端）
    tta_client_ids = [int(x.strip()) for x in args.tta_clients.split(',')]
    # 解析困难客户端
    hard_client_ids = [int(x.strip()) for x in args.hard_clients.split(',')]
    
    # 验证阶段比例
    phase_ratios = [float(x) for x in args.phase_ratios.split(',')]
    assert len(phase_ratios) == 3, "Phase ratios must have exactly 3 values"
    assert all(r > 0 for r in phase_ratios), "All phase ratios must be positive"
    assert abs(sum(phase_ratios) - 1.0) < 1e-6, "Phase ratios must sum to 1.0"

    temp_federated_dir = None
    try:
        # 1. 初始化配置
        config = setup_config(args)

        # 2. 设置实验环境
        exp_dir, logger = setup_experiment(args, config)

        # 3. 数据准备
        train_client_ids, test_client_ids, temp_federated_dir, scheduler, client_loaders, test_client_loaders, global_test_loader, val_loader, federated_dir, calib_loader = setup_data(args, config, logger)

        # 3.5 为每个目标客户端构建校准集加载器（用于少样本微调和主动学习查询）
        calibration_loaders = {}
        for cid in test_client_ids:
            cal_ldr = build_client_calibration_loader(cid, federated_dir, config.BATCH_SIZE)
            if cal_ldr is not None:
                calibration_loaders[cid] = cal_ldr
                logger.info(f"Client {cid}: calibration loader built with {len(cal_ldr.dataset)} samples")
            else:
                logger.warning(f"Client {cid}: calibration data not found, will fallback to full test data")

        # 4. 初始化模型、服务器、客户端
        if args.regression_only:
            logger.info("=== Regression-only mode: skip federated classification training ===")
            run_regression_only_pipeline(
                args=args, config=config, train_client_ids=train_client_ids, test_client_ids=test_client_ids,
                temp_federated_dir=temp_federated_dir, calibration_loaders=calibration_loaders,
                test_client_loaders=test_client_loaders, logger=logger
            )
            logger.info("Regression-only experiment finished.")
            return

        global_model, server, clients = setup_model(config, global_test_loader, test_client_loaders, test_client_ids, train_client_ids, val_loader, logger, calib_loader=calib_loader)

        # === 诊断：随机初始化模型的各阶段准确率 ===
        from utils import evaluate_model_with_phase
        logger.info("=== Evaluating random initialized model on global test set ===")
        random_acc = evaluate_model_with_phase(global_model, global_test_loader, config.DEVICE)
        logger.info(f"Random model - Early: {random_acc['early']:.4f}, "
                    f"Middle: {random_acc['middle']:.4f}, Late: {random_acc['late']:.4f}")

        # 5. 预热与原型初始化
        warmup_and_init_prototypes(clients, scheduler, client_loaders, server, config, logger, args)

        # 6. 训练历史记录容器
        history: List[Dict] = []
        weights_history: List[Dict] = []
        peak_acc = {'early': 0.0, 'middle': 0.0}
        peak_round = {'early': 0, 'middle': 0}

        best_test_accs = {cid: 0.0 for cid in test_client_ids}
        best_rounds = {cid: 0 for cid in test_client_ids}
        best_model_states: Dict[int, Optional[Dict[str, torch.Tensor]]] = {cid: None for cid in test_client_ids}
        
        # 早停相关变量初始化
        patience = args.early_stop_patience
        wait = 0
        best_val_metric = 0.0
        best_val_acc = 0.0  # 用于模型选择
        best_model_state_for_all = None  # 最佳模型状态
        best_round_for_all = None  # 最佳模型对应的轮次
        early_stop_enabled = patience > 0

        mmd_history = []
        if args.compute_mmd:
            train_feats = extract_features_batch(global_model, global_test_loader, config.DEVICE)
            for cid in test_client_ids:
                test_feats = extract_features_batch(global_model, test_client_loaders[cid], config.DEVICE)
                mmd_val = compute_mmd(torch.from_numpy(train_feats), torch.from_numpy(test_feats)).item()
                mmd_history.append({'round': 0, 'mmd': mmd_val, 'test_client_id': cid})
                logger.info(f"Initial MMD client {cid}: {mmd_val:.4f}")

        # 7. 联邦主循环
        start_time = time.time()
        for round_idx in range(1, config.GLOBAL_ROUNDS + 1):
            # 执行一轮训练
            result = run_round(round_idx, config, server, clients, scheduler, client_loaders, test_client_ids, test_client_loaders, global_test_loader, val_loader, args, logger, tta_client_ids)
            if result is None or result[0] is None:
                continue
            
            acc_early, acc_middle, acc_late, acc_test, test_accs, weight_dict, align_loss, current_mmds, reg_metrics = result

            # 更新峰值
            if acc_early is not None and acc_early > peak_acc['early']:
                peak_acc['early'] = acc_early
                peak_round['early'] = round_idx
            if acc_middle is not None and acc_middle > peak_acc['middle']:
                peak_acc['middle'] = acc_middle
                peak_round['middle'] = round_idx

            # 每轮都评估验证集分类准确率（无论是否 model_selection）
            from utils import evaluate_model_with_phase
            val_acc_dict = evaluate_model_with_phase(server.global_model, val_loader, config.DEVICE)
            val_cls_acc = val_acc_dict['global']

            # 模型选择（使用验证集分类 + 回归综合指标）
            if args.model_selection:
                composite_score = val_cls_acc
                if config.USE_REG_LOSS:
                    try:
                        from utils import evaluate_regression_metrics
                        val_reg_metrics, val_reg_overall = evaluate_regression_metrics(
                            server.global_model, val_loader, config.DEVICE,
                            tolerance=0.1, enable_calibration=False
                        )
                        val_reg_rmse = val_reg_overall.get('RMSE', 99.0)
                        cls_weight = getattr(config, 'MODEL_SELECTION_CLS_WEIGHT', 0.7)
                        reg_weight = getattr(config, 'MODEL_SELECTION_REG_WEIGHT', 0.3)
                        reg_score = 1.0 / (1.0 + val_reg_rmse)
                        composite_score = cls_weight * val_cls_acc + reg_weight * reg_score
                    except Exception as e:
                        logger.warning(f"Could not compute regression metrics for model selection: {e}")
                if composite_score > best_val_acc:
                    best_val_acc = composite_score
                    best_model_state_for_all = copy.deepcopy(server.global_model.state_dict())
                    best_round_for_all = round_idx
                    # 更新每个测试客户端的最佳模型
                    for cid in test_client_ids:
                        best_model_states[cid] = copy.deepcopy(server.global_model.state_dict())
                        best_test_accs[cid] = test_accs.get(cid, 0.0)
                        best_rounds[cid] = round_idx
                    logger.info(f"New best model on validation: composite={composite_score:.4f} (cls={val_cls_acc:.4f}) at round {round_idx}")

            # 早停（基于验证集分类准确率）
            if args.early_stop_patience > 0:
                if val_cls_acc > best_val_metric + 1e-6:
                    best_val_metric = val_cls_acc
                    wait = 0
                else:
                    wait += 1
                if wait >= args.early_stop_patience:
                    logger.info(f"Early stopping at round {round_idx}")
                    break

            # 保存 MMD 历史
            if args.compute_mmd and round_idx % args.mmd_interval == 0 and current_mmds:
                for cid in test_client_ids:
                    if cid in current_mmds:
                        mmd_history.append({'round': round_idx, 'mmd': current_mmds[cid], 'test_client_id': cid})

            # 保存历史
            record_entry = {
                'round': round_idx, 'acc_early': acc_early, 'acc_middle': acc_middle, 'acc_late': acc_late,
                'test_acc': acc_test, 'test_client_acc': None, 'test_client_id': None,
                'align_loss': align_loss, 'mmd': None, 'weights': weight_dict
            }
            # 回归指标
            if reg_metrics:
                reg_record = {}
                for cid, rm in reg_metrics.items():
                    ro = rm['overall']
                    if ro and ro.get('n_samples', 0) > 0:
                        reg_record[f"client_{cid}"] = {'R2': float(ro['R2']), 'RMSE': float(ro['RMSE']), 'MAE': float(ro['MAE'])}
                record_entry['regression'] = reg_record
            if test_accs:
                for cid, acc in test_accs.items():
                    entry = record_entry.copy()
                    entry['test_client_acc'] = acc
                    entry['test_client_id'] = cid
                    history.append(entry)
            else:
                history.append(record_entry)

            # 记录权重
            if weight_dict:
                weights_history.append(weight_dict)

            # 每5轮保存一次中间检查点（防止崩溃丢失全部结果）
            if round_idx % 5 == 0:
                checkpoint_dir = Path(config.MODEL_SAVE_DIR)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = checkpoint_dir / f"checkpoint_r{round_idx}.pth"
                ckpt = {
                    'round': round_idx,
                    'model_state': server.global_model.state_dict(),
                    'shared_reg_state': getattr(server, 'shared_reg_state', {}),
                    'protos': {k: v.data.cpu() for k, v in server.semantic_protos.items()},
                    'proto_vars': {k: v.data.cpu() for k, v in server.semantic_proto_vars.items()},
                    'history': history,
                    'weights_history': weights_history,
                }
                if hasattr(server, 'optimizer') and server.optimizer is not None:
                    ckpt['optimizer_state'] = server.optimizer.state_dict()
                torch.save(ckpt, ckpt_path)
                logger.info(f"Checkpoint saved at round {round_idx} -> {ckpt_path}")

        training_time = time.time() - start_time
        logger.info(f"Training completed in {format_time(training_time)}")

        # === 部署前少样本微调（使在线适应从校准后的模型开始） ===
        if args.deploy_rounds > 0 and args.few_shot_classification:
            logger.info("=== Pre-deployment Few-Shot Fine-Tuning ===")
            for cid in test_client_ids:
                # 从云端维护的校准集中抽取少样本（而非全量测试数据）
                calib_loader = calibration_loaders.get(cid) if calibration_loaders else None
                source_loader = calib_loader if calib_loader is not None else test_client_loaders.get(cid)
                if source_loader is None:
                    continue
                fs_loader, _ = build_few_shot_and_test_loaders(source_loader, args.few_shot_samples, config.BATCH_SIZE)
                deploy_model = copy.deepcopy(server.global_model)
                few_shot_finetune_classification(
                    deploy_model, fs_loader, config.DEVICE,
                    epochs=5, lr=1e-3,
                    finetune_feat_lr=args.few_shot_cls_feat_lr
                )
                # 将微调后的模型作为服务器模型（单测试客户端下可直接覆盖）
                server.global_model.load_state_dict(deploy_model.state_dict())
                logger.info(f"Client {cid}: few-shot fine-tuned before deployment (from calibration set).")
        # ============================================================

        # 部署在线自适应阶段（Stage 2）
        if args.deploy_rounds > 0:
            logger.info("\n=== Starting Deployment Phase ===")
            from data_scheduler import CumulativeDataScheduler
            deploy_client_dirs = [os.path.join(federated_dir, f'client_{cid}') for cid in test_client_ids]
            deploy_scheduler = CumulativeDataScheduler(
                deploy_client_dirs,
                batch_size=config.BATCH_SIZE,
                phase_ratios=[float(x) for x in args.phase_ratios.split(',')],
                device=config.DEVICE
            )
            phase_ratios = [float(x) for x in args.phase_ratios.split(',')]
            deploy_history = deployment_phase(
                server, clients, test_client_ids, test_client_loaders,
                config, args, logger, deploy_scheduler, phase_ratios,
                active_query_freq=args.active_query_freq,
                active_query_samples=args.active_query_samples,
                calibration_loaders=calibration_loaders
            )
            history.extend(deploy_history)

        # 8. 最终评估与模型选择（使用验证集选择的最佳模型）
        final_accs, regression_metrics_all, classification_metrics_all = final_evaluation(args, config, server, test_client_ids, test_client_loaders, best_model_states, best_rounds, logger, global_test_loader, tta_client_ids, best_model_state_for_all, best_round_for_all, calibration_loaders=calibration_loaders)

        if args.regression_mode == 'separate':
            logger.info("\n=== Separate Supervised Regression Pipeline ===")
            classifier_state = (best_model_state_for_all if args.model_selection and best_model_state_for_all is not None
                                else copy.deepcopy(server.global_model.state_dict()))
            separate_metrics = run_separate_regression_pipeline(
                args=args, config=config, classifier_state=classifier_state,
                train_client_ids=train_client_ids, temp_federated_dir=temp_federated_dir,
                calibration_loaders=calibration_loaders, test_client_loaders=test_client_loaders,
                logger=logger,
                semantic_protos={k: v.detach().cpu() for k, v in server.semantic_protos.items()},
                target_classifier_states=getattr(server, 'target_classifier_states', None)
            )
            regression_metrics_all['separate_regression'] = separate_metrics

        # ========= 源域评估与可视化 =========
        source_regression_metrics = {}
        source_cls_accs = {}
        logger.info("\n=== Source Domain Evaluation ===")
        for client in clients:
            cid = client.client_id
            # 构建该客户端的本地测试集加载器
            client_dir = os.path.join(temp_federated_dir, f'client_{cid}')
            from federated_dataset import GasSensorWindowDataset
            import numpy as np
            from torch.utils.data import DataLoader
            client_path = Path(client_dir)
            normalize = False
            mean_std = None
            test_features = np.load(os.path.join(client_dir, 'test_features.npy'))
            test_reg = np.load(os.path.join(client_dir, 'test_regression_labels.npy'))
            test_cls = np.load(os.path.join(client_dir, 'test_classification_labels.npy'))
            test_phase = np.load(os.path.join(client_dir, 'test_phase_labels.npy'), allow_pickle=True)
            dataset = GasSensorWindowDataset(test_features, test_reg, test_cls, test_phase, normalize=normalize, mean_std=mean_std)
            test_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            # 使用客户端的最终本地模型评估
            eval_model = client.model
            eval_model.eval()

            # 1. 源域分类评估（使用软聚合）——始终执行，不依赖 USE_REG_LOSS
            from utils import evaluate_model_with_phase_and_soft_agg
            sem_protos = {k: v.detach().cpu() for k, v in server.semantic_protos.items()}
            # 马氏距离推理方差
            use_mahalanobis_src = getattr(config, 'USE_MAHALANOBIS_INFERENCE', False)
            proto_vars_src = {
                k: v.detach().cpu() for k, v in server.semantic_proto_vars.items()
            } if (use_mahalanobis_src and server.semantic_proto_vars) else None
            phase_result = evaluate_model_with_phase_and_soft_agg(
                eval_model, test_loader, config.DEVICE,
                semantic_protos=sem_protos,
                device_residuals=None,
                soft_agg_temp=config.SOFT_AGG_TEMPERATURE,
                prior_weight=config.SOFT_AGG_PRIOR_WEIGHT,
                num_classes=config.NUM_CLASSES,
                use_mahalanobis_inference=use_mahalanobis_src,
                semantic_proto_vars=proto_vars_src
            )
            source_cls_accs[cid] = phase_result['global']
            logger.info(f"Client {cid} (source) classification accuracy: {source_cls_accs[cid]:.4f}")

            # 2. 源域回归评估——仅在启用回归时执行
            if config.USE_REG_LOSS:
                from utils import evaluate_regression_metrics
                reg_metrics, reg_overall = evaluate_regression_metrics(
                    eval_model, test_loader, config.DEVICE, tolerance=0.1
                )
                source_regression_metrics[cid] = {
                    'per_class': reg_metrics,
                    'overall': reg_overall
                }
                logger.info(f"Client {cid} (source) regression: R²={reg_overall['R2']:.4f}, RMSE={reg_overall['RMSE']:.2f}")

                # 3. 源域回归可视化
                from utils import plot_concentration_feature_correlation, plot_regression_scatter
                plot_concentration_feature_correlation(
                    eval_model, test_loader, config.DEVICE, config.PLOT_SAVE_DIR,
                    filename=f"conc_feat_corr_source_client{cid}.png"
                )
                plot_regression_scatter(
                    eval_model, test_loader, config.DEVICE, config.PLOT_SAVE_DIR,
                    filename=f"reg_scatter_source_client{cid}.png", tolerance=0.1
                )
                logger.info(f"Source regression plots saved for client {cid}")

        # 将源域回归指标合并到总结果中
        if source_regression_metrics:
            regression_metrics_all['source'] = source_regression_metrics

        # 补丁1：当未启用模型选择时，用最终准确率填充best_test_accs和best_rounds
        if not args.model_selection:
            for cid in test_client_ids:
                if cid in final_accs:
                    best_test_accs[cid] = final_accs[cid]
                    best_rounds[cid] = config.GLOBAL_ROUNDS

        # 9. 保存结果与可视化
        save_and_visualize(args, config, exp_dir, history, weights_history, mmd_history, final_accs, peak_acc, peak_round, training_time, test_client_ids, temp_federated_dir, logger, best_test_accs, best_rounds, server, test_client_loaders, global_test_loader, regression_metrics_all, tta_client_ids, source_cls_accs, classification_metrics_all)

        # 10. 保存服务器检查点（用于后续验证）
        server_checkpoint = {
            'model_state': server.global_model.state_dict(),
            'shared_reg_state': getattr(server, 'shared_reg_state', {}),
            'protos': {k: v.data.cpu() for k, v in server.semantic_protos.items()},
            'config': config
        }
        torch.save(server_checkpoint, Path(config.MODEL_SAVE_DIR) / "server_checkpoint.pth")
        logger.info(f"Server checkpoint saved to {Path(config.MODEL_SAVE_DIR) / 'server_checkpoint.pth'}")

    finally:
        # 清理临时目录
        if temp_federated_dir and os.path.exists(temp_federated_dir):
            shutil.rmtree(temp_federated_dir)
            logger.info(f"Cleaned up temporary directory: {temp_federated_dir}")


if __name__ == "__main__":
    main()
