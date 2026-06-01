"""
数据调度器模块
实现时序漂移模拟：按早期→中期→晚期顺序渐进入场
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader
import logging
import torch

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 阶段字符串到整数的映射
PHASE_MAP = {'early': 0, 'middle': 1, 'late': 2}


class GasSensorPhaseDataset(Dataset):
    """
    气体传感器按阶段划分的数据集
    返回格式：(features, cls_labels, reg_labels, phase_labels) 四元组
    phase_labels 为整数 (0=early, 1=middle, 2=late)
    """

    def __init__(self, features: np.ndarray, cls_labels: np.ndarray, reg_labels: np.ndarray,
                 phase_labels: np.ndarray, phase: Optional[str] = None):
        """
        初始化数据集

        Args:
            features: 特征数组 (N, 100, 8)
            cls_labels: 分类标签 (N,)
            reg_labels: 回归标签 (N, 4)
            phase_labels: 阶段标签 (N,)，可以是字符串或整数
            phase: 指定阶段字符串 ('early'/'middle'/'late')，None表示使用所有阶段
        """
        # 将阶段标签转换为整数编码
        if phase_labels.dtype == object or phase_labels.dtype.kind in 'US':
            phase_labels = np.array([PHASE_MAP.get(str(p), 0) for p in phase_labels], dtype=np.int64)
        else:
            phase_labels = phase_labels.astype(np.int64)

        self.features = features
        self.cls_labels = cls_labels
        self.reg_labels = reg_labels
        self.phase_labels = phase_labels

        if phase is not None:
            target_phase = PHASE_MAP[phase]
            mask = self.phase_labels == target_phase
            self.features = self.features[mask]
            self.cls_labels = self.cls_labels[mask]
            self.reg_labels = self.reg_labels[mask]
            self.phase_labels = self.phase_labels[mask]
            if len(self.features) == 0:
                logger.warning(f"阶段 {phase} 没有样本")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        cls_label = torch.tensor(self.cls_labels[idx], dtype=torch.long)
        reg_label = torch.tensor(self.reg_labels[idx], dtype=torch.float32)
        phase_label = torch.tensor(self.phase_labels[idx], dtype=torch.long)
        return feature, cls_label, reg_label, phase_label


class GasSensorMultiPhaseDataset(Dataset):
    """
    气体传感器多阶段数据集
    支持同时加载多个阶段的数据
    返回格式：(features, cls_labels, reg_labels, phase_labels) 四元组
    """

    def __init__(self, features: np.ndarray, cls_labels: np.ndarray, reg_labels: np.ndarray,
                 phase_labels: np.ndarray, phase_list: Optional[List[str]] = None):
        """
        初始化数据集

        Args:
            features: 特征数组 (N, 100, 8)
            cls_labels: 分类标签 (N,)
            reg_labels: 回归标签 (N, 4)
            phase_labels: 阶段标签 (N,)
            phase_list: 阶段列表，如 ['early', 'middle']
        """
        # 将阶段标签转换为整数编码
        if phase_labels.dtype == object or phase_labels.dtype.kind in 'US':
            phase_labels = np.array([PHASE_MAP.get(str(p), 0) for p in phase_labels], dtype=np.int64)
        else:
            phase_labels = phase_labels.astype(np.int64)

        self.features = features
        self.cls_labels = cls_labels
        self.reg_labels = reg_labels
        self.phase_labels = phase_labels

        if phase_list:
            target_phases = [PHASE_MAP[p] for p in phase_list]
            mask = np.isin(self.phase_labels, target_phases)
            self.features = self.features[mask]
            self.cls_labels = self.cls_labels[mask]
            self.reg_labels = self.reg_labels[mask]
            self.phase_labels = self.phase_labels[mask]
            if len(self.features) == 0:
                logger.warning(f"阶段 {phase_list} 没有样本")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        cls_label = torch.tensor(self.cls_labels[idx], dtype=torch.long)
        reg_label = torch.tensor(self.reg_labels[idx], dtype=torch.float32)
        phase_label = torch.tensor(self.phase_labels[idx], dtype=torch.long)
        return feature, cls_label, reg_label, phase_label


class DataScheduler:
    """
    数据调度器
    根据当前联邦轮次，动态切换使用的数据阶段
    """

    def __init__(self, client_dirs: List[str], batch_size: int = 32,
                 phase_ratios: List[float] = [0.3, 0.4, 0.3], phase_fallback: bool = False, device: str = 'cpu'):
        """
        初始化数据调度器

        Args:
            client_dirs: 客户端数据目录列表
            batch_size: 批大小
            phase_ratios: 阶段轮次比例 [early, middle, late]
            phase_fallback: 是否启用阶段回退机制
            device: 设备类型
        """
        self.client_dirs = client_dirs
        self.batch_size = batch_size
        self.phase_ratios = phase_ratios
        self.phase_fallback = phase_fallback
        self.device = device
        self.client_data = {}

        # 加载所有客户端的数据
        for client_dir in client_dirs:
            client_id = Path(client_dir).name
            self._load_client_data(client_dir, client_id)

    def _load_client_data(self, client_dir: str, client_id: str):
        """加载单个客户端的数据"""
        try:
            features = np.load(Path(client_dir) / 'train_features.npy')
            cls_labels = np.load(Path(client_dir) / 'train_classification_labels.npy')
            reg_labels = np.load(Path(client_dir) / 'train_regression_labels.npy')
            phase_labels = np.load(Path(client_dir) / 'train_phase_labels.npy', allow_pickle=True)

            self.client_data[client_id] = {
                'features': features,
                'cls_labels': cls_labels,
                'reg_labels': reg_labels,
                'phase_labels': phase_labels
            }
            logger.info(f"加载客户端 {client_id} 数据完成")
        except Exception as e:
            logger.error(f"加载客户端 {client_id} 数据失败: {e}")

    def get_client_loaders(self, current_round: int, total_rounds: int) -> Dict[str, DataLoader]:
        """
        根据当前轮次获取客户端数据加载器

        Args:
            current_round: 当前轮次（从0开始）
            total_rounds: 总轮次

        Returns:
            Dict[str, DataLoader]: 客户端数据加载器字典，键为客户端ID
        """
        phase = self._get_current_phase(current_round, total_rounds)
        logger.info(f"当前轮次 {current_round}/{total_rounds}，使用阶段: {phase}")

        client_loaders = {}
        for client_id in sorted(self.client_data.keys()):
            data = self.client_data[client_id]
            dataset = GasSensorPhaseDataset(
                data['features'],
                data['cls_labels'],
                data['reg_labels'],
                data['phase_labels'],
                phase
            )

            if len(dataset) == 0 and self.phase_fallback:
                logger.info(f"客户端 {client_id} 阶段 {phase} 没有样本，尝试回退机制")
                dataset = self._fallback_to_previous_phase(data, phase, client_id)

            if len(dataset) == 0:
                logger.info(f"客户端 {client_id} 无数据，本轮不参与训练")
                continue
            else:
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0
                )
                client_loaders[client_id] = loader

        return client_loaders

    def _fallback_to_previous_phase(self, data: Dict, phase: str, client_id: str) -> GasSensorPhaseDataset:
        """阶段回退机制：当指定阶段无样本时，逐级回退到前一阶段"""
        if phase == 'middle':
            logger.info(f"客户端 {client_id} 阶段 {phase} 没有样本，使用 early 阶段")
            return GasSensorPhaseDataset(
                data['features'], data['cls_labels'], data['reg_labels'],
                data['phase_labels'], 'early'
            )
        elif phase == 'late':
            logger.info(f"客户端 {client_id} 阶段 {phase} 没有样本，尝试 middle 阶段")
            dataset = GasSensorPhaseDataset(
                data['features'], data['cls_labels'], data['reg_labels'],
                data['phase_labels'], 'middle'
            )
            if len(dataset) == 0:
                logger.info(f"客户端 {client_id} 阶段 middle 也没有样本，使用 early 阶段")
                dataset = GasSensorPhaseDataset(
                    data['features'], data['cls_labels'], data['reg_labels'],
                    data['phase_labels'], 'early'
                )
            return dataset
        else:
            return GasSensorPhaseDataset(
                data['features'], data['cls_labels'], data['reg_labels'],
                data['phase_labels'], 'early'
            )

    def _get_current_phase(self, current_round: int, total_rounds: int) -> str:
        """根据当前轮次获取当前阶段"""
        early_end = int(total_rounds * self.phase_ratios[0])
        middle_end = int(total_rounds * (self.phase_ratios[0] + self.phase_ratios[1]))

        # 检查边界条件
        if early_end <= 0:
            logger.warning(f"early_end 计算结果为 {early_end}，可能导致早期阶段轮次不足")
        if middle_end <= early_end:
            logger.warning(f"middle_end ({middle_end}) 小于或等于 early_end ({early_end})，可能导致中期阶段轮次不足")
        if middle_end >= total_rounds:
            logger.warning(f"middle_end ({middle_end}) 大于或等于总轮次 ({total_rounds})，可能导致晚期阶段轮次不足")

        if current_round < early_end:
            return 'early'
        elif current_round < middle_end:
            return 'middle'
        else:
            return 'late'

    def get_phase_stats(self) -> Dict[str, Dict[str, int]]:
        """获取各客户端各阶段的数据统计"""
        stats = {}
        for client_id, data in self.client_data.items():
            # 转换阶段标签为整数以便统计
            phase_labels = data['phase_labels']
            if phase_labels.dtype == object or phase_labels.dtype.kind in 'US':
                phase_labels = np.array([PHASE_MAP.get(str(p), 0) for p in phase_labels], dtype=np.int64)
            else:
                phase_labels = phase_labels.astype(np.int64)

            phase_counts = {
                'early': int(np.sum(phase_labels == 0)),
                'middle': int(np.sum(phase_labels == 1)),
                'late': int(np.sum(phase_labels == 2))
            }
            stats[client_id] = phase_counts
        return stats


class CumulativeDataScheduler(DataScheduler):
    """
    累积数据调度器
    实现数据累积策略：早期阶段使用early数据；中期阶段使用early+middle；晚期阶段使用所有数据
    模拟传感器持续运行场景，历史数据一直可用，但新数据分布发生变化
    """

    def __init__(self, client_dirs: List[str], batch_size: int = 32,
                 phase_ratios: List[float] = [0.3, 0.4, 0.3], device: str = 'cpu'):
        super().__init__(client_dirs, batch_size, phase_ratios, phase_fallback=False, device=device)

    def get_client_loaders(self, current_round: int, total_rounds: int) -> List[DataLoader]:
        """根据当前轮次获取客户端数据加载器（累积模式）"""
        phase_list = self._get_cumulative_phases(current_round, total_rounds)
        logger.info(f"当前轮次 {current_round}/{total_rounds}，使用阶段: {phase_list}")

        client_loaders = []
        for client_id in sorted(self.client_data.keys()):
            data = self.client_data[client_id]
            dataset = GasSensorMultiPhaseDataset(
                data['features'], data['cls_labels'], data['reg_labels'],
                data['phase_labels'], phase_list
            )

            if len(dataset) == 0:
                logger.warning(f"客户端 {client_id} 没有可用数据，跳过该客户端")
                continue  # 跳过空数据集的客户端
            else:
                logger.info(f"客户端 {client_id} 加载了 {len(dataset)} 个样本")

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )
            client_loaders.append((client_id, loader))

        client_loaders.sort(key=lambda x: x[0])
        return {client_id: loader for client_id, loader in client_loaders}


    def _get_cumulative_phases(self, current_round: int, total_rounds: int) -> List[str]:
        """根据当前轮次获取累积阶段列表"""
        early_end = int(total_rounds * self.phase_ratios[0])
        middle_end = int(total_rounds * (self.phase_ratios[0] + self.phase_ratios[1]))

        if current_round < early_end:
            return ['early']
        elif current_round < middle_end:
            return ['early', 'middle']
        else:
            return ['early', 'middle', 'late']


def create_data_scheduler(data_dir: str, batch_size: int = 32,
                          phase_ratios: List[float] = [0.3, 0.4, 0.3],
                          phase_fallback: bool = False, device: str = 'cpu',
                          use_cumulative: bool = False) -> DataScheduler:
    """
    创建数据调度器

    Args:
        data_dir: 数据目录
        batch_size: 批大小
        phase_ratios: 阶段轮次比例 [early, middle, late]
        phase_fallback: 是否启用阶段回退机制
        device: 设备类型
        use_cumulative: 是否使用累积数据调度器

    Returns:
        DataScheduler: 数据调度器实例
    """
    client_dirs = []
    data_path = Path(data_dir)
    # 自动检测所有客户端目录
    for client_dir in data_path.glob('client_*'):
        if client_dir.is_dir():
            client_dirs.append(str(client_dir))
    
    if not client_dirs:
        raise FileNotFoundError(f"No client directories found in {data_dir}")

    if use_cumulative:
        return CumulativeDataScheduler(client_dirs, batch_size, phase_ratios, device)
    else:
        return DataScheduler(client_dirs, batch_size, phase_ratios, phase_fallback, device)