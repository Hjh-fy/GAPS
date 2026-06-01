"""
联邦学习数据加载器（时序窗口版本）
支持客户端训练集、全局测试集、校准集、Unit5测试集
输入形状：(batch, 100, 8)
输出：(features, (regression_labels, classification_labels))
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class GasSensorWindowDataset(Dataset):
    """时序窗口数据集"""
    def __init__(self,
                 features: np.ndarray,
                 regression_labels: Optional[np.ndarray] = None,
                 classification_labels: Optional[np.ndarray] = None,
                 phase_labels: Optional[np.ndarray] = None,
                 transform=None,
                 normalize: bool = True,
                 mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Args:
            features: (N, 100, 8)
            regression_labels: (N, 4)
            classification_labels: (N,)
            phase_labels: (N,) 阶段标签
            transform: 数据增强函数，接收并返回 numpy 数组
            normalize: 是否使用 Z-score 标准化
            mean_std: (mean, std) 用于标准化，当 normalize=True 时必须传入
        """
        self.features = features.astype(np.float32)
        self.regression_labels = regression_labels.astype(np.float32) if regression_labels is not None else None
        self.classification_labels = classification_labels.astype(np.int64) if classification_labels is not None else None
        
        # 处理 phase_labels
        if phase_labels is not None:
            # 尝试转换为整数
            try:
                phase_labels = phase_labels.astype(np.int64)
            except ValueError:
                # 如果转换失败，说明是字符串类型
                phase_map = {'early': 0, 'middle': 1, 'late': 2, '0': 0, '1': 1, '2': 2, '-1': -1}
                # 处理数组中的每个元素
                if isinstance(phase_labels, np.ndarray):
                    phase_labels = np.array([phase_map.get(str(p), -1) for p in phase_labels], dtype=np.int64)
                else:
                    # 单个值的情况
                    phase_labels = np.array([phase_map.get(str(phase_labels), -1)], dtype=np.int64)
        self.phase_labels = phase_labels
        
        self.transform = transform

        if normalize:
            # 强制要求normalize=True时必须传入mean_std
            if mean_std is None:
                raise ValueError("Normalization requires mean_std from training set")
            mean, std = mean_std
            self.features = (self.features - mean) / std
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]  # (100, 8)
        if self.transform:
            x = self.transform(x)
        x = torch.from_numpy(x).float()

        if self.regression_labels is not None and self.classification_labels is not None:
            reg = torch.from_numpy(self.regression_labels[idx]).float()
            cls = torch.tensor(self.classification_labels[idx], dtype=torch.long)
            if self.phase_labels is not None:
                phase = torch.tensor(self.phase_labels[idx], dtype=torch.long)
            else:
                # 如果没有阶段标签，返回-1作为标记
                phase = torch.tensor(-1, dtype=torch.long)
            return x, cls, reg, phase  # 四元组格式：特征, 分类, 回归, 阶段
        elif self.classification_labels is not None:
            cls = torch.tensor(self.classification_labels[idx], dtype=torch.long)
            reg = torch.zeros(4, dtype=torch.float)  # 默认回归标签
            if self.phase_labels is not None:
                phase = torch.tensor(self.phase_labels[idx], dtype=torch.long)
            else:
                phase = torch.tensor(-1, dtype=torch.long)
            return x, cls, reg, phase
        elif self.regression_labels is not None:
            reg = torch.from_numpy(self.regression_labels[idx]).float()
            cls = torch.tensor(0, dtype=torch.long)  # 默认分类标签
            if self.phase_labels is not None:
                phase = torch.tensor(self.phase_labels[idx], dtype=torch.long)
            else:
                phase = torch.tensor(-1, dtype=torch.long)
            return x, cls, reg, phase
        else:
            cls = torch.tensor(0, dtype=torch.long)  # 默认分类标签
            reg = torch.zeros(4, dtype=torch.float)  # 默认回归标签
            if self.phase_labels is not None:
                phase = torch.tensor(self.phase_labels[idx], dtype=torch.long)
            else:
                phase = torch.tensor(-1, dtype=torch.long)
            return x, cls, reg, phase


def create_train_loader(client_dir: Union[str, Path],
                        batch_size: int = 32,
                        shuffle: bool = True,
                        transform=None,
                        normalize: bool = True,
                        mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        num_workers: int = 0) -> DataLoader:
    """创建客户端训练数据加载器"""
    client_path = Path(client_dir)
    
    # 尝试从父目录加载归一化统计量
    if mean_std is None:
        norm_stats_path = client_path.parent / "norm_stats.npz"
        if norm_stats_path.exists():
            norm_data = np.load(norm_stats_path)
            mean = norm_data['mean']
            std = norm_data['std']
            mean_std = (mean, std)
    
    features = np.load(client_path / "train_features.npy")
    regression_labels = np.load(client_path / "train_regression_labels.npy")
    classification_labels = np.load(client_path / "train_classification_labels.npy")
    # 尝试加载 phase_labels，如果存在的话
    phase_labels = None
    phase_path = client_path / "train_phase_labels.npy"
    if phase_path.exists():
        phase_labels = np.load(phase_path)
        # 将字符串类型的 phase_labels 转换为整数
        if phase_labels.dtype == object:
            phase_map = {'early': 0, 'middle': 1, 'late': 2, '0': 0, '1': 1, '2': 2, '-1': -1}
            phase_labels = np.array([phase_map.get(str(p), -1) for p in phase_labels], dtype=np.int64)
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=regression_labels,
        classification_labels=classification_labels,
        phase_labels=phase_labels,
        transform=transform,
        normalize=normalize,
        mean_std=mean_std
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def create_client_full_test_loader(client_dir: Union[str, Path],
                                 batch_size: int = 32,
                                 mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                 num_workers: int = 0) -> DataLoader:
    """创建单个客户端的完整测试集加载器（包含训练集、测试集和校准集）
    
    Args:
        client_dir: 客户端目录
        batch_size: 批次大小
        mean_std: 标准化参数
        num_workers: 工作线程数
        
    Returns:
        DataLoader: 包含该客户端全部数据的测试集加载器
    """
    client_path = Path(client_dir)
    
    # 尝试从父目录加载归一化统计量
    if mean_std is None:
        norm_stats_path = client_path.parent / "norm_stats.npz"
        if norm_stats_path.exists():
            norm_data = np.load(norm_stats_path)
            mean = norm_data['mean']
            std = norm_data['std']
            mean_std = (mean, std)
    
    all_features, all_regression_labels, all_classification_labels, all_phase_labels = [], [], [], []
    
    # 加载训练数据
    if (client_path / "train_features.npy").exists():
        features = np.load(client_path / "train_features.npy")
        regression_labels = np.load(client_path / "train_regression_labels.npy")
        classification_labels = np.load(client_path / "train_classification_labels.npy")
        all_features.append(features)
        all_regression_labels.append(regression_labels)
        all_classification_labels.append(classification_labels)
        # 加载阶段标签
        if (client_path / "train_phase_labels.npy").exists():
            phase_labels = np.load(client_path / "train_phase_labels.npy")
            all_phase_labels.append(phase_labels)
    
    # 加载测试集数据
    if (client_path / "test_features.npy").exists():
        features = np.load(client_path / "test_features.npy")
        regression_labels = np.load(client_path / "test_regression_labels.npy")
        classification_labels = np.load(client_path / "test_classification_labels.npy")
        all_features.append(features)
        all_regression_labels.append(regression_labels)
        all_classification_labels.append(classification_labels)
        # 加载阶段标签
        if (client_path / "test_phase_labels.npy").exists():
            phase_labels = np.load(client_path / "test_phase_labels.npy")
            all_phase_labels.append(phase_labels)
    
    # 加载校准集数据
    if (client_path / "calibration_features.npy").exists():
        features = np.load(client_path / "calibration_features.npy")
        regression_labels = np.load(client_path / "calibration_regression_labels.npy")
        classification_labels = np.load(client_path / "calibration_classification_labels.npy")
        all_features.append(features)
        all_regression_labels.append(regression_labels)
        all_classification_labels.append(classification_labels)
        # 加载阶段标签
        if (client_path / "calibration_phase_labels.npy").exists():
            phase_labels = np.load(client_path / "calibration_phase_labels.npy")
            all_phase_labels.append(phase_labels)
    
    # 合并数据
    if not all_features:
        raise FileNotFoundError(f"客户端目录 {client_dir} 中没有找到任何数据文件")
    
    features = np.concatenate(all_features, axis=0)
    regression_labels = np.concatenate(all_regression_labels, axis=0)
    classification_labels = np.concatenate(all_classification_labels, axis=0)
    
    # 合并阶段标签
    phase_labels = None
    if all_phase_labels:
        phase_labels = np.concatenate(all_phase_labels, axis=0)
    
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=regression_labels,
        classification_labels=classification_labels,
        phase_labels=phase_labels,
        transform=None,
        normalize=False,
        mean_std=mean_std
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def create_client_test_only_loader(client_dir: Union[str, Path],
                                   batch_size: int = 32,
                                   mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                   num_workers: int = 0) -> DataLoader:
    """创建单个客户端的严格测试集加载器（仅 test split）。

    与 create_client_full_test_loader 不同，本函数只读取 test_*.npy，
    不混入 train/calibration split，用于有标签校准后的最终无泄露评估。
    """
    client_path = Path(client_dir)

    if mean_std is None:
        norm_stats_path = client_path.parent / "norm_stats.npz"
        if norm_stats_path.exists():
            norm_data = np.load(norm_stats_path)
            mean = norm_data['mean']
            std = norm_data['std']
            mean_std = (mean, std)

    feat_path = client_path / "test_features.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"客户端目录 {client_dir} 中没有找到 test_features.npy")

    features = np.load(feat_path)
    regression_labels = np.load(client_path / "test_regression_labels.npy")
    classification_labels = np.load(client_path / "test_classification_labels.npy")

    phase_path = client_path / "test_phase_labels.npy"
    if phase_path.exists():
        phase_labels = np.load(phase_path, allow_pickle=True)
    else:
        phase_labels = np.full(len(features), -1, dtype=np.int64)

    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=regression_labels,
        classification_labels=classification_labels,
        phase_labels=phase_labels,
        transform=None,
        normalize=False,
        mean_std=mean_std
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def create_merged_test_loader(
    client_dirs: Union[List[str], List[Path]],
    batch_size: int = 32,
    mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    从多个客户端目录中合并所有测试数据，形成全局测试集（包含阶段标签）
    """
    # 尝试从第一个客户端的父目录加载归一化统计量
    if mean_std is None and client_dirs:
        first_client_path = Path(client_dirs[0])
        norm_stats_path = first_client_path.parent / "norm_stats.npz"
        if norm_stats_path.exists():
            norm_data = np.load(norm_stats_path)
            mean = norm_data['mean']
            std = norm_data['std']
            mean_std = (mean, std)
    
    all_features, all_reg_labels, all_cls_labels, all_phase_labels = [], [], [], []
    for client_dir in client_dirs:
        client_path = Path(client_dir)
        test_feat = client_path / "test_features.npy"
        if not test_feat.exists():
            continue
        features = np.load(test_feat)
        reg_labels = np.load(client_path / "test_regression_labels.npy")
        cls_labels = np.load(client_path / "test_classification_labels.npy")
        
        # 加载阶段标签（如果存在）
        phase_path = client_path / "test_phase_labels.npy"
        if phase_path.exists():
            phase_labels = np.load(phase_path, allow_pickle=True)
        else:
            # 如果没有阶段标签文件，创建默认的阶段标签数组（全部设为-1）
            # 这样可以确保所有客户端都有阶段标签，避免后续合并时的问题
            phase_labels = np.full(len(features), -1, dtype=np.int64)
        
        all_features.append(features)
        all_reg_labels.append(reg_labels)
        all_cls_labels.append(cls_labels)
        all_phase_labels.append(phase_labels)

    if not all_features:
        raise ValueError("No test data found in given client directories.")

    features = np.concatenate(all_features, axis=0)
    reg_labels = np.concatenate(all_reg_labels, axis=0)
    cls_labels = np.concatenate(all_cls_labels, axis=0)
    phase_labels = np.concatenate(all_phase_labels, axis=0) if all_phase_labels else None

    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=reg_labels,
        classification_labels=cls_labels,
        phase_labels=phase_labels,          # 关键：传递阶段标签
        normalize=False,
        mean_std=mean_std
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


def create_merged_calibration_loader(
    client_dirs: Union[List[str], List[Path]],
    batch_size: int = 32,
    mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    num_workers: int = 0
) -> DataLoader:
    """
    从多个客户端目录中合并所有校准数据，形成全局校准集（包含阶段标签）
    """
    all_features, all_reg_labels, all_cls_labels, all_phase_labels = [], [], [], []
    for client_dir in client_dirs:
        client_path = Path(client_dir)
        cal_feat = client_path / "calibration_features.npy"
        if not cal_feat.exists():
            continue
        features = np.load(cal_feat)
        reg_labels = np.load(client_path / "calibration_regression_labels.npy")
        cls_labels = np.load(client_path / "calibration_classification_labels.npy")
        
        # 加载阶段标签（如果存在）
        phase_path = client_path / "calibration_phase_labels.npy"
        if phase_path.exists():
            phase_labels = np.load(phase_path, allow_pickle=True)
        else:
            # 如果没有阶段标签文件，创建默认的阶段标签数组（全部设为-1）
            # 这样可以确保所有客户端都有阶段标签，避免后续合并时的问题
            phase_labels = np.full(len(features), -1, dtype=np.int64)
        
        all_features.append(features)
        all_reg_labels.append(reg_labels)
        all_cls_labels.append(cls_labels)
        all_phase_labels.append(phase_labels)

    if not all_features:
        raise ValueError("No calibration data found in given client directories.")

    features = np.concatenate(all_features, axis=0)
    reg_labels = np.concatenate(all_reg_labels, axis=0)
    cls_labels = np.concatenate(all_cls_labels, axis=0)
    phase_labels = np.concatenate(all_phase_labels, axis=0) if all_phase_labels else None

    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=reg_labels,
        classification_labels=cls_labels,
        phase_labels=phase_labels,
        normalize=False,
        mean_std=mean_std
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


# 不使用
def create_global_test_loader_with_phase(data_dir: Union[str, Path],
                                       batch_size: int = 32,
                                       mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                       num_workers: int = 0) -> DataLoader:
    """创建包含阶段标签的全局测试集加载器"""
    data_path = Path(data_dir) / "global_test"
    features = np.load(data_path / "features.npy")
    regression_labels = np.load(data_path / "regression_labels.npy")
    classification_labels = np.load(data_path / "classification_labels.npy")
    # 尝试加载阶段标签
    phase_labels = None
    phase_path = data_path / "phase_labels.npy"
    if phase_path.exists():
        phase_labels = np.load(phase_path)
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=regression_labels,
        classification_labels=classification_labels,
        phase_labels=phase_labels,
        transform=None,
        normalize=False,
        mean_std=mean_std
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

# 不使用
def create_calibration_loader(data_dir: Union[str, Path],
                              batch_size: int = 32,
                              mean_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                              num_workers: int = 0) -> DataLoader:
    """创建校准集加载器"""
    data_path = Path(data_dir) / "calibration"
    features = np.load(data_path / "features.npy")
    regression_labels = np.load(data_path / "regression_labels.npy")
    classification_labels = np.load(data_path / "classification_labels.npy")
    dataset = GasSensorWindowDataset(
        features=features,
        regression_labels=regression_labels,
        classification_labels=classification_labels,
        transform=None,
        normalize=False,
        mean_std=mean_std
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)

