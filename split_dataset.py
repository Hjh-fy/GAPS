import sys
import numpy as np
import json
import random
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =========================
# 枚举类型定义
# =========================

class SplitMethod(Enum):
    UNIT_BASED = "unit_based"
    IID = "iid"
    GAS_BASED = "gas_based"


class GlobalTestSampling(Enum):
    STRATIFIED = "stratified"
    RANDOM = "random"
    PROPORTIONAL = "proportional"


# =========================
# 配置类定义
# =========================

@dataclass
class FederatedSplitConfig:
    """联邦学习数据集三路划分配置类
    定义了数据集划分的参数，包括训练集、全局测试集、校准集的比例，
    以及划分方法、随机种子、是否分层采样、最小样本数等。
    """
    train_ratio: float = 0.70
    global_test_ratio: float = 0.20
    calibration_ratio: float = 0.10
    split_method: SplitMethod = SplitMethod.UNIT_BASED
    seed: int = 42
    shuffle: bool = True
    stratify: bool = True
    min_samples_per_client: int = 100
    global_test_sampling: GlobalTestSampling = GlobalTestSampling.STRATIFIED
    ensure_balance: bool = True
    min_samples_per_class: int = 20
    target_samples_per_class: int = 940
    save_global_test: bool = True
    save_visualizations: bool = True
    verbose: bool = True
    # Regression-FullGrid split: prioritize full concentration-grid coverage.
    # Role-aware mode is useful for experiments such as source clients 4/5 and
    # target clients 1/2/3. Source clients keep every concentration in train;
    # target clients keep every concentration in calibration and test when
    # enough repeated files exist.
    regression_full_grid: bool = False
    # Window-level FullGrid is an upper-bound/diagnostic split. It stratifies
    # individual windows by class/concentration, so the same source file may
    # appear in train/calibration/test. Do not use as the strict main result.
    regression_window_full_grid: bool = False
    full_grid_source_clients: Optional[List[int]] = None
    full_grid_target_clients: Optional[List[int]] = None
    
    def __post_init__(self):
        self._validate_ratios()
        self._normalize_methods()
    
    def _validate_ratios(self):
        total = self.train_ratio + self.global_test_ratio + self.calibration_ratio
        if abs(total - 1.0) > 1e-10:
            raise ValueError(f"比例之和必须为1.0，当前为{total:.4f}")
        for name, ratio in [("训练集", self.train_ratio), ("全局测试集", self.global_test_ratio), ("校准集", self.calibration_ratio)]:
            if ratio <= 0 or ratio >= 1:
                raise ValueError(f"{name}比例必须在0和1之间，当前为{ratio}")
    
    def _normalize_methods(self):
        if isinstance(self.split_method, str):
            self.split_method = SplitMethod(self.split_method)
        if isinstance(self.global_test_sampling, str):
            self.global_test_sampling = GlobalTestSampling(self.global_test_sampling)
    
    def to_dict(self):
        return {
            "train_ratio": self.train_ratio,
            "global_test_ratio": self.global_test_ratio,
            "calibration_ratio": self.calibration_ratio,
            "split_method": self.split_method.value,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "stratify": self.stratify,
            "min_samples_per_client": self.min_samples_per_client,
            "global_test_sampling": self.global_test_sampling.value,
            "ensure_balance": self.ensure_balance,
            "min_samples_per_class": self.min_samples_per_class,
            "target_samples_per_class": self.target_samples_per_class,
            "save_global_test": self.save_global_test,
            "save_visualizations": self.save_visualizations,
            "verbose": self.verbose,
            "regression_full_grid": self.regression_full_grid,
            "regression_window_full_grid": self.regression_window_full_grid,
            "full_grid_source_clients": self.full_grid_source_clients,
            "full_grid_target_clients": self.full_grid_target_clients
        }


# =========================
# 联邦数据集划分器
# =========================

class FederatedDatasetSplitter:
    """联邦学习数据集划分器
    
    负责将预处理后的数据集划分为联邦学习格式，支持按单元、IID和气体类型划分
    每个客户端数据按比例划分为训练集、测试集和校准集，并构建全局测试集和校准集
    """
    def __init__(self, config: Optional[FederatedSplitConfig] = None):
        """初始化数据集划分器
        
        Args:
            config: 划分配置，若为None则使用默认配置
        """
        self.config = config or FederatedSplitConfig()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.clients_data = {}
        self.global_test_set = None
        self.stats = {}
        if self.config.verbose:
            print("=" * 60)
            print("联邦学习数据集划分器初始化完成")
            print("=" * 60)
            print(f"划分方法: {self.config.split_method.value}")
            print(f"训练集: {self.config.train_ratio:.1%}")
            print(f"全局测试集: {self.config.global_test_ratio:.1%}")
            print(f"校准集: {self.config.calibration_ratio:.1%}")
            print(f"随机种子: {self.config.seed}")
            print("=" * 60)
    
    def load_processed_data(self, processed_dir: str) -> Dict[int, Dict[str, Any]]:
        """加载预处理后的数据集
        
        从指定目录加载预处理后的数据集，包括特征、标签和实验信息
        
        Args:
            processed_dir: 预处理数据目录
            
        Returns:
            单元数据字典，键为单元ID，值为包含数据和信息的字典
        """
        processed_path = Path(processed_dir)
        if not processed_path.exists():
            raise FileNotFoundError(f"预处理数据目录不存在: {processed_dir}")
        unit_dirs = list(processed_path.glob("unit_*"))
        if not unit_dirs:
            raise FileNotFoundError(f"在目录 {processed_dir} 中未找到单元数据")
        unit_data = {}
        for unit_dir in unit_dirs:
            try:
                unit_id = int(unit_dir.name.split("_")[1])
                required_files = {
                    "features": unit_dir / "features.npy",
                    "regression_labels": unit_dir / "regression_labels.npy",
                    "classification_labels": unit_dir / "classification_labels.npy",
                    "phase_labels": unit_dir / "phase_labels.npy"
                }
                missing_files = [name for name, filepath in required_files.items() if not filepath.exists()]
                if missing_files:
                    if self.config.verbose:
                        warnings.warn(f"单元 {unit_id} 缺少文件: {missing_files}")
                    continue
                features = np.load(required_files["features"])
                regression_labels = np.load(required_files["regression_labels"])
                classification_labels = np.load(required_files["classification_labels"])
                phase_labels = np.load(required_files["phase_labels"])
                experiment_info = []
                info_file = unit_dir / "experiment_info.json"
                if info_file.exists():
                    with open(info_file, "r", encoding='utf-8') as f:
                        experiment_info = json.load(f)
                n_samples = len(features)
                if len(regression_labels) != n_samples or len(classification_labels) != n_samples or len(phase_labels) != n_samples:
                    warnings.warn(f"单元 {unit_id} 数据维度不一致，跳过")
                    continue
                if features.shape[1:] != (100, 8):
                    warnings.warn(f"单元 {unit_id} 特征形状异常: {features.shape}")
                unit_data[unit_id] = {
                    "features": features, "regression_labels": regression_labels,
                    "classification_labels": classification_labels, "phase_labels": phase_labels,
                    "experiment_info": experiment_info, "n_samples": n_samples, "unit_id": unit_id
                }
                if self.config.verbose:
                    print(f" 加载单元 {unit_id}: {n_samples:,} 个样本")
            except Exception as e:
                print(f" 加载单元 {unit_dir.name} 数据失败: {e}")
                continue
        if not unit_data:
            raise ValueError("没有成功加载任何单元数据")
        total_samples = sum(data["n_samples"] for data in unit_data.values())
        if self.config.verbose:
            print(f"\n 数据加载完成: {len(unit_data)} 个单元, {total_samples:,} 个总样本")
            for unit_id, data in unit_data.items():
                print(f"   单元 {unit_id}: {data['n_samples']:,} 样本")
        return unit_data
    
    def create_federated_dataset(self, unit_data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """创建联邦学习数据集
        
        根据配置的划分方法，将单元数据划分为客户端数据，并构建全局测试集和校准集
        
        Args:
            unit_data: 单元数据字典
            
        Returns:
            包含客户端数据、全局测试集、校准集和统计信息的字典
        """
        if self.config.verbose:
            print(f"\n 开始创建联邦学习数据集")
            print(f"   使用 {self.config.split_method.value} 划分策略")
        if self.config.split_method == SplitMethod.UNIT_BASED:
            clients_data, global_contributions, calibration_contributions = self._split_by_unit(unit_data)
        elif self.config.split_method == SplitMethod.IID:
            clients_data, global_contributions, calibration_contributions = self._split_iid(unit_data)
        elif self.config.split_method == SplitMethod.GAS_BASED:
            clients_data, global_contributions, calibration_contributions = self._split_by_gas(unit_data)
        else:
            raise ValueError(f"不支持的划分方法: {self.config.split_method}")
        if global_contributions:
            self.global_test_set = self._build_global_test_set(global_contributions)
        if calibration_contributions:
            self.calibration_set = self._build_calibration_set(calibration_contributions)
        self.clients_data = clients_data
        self._compute_overall_statistics()
        self._print_statistics_summary()
        return {
            "clients_data": clients_data,
            "global_test_set": self.global_test_set,
            "calibration_set": getattr(self, 'calibration_set', None),
            "config": self.config.to_dict(),
            "stats": self.stats
        }
    
    def _split_by_unit(self, unit_data: Dict[int, Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """按单元划分数据集
        
        将每个单元的数据分配给一个客户端，每个客户端数据按比例划分为训练集、测试集和校准集
        
        Args:
            unit_data: 单元数据字典
            
        Returns:
            客户端数据字典、全局测试贡献列表、校准集贡献列表
        """
        clients_data = {}
        all_global_contributions = []
        all_calibration_contributions = []
        for unit_id, data in unit_data.items():
            if self.config.verbose:
                print(f"\n 处理单元 {unit_id} -> 客户端 {unit_id}")
                print(f"   原始样本数: {data['n_samples']:,}")
            if data["n_samples"] < self.config.min_samples_per_client:
                warnings.warn(f"单元 {unit_id} 只有 {data['n_samples']} 个样本，少于最小要求 {self.config.min_samples_per_client}")
            client_data, global_contribution, calibration_contribution = self._split_single_client(
                features=data["features"], regression_labels=data["regression_labels"],
                classification_labels=data["classification_labels"], phase_labels=data["phase_labels"],
                experiment_info=data["experiment_info"], source_info={"unit_id": unit_id, "client_id": unit_id}
            )
            client_data["unit_id"] = unit_id
            client_data["source_info"] = {"unit_id": unit_id, "original_samples": data["n_samples"]}
            clients_data[unit_id] = client_data
            if global_contribution:
                all_global_contributions.append(global_contribution)
            if calibration_contribution:
                all_calibration_contributions.append(calibration_contribution)
            stats = client_data["stats"]
            if self.config.verbose:
                print(f"    划分完成:")
                print(f"      训练集: {stats['n_train']:,} 样本 ({stats['train_ratio']:.1%})")
                print(f"      测试集: {stats['n_test']:,} 样本 ({stats['test_ratio']:.1%})")
                print(f"      校准集: {stats['n_calibration']:,} 样本 ({stats['calibration_ratio']:.1%})")
        return clients_data, all_global_contributions, all_calibration_contributions
    
    def _normalize_phase_labels(self, phase_labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert string/int phase labels to {-1,0,1,2}."""
        if phase_labels is None:
            return None
        phase_map = {'early': 0, 'middle': 1, 'late': 2, '0': 0, '1': 1, '2': 2, '-1': -1}
        result = []
        for value in np.asarray(phase_labels).reshape(-1):
            try:
                result.append(int(value))
            except Exception:
                result.append(phase_map.get(str(value), -1))
        return np.asarray(result, dtype=np.int64)

    def _evenly_spaced_values(self, values: List[float], quota: int) -> List[float]:
        """Return values ordered to cover the full concentration range early."""
        if quota <= 0 or not values:
            return []
        ordered = sorted(values)
        if quota >= len(ordered):
            return ordered
        positions = np.linspace(0, len(ordered) - 1, quota)
        picked_indices = []
        for pos in positions:
            idx = int(round(pos))
            while idx in picked_indices and idx + 1 < len(ordered):
                idx += 1
            while idx in picked_indices and idx - 1 >= 0:
                idx -= 1
            if idx not in picked_indices:
                picked_indices.append(idx)
        picked = [ordered[i] for i in picked_indices]
        remaining = [v for v in ordered if v not in picked]
        return picked + remaining

    def _pop_balanced_files(self, conc_to_files: Dict[float, List[str]], quota: int,
                            preferred_concs: Optional[List[float]] = None) -> List[str]:
        """Pop files while spreading selections across concentration levels."""
        selected = []
        if quota <= 0:
            return selected

        preferred_concs = preferred_concs or []
        for conc in preferred_concs:
            if len(selected) >= quota:
                break
            bucket = conc_to_files.get(conc, [])
            if bucket:
                selected.append(bucket.pop())

        while len(selected) < quota:
            available = [conc for conc, files in conc_to_files.items() if files]
            if not available:
                break
            made_progress = False
            for conc in self._evenly_spaced_values(available, len(available)):
                if len(selected) >= quota:
                    break
                bucket = conc_to_files.get(conc, [])
                if bucket:
                    selected.append(bucket.pop())
                    made_progress = True
            if not made_progress:
                break
        return selected

    def _group_files_by_class_concentration(
        self,
        file_groups: Dict[str, List[int]],
        classification_labels: np.ndarray,
        regression_labels: np.ndarray
    ) -> Dict[int, Dict[float, List[str]]]:
        """Group file names by dominant gas class and concentration."""
        by_class_conc: Dict[int, Dict[float, List[str]]] = {}
        for filename, indices in file_groups.items():
            idx = np.asarray(indices, dtype=np.int64)
            labels = classification_labels[idx].astype(np.int64)
            if labels.size == 0:
                continue
            counts = np.bincount(labels, minlength=4)
            cls = int(np.argmax(counts))
            conc = float(np.median(regression_labels[idx, cls]))
            by_class_conc.setdefault(cls, {}).setdefault(conc, []).append(filename)
        return by_class_conc

    def _split_files_regression_full_grid(
        self,
        file_groups: Dict[str, List[int]],
        classification_labels: np.ndarray,
        regression_labels: np.ndarray,
        source_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        """Role-aware FullGrid split for regression experiments.

        Source clients:
          - Keep at least one file for every class/concentration in train.
          - With 4 repeats, also keep one calibration and one test file per
            concentration; with 2 repeats, split the remaining repeat between
            test/calibration across concentrations so neither split is empty.

        Target clients:
          - Keep one calibration and one test file per class/concentration
            whenever at least two repeated files exist.
          - Remaining files go to train, but target train is not used by the
            supervised calibration/evaluation route.
        """
        source_info = source_info or {}
        client_id = source_info.get('client_id', source_info.get('unit_id'))
        try:
            client_id = int(client_id)
        except Exception:
            client_id = None

        source_clients = set(self.config.full_grid_source_clients or [])
        target_clients = set(self.config.full_grid_target_clients or [])
        is_target = client_id in target_clients if target_clients else False
        is_source = client_id in source_clients if source_clients else not is_target

        by_class_conc = self._group_files_by_class_concentration(
            file_groups, classification_labels, regression_labels
        )
        train_files: List[str] = []
        test_files: List[str] = []
        calibration_files: List[str] = []

        for cls in sorted(by_class_conc.keys()):
            conc_items = sorted(by_class_conc[cls].items(), key=lambda item: item[0])
            for conc_idx, (_conc, files) in enumerate(conc_items):
                bucket = list(files)
                if self.config.shuffle:
                    np.random.shuffle(bucket)
                n_files = len(bucket)
                if n_files == 0:
                    continue

                if is_target:
                    if n_files >= 2:
                        calibration_files.append(bucket.pop())
                        test_files.append(bucket.pop())
                        train_files.extend(bucket)
                    else:
                        test_files.append(bucket.pop())
                elif is_source:
                    if n_files >= 3:
                        calibration_files.append(bucket.pop())
                        test_files.append(bucket.pop())
                        train_files.extend(bucket)
                    elif n_files == 2:
                        train_files.append(bucket.pop())
                        # C4/C5 only have two repeats per concentration. We keep
                        # train full-grid, then alternate the remaining repeat to
                        # provide both source validation and source test coverage.
                        if conc_idx % 2 == 0:
                            test_files.append(bucket.pop())
                        else:
                            calibration_files.append(bucket.pop())
                    else:
                        train_files.append(bucket.pop())
                else:
                    # Generic fallback: same as target if possible, otherwise train.
                    if n_files >= 3:
                        calibration_files.append(bucket.pop())
                        test_files.append(bucket.pop())
                        train_files.extend(bucket)
                    elif n_files == 2:
                        train_files.append(bucket.pop())
                        test_files.append(bucket.pop())
                    else:
                        train_files.append(bucket.pop())

        if self.config.shuffle:
            np.random.shuffle(train_files)
            np.random.shuffle(test_files)
            np.random.shuffle(calibration_files)
        return train_files, test_files, calibration_files

    def _split_indices_regression_window_full_grid(
        self,
        classification_labels: np.ndarray,
        regression_labels: np.ndarray,
        source_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Window-level FullGrid split for upper-bound regression diagnostics.

        This deliberately ignores file boundaries after grouping by
        class/concentration. It is useful to estimate the best possible
        calibration performance when every split has dense concentration
        coverage, but it is not a leakage-free evaluation protocol.
        """
        source_info = source_info or {}
        client_id = source_info.get('client_id', source_info.get('unit_id'))
        try:
            client_id = int(client_id)
        except Exception:
            client_id = None

        source_clients = set(self.config.full_grid_source_clients or [])
        target_clients = set(self.config.full_grid_target_clients or [])
        is_target = client_id in target_clients if target_clients else False
        is_source = client_id in source_clients if source_clients else not is_target

        buckets: Dict[int, Dict[float, List[int]]] = {}
        for idx, cls_value in enumerate(classification_labels.astype(np.int64)):
            cls = int(cls_value)
            conc = float(regression_labels[idx, cls])
            buckets.setdefault(cls, {}).setdefault(conc, []).append(idx)

        train_idx: List[int] = []
        test_idx: List[int] = []
        calibration_idx: List[int] = []

        for cls in sorted(buckets.keys()):
            conc_items = sorted(buckets[cls].items(), key=lambda item: item[0])
            for _conc, indices in conc_items:
                bucket = list(indices)
                if self.config.shuffle:
                    np.random.shuffle(bucket)
                n = len(bucket)
                if n == 0:
                    continue

                if is_target:
                    n_calib = max(1, int(round(n * self.config.calibration_ratio)))
                    n_test = max(1, int(round(n * self.config.global_test_ratio)))
                    if n_calib + n_test >= n:
                        n_calib = min(n_calib, max(1, n // 4))
                        n_test = min(n_test, max(1, n // 4))
                    calibration_idx.extend(bucket[:n_calib])
                    test_idx.extend(bucket[n_calib:n_calib + n_test])
                    train_idx.extend(bucket[n_calib + n_test:])
                elif is_source:
                    n_test = max(1, int(round(n * self.config.global_test_ratio)))
                    n_calib = max(1, int(round(n * self.config.calibration_ratio)))
                    if n_test + n_calib >= n:
                        n_test = min(n_test, max(1, n // 5))
                        n_calib = min(n_calib, max(1, n // 5))
                    test_idx.extend(bucket[:n_test])
                    calibration_idx.extend(bucket[n_test:n_test + n_calib])
                    train_idx.extend(bucket[n_test + n_calib:])
                else:
                    n_test = max(1, int(round(n * self.config.global_test_ratio)))
                    n_calib = max(1, int(round(n * self.config.calibration_ratio)))
                    if n_test + n_calib >= n:
                        n_test = min(n_test, max(1, n // 5))
                        n_calib = min(n_calib, max(1, n // 5))
                    test_idx.extend(bucket[:n_test])
                    calibration_idx.extend(bucket[n_test:n_test + n_calib])
                    train_idx.extend(bucket[n_test + n_calib:])

        train_arr = np.array(train_idx, dtype=np.int64)
        test_arr = np.array(test_idx, dtype=np.int64)
        calibration_arr = np.array(calibration_idx, dtype=np.int64)
        if self.config.shuffle:
            np.random.shuffle(train_arr)
            np.random.shuffle(test_arr)
            np.random.shuffle(calibration_arr)
        return train_arr, test_arr, calibration_arr

    def _split_files_stratified_by_concentration(
        self,
        file_groups: Dict[str, List[int]],
        classification_labels: np.ndarray,
        regression_labels: np.ndarray
    ) -> Tuple[List[str], List[str], List[str]]:
        """File-level split stratified by gas class and concentration.

        The old implementation shuffled all files globally and then sliced 70/20/10.
        That preserved file isolation but produced sparse and uneven regression test
        concentrations. Here we keep file isolation while allocating per-class quotas
        and selecting files across concentration levels as evenly as possible.
        """
        by_class_conc = self._group_files_by_class_concentration(
            file_groups, classification_labels, regression_labels
        )

        train_files, test_files, calibration_files = [], [], []
        for cls in sorted(by_class_conc.keys()):
            conc_to_files = {conc: list(files) for conc, files in by_class_conc[cls].items()}
            for files in conc_to_files.values():
                if self.config.shuffle:
                    np.random.shuffle(files)

            n_class_files = sum(len(files) for files in conc_to_files.values())
            if n_class_files == 0:
                continue
            n_test = int(round(n_class_files * self.config.global_test_ratio))
            n_calib = int(round(n_class_files * self.config.calibration_ratio))
            if n_class_files >= 3:
                n_test = max(1, n_test)
                n_calib = max(1, n_calib)
            if n_test + n_calib >= n_class_files:
                overflow = n_test + n_calib - (n_class_files - 1)
                n_test = max(0, n_test - max(0, overflow))
            n_train = n_class_files - n_test - n_calib
            if n_train < 1 and n_class_files > 0:
                need = 1 - n_train
                if n_calib >= need:
                    n_calib -= need
                else:
                    n_test = max(0, n_test - (need - n_calib))
                    n_calib = 0

            concs = sorted(conc_to_files.keys())
            test_pref = self._evenly_spaced_values(concs, n_test)
            picked_test = self._pop_balanced_files(conc_to_files, n_test, test_pref)

            # Calibration also covers the range, but uses remaining files only.
            available_concs = [conc for conc in concs if conc_to_files.get(conc)]
            calib_pref = self._evenly_spaced_values(available_concs, n_calib)
            picked_calib = self._pop_balanced_files(conc_to_files, n_calib, calib_pref)

            remaining_train = []
            for conc in sorted(conc_to_files.keys()):
                remaining_train.extend(conc_to_files[conc])

            train_files.extend(remaining_train)
            test_files.extend(picked_test)
            calibration_files.extend(picked_calib)

        if self.config.shuffle:
            np.random.shuffle(train_files)
            np.random.shuffle(test_files)
            np.random.shuffle(calibration_files)
        return train_files, test_files, calibration_files

    def _split_iid(self, unit_data: Dict[int, Dict[str, Any]], n_clients: int = 5) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """按IID划分数据集
        
        将所有单元的数据合并后随机分配给n_clients个客户端，实现IID划分
        
        Args:
            unit_data: 单元数据字典
            n_clients: 客户端数量
            
        Returns:
            客户端数据字典、全局测试贡献列表、校准集贡献列表
        """
        # 合并所有单元的数据
        all_features, all_regression_labels, all_classification_labels, all_phase_labels = [], [], [], []
        all_experiment_info, all_unit_ids = [], []
        for unit_id, data in unit_data.items():
            n_samples = data["n_samples"]
            all_features.append(data["features"])
            all_regression_labels.append(data["regression_labels"])
            all_classification_labels.append(data["classification_labels"])
            all_phase_labels.append(data["phase_labels"])
            if data["experiment_info"]:
                all_experiment_info.extend(data["experiment_info"])
            else:
                all_experiment_info.extend([{"unit_id": unit_id}] * n_samples)
            all_unit_ids.extend([unit_id] * n_samples)
        features = np.concatenate(all_features, axis=0)
        regression_labels = np.concatenate(all_regression_labels, axis=0)
        classification_labels = np.concatenate(all_classification_labels, axis=0)
        phase_labels = np.concatenate(all_phase_labels, axis=0)
        n_samples = len(features)
        indices = np.arange(n_samples)
        if self.config.shuffle:
            np.random.shuffle(indices)
        features = features[indices]
        regression_labels = regression_labels[indices]
        classification_labels = classification_labels[indices]
        phase_labels = phase_labels[indices]
        experiment_info = [all_experiment_info[i] for i in indices]
        unit_ids = [all_unit_ids[i] for i in indices]
        samples_per_client = n_samples // n_clients
        clients_data = {}
        all_global_contributions = []
        all_calibration_contributions = []
        for client_id in range(n_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client if client_id < n_clients - 1 else n_samples
            n_client_samples = end_idx - start_idx
            if self.config.verbose:
                print(f"\n 创建IID客户端 {client_id}")
                print(f"   分配样本数: {n_client_samples:,}")
            if n_client_samples < self.config.min_samples_per_client:
                warnings.warn(f"IID客户端 {client_id} 只有 {n_client_samples} 个样本，少于最小要求 {self.config.min_samples_per_client}")
            client_features = features[start_idx:end_idx]
            client_regression_labels = regression_labels[start_idx:end_idx]
            client_classification_labels = classification_labels[start_idx:end_idx]
            client_phase_labels = phase_labels[start_idx:end_idx]
            client_experiment_info = experiment_info[start_idx:end_idx]
            client_unit_ids = unit_ids[start_idx:end_idx]
            main_unit_id = max(set(client_unit_ids), key=client_unit_ids.count) if client_unit_ids else -1
            client_data, global_contribution, calibration_contribution = self._split_single_client(
                features=client_features, regression_labels=client_regression_labels,
                classification_labels=client_classification_labels, phase_labels=client_phase_labels,
                experiment_info=client_experiment_info,
                source_info={"unit_id": main_unit_id, "client_id": client_id, "is_iid": True}
            )
            client_data["unit_id"] = main_unit_id
            client_data["unit_ids"] = list(set(client_unit_ids))
            client_data["source_info"] = {"unit_id": main_unit_id, "unit_ids": list(set(client_unit_ids)), "is_iid": True, "original_samples": n_client_samples}
            clients_data[client_id] = client_data
            if global_contribution:
                all_global_contributions.append(global_contribution)
            if calibration_contribution:
                all_calibration_contributions.append(calibration_contribution)
        return clients_data, all_global_contributions, all_calibration_contributions
    
    def _split_by_gas(self, unit_data: Dict[int, Dict[str, Any]], n_clients: int = 4) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """按气体类型划分数据集
        
        将所有单元的数据按气体类型分配给n_clients个客户端
        
        Args:
            unit_data: 单元数据字典
            n_clients: 客户端数量
            
        Returns:
            客户端数据字典、全局测试贡献列表、校准集贡献列表
        """
        n_clients = min(n_clients, 4)
        gas_data = {i: {"features": [], "regression_labels": [], "classification_labels": [], "phase_labels": [], "experiment_info": []} for i in range(4)}
        for unit_id, data in unit_data.items():
            features = data["features"]
            regression_labels = data["regression_labels"]
            classification_labels = data["classification_labels"]
            phase_labels = data["phase_labels"]
            experiment_info = data["experiment_info"]
            for i in range(len(features)):
                gas_type = int(classification_labels[i])
                gas_data[gas_type]["features"].append(features[i])
                gas_data[gas_type]["regression_labels"].append(regression_labels[i])
                gas_data[gas_type]["classification_labels"].append(classification_labels[i])
                gas_data[gas_type]["phase_labels"].append(phase_labels[i])
                if experiment_info and i < len(experiment_info):
                    gas_data[gas_type]["experiment_info"].append(experiment_info[i])
                else:
                    gas_data[gas_type]["experiment_info"].append({"unit_id": unit_id})
        clients_data = {}
        all_global_contributions = []
        all_calibration_contributions = []
        client_id = 0
        for gas_type in range(4):
            if len(gas_data[gas_type]["features"]) > 0:
                features = np.array(gas_data[gas_type]["features"])
                regression_labels = np.array(gas_data[gas_type]["regression_labels"])
                classification_labels = np.array(gas_data[gas_type]["classification_labels"])
                phase_labels = np.array(gas_data[gas_type]["phase_labels"])
                experiment_info = gas_data[gas_type]["experiment_info"]
                if self.config.verbose:
                    gas_name = ["乙醇", "一氧化碳", "乙烯", "甲烷"][gas_type]
                    print(f"\n 创建气体客户端 {client_id} ({gas_name})")
                    print(f"   样本数: {len(features):,}")
                if len(features) < self.config.min_samples_per_client:
                    warnings.warn(f"气体类型 {gas_type} 只有 {len(features)} 个样本，少于最小要求 {self.config.min_samples_per_client}")
                client_data, global_contribution, calibration_contribution = self._split_single_client(
                    features=features, regression_labels=regression_labels,
                    classification_labels=classification_labels, phase_labels=phase_labels,
                    experiment_info=experiment_info, source_info={"gas_type": gas_type, "client_id": client_id}
                )
                gas_name = ["ethanol", "carbon_monoxide", "ethylene", "methane"][gas_type]
                client_data["gas_type"] = gas_type
                client_data["gas_name"] = gas_name
                client_data["source_info"] = {"gas_type": gas_type, "gas_name": gas_name, "original_samples": len(features)}
                clients_data[client_id] = client_data
                if global_contribution:
                    all_global_contributions.append(global_contribution)
                if calibration_contribution:
                    all_calibration_contributions.append(calibration_contribution)
                client_id += 1
                if client_id >= n_clients:
                    break
        return clients_data, all_global_contributions, all_calibration_contributions
    
    def _split_single_client(self, features: np.ndarray, regression_labels: np.ndarray,
                             classification_labels: np.ndarray, phase_labels: Optional[np.ndarray],
                             experiment_info: List[Dict], source_info: Dict) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """按单客户端划分数据集
        
        将单个客户端的数据按比例划分为训练集、测试集和校准集
        
        Args:
            features: 特征数组
            regression_labels: 回归标签数组
            classification_labels: 分类标签数组
            phase_labels: 阶段标签数组
            experiment_info: 实验信息列表
            source_info: 源信息字典
            
        Returns:
            客户端数据字典、全局测试贡献字典、校准集贡献字典
        """
        n_samples = len(features)
        # 按文件分组
        file_groups = {}
        for idx in range(n_samples):
            if experiment_info and idx < len(experiment_info):
                filename = experiment_info[idx].get('filename', f'unknown_{idx}')
            else:
                filename = f'unknown_{idx}'
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append(idx)
        filenames = list(file_groups.keys())
        n_files = len(filenames)
        split_level = "file"
        if self.config.stratify:
            if getattr(self.config, 'regression_window_full_grid', False):
                train_idx, global_test_idx, calibration_idx = self._split_indices_regression_window_full_grid(
                    classification_labels, regression_labels, source_info
                )
                train_files = sorted({experiment_info[i].get('filename', f'unknown_{i}') for i in train_idx}) if experiment_info else []
                global_test_files = sorted({experiment_info[i].get('filename', f'unknown_{i}') for i in global_test_idx}) if experiment_info else []
                calibration_files = sorted({experiment_info[i].get('filename', f'unknown_{i}') for i in calibration_idx}) if experiment_info else []
                split_level = "window"
            elif getattr(self.config, 'regression_full_grid', False):
                train_files, global_test_files, calibration_files = self._split_files_regression_full_grid(
                    file_groups, classification_labels, regression_labels, source_info
                )
                train_idx = []
                global_test_idx = []
                calibration_idx = []
                for filename in train_files:
                    train_idx.extend(file_groups[filename])
                for filename in global_test_files:
                    global_test_idx.extend(file_groups[filename])
                for filename in calibration_files:
                    calibration_idx.extend(file_groups[filename])
                train_idx = np.array(train_idx, dtype=np.int64)
                global_test_idx = np.array(global_test_idx, dtype=np.int64)
                calibration_idx = np.array(calibration_idx, dtype=np.int64)
            else:
                train_files, global_test_files, calibration_files = self._split_files_stratified_by_concentration(
                    file_groups, classification_labels, regression_labels
                )
                train_idx = []
                global_test_idx = []
                calibration_idx = []
                for filename in train_files:
                    train_idx.extend(file_groups[filename])
                for filename in global_test_files:
                    global_test_idx.extend(file_groups[filename])
                for filename in calibration_files:
                    calibration_idx.extend(file_groups[filename])
                train_idx = np.array(train_idx, dtype=np.int64)
                global_test_idx = np.array(global_test_idx, dtype=np.int64)
                calibration_idx = np.array(calibration_idx, dtype=np.int64)
        else:
            if self.config.shuffle:
                np.random.shuffle(filenames)
            n_train_files = int(n_files * self.config.train_ratio)
            n_global_test_files = int(n_files * self.config.global_test_ratio)
            train_files = filenames[:n_train_files]
            global_test_files = filenames[n_train_files:n_train_files + n_global_test_files]
            calibration_files = filenames[n_train_files + n_global_test_files:]
            train_idx = []
            global_test_idx = []
            calibration_idx = []
            for filename in train_files:
                train_idx.extend(file_groups[filename])
            for filename in global_test_files:
                global_test_idx.extend(file_groups[filename])
            for filename in calibration_files:
                calibration_idx.extend(file_groups[filename])
            train_idx = np.array(train_idx, dtype=np.int64)
            global_test_idx = np.array(global_test_idx, dtype=np.int64)
            calibration_idx = np.array(calibration_idx, dtype=np.int64)
        if self.config.shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(global_test_idx)
            np.random.shuffle(calibration_idx)
        if self.config.verbose:
            print(f"客户端 {source_info.get('client_id')} 各部分标签分布:")
            splits = [("训练集", train_idx, len(train_idx)), ("全局测试集贡献", global_test_idx, len(global_test_idx)), ("校准集贡献", calibration_idx, len(calibration_idx))]
            for name, idx, target_size in splits:
                if len(idx) > 0:
                    labels = classification_labels[idx]
                    unique, counts = np.unique(labels, return_counts=True)
                    gas_names = ["乙醇", "一氧化碳", "乙烯", "甲烷"]
                    print(f"  {name} ({len(idx)}/{target_size}): ", end="")
                    for label, count in zip(unique, counts):
                        gas_name = gas_names[int(label)] if int(label) < len(gas_names) else f"标签{label}"
                        print(f"{gas_name}:{count}({count/len(idx)*100:.1f}%) ", end="")
                    print()
        # 构建客户端数据字典，包含训练集、测试集和校准集
        phase_labels_numeric = self._normalize_phase_labels(phase_labels) if phase_labels is not None else None
        client_data = {
            "train": {
                "features": features[train_idx],
                "regression_labels": regression_labels[train_idx],
                "classification_labels": classification_labels[train_idx],
                "experiment_info": [experiment_info[i] if i < len(experiment_info) else {} for i in train_idx]
            },
            "test": {
                "features": features[global_test_idx],
                "regression_labels": regression_labels[global_test_idx],
                "classification_labels": classification_labels[global_test_idx],
                "experiment_info": [experiment_info[i] if i < len(experiment_info) else {} for i in global_test_idx]
            },
            "calibration": {
                "features": features[calibration_idx],
                "regression_labels": regression_labels[calibration_idx],
                "classification_labels": classification_labels[calibration_idx],
                "experiment_info": [experiment_info[i] if i < len(experiment_info) else {} for i in calibration_idx]
            },
            "stats": {}
        }
        if phase_labels_numeric is not None:
            client_data["train"]["phase_labels"] = phase_labels_numeric[train_idx]
            client_data["test"]["phase_labels"] = phase_labels_numeric[global_test_idx]
            client_data["calibration"]["phase_labels"] = phase_labels_numeric[calibration_idx]
        stats = client_data["stats"]
        stats.update({
            "n_total": n_samples, "n_train": len(train_idx), "n_test": len(global_test_idx),
            "n_calibration": len(calibration_idx), "train_ratio": len(train_idx) / n_samples,
            "test_ratio": len(global_test_idx) / n_samples, "calibration_ratio": len(calibration_idx) / n_samples,
            "n_files": n_files, "n_train_files": len(train_files), "n_test_files": len(global_test_files),
            "n_calibration_files": len(calibration_files), "split_level": split_level,
            "allows_file_overlap": split_level == "window"
        })
        stats["classification_distribution"] = self._calculate_label_distribution(classification_labels)
        stats["concentration_stats"] = self._calculate_concentration_stats(regression_labels)
        stats["source_info"] = source_info
        # 构建全局测试贡献
        global_contribution = None
        if len(global_test_idx) > 0:
            global_experiment_info = []
            if experiment_info:
                for i in global_test_idx:
                    if i < len(experiment_info):
                        global_experiment_info.append(experiment_info[i])
                    else:
                        global_experiment_info.append({})
            global_contribution = {
                "features": features[global_test_idx], "regression_labels": regression_labels[global_test_idx],
                "classification_labels": classification_labels[global_test_idx],
                "phase_labels": phase_labels_numeric[global_test_idx] if phase_labels_numeric is not None else None,
                "experiment_info": global_experiment_info, "source_info": source_info, "n_samples": len(global_test_idx)
            }
        # 构建校准集贡献
        calibration_contribution = None
        if len(calibration_idx) > 0:
            calibration_experiment_info = []
            if experiment_info:
                for i in calibration_idx:
                    if i < len(experiment_info):
                        calibration_experiment_info.append(experiment_info[i])
                    else:
                        calibration_experiment_info.append({})
            calibration_contribution = {
                "features": features[calibration_idx], "regression_labels": regression_labels[calibration_idx],
                "classification_labels": classification_labels[calibration_idx],
                "phase_labels": phase_labels_numeric[calibration_idx] if phase_labels_numeric is not None else None,
                "experiment_info": calibration_experiment_info, "source_info": source_info, "n_samples": len(calibration_idx)
            }
        return client_data, global_contribution, calibration_contribution
    
    def _build_client_data_dict(self, features: np.ndarray, regression_labels: np.ndarray,
                                classification_labels: np.ndarray, phase_labels: Optional[np.ndarray],
                                experiment_info: List[Dict], train_idx: np.ndarray) -> Dict[str, Any]:
        """构建客户端数据字典
        
        根据训练索引构建客户端训练数据字典
        
        Args:
            features: 特征数组
            regression_labels: 回归标签数组
            classification_labels: 分类标签数组
            phase_labels: 阶段标签数组
            experiment_info: 实验信息列表
            train_idx: 训练样本索引数组
            
        Returns:
            客户端数据字典
        """
        train_experiment_info = []
        if experiment_info:
            for i in train_idx:
                if i < len(experiment_info):
                    train_experiment_info.append(experiment_info[i])
                else:
                    train_experiment_info.append({})
        result = {
            "train": {
                "features": features[train_idx],
                "regression_labels": regression_labels[train_idx],
                "classification_labels": classification_labels[train_idx],
                "experiment_info": train_experiment_info
            },
            "stats": {}
        }
        if phase_labels is not None:
            result["train"]["phase_labels"] = phase_labels[train_idx]
        return result
    
    def _build_calibration_set(self, calibration_contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建校准集
        
        收集所有客户端的校准集贡献，合并构建全局校准集
        
        Args:
            calibration_contributions: 校准集贡献列表
            
        Returns:
            校准集字典
        """
        if not calibration_contributions:
            return {}
        if self.config.verbose:
            print(f"\n 构建校准集")
            print(f"   收集到 {len(calibration_contributions)} 个客户端的贡献")
        all_features, all_regression_labels, all_classification_labels, all_phase_labels = [], [], [], []
        all_experiment_info = []
        source_stats = []
        total_contributions = 0
        for contribution in calibration_contributions:
            n_samples = contribution["n_samples"]
            all_features.append(contribution["features"])
            all_regression_labels.append(contribution["regression_labels"])
            all_classification_labels.append(contribution["classification_labels"])
            if contribution["phase_labels"] is not None:
                all_phase_labels.append(contribution["phase_labels"])
            all_experiment_info.extend(contribution["experiment_info"])
            source_info = contribution["source_info"]
            source_stats.append({"client_id": source_info.get("client_id"), "unit_id": source_info.get("unit_id"),
                                "gas_type": source_info.get("gas_type"), "n_samples": n_samples})
            total_contributions += n_samples
        calibration_features = np.concatenate(all_features, axis=0)
        calibration_regression_labels = np.concatenate(all_regression_labels, axis=0)
        calibration_classification_labels = np.concatenate(all_classification_labels, axis=0)
        if all_phase_labels:
            calibration_phase_labels = self._normalize_phase_labels(np.concatenate(all_phase_labels, axis=0))
        else:
            calibration_phase_labels = None
        if self.config.verbose:
            print(f"   校准集: {len(calibration_features):,} 个样本")
        label_distribution = self._calculate_label_distribution(calibration_classification_labels)
        calibration_set = {
            "features": calibration_features, "regression_labels": calibration_regression_labels,
            "classification_labels": calibration_classification_labels, "phase_labels": calibration_phase_labels,
            "experiment_info": all_experiment_info,
            "stats": {"n_samples": len(calibration_features), "n_contributions": len(calibration_contributions),
                      "total_contributions": total_contributions, "label_distribution": label_distribution,
                      "source_stats": source_stats}
        }
        if self.config.verbose:
            print(f"    校准集构建完成")
            print(f"      最终样本数: {len(calibration_features):,}")
            print(f"      类别分布:")
            for gas_name, info in label_distribution.items():
                print(f"        {gas_name}: {info['count']:,} ({info['percentage']:.1f}%)")
        return calibration_set
    
    def _build_global_test_set(self, global_contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建全局测试集
        
        收集所有客户端的测试集贡献，合并构建全局测试集，并创建分阶段测试子集
        
        Args:
            global_contributions: 全局测试集贡献列表
            
        Returns:
            全局测试集字典，包含分阶段测试子集
        """
        if not global_contributions:
            return {}
        if self.config.verbose:
            print(f"\n 构建全局测试集")
            print(f"   收集到 {len(global_contributions)} 个客户端的贡献")
        
        # 统计贡献为0的客户端
        zero_contribution_clients = [c for c in global_contributions if c.get('n_samples', 0) == 0]
        if zero_contribution_clients:
            logger.warning(f"全局测试集构建中，有 {len(zero_contribution_clients)} 个客户端的贡献为0，可能被平衡采样排除。")
        all_features, all_regression_labels, all_classification_labels, all_phase_labels = [], [], [], []
        all_experiment_info = []
        source_stats = []
        total_contributions = 0
        for contribution in global_contributions:
            n_samples = contribution["n_samples"]
            all_features.append(contribution["features"])
            all_regression_labels.append(contribution["regression_labels"])
            all_classification_labels.append(contribution["classification_labels"])
            if contribution["phase_labels"] is not None:
                all_phase_labels.append(contribution["phase_labels"])
            all_experiment_info.extend(contribution["experiment_info"])
            source_info = contribution["source_info"]
            source_stats.append({"client_id": source_info.get("client_id"), "unit_id": source_info.get("unit_id"),
                                "gas_type": source_info.get("gas_type"), "n_samples": n_samples})
            total_contributions += n_samples
        global_features = np.concatenate(all_features, axis=0)
        global_regression_labels = np.concatenate(all_regression_labels, axis=0)
        global_classification_labels = np.concatenate(all_classification_labels, axis=0)
        if all_phase_labels:
            global_phase_labels = self._normalize_phase_labels(np.concatenate(all_phase_labels, axis=0))
        else:
            global_phase_labels = None
        if self.config.verbose:
            print(f"   原始全局测试集: {len(global_features):,} 个样本")
        if self.config.global_test_sampling == GlobalTestSampling.STRATIFIED and self.config.ensure_balance:
            balanced_data = self._balance_global_test_set(
                global_features, global_regression_labels, global_classification_labels,
                global_phase_labels, all_experiment_info
            )
            if balanced_data:
                global_features = balanced_data["features"]
                global_regression_labels = balanced_data["regression_labels"]
                global_classification_labels = balanced_data["classification_labels"]
                global_phase_labels = balanced_data["phase_labels"]
                all_experiment_info = balanced_data["experiment_info"]
                if self.config.verbose:
                    print(f"   平衡后全局测试集: {len(global_features):,} 个样本")
        label_distribution = self._calculate_label_distribution(global_classification_labels)
        # 创建分阶段测试子集
        phase_test_sets = {}
        if self.config.verbose:
            print(f"   全局 phase_labels 类型: {type(global_phase_labels)}")
            if global_phase_labels is not None:
                print(f"   全局 phase_labels 形状: {global_phase_labels.shape}")
                print(f"   全局 phase_labels 唯一值: {np.unique(global_phase_labels)}")
        
        if global_phase_labels is not None:
            # 直接使用整数格式的 phase_labels
            phase_labels_numeric = self._normalize_phase_labels(global_phase_labels)
            
            # 只对已知阶段（0,1,2）构建测试子集；未知 -1 被排除
            for phase_value, phase_name in zip([0,1,2], ['early','middle','late']):
                mask = (phase_labels_numeric == phase_value)
                if np.any(mask):
                    phase_test_sets[phase_name] = {
                        'features': global_features[mask],
                        'regression_labels': global_regression_labels[mask],
                        'classification_labels': global_classification_labels[mask],
                        'phase_labels': phase_labels_numeric[mask]
                    }
            
            if np.any(phase_labels_numeric == -1):
                if self.config.verbose:
                    print(f"   未知阶段标签(-1)样本数: {np.sum(phase_labels_numeric==-1)}")
                logger.warning(f"全局测试集中有 {np.sum(phase_labels_numeric==-1)} 个样本缺少阶段标签，已排除在分阶段评估之外。")
        
        if self.config.verbose:
            print(f"   分阶段测试集数量: {len(phase_test_sets)}")
            for phase, test_set in phase_test_sets.items():
                print(f"   {phase} 阶段测试集样本数: {len(test_set['features'])}")
        
        global_test_set = {
            "features": global_features, "regression_labels": global_regression_labels,
            "classification_labels": global_classification_labels, "phase_labels": global_phase_labels,
            "experiment_info": all_experiment_info,
            "phase_test_sets": phase_test_sets,
            "stats": {"n_samples": len(global_features), "n_contributions": len(global_contributions),
                      "total_contributions": total_contributions, "label_distribution": label_distribution,
                      "source_stats": source_stats, "sampling_method": self.config.global_test_sampling.value,
                      "is_balanced": self.config.ensure_balance},
            "config": {"global_test_sampling": self.config.global_test_sampling.value,
                       "ensure_balance": self.config.ensure_balance,
                       "min_samples_per_class": self.config.min_samples_per_class}
        }
        if self.config.verbose:
            print(f"    全局测试集构建完成")
            print(f"      最终样本数: {len(global_features):,}")
            print(f"      类别分布:")
            for gas_name, info in label_distribution.items():
                print(f"        {gas_name}: {info['count']:,} ({info['percentage']:.1f}%)")
        return global_test_set
    
    def _balance_global_test_set(self, features: np.ndarray, regression_labels: np.ndarray,
                                 classification_labels: np.ndarray, phase_labels: Optional[np.ndarray],
                                 experiment_info: List[Dict]) -> Optional[Dict[str, Any]]:
        """平衡全局测试集
        
        对全局测试集进行类别平衡，确保每个类别的样本数大致相同
        
        Args:
            features: 特征数组
            regression_labels: 回归标签数组
            classification_labels: 分类标签数组
            phase_labels: 阶段标签数组
            experiment_info: 实验信息列表
            
        Returns:
            平衡后的数据集字典
        """
        unique_classes, class_counts = np.unique(classification_labels, return_counts=True)
        if self.config.verbose:
            print(f"   类别分布分析:")
            for class_label, count in zip(unique_classes, class_counts):
                gas_names = ["乙醇", "一氧化碳", "乙烯", "甲烷"]
                class_idx = int(class_label)
                gas_name = gas_names[class_idx] if 0 <= class_idx < len(gas_names) else f"标签{class_label}"
                print(f"     {gas_name}: {count:,} 样本")
        target_per_class = self.config.target_samples_per_class if self.config.target_samples_per_class > 0 else max(self.config.min_samples_per_class, min(class_counts))
        if self.config.verbose:
            print(f"   目标每类样本数: {target_per_class:,}")
        balanced_indices = []
        for class_label in unique_classes:
            class_indices = np.where(classification_labels == class_label)[0]
            n_class_samples = len(class_indices)
            if n_class_samples <= target_per_class:
                balanced_indices.extend(class_indices.tolist())
                if self.config.verbose:
                    gas_names = ["乙醇", "一氧化碳", "乙烯", "甲烷"]
                    class_idx = int(class_label)
                    gas_name = gas_names[class_idx] if 0 <= class_idx < len(gas_names) else f"标签{class_label}"
                    print(f"     {gas_name}: 样本不足，使用全部 {n_class_samples:,} 个样本")
            else:
                sampled_indices = np.random.choice(class_indices, size=target_per_class, replace=False)
                balanced_indices.extend(sampled_indices.tolist())
                if self.config.verbose:
                    gas_names = ["乙醇", "一氧化碳", "乙烯", "甲烷"]
                    class_idx = int(class_label)
                    gas_name = gas_names[class_idx] if 0 <= class_idx < len(gas_names) else f"标签{class_label}"
                    print(f"     {gas_name}: 从 {n_class_samples:,} 个样本中采样 {target_per_class:,} 个")
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        balanced_experiment_info = []
        if experiment_info:
            for i in balanced_indices:
                if i < len(experiment_info):
                    balanced_experiment_info.append(experiment_info[i])
                else:
                    balanced_experiment_info.append({})
        return {
            "features": features[balanced_indices], "regression_labels": regression_labels[balanced_indices],
            "classification_labels": classification_labels[balanced_indices],
            "phase_labels": phase_labels[balanced_indices] if phase_labels is not None else None,
            "experiment_info": balanced_experiment_info
        }
    
    def _calculate_label_distribution(self, labels: np.ndarray) -> Dict:
        """计算标签分布
        
        计算分类标签的分布情况
        
        Args:
            labels: 分类标签数组
            
        Returns:
            标签分布字典
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {}
        total = len(labels)
        gas_names = ["乙醇", "一氧化碳", "乙烯", "甲烷"]
        for label, count in zip(unique, counts):
            gas_name = gas_names[int(label)] if int(label) < len(gas_names) else f"label_{label}"
            distribution[gas_name] = {"count": int(count), "percentage": float(count / total * 100)}
        return distribution
    
    def _calculate_concentration_stats(self, regression_labels: np.ndarray) -> Dict:
        """计算浓度统计信息
        
        计算各气体浓度的统计信息
        
        Args:
            regression_labels: 回归标签数组
            
        Returns:
            浓度统计信息字典
        """
        stats = {}
        gas_names = ["乙醇", "一氧化碳", "乙烯", "甲烷"]
        for i, gas_name in enumerate(gas_names):
            concentrations = regression_labels[:, i]
            nonzero_indices = concentrations > 0
            if np.any(nonzero_indices):
                nonzero_concentrations = concentrations[nonzero_indices]
                stats[gas_name] = {
                    "nonzero_count": int(np.sum(nonzero_indices)), "mean": float(np.mean(nonzero_concentrations)),
                    "std": float(np.std(nonzero_concentrations)), "min": float(np.min(nonzero_concentrations)),
                    "max": float(np.max(nonzero_concentrations)), "median": float(np.median(nonzero_concentrations)),
                    "q25": float(np.percentile(nonzero_concentrations, 25)), "q75": float(np.percentile(nonzero_concentrations, 75))
                }
            else:
                stats[gas_name] = {"nonzero_count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0}
        return stats
    
    def _compute_overall_statistics(self):
        """计算总体统计信息
        
        计算联邦数据集的总体统计信息
        """
        if not self.clients_data:
            return
        n_clients = len(self.clients_data)
        total_samples = train_samples = global_test_contributions = calibration_contributions = 0
        client_samples = []
        client_stats = {}
        for client_id, data in self.clients_data.items():
            stats = data["stats"]
            total_samples += stats["n_total"]
            train_samples += stats["n_train"]
            global_test_contributions += stats["n_test"]
            calibration_contributions += stats["n_calibration"]
            client_samples.append(stats["n_total"])
            client_stats[client_id] = {"n_total": stats["n_total"], "n_train": stats["n_train"],
                                       "n_test": stats["n_test"], "n_calibration": stats["n_calibration"]}
        global_test_samples = self.global_test_set["stats"]["n_samples"] if self.global_test_set else 0
        calibration_samples = self.calibration_set["stats"]["n_samples"] if hasattr(self, 'calibration_set') and self.calibration_set else 0
        self.stats = {
            "overall": {
                "n_clients": n_clients, "total_samples": total_samples, "train_samples": train_samples,
                "global_test_contributions": global_test_contributions, "global_test_samples": global_test_samples,
                "calibration_contributions": calibration_contributions, "calibration_samples": calibration_samples,
                "avg_samples_per_client": float(np.mean(client_samples)), "std_samples_per_client": float(np.std(client_samples)),
                "min_samples_per_client": float(np.min(client_samples)), "max_samples_per_client": float(np.max(client_samples)),
                "train_ratio": train_samples / total_samples, "global_test_ratio": global_test_contributions / total_samples,
                "calibration_ratio": calibration_contributions / total_samples
            },
            "clients": client_stats, "config": self.config.to_dict()
        }
        if self.global_test_set:
            self.stats["global_test"] = self.global_test_set["stats"]
        if hasattr(self, 'calibration_set') and self.calibration_set:
            self.stats["calibration"] = self.calibration_set["stats"]
    
    def _print_statistics_summary(self):
        """打印统计摘要
        
        打印联邦数据集的统计摘要信息
        """
        if not self.config.verbose:
            return
        stats = self.stats["overall"]
        print("\n" + "=" * 60)
        print(" 联邦学习数据集统计摘要")
        print("=" * 60)
        print(f"客户端数量: {stats['n_clients']}")
        print(f" 总样本数: {stats['total_samples']:,}")
        print(f" 样本分布:")
        print(f"   训练集: {stats['train_samples']:,} ({stats['train_ratio']:.1%})")
        print(f"   全局测试集贡献: {stats['global_test_contributions']:,} ({stats['global_test_ratio']:.1%})")
        print(f"   实际全局测试集: {stats['global_test_samples']:,}")
        print(f"   校准集贡献: {stats['calibration_contributions']:,} ({stats['calibration_ratio']:.1%})")
        print(f"   实际校准集: {stats['calibration_samples']:,}")
        if stats['global_test_samples'] > 0:
            print(f"   全局测试集利用率: {stats['global_test_samples']/stats['global_test_contributions']:.1%}")
        if stats['calibration_samples'] > 0:
            print(f"   校准集利用率: {stats['calibration_samples']/stats['calibration_contributions']:.1%}")
        print(f"\n 客户端样本分布:")
        print(f"   平均样本数/客户端: {stats['avg_samples_per_client']:.0f}")
        print(f"   样本数标准差: {stats['std_samples_per_client']:.0f}")
        print(f"   最小样本数: {stats['min_samples_per_client']}")
        print(f"   最大样本数: {stats['max_samples_per_client']}")
        if self.global_test_set:
            print(f"\n 全局测试集类别分布:")
            for gas_name, info in self.global_test_set["stats"]["label_distribution"].items():
                print(f"   {gas_name}: {info['count']:,} ({info['percentage']:.1f}%)")
        if hasattr(self, 'calibration_set') and self.calibration_set:
            print(f"\n 校准集类别分布:")
            for gas_name, info in self.calibration_set["stats"]["label_distribution"].items():
                print(f"   {gas_name}: {info['count']:,} ({info['percentage']:.1f}%)")
        print("=" * 60)
    
    def save_federated_dataset(self, output_dir: str = "dataset/client_data_federated"):
        """保存联邦学习数据集
        
        将联邦学习数据集保存到指定目录
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if self.config.verbose:
            print(f"\n 保存联邦学习数据集到: {output_path}")
        # 保存每个客户端的数据
        for unit_id, data in self.clients_data.items():
            client_dir = output_path / f"client_{unit_id}"
            client_dir.mkdir(exist_ok=True)
            if self.config.verbose:
                print(f"   保存客户端 {unit_id}...")
            # 保存训练集
            np.save(client_dir / "train_features.npy", data["train"]["features"])
            np.save(client_dir / "train_regression_labels.npy", data["train"]["regression_labels"])
            np.save(client_dir / "train_classification_labels.npy", data["train"]["classification_labels"])
            if "phase_labels" in data["train"]:
                np.save(client_dir / "train_phase_labels.npy", data["train"]["phase_labels"])
            with open(client_dir / "train_experiment_info.json", "w", encoding='utf-8') as f:
                json.dump(data["train"]["experiment_info"], f, indent=2, ensure_ascii=False)
            # 保存测试集
            if "test" in data and len(data["test"]["features"]) > 0:
                np.save(client_dir / "test_features.npy", data["test"]["features"])
                np.save(client_dir / "test_regression_labels.npy", data["test"]["regression_labels"])
                np.save(client_dir / "test_classification_labels.npy", data["test"]["classification_labels"])
                if "phase_labels" in data["test"]:
                    np.save(client_dir / "test_phase_labels.npy", data["test"]["phase_labels"])
                with open(client_dir / "test_experiment_info.json", "w", encoding='utf-8') as f:
                    json.dump(data["test"]["experiment_info"], f, indent=2, ensure_ascii=False)
            # 保存校准集
            if "calibration" in data and len(data["calibration"]["features"]) > 0:
                np.save(client_dir / "calibration_features.npy", data["calibration"]["features"])
                np.save(client_dir / "calibration_regression_labels.npy", data["calibration"]["regression_labels"])
                np.save(client_dir / "calibration_classification_labels.npy", data["calibration"]["classification_labels"])
                if "phase_labels" in data["calibration"]:
                    np.save(client_dir / "calibration_phase_labels.npy", data["calibration"]["phase_labels"])
                with open(client_dir / "calibration_experiment_info.json", "w", encoding='utf-8') as f:
                    json.dump(data["calibration"]["experiment_info"], f, indent=2, ensure_ascii=False)
            with open(client_dir / "stats.json", "w", encoding='utf-8') as f:
                json.dump(data["stats"], f, indent=2, ensure_ascii=False)
        # 保存全局测试集
        if self.config.save_global_test and self.global_test_set:
            global_test_dir = output_path / "global_test"
            global_test_dir.mkdir(exist_ok=True)
            np.save(global_test_dir / "features.npy", self.global_test_set["features"])
            np.save(global_test_dir / "regression_labels.npy", self.global_test_set["regression_labels"])
            np.save(global_test_dir / "classification_labels.npy", self.global_test_set["classification_labels"])
            if self.global_test_set["phase_labels"] is not None:
                np.save(global_test_dir / "phase_labels.npy", self.global_test_set["phase_labels"])
            with open(global_test_dir / "experiment_info.json", "w", encoding='utf-8') as f:
                json.dump(self.global_test_set["experiment_info"], f, indent=2, ensure_ascii=False)
            with open(global_test_dir / "stats.json", "w", encoding='utf-8') as f:
                json.dump(self.global_test_set["stats"], f, indent=2, ensure_ascii=False)
            if self.config.verbose:
                print(f"   全局测试集已保存: {len(self.global_test_set['features']):,} 个样本")
            
            # 保存分阶段测试集
            phase_test_sets = self.global_test_set.get("phase_test_sets", {})
            for phase, test_set in phase_test_sets.items():
                phase_dir = output_path / f"global_test_{phase}"
                phase_dir.mkdir(exist_ok=True)
                np.save(phase_dir / "features.npy", test_set["features"])
                np.save(phase_dir / "regression_labels.npy", test_set["regression_labels"])
                np.save(phase_dir / "classification_labels.npy", test_set["classification_labels"])
                np.save(phase_dir / "phase_labels.npy", test_set["phase_labels"])
                if self.config.verbose:
                    print(f"   {phase}阶段测试集已保存: {len(test_set['features']):,} 个样本")
        # 保存校准集
        if hasattr(self, 'calibration_set') and self.calibration_set:
            calibration_dir = output_path / "calibration"
            calibration_dir.mkdir(exist_ok=True)
            np.save(calibration_dir / "features.npy", self.calibration_set["features"])
            np.save(calibration_dir / "regression_labels.npy", self.calibration_set["regression_labels"])
            np.save(calibration_dir / "classification_labels.npy", self.calibration_set["classification_labels"])
            if self.calibration_set["phase_labels"] is not None:
                np.save(calibration_dir / "phase_labels.npy", self.calibration_set["phase_labels"])
            with open(calibration_dir / "experiment_info.json", "w", encoding='utf-8') as f:
                json.dump(self.calibration_set["experiment_info"], f, indent=2, ensure_ascii=False)
            with open(calibration_dir / "stats.json", "w", encoding='utf-8') as f:
                json.dump(self.calibration_set["stats"], f, indent=2, ensure_ascii=False)
            if self.config.verbose:
                print(f"   校准集已保存: {len(self.calibration_set['features']):,} 个样本")

        # 保存 split_info.json
        split_info = {
            "n_clients": len(self.clients_data), "split_method": self.config.split_method.value,
            "split_ratios": {"train": self.config.train_ratio, "global_test": self.config.global_test_ratio, "calibration": self.config.calibration_ratio},
            "seed": self.config.seed, "client_ids": list(self.clients_data.keys()),
            "has_global_test": self.global_test_set is not None, "has_calibration": hasattr(self, 'calibration_set') and self.calibration_set is not None,
            "regression_full_grid": self.config.regression_full_grid,
            "regression_window_full_grid": self.config.regression_window_full_grid,
            "split_level": "window" if self.config.regression_window_full_grid else "file",
            "allows_file_overlap": bool(self.config.regression_window_full_grid),
            "full_grid_source_clients": self.config.full_grid_source_clients,
            "full_grid_target_clients": self.config.full_grid_target_clients,
            "creation_time": Path(__file__).stat().st_ctime
        }
        with open(output_path / "split_info.json", "w", encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        # 计算并保存训练客户端的归一化统计量（从所有训练客户端合并）
        train_features = []
        for unit_id in self.clients_data.keys():
            train_features.append(self.clients_data[unit_id]['train']['features'])
        
        if train_features:
            all_features = np.concatenate(train_features, axis=0)
            mean = all_features.mean(axis=(0, 1), keepdims=True)  # shape (1, 1, 8)
            std = all_features.std(axis=(0, 1), keepdims=True) + 1e-8
            
            norm_stats_path = output_path / "norm_stats.npz"
            np.savez(norm_stats_path, mean=mean, std=std)
            
            if self.config.verbose:
                print(f"   训练集归一化统计量已保存: {norm_stats_path}")
                print(f"   均值形状: {mean.shape}, 标准差形状: {std.shape}")
        
        if self.config.save_visualizations:
            self._create_visualizations(output_path)
        if self.config.verbose:
            print(f"\n 数据集保存完成!")
            print(f"   输出目录: {output_path}")
            print(f"   客户端数量: {len(self.clients_data)}")
            if self.global_test_set:
                print(f"   全局测试集: {len(self.global_test_set['features']):,} 个样本")
            if hasattr(self, 'calibration_set') and self.calibration_set:
                print(f"   校准集: {len(self.calibration_set['features']):,} 个样本")
    
    def _create_visualizations(self, output_path: Path):
        # 省略可视化代码（保留原功能，不影响核心流程）
        pass


# =========================
# 高级API接口
# =========================

def create_federated_dataset(processed_dir: str, output_dir: str = "dataset/client_data_federated",
                             config: Optional[FederatedSplitConfig] = None) -> Dict[str, Any]:
    """创建联邦学习数据集
    
    从预处理数据创建联邦学习数据集，支持多种划分方法
    
    Args:
        processed_dir: 预处理数据目录
        output_dir: 输出目录
        config: 划分配置
        
    Returns:
        数据集创建结果字典
    """
    print("=" * 60)
    print(" 创建联邦学习数据集")
    print("=" * 60)
    if config is None:
        config = FederatedSplitConfig()
        print(f" 使用默认配置")
    splitter = FederatedDatasetSplitter(config)
    print(f" 加载预处理数据...")
    try:
        unit_data = splitter.load_processed_data(processed_dir)
        print(f" 数据加载成功: {len(unit_data)} 个单元")
    except Exception as e:
        print(f" 数据加载失败: {e}")
        raise
    print(f" 创建联邦数据集...")
    try:
        result = splitter.create_federated_dataset(unit_data)
        print(f" 联邦数据集创建成功")
    except Exception as e:
        print(f" 联邦数据集创建失败: {e}")
        raise
    print(f" 保存数据集...")
    try:
        splitter.save_federated_dataset(output_dir)
        print(f" 数据集保存成功: {output_dir}")
    except Exception as e:
        print(f" 数据集保存失败: {e}")
        raise
    print("=" * 60)
    print(" 联邦学习数据集创建完成!")
    print("=" * 60)
    return result


if __name__ == "__main__":
    processed_dir = "./dataset/processed"
    output_dir = "./dataset/client_data_federated"
    config = FederatedSplitConfig(
        split_method=SplitMethod.UNIT_BASED,
        train_ratio=0.70, global_test_ratio=0.20, calibration_ratio=0.10,
        stratify=True, ensure_balance=True, global_test_sampling=GlobalTestSampling.STRATIFIED,
        verbose=True, save_visualizations=True
    )
    try:
        result = create_federated_dataset(processed_dir, output_dir, config)
    except Exception as e:
        print(f" 错误: {e}")
        import traceback
        traceback.print_exc()
