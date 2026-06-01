"""气体传感器数据预处理模块

实现用户要求的数据预处理流程：
1. 去除不稳定阶段（前20秒）
2. 降采样（100Hz → 10Hz）
3. 基线校正（计算相对电导率）
4. 截取响应区间（30秒 ~ 330秒）
5. 滑动窗口生成（窗口长度100，步长50）
6. 标签与元信息生成
7. 数据归一化（Z-score）
8. 数据划分（按文件级）
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GasSensorPreprocessor")

# =========================
# 配置类和数据类定义
# =========================

@dataclass
class PreprocessConfig:
    """预处理配置参数"""
    original_fs: int = 100              # 原始采样率 (Hz)
    target_fs: int = 10                 # 目标采样率 (Hz)
    unstable_duration: int = 20         # 去除不稳定阶段时长 (秒)
    baseline_duration: int = 30         # 基线计算时长 (秒)
    response_start: int = 40            # 响应开始时间 (秒)
    response_end: int = 150             # 响应结束时间 (秒)
    window_size: int =100               # 时间窗口长度 (目标采样率下的点数)
    step_size: int = 50                 # 滑动步长 (50%重叠)
    normalization: str = "zscore"       # 归一化方法: "zscore", "none"
    use_relative_conductance: bool = True  # 是否使用相对电导率
    seed: int = 42                      # 随机种子
    expected_duration: int = 600        # 期望的总时长 (秒)
    data_completeness_threshold: float = 0.95  # 数据完整度阈值 (低于此值记录警告)
    phase_early_max_days: int = 5       # 早期阶段最大天数
    phase_middle_max_days: int = 15     # 中期阶段最大天数


@dataclass
class ProcessingStats:
    """处理统计信息"""
    filename: str
    unit_id: int
    gas_type: str
    concentration: float
    original_points: int = 0
    stable_points: int = 0
    downsampled_points: int = 0
    response_points: int = 0
    window_count: int = 0
    status: str = "success"
    error_message: str = ""
    processing_time: float = 0.0


@dataclass
class ExperimentInfo:
    """实验元信息"""
    unit_id: int                        # 单元编号 (1-5)
    gas_code: str                      # 气体代码: "Ea", "CO", "Ey", "Me"
    gas_type: str                      # 气体类型全称
    concentration: float               # 浓度值 (ppm)
    concentration_code: str            # 浓度代码: "010"-"100"
    repeat_id: int                     # 重复实验编号
    filename: str                      # 原始文件名
    processing_stats: ProcessingStats  # 处理统计信息
    
    @property
    def classification_label(self) -> int:
        """获取分类标签 (0-3)"""
        return GAS_CODE_TO_LABEL[self.gas_code]
    
    @property
    def regression_label(self) -> np.ndarray:
        """获取回归标签 (4种气体的浓度)"""
        label = np.zeros(4, dtype=np.float32)
        label[self.classification_label] = self.concentration
        return label
    
    @property
    def phase_label(self) -> str:
        """获取阶段标签"""
        # 测试天数映射（根据用户提供的数据集信息）
        unit_days_map = {
            1: {1: 4, 2: 10, 3: 15, 4: 21},  # Unit 1: R1=4天, R2=10天, R3=15天, R4=21天
            2: {1: 1, 2: 7, 3: 11, 4: 16},  # Unit 2: R1=1天, R2=7天, R3=11天, R4=16天
            3: {1: 2, 2: 8, 3: 14, 4: 17},  # Unit 3: R1=2天, R2=8天, R3=14天, R4=17天
            4: {1: 3, 2: 9},               # Unit 4: R1=3天, R2=9天
            5: {1: 18, 2: 22}              # Unit 5: R1=18天, R2=22天
        }
        
        # 获取实际测试天数
        unit_repeats = unit_days_map.get(self.unit_id)
        if unit_repeats is None:
            raise ValueError(f"单元 {self.unit_id} 不存在于 unit_days_map 中，请补全映射或检查数据。")
        
        days = unit_repeats.get(self.repeat_id)
        if days is None:
            raise ValueError(f"单元 {self.unit_id} 的 Repeat {self.repeat_id} 不存在于 unit_days_map 中。")
        
        # 获取配置参数
        from preprocessor import PreprocessConfig
        config = PreprocessConfig()
        
        # 根据天数划分阶段
        if days <= config.phase_early_max_days:
            return "early"
        elif days <= config.phase_middle_max_days:
            return "middle"
        else:
            return "late"

    # @property
    # def phase_label(self) -> str:
    #     # 获取该单元的所有repeat天数（从预定义的映射中提取）
    #     unit_days_map = {
    #         1: {1: 4, 2: 10, 3: 15, 4: 21},
    #         2: {1: 1, 2: 7, 3: 11, 4: 16},
    #         3: {1: 2, 2: 8, 3: 14, 4: 17},
    #         4: {1: 3, 2: 9},
    #         5: {1: 18, 2: 22}
    #     }
    #     unit_repeats = unit_days_map.get(self.unit_id, {})
    #     if not unit_repeats:
    #         return "early"
        
    #     # 获取当前repeat的天数
    #     current_days = unit_repeats[self.repeat_id]
    #     all_days = sorted(unit_repeats.values())
    #     min_days, max_days = all_days[0], all_days[-1]
        
    #     # 归一化相对时间进度 (0~1)
    #     if max_days == min_days:
    #         norm_progress = 0.0
    #     else:
    #         norm_progress = (current_days - min_days) / (max_days - min_days)
        
    #     # 映射到phase
    #     if norm_progress <= 0.33:
    #         return "early"
    #     elif norm_progress <= 0.66:
    #         return "middle"
    #     else:
    #         return "late"

# =========================
# 全局映射定义
# =========================

# 气体类型映射
GAS_TYPE_MAP = {
    "Ea": "ethanol",
    "CO": "carbon_monoxide",
    "Ey": "ethylene",
    "Me": "methane",
}

# 气体代码到标签索引的映射
GAS_CODE_TO_LABEL = {
    "Ea": 0,    # 乙醇 12.5-125ppm
    "CO": 1,    # 一氧化碳 25.0-250ppm
    "Ey": 2,    # 乙烯 12.5-125ppm
    "Me": 3,    # 甲烷 25.0-250ppm
}

# 气体标签到代码的反向映射
LABEL_TO_GAS_CODE = {v: k for k, v in GAS_CODE_TO_LABEL.items()}


def get_concentration_value(gas_code: str, concentration_code: str) -> float:
    """
    根据气体类型和浓度代码获取实际浓度值(ppm)
    
    Args:
        gas_code: 气体代码 ("Ea", "CO", "Ey", "Me")
        concentration_code: 浓度代码 ("010", "020", ..., "100")
        
    Returns:
        浓度值 (ppm)
        
    Raises:
        ValueError: 如果浓度代码不在有效范围内
    """
    # 浓度等级映射 010→0, 020→1 ... 100→9
    try:
        conc_num = int(concentration_code)
        level = (conc_num // 10) - 1
    except:
        raise ValueError(f"浓度代码 {concentration_code} 格式错误")
        
    # 检查等级索引是否在有效范围内
    if not (0 <= level <= 9):
        raise ValueError(f"浓度代码必须是 010~100，得到：{concentration_code}")
    
    # 乙醇和乙烯的浓度映射
    if gas_code in ["Ea", "Ey"]:
        concentrations = [12.5, 25.0, 37.5, 50.0, 62.5,
                         75.0, 87.5, 100.0, 112.5, 125.0]
    # 一氧化碳和甲烷的浓度映射
    elif gas_code in ["CO", "Me"]:
        concentrations = [25.0, 50.0, 75.0, 100.0, 125.0,
                         150.0, 175.0, 200.0, 225.0, 250.0]
    else:
        raise ValueError(f"未知气体代码: {gas_code}")
    
    return concentrations[level]


# =========================
# 主要预处理器类
# =========================

class GasSensorPreprocessor:
    """
    气体传感器数据预处理器
    
    实现用户要求的预处理流程
    """
    
    def __init__(self, config: Optional[PreprocessConfig] = None):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置，如果为None则使用默认配置
        """
        self.config = config or PreprocessConfig()
        self._validate_config()
        
        # 设置随机种子
        np.random.seed(self.config.seed)
        
        # 计算期望的数据点数
        self.expected_original_points = self.config.expected_duration * self.config.original_fs
        
        logger.info(f"初始化预处理器，配置: {self.config}")
        logger.info(f"期望原始数据点数: {self.expected_original_points}")
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        if self.config.target_fs <= 0:
            raise ValueError("目标采样率必须大于0")
        if self.config.original_fs % self.config.target_fs != 0:
            logger.warning(f"原始采样率{self.config.original_fs}Hz不是目标采样率"
                         f"{self.config.target_fs}Hz的整数倍，降采样可能不精确")
        if self.config.window_size <= 0:
            raise ValueError("窗口大小必须大于0")
        if self.config.step_size <= 0:
            raise ValueError("步长必须大于0")
        if self.config.normalization not in ["zscore", "none"]:
            raise ValueError("归一化方法必须是'zscore'或'none'")
        
        # 检查响应时间范围
        if self.config.response_start >= self.config.response_end:
            raise ValueError("响应开始时间必须小于响应结束时间")
    
    def parse_filename(self, filename: str) -> ExperimentInfo:
        """
        从文件名中解析实验信息
        
        Args:
            filename: 文件名，如"B1_GEa_F050_R2.txt"
            
        Returns:
            ExperimentInfo: 实验信息对象
            
        Raises:
            ValueError: 如果文件名格式不符合规范
        """
        pattern = r"B(\d+)_G([A-Za-z]+)_F(\d+)_R(\d+)"
        match = re.search(pattern, filename)
        
        if match is None:
            raise ValueError(f"文件名格式不符合规范: {filename}")
        
        unit_id = int(match.group(1))
        gas_code = match.group(2)
        conc_code = match.group(3)
        repeat_id = int(match.group(4))
        
        # 验证气体代码
        if gas_code not in GAS_CODE_TO_LABEL:
            raise ValueError(f"未知气体代码: {gas_code}")
        
        # 验证浓度代码格式
        if not re.match(r"\d{3}", conc_code):
            raise ValueError(f"浓度代码格式错误: {conc_code}")
        
        # 获取气体类型全称
        gas_type = GAS_TYPE_MAP.get(gas_code, gas_code)
        
        # 获取浓度值
        concentration = get_concentration_value(gas_code, conc_code)
        
        # 创建处理统计信息
        processing_stats = ProcessingStats(
            filename=filename,
            unit_id=unit_id,
            gas_type=gas_type,
            concentration=concentration
        )
        
        return ExperimentInfo(
            unit_id=unit_id,
            gas_code=gas_code,
            gas_type=gas_type,
            concentration=concentration,
            concentration_code=conc_code,
            repeat_id=repeat_id,
            filename=filename,
            processing_stats=processing_stats
        )
    
    def load_raw_data(self, filepath: str) -> np.ndarray:
        """
        加载原始数据文件，并进行完整性检查
        
        Args:
            filepath: 文件路径
            
        Returns:
            np.ndarray: 传感器数据，形状 (n_samples, 8)
        """
        filename = os.path.basename(filepath)
        
        try:
            # 尝试不同分隔符加载数据
            for delimiter in [None, "\t", " ", ","]:
                try:
                    data = np.loadtxt(filepath, delimiter=delimiter)
                    if data.shape[1] == 9:  # 时间列 + 8个传感器
                        break
                except:
                    continue
            
            # 检查数据形状
            if data.shape[1] != 9:
                # 尝试手动解析
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                # 解析每一行
                parsed_data = []
                for line in lines:
                    # 分割数字（支持空格、制表符、多个空格）
                    values = re.split(r'\s+', line.strip())
                    if len(values) >= 9:
                        parsed_data.append([float(v) for v in values[:9]])
                
                data = np.array(parsed_data)
                
                if data.shape[1] != 9:
                    raise ValueError(f"数据形状错误: 期望9列(时间+8传感器)，实际{data.shape[1]}列")
            
            # 提取传感器数据（去掉时间列）
            sensor_data = data[:, 1:]  # 形状 (n_samples, 8)
            
            # 检查数据完整性
            n_samples = sensor_data.shape[0]
            completeness = n_samples / self.expected_original_points
            
            if completeness < self.config.data_completeness_threshold:
                logger.warning(
                    f"文件 {filename}: 数据不完整，"
                    f"期望 {self.expected_original_points} 点，"
                    f"实际 {n_samples} 点，"
                    f"完整度: {completeness:.2%}"
                )
            
            # 记录原始数据点数
            if hasattr(self, 'current_stats'):
                self.current_stats.original_points = n_samples
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"加载文件 {filepath} 失败: {e}")
            raise
    
    def remove_unstable_phase(self, data: np.ndarray) -> np.ndarray:
        """
        去除初始不稳定阶段
        
        Args:
            data: 原始数据，形状 (n_samples, n_sensors)
            
        Returns:
            np.ndarray: 去除不稳定阶段后的数据
        """
        unstable_points = self.config.unstable_duration * self.config.original_fs
        
        # 检查是否有足够的数据点
        if unstable_points >= data.shape[0]:
            logger.warning(
                f"数据长度 ({data.shape[0]}) 不足以去除不稳定阶段 ({unstable_points} 点)"
            )
            return data
        
        stable_data = data[unstable_points:, :]
        
        # 记录稳定数据点数
        if hasattr(self, 'current_stats'):
            self.current_stats.stable_points = len(stable_data)
        
        return stable_data
    
    def downsample_data(self, data: np.ndarray) -> np.ndarray:
        """
        降采样数据
        
        Args:
            data: 原始数据，形状 (n_samples, n_sensors)
            
        Returns:
            np.ndarray: 降采样后的数据
        """
        if self.config.original_fs == self.config.target_fs:
            return data
        
        downsample_ratio = self.config.original_fs // self.config.target_fs
        
        # 确保数据长度是降采样比率的整数倍
        n_samples = data.shape[0]
        truncated_length = (n_samples // downsample_ratio) * downsample_ratio
        
        if truncated_length == 0:
            raise ValueError(f"数据长度({n_samples})不足以进行{downsample_ratio}倍降采样")
        
        # 如果有截断，记录日志
        if truncated_length < n_samples:
            logger.info(f"降采样前截断数据: {n_samples} -> {truncated_length}")
        
        data = data[:truncated_length, :]
        
        # 重塑并取均值降采样（抗混叠）
        n_new_samples = truncated_length // downsample_ratio
        downsampled = data.reshape(n_new_samples, downsample_ratio, -1).mean(axis=1)
        
        # 记录降采样后点数
        if hasattr(self, 'current_stats'):
            self.current_stats.downsampled_points = len(downsampled)
        
        return downsampled
    
    def calculate_relative_conductance(self, data: np.ndarray) -> np.ndarray:
        """
        计算相对电导率变化 (ΔG/G₀)
        
        电导率 G = 1/R，其中R为电阻值(KΩ)
        相对变化 = (G - G₀) / G₀，其中G₀为基线电导率
        
        Args:
            data: 原始电阻数据，形状 (n_samples, n_sensors)
            
        Returns:
            np.ndarray: 相对电导率变化
        """
        # 转换为电导率 (G = 1/R)
        # 添加小常数避免除以0
        conductance = 1 / (data + 1e-10)
        
        # 计算基线平均值（前baseline_duration秒）
        baseline_samples = self.config.baseline_duration * self.config.target_fs
        
        # 确保有足够的基线数据
        if baseline_samples > len(conductance):
            logger.warning(
                f"数据长度 ({len(conductance)}) 不足以计算基线 ({baseline_samples} 点)，"
                f"使用前50%数据作为基线"
            )
            baseline_samples = len(conductance) // 2
        
        baseline = np.mean(conductance[:baseline_samples, :], axis=0, keepdims=True)
        
        # 避免基线为0
        baseline = np.where(baseline == 0, 1e-10, baseline)
        
        # 计算相对变化
        relative_conductance = (conductance - baseline) / baseline
        
        return relative_conductance
    
    def extract_response_region(self, data: np.ndarray) -> np.ndarray:
        """
        提取关键响应区域（降采样后时间轴）
        默认截取第 40 秒到第 150 秒（对应原始实验第 60~170 秒）
        
        Args:
            data: 输入数据，形状 (n_samples, n_sensors)
            
        Returns:
            np.ndarray: 响应区域数据
        """
        start_idx = self.config.response_start * self.config.target_fs
        end_idx = self.config.response_end * self.config.target_fs
        
        # 检查起始索引
        if start_idx >= len(data):
            raise ValueError(f"响应开始时间({self.config.response_start}s)超出数据范围")
        
        # 检查结束索引
        if end_idx > len(data):
            logger.warning(
                f"响应结束时间({self.config.response_end}s)超出数据范围({len(data)}点)，"
                f"使用数据末尾"
            )
            end_idx = len(data)
        
        response_data = data[start_idx:end_idx, :]
        
        # 记录响应区域点数
        if hasattr(self, 'current_stats'):
            self.current_stats.response_points = len(response_data)
        
        return response_data
    
    def create_time_windows(self, sequence: np.ndarray) -> np.ndarray:
        """
        将时间序列分割成重叠窗口
        
        Args:
            sequence: 输入序列，形状 (n_samples, n_sensors)
            
        Returns:
            np.ndarray: 窗口数据，形状 (n_windows, window_size, n_sensors)
        """
        n_samples, n_sensors = sequence.shape
        window_size = self.config.window_size
        step_size = self.config.step_size
        
        # 如果序列太短，进行零填充
        if n_samples < window_size:
            logger.warning(
                f"序列长度({n_samples})小于窗口大小({window_size})，进行零填充。"
                f"响应区间长度不足，将生成1个窗口。"
            )
            padded = np.zeros((window_size, n_sensors))
            padded[:n_samples, :] = sequence
            windows = padded[np.newaxis, ...]
        else:
            windows = []
            for start in range(0, n_samples - window_size + 1, step_size):
                window = sequence[start:start + window_size, :]
                windows.append(window)
            
            if not windows:
                # 如果没有任何窗口，创建一个
                logger.warning(
                    f"序列长度({n_samples})不足以创建窗口(窗口大小{window_size}，步长{step_size})，"
                    f"使用前{window_size}个点创建单个窗口"
                )
                window = sequence[:window_size, :]
                windows.append(window)
            
            windows = np.array(windows)
        
        # 记录窗口数
        if hasattr(self, 'current_stats'):
            self.current_stats.window_count = len(windows)
        
        return windows
    
    def normalize_data(self, data: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        数据归一化
        
        Args:
            data: 输入数据
            mean: 均值，如果为None则计算
            std: 标准差，如果为None则计算
            
        Returns:
            Tuple: (归一化后的数据, 均值, 标准差)
        """
        if self.config.normalization == "none":
            return data, np.zeros(data.shape[2]), np.ones(data.shape[2])
        
        if self.config.normalization == "zscore":
            # Z-score标准化（每个传感器独立标准化）
            if mean is None or std is None:
                mean = np.mean(data, axis=(0, 1), keepdims=True)
                std = np.std(data, axis=(0, 1), keepdims=True) + 1e-10
            
            normalized = (data - mean) / std
            return normalized, mean, std
        
        else:
            raise ValueError(f"不支持的归一化方法: {self.config.normalization}")
    
    def process_single_experiment(self, filepath: str) -> Dict[str, Any]:
        """
        处理单个实验文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            Dict: 包含处理后的数据和元信息
        """
        import time
        start_time = time.time()
        
        filename = os.path.basename(filepath)
        logger.info(f"开始处理文件: {filename}")
        
        try:
            # 解析文件名
            experiment_info = self.parse_filename(filename)
            self.current_stats = experiment_info.processing_stats
            
            logger.info(
                f"解析信息 - 单元: {experiment_info.unit_id}, "
                f"气体: {experiment_info.gas_type}, "
                f"浓度: {experiment_info.concentration}ppm, "
                f"阶段: {experiment_info.phase_label}"
            )
            
            # 加载原始数据
            raw_data = self.load_raw_data(filepath)
            
            # 去除初始不稳定阶段
            stable_data = self.remove_unstable_phase(raw_data)
            
            # 降采样
            downsampled = self.downsample_data(stable_data)
            
            # 计算相对电导率变化
            if self.config.use_relative_conductance:
                processed_data = self.calculate_relative_conductance(downsampled)
            else:
                processed_data = downsampled
            
            # 提取响应区域
            response_data = self.extract_response_region(processed_data)
            
            # 创建时间窗口
            windows = self.create_time_windows(response_data)
            
            # 生成标签
            n_windows = len(windows)
            
            # 回归标签：4种气体的浓度
            regression_labels = np.tile(
                experiment_info.regression_label,
                (n_windows, 1)
            )
            
            # 分类标签：气体类型
            classification_labels = np.full(n_windows, experiment_info.classification_label)
            
            # 阶段标签：根据重复实验编号
            phase_labels = np.full(n_windows, experiment_info.phase_label)
            
            # 实验信息（每个窗口一份）
            meta_list = [experiment_info] * n_windows
            
            # 计算处理时间
            processing_time = time.time() - start_time
            self.current_stats.processing_time = processing_time
            
            logger.info(
                f"文件处理完成: {filename}, "
                f"窗口数: {n_windows}, "
                f"阶段: {experiment_info.phase_label}, "
                f"处理时间: {processing_time:.2f}秒"
            )
            
            result = {
                "features": windows,            # 形状 (n_windows, window_size, 8)
                "regression_labels": regression_labels,    # 形状 (n_windows, 4)
                "classification_labels": classification_labels,  # 形状 (n_windows,)
                "phase_labels": phase_labels,              # 形状 (n_windows,)
                "experiment_info": meta_list,              # 列表，长度 n_windows
                "unit_id": experiment_info.unit_id,
                "filename": filename,
                "n_windows": n_windows,
                "processing_stats": self.current_stats
            }
            
            # 记录详细统计信息
            logger.debug(
                f"处理详情 - {filename}:\n"
                f"  原始点数: {self.current_stats.original_points}\n"
                f"  稳定后点数: {self.current_stats.stable_points}\n"
                f"  降采样后点数: {self.current_stats.downsampled_points}\n"
                f"  响应区域点数: {self.current_stats.response_points}\n"
                f"  窗口数: {self.current_stats.window_count}"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"处理文件 {filename} 失败: {e}")
            
            # 更新统计信息为失败状态
            if hasattr(self, 'current_stats'):
                self.current_stats.status = "failed"
                self.current_stats.error_message = str(e)
                self.current_stats.processing_time = processing_time
            
            raise
    
    def process_dataset(self, data_dir: str, 
                       output_dir: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
        """
        处理整个数据集目录
        
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录，如果提供则保存处理后的数据
            
        Returns:
            Dict: 按单元分组的数据字典
        """
        logger.info(f"开始处理数据集目录: {data_dir}")
        
        data_path = Path(data_dir)
        
        # 查找所有.txt文件
        txt_files = list(data_path.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"在目录 {data_dir} 中未找到.txt文件")
        
        logger.info(f"找到 {len(txt_files)} 个数据文件")
        
        # 按单元分组处理
        unit_data = {}
        processing_stats_list = []
        processed_count = 0
        error_count = 0
        
        for file_path in txt_files:
            try:
                # 处理单个文件
                result = self.process_single_experiment(str(file_path))
                unit_id = result["unit_id"]
                
                # 收集处理统计信息
                if "processing_stats" in result:
                    processing_stats_list.append(result["processing_stats"])
                
                # 初始化单元数据字典
                if unit_id not in unit_data:
                    unit_data[unit_id] = {
                        "features": [],
                        "regression_labels": [],
                        "classification_labels": [],
                        "phase_labels": [],
                        "experiment_info": [],
                        "filenames": [],
                        "n_windows_per_file": [],
                        "processing_stats": []
                    }
                
                # 添加到单元数据
                unit_data[unit_id]["features"].append(result["features"])
                unit_data[unit_id]["regression_labels"].append(result["regression_labels"])
                unit_data[unit_id]["classification_labels"].append(result["classification_labels"])
                unit_data[unit_id]["phase_labels"].append(result["phase_labels"])
                unit_data[unit_id]["experiment_info"].extend(result["experiment_info"])
                unit_data[unit_id]["filenames"].append(result["filename"])
                unit_data[unit_id]["n_windows_per_file"].append(result["n_windows"])
                unit_data[unit_id]["processing_stats"].append(result.get("processing_stats"))
                
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                continue
        
        logger.info(f"处理完成: 成功 {processed_count} 个文件，失败 {error_count} 个文件")
        
        # 合并每个单元的数据
        processed_data = {}
        for unit_id, data in unit_data.items():
            if data["features"]:
                features = np.concatenate(data["features"], axis=0)
                regression_labels = np.concatenate(data["regression_labels"], axis=0)
                classification_labels = np.concatenate(data["classification_labels"], axis=0)
                phase_labels = np.concatenate(data["phase_labels"], axis=0)
                
                processed_data[unit_id] = {
                    "features": features,
                    "regression_labels": regression_labels,
                    "classification_labels": classification_labels,
                    "phase_labels": phase_labels,
                    "experiment_info": data["experiment_info"],
                    "filenames": data["filenames"],
                    "n_samples": len(features),
                    "n_windows_per_file": data["n_windows_per_file"],
                    "processing_stats": data["processing_stats"]
                }
                
                logger.info(
                    f"单元 {unit_id}: {len(features)} 个样本, "
                    f"来自 {len(data['filenames'])} 个文件"
                )
        
        # 生成处理统计报告
        self._generate_processing_report(processing_stats_list, output_dir)
        
        # 保存处理后的数据
        if output_dir:
            self.save_processed_data(processed_data, output_dir)
        
        return processed_data
    
    def _generate_processing_report(self, stats_list: List[ProcessingStats], 
                                   output_dir: Optional[str] = None):
        """生成处理统计报告"""
        if not stats_list:
            return
        
        successful_stats = [s for s in stats_list if s.status == "success"]
        failed_stats = [s for s in stats_list if s.status == "failed"]
        
        # 按单元分组统计
        unit_stats = {}
        for stat in successful_stats:
            unit_id = stat.unit_id
            if unit_id not in unit_stats:
                unit_stats[unit_id] = []
            unit_stats[unit_id].append(stat)
        
        # 生成报告
        report = {
            "summary": {
                "total_files": len(stats_list),
                "successful_files": len(successful_stats),
                "failed_files": len(failed_stats),
                "success_rate": len(successful_stats) / len(stats_list) if stats_list else 0
            },
            "unit_statistics": {},
            "data_completeness": {},
            "detailed_statistics": []
        }
        
        # 单元统计
        for unit_id, stats in unit_stats.items():
            window_counts = [s.window_count for s in stats]
            original_points = [s.original_points for s in stats]
            
            report["unit_statistics"][unit_id] = {
                "file_count": len(stats),
                "total_windows": sum(window_counts),
                "avg_windows_per_file": np.mean(window_counts),
                "std_windows_per_file": np.std(window_counts),
                "min_windows_per_file": min(window_counts),
                "max_windows_per_file": max(window_counts),
                "avg_original_points": np.mean(original_points),
                "completeness_rate": np.mean([p / self.expected_original_points for p in original_points])
            }
        
        # 数据完整性统计
        completeness_rates = []
        for stat in successful_stats:
            if stat.original_points > 0:
                completeness = stat.original_points / self.expected_original_points
                completeness_rates.append(completeness)
        
        if completeness_rates:
            report["data_completeness"] = {
                "mean_completeness": np.mean(completeness_rates),
                "std_completeness": np.std(completeness_rates),
                "min_completeness": min(completeness_rates),
                "max_completeness": max(completeness_rates),
                "files_below_threshold": sum(1 for rate in completeness_rates 
                                           if rate < self.config.data_completeness_threshold)
            }
        
        # 详细统计
        for stat in stats_list:
            report["detailed_statistics"].append({
                "filename": stat.filename,
                "unit_id": stat.unit_id,
                "gas_type": stat.gas_type,
                "concentration": stat.concentration,
                "original_points": stat.original_points,
                "window_count": stat.window_count,
                "status": stat.status,
                "error_message": stat.error_message,
                "processing_time": stat.processing_time
            })
        
        # 输出报告
        logger.info("\n" + "="*60)
        logger.info("处理统计报告")
        logger.info("="*60)
        logger.info(f"总文件数: {report['summary']['total_files']}")
        logger.info(f"成功处理: {report['summary']['successful_files']}")
        logger.info(f"处理失败: {report['summary']['failed_files']}")
        logger.info(f"成功率: {report['summary']['success_rate']:.2%}")
        
        for unit_id, stats in report["unit_statistics"].items():
            logger.info(f"\n单元 {unit_id}:")
            logger.info(f"  文件数: {stats['file_count']}")
            logger.info(f"  总窗口数: {stats['total_windows']}")
            logger.info(f"  平均窗口/文件: {stats['avg_windows_per_file']:.1f}")
            logger.info(f"  数据完整率: {stats['completeness_rate']:.2%}")
        
        if report["data_completeness"]:
            logger.info(f"\n数据完整性:")
            logger.info(f"  平均完整度: {report['data_completeness']['mean_completeness']:.2%}")
            logger.info(f"  低于阈值文件数: {report['data_completeness']['files_below_threshold']}")
        
        # 保存报告到文件
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            report_path = output_path / "processing_report.json"
            with open(report_path, "w") as f:
                # 将dataclass对象转换为字典
                def convert_to_dict(obj):
                    if isinstance(obj, ProcessingStats):
                        return obj.__dict__
                    return obj
                
                json.dump(report, f, default=convert_to_dict, indent=2)
            
            logger.info(f"\n详细报告已保存到: {report_path}")
    
    def compute_and_save_norm_stats(self, features_list: List[np.ndarray], save_path: str):
        """
        计算并保存全局归一化统计量
        
        Args:
            features_list: 特征列表，每个元素是一个单元的特征数组
            save_path: 保存路径
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (均值, 标准差)
        """
        all_features = np.concatenate(features_list, axis=0)
        mean = all_features.mean(axis=(0, 1), keepdims=True)  # shape (1, 1, 8)
        std = all_features.std(axis=(0, 1), keepdims=True) + 1e-8
        
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        np.savez(save_path, mean=mean, std=std)
        logger.info(f"保存归一化统计量到: {save_path}")
        logger.info(f"均值形状: {mean.shape}, 标准差形状: {std.shape}")
        return mean, std
    
    def save_processed_data(self, processed_data: Dict[int, Dict], output_dir: str):
        """
        保存处理后的数据
        
        Args:
            processed_data: 处理后的数据字典
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存处理后的数据到: {output_dir}")
        
        # 保存配置
        config_dict = {
            "original_fs": self.config.original_fs,
            "target_fs": self.config.target_fs,
            "unstable_duration": self.config.unstable_duration,
            "baseline_duration": self.config.baseline_duration,
            "response_start": self.config.response_start,
            "response_end": self.config.response_end,
            "window_size": self.config.window_size,
            "step_size": self.config.step_size,
            "normalization": self.config.normalization,
            "use_relative_conductance": self.config.use_relative_conductance,
            "seed": self.config.seed,
            "expected_duration": self.config.expected_duration,
            "data_completeness_threshold": self.config.data_completeness_threshold
        }
        
        with open(output_path / "preprocess_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # 计算并保存全局归一化统计量
        features_list = []
        for unit_id, data in processed_data.items():
            features_list.append(data["features"])
        
        norm_stats_path = output_path / "norm_stats.npz"
        self.compute_and_save_norm_stats(features_list, str(norm_stats_path))
        
        # 保存每个单元的数据
        for unit_id, data in processed_data.items():
            unit_dir = output_path / f"unit_{unit_id}"
            unit_dir.mkdir(exist_ok=True)
            
            logger.info(f"保存单元 {unit_id} 数据: {data['n_samples']} 个样本")
            
            np.save(unit_dir / "features.npy", data["features"])
            np.save(unit_dir / "regression_labels.npy", data["regression_labels"])
            np.save(unit_dir / "classification_labels.npy", data["classification_labels"])
            # 将阶段标签转换为整数格式保存
            phase_map = {'early': 0, 'middle': 1, 'late': 2}
            phase_int = np.array([phase_map[p] for p in data["phase_labels"]], dtype=np.int8)
            np.save(unit_dir / "phase_labels.npy", phase_int)
            
            # 保存实验信息（简化版本）
            simple_meta = []
            for info in data["experiment_info"]:
                simple_meta.append({
                    "unit_id": info.unit_id,
                    "gas_code": info.gas_code,
                    "gas_type": info.gas_type,
                    "concentration": info.concentration,
                    "concentration_code": info.concentration_code,
                    "repeat_id": info.repeat_id,
                    "filename": info.filename,
                    "phase_label": info.phase_label,
                    "classification_label": info.classification_label,
                    "regression_label": info.regression_label.tolist()
                })
            
            with open(unit_dir / "experiment_info.json", "w") as f:
                json.dump(simple_meta, f, indent=2)
            
            # 保存文件列表和窗口统计
            file_info = {
                "filenames": data["filenames"],
                "n_windows_per_file": data["n_windows_per_file"],
                "total_samples": data["n_samples"]
            }
            
            with open(unit_dir / "file_info.json", "w") as f:
                json.dump(file_info, f, indent=2)
            
            # 保存处理统计
            if "processing_stats" in data:
                stats_list = []
                for stat in data["processing_stats"]:
                    if stat:
                        stats_list.append({
                            "filename": stat.filename,
                            "unit_id": stat.unit_id,
                            "gas_type": stat.gas_type,
                            "concentration": stat.concentration,
                            "original_points": stat.original_points,
                            "window_count": stat.window_count,
                            "status": stat.status,
                            "error_message": stat.error_message,
                            "processing_time": stat.processing_time
                        })
                
                with open(unit_dir / "processing_stats.json", "w") as f:
                    json.dump(stats_list, f, indent=2)




if __name__ == "__main__":
    # 测试数据预处理
    data_dir = "./dataset/data1"
    output_dir = "./dataset/processed"
    
    # 处理数据集
    preprocessor = GasSensorPreprocessor()
    processed_data = preprocessor.process_dataset(data_dir, output_dir)
    
    print(f"处理完成，共处理 {len(processed_data)} 个单元")
    for unit_id, data in processed_data.items():
        print(f"  单元 {unit_id}: {data['n_samples']} 个样本")