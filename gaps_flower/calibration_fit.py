"""Phase 6: 目标域校准参数拟合 (独立脚本)

从 exp_improved.py 移植 per-class affine/bias/phase_affine 校准拟合逻辑,
在目标域 (校准集) 的 oracle 模式 (使用真实类别选择回归头) 下,
对回归头输出进行 ppm 空间校准。

校准公式:
    bias_only:   y_calibrated = y_pred + b
    affine_only: y_calibrated = a * y_pred + b

输出:
    - routing_config.json: 类别→校准模式映射 + per-class 校准参数
    - 兼容 gaps_deploy/calibration.py 中的 RegressionCalibrator

用法:
    python -m gaps_flower.calibration_fit \
        --classifier_ckpt gaps_flower/outputs/classifier_global.pt \
        --regression_ckpt gaps_flower/outputs/regression/regression_fedavg_global.pt \
        --calib_data_dir gaps_flower/data/target_calib \
        --output_dir gaps_flower/outputs/calibration \
        --mode affine_only [--cpu]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import FLConfig
from federated_dataset import create_merged_calibration_loader, create_train_loader
from gaps_flower.regression_task import create_regression_model, load_classifier_weights, make_regression_config
from model import FedGasMultiTaskModel
from utils import CONC_STATS, create_model_by_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("calibration_fit")

# 气体名称列表
GAS_NAMES = ["Ethanol", "CO", "Ethylene", "Methane"]


def _create_target_calibration_loader(
    data_dir: Union[str, Path],
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Load target calibration split when available, otherwise keep legacy train fallback."""
    data_path = Path(data_dir)
    if (data_path / "calibration_features.npy").exists():
        return create_merged_calibration_loader(
            [data_path],
            batch_size=batch_size,
            num_workers=0,
        )
    return create_train_loader(
        data_path,
        batch_size=batch_size,
        shuffle=False,
        normalize=False,
        num_workers=0,
    )


def _denormalize_by_class_numpy(pred_norm: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """将归一化浓度 [0, 1] 按类别反归一化到 ppm 空间 (NumPy 版本)

    参数:
        pred_norm: (N,) 归一化浓度
        class_ids: (N,) 类别索引

    返回:
        (N,) ppm 浓度
    """
    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    class_ids = np.asarray(class_ids, dtype=int)
    ppm = np.zeros_like(pred_norm)
    for cls_id in sorted(CONC_STATS.keys()):
        mask = class_ids == cls_id
        if mask.any():
            ppm[mask] = pred_norm[mask] * (CONC_STATS[cls_id]["max"] - CONC_STATS[cls_id]["min"]) + CONC_STATS[cls_id]["min"]
    return ppm


def _denormalize_by_class_torch(pred_norm: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本的按类别反归一化 (用于 GPU 加速)"""
    pred_norm = pred_norm.view(-1)
    class_ids = class_ids.view(-1).long()
    ppm = torch.zeros_like(pred_norm)
    for cls_id in sorted(CONC_STATS.keys()):
        mask = class_ids == cls_id
        if mask.any():
            ppm[mask] = pred_norm[mask] * (CONC_STATS[cls_id]["max"] - CONC_STATS[cls_id]["min"]) + CONC_STATS[cls_id]["min"]
    return ppm


def fit_per_class_affine_params(
    data_dir: Union[str, Path],
    reg_model: FedGasMultiTaskModel,
    classifier_model: FedGasMultiTaskModel,
    device: torch.device,
    num_classes: int = 4,
    mode: str = "affine_only",
    batch_size: int = 32,
) -> Dict[int, Dict[str, Any]]:
    """在 ppm 原始浓度空间，为每个类别拟合 affine/bias 校准参数

    使用 oracle 模式 (真实类别选择回归头) 进行拟合，确保校准参数不受分类错误影响。

    算法:
        1. 前向传播获取归一化预测 pred_norm
        2. 反归一化到 ppm 空间: pred_ppm
        3. 对每个类别 c:
           - bias_only:  y_true_c = y_pred_c + b_c
           - affine_only: y_true_c = a_c * y_pred_c + b_c
        4. 用最小二乘法拟合 {a_c, b_c}

    参数:
        data_dir: 校准数据目录 (含 train_features.npy, train_regression_labels.npy 等)
        reg_model: 回归模型 B (FedGasMultiTaskModel)
        classifier_model: 分类模型 A (FedGasMultiTaskModel)
        device: 计算设备
        num_classes: 类别数
        mode: 校准模式 ("bias_only" | "affine_only")
        batch_size: 批次大小

    返回:
        {class_id: {"a": float, "b": float, "n_samples": int, "calib_r2": float, "calib_mae": float}}
    """
    reg_model.eval()
    classifier_model.eval()

    # 构建校准数据加载器
    loader = _create_target_calibration_loader(data_dir, batch_size)
    logger.info(f"校准拟合: 加载校准数据 {len(loader.dataset)} 条")

    # 收集每个类别的真实值和预测值
    stores: Dict[int, Dict[str, List[float]]] = {c: {"true": [], "pred": []} for c in range(num_classes)}

    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device)

            # 前向传播 (回归模型)
            _, _, reg_feat = reg_model(x)

            # oracle 模式: 使用真实类别选择回归头
            pred_norm = reg_model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)

            # 提取当前类别的回归标签
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]

            # 反归一化到 ppm 空间
            pred_ppm = _denormalize_by_class_torch(pred_norm, y_cls)

            for c in range(num_classes):
                mask = y_cls == c
                if mask.any():
                    stores[c]["true"].extend(y_true[mask].cpu().numpy().tolist())
                    stores[c]["pred"].extend(pred_ppm[mask].cpu().numpy().tolist())

    # 按类别拟合
    params: Dict[int, Dict[str, Any]] = {}
    for c in range(num_classes):
        y_true = np.asarray(stores[c]["true"], dtype=np.float64)
        y_pred = np.asarray(stores[c]["pred"], dtype=np.float64)
        n = len(y_true)

        if n < 2:
            logger.warning(f"校准拟合: 类别 {c} 样本数={n} (<2), 使用 identity")
            params[c] = {"a": 1.0, "b": 0.0, "mode": mode, "n_samples": n, "calib_r2": 0.0, "calib_mae": 0.0}
            continue

        residual = y_true - y_pred

        if mode == "bias_only":
            # 只拟合偏置: y_true = y_pred + b
            b = float(np.mean(residual))
            a = 1.0
        else:
            # affine: y_true = a * y_pred + b
            y_pred_var = np.var(y_pred)
            if y_pred_var < 1e-12:
                # 预测方差接近零，退化到 bias_only
                b = float(np.mean(residual))
                a = 1.0
            else:
                A = np.column_stack([y_pred, np.ones_like(y_pred)])
                coeffs, _, _, _ = np.linalg.lstsq(A, y_true, rcond=None)
                a, b = float(coeffs[0]), float(coeffs[1])

        # 评估校准后 R² 和 MAE
        y_adj = a * y_pred + b
        ss_res = np.sum((y_true - y_adj) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        calib_r2 = float(1.0 - ss_res / max(ss_tot, 1e-10))
        calib_mae = float(np.mean(np.abs(y_true - y_adj)))

        params[c] = {
            "a": a, "b": b, "mode": mode, "n_samples": n,
            "calib_r2": calib_r2, "calib_mae": calib_mae,
        }

        logger.info(
            f"  类别 {c} ({GAS_NAMES[c]}): a={a:.4f}, b={b:.4f}, "
            f"n={n}, calib_R²={calib_r2:.4f}, calib_MAE={calib_mae:.4f}"
        )

    return params


def build_routing_config(
    per_class_params: Dict[int, Dict[str, Any]],
    mode: str = "affine_only",
) -> Dict[str, Any]:
    """从 per-class 校准参数构建 routing_config.json 兼容格式

    参数:
        per_class_params: fit_per_class_affine_params 的输出
        mode: 校准模式

    返回:
        routing_config 字典 (兼容 gaps_deploy/calibration.py)
    """
    selected_modes: Dict[str, str] = {}
    affine_params: Dict[str, Dict[str, Any]] = {}

    for cls_id, params in sorted(per_class_params.items()):
        cls_key = str(cls_id)
        # 如果校准后 R² 极低 (< -10)，说明该类别校准不适用, 退化为 none
        if params.get("calib_r2", 0.0) < -10.0:
            selected_modes[cls_key] = "none"
        else:
            selected_modes[cls_key] = mode
        affine_params[cls_key] = {
            "a": params["a"],
            "b": params["b"],
            "mode": params["mode"],
            "n_samples": params["n_samples"],
            "calib_r2": params["calib_r2"],
            "calib_mae": params["calib_mae"],
        }

    return {
        "selected_modes": selected_modes,
        "affine_params": affine_params,
        "phase_affine_params": {},
        "routing_mode": mode,
        "num_classes": len(per_class_params),
    }


def compute_regression_metrics(
    data_dir: Union[str, Path],
    reg_model: FedGasMultiTaskModel,
    device: torch.device,
    num_classes: int = 4,
    batch_size: int = 32,
    affine_params: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """在 oracle 模式下评估回归模型性能

    参数:
        data_dir: 数据目录
        reg_model: 回归模型
        device: 计算设备
        num_classes: 类别数
        batch_size: 批次大小
        affine_params: 可选校准参数 (用于评估校准后性能)

    返回:
        指标字典 {"overall_R2": ..., "overall_MAE": ..., "per_class": [...]}
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    reg_model.eval()
    loader = _create_target_calibration_loader(data_dir, batch_size)

    all_true: List[float] = []
    all_pred_raw: List[float] = []
    all_cls: List[int] = []
    per_class_stores: Dict[int, Dict[str, List[float]]] = {c: {"true": [], "pred_raw": []} for c in range(num_classes)}

    with torch.no_grad():
        for x, y_cls, y_reg_full, y_phase in loader:
            x = x.to(device)
            y_cls = y_cls.to(device)
            y_reg_full = y_reg_full.to(device)
            y_phase = y_phase.to(device)

            _, _, reg_feat = reg_model(x)
            pred_norm = reg_model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
            y_true = y_reg_full[torch.arange(y_cls.size(0), device=device), y_cls]
            pred_ppm_raw = _denormalize_by_class_torch(pred_norm, y_cls)

            all_true.extend(y_true.cpu().numpy().tolist())
            all_pred_raw.extend(pred_ppm_raw.cpu().numpy().tolist())
            all_cls.extend(y_cls.cpu().numpy().astype(int).tolist())

            for c in range(num_classes):
                mask = y_cls == c
                if mask.any():
                    per_class_stores[c]["true"].extend(y_true[mask].cpu().numpy().tolist())
                    per_class_stores[c]["pred_raw"].extend(pred_ppm_raw[mask].cpu().numpy().tolist())

    y_true_arr = np.asarray(all_true, dtype=np.float64)
    y_pred_raw_arr = np.asarray(all_pred_raw, dtype=np.float64)
    y_cls_arr = np.asarray(all_cls, dtype=int)

    # 校准前
    overall_r2_raw = float(r2_score(y_true_arr, y_pred_raw_arr))
    overall_mae_raw = float(mean_absolute_error(y_true_arr, y_pred_raw_arr))
    overall_rmse_raw = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_raw_arr)))

    metrics: Dict[str, Any] = {
        "overall_R2_raw": overall_r2_raw,
        "overall_MAE_raw": overall_mae_raw,
        "overall_RMSE_raw": overall_rmse_raw,
        "total_samples": len(all_true),
        "per_class": [],
    }

    # 校准后
    per_class_cal: Dict[int, List[float]] = {}
    if affine_params is not None:
        y_pred_cal_arr = y_pred_raw_arr.copy()
        per_class_cal = {c: [] for c in range(num_classes)}
        for c in range(num_classes):
            true_arr = np.asarray(per_class_stores[c]["true"], dtype=np.float64)
            pred_arr = np.asarray(per_class_stores[c]["pred_raw"], dtype=np.float64)
            if c in affine_params:
                a, b = affine_params[c]["a"], affine_params[c]["b"]
                cal_arr = a * pred_arr + b
            else:
                cal_arr = pred_arr
            per_class_cal[c] = cal_arr.tolist()
            class_mask = y_cls_arr == c
            if class_mask.any():
                y_pred_cal_arr[class_mask] = cal_arr

        metrics["overall_R2_cal"] = float(r2_score(y_true_arr, y_pred_cal_arr))
        metrics["overall_MAE_cal"] = float(mean_absolute_error(y_true_arr, y_pred_cal_arr))
        metrics["overall_RMSE_cal"] = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_cal_arr)))

    # 按类别指标
    for c in range(num_classes):
        true_c = np.asarray(per_class_stores[c]["true"], dtype=np.float64)
        pred_c = np.asarray(per_class_stores[c]["pred_raw"], dtype=np.float64)
        if len(true_c) < 2:
            metrics["per_class"].append({"class": c, "gas": GAS_NAMES[c], "n_samples": len(true_c)})
            continue
        per_metrics = {
            "class": c,
            "gas": GAS_NAMES[c],
            "n_samples": len(true_c),
            "R2": float(r2_score(true_c, pred_c)),
            "MAE": float(mean_absolute_error(true_c, pred_c)),
            "RMSE": float(np.sqrt(mean_squared_error(true_c, pred_c))),
        }
        if per_class_cal:
            cal_c = np.asarray(per_class_cal[c], dtype=np.float64)
            per_metrics["R2_cal"] = float(r2_score(true_c, cal_c))
            per_metrics["MAE_cal"] = float(mean_absolute_error(true_c, cal_c))
            per_metrics["RMSE_cal"] = float(np.sqrt(mean_squared_error(true_c, cal_c)))
        metrics["per_class"].append(per_metrics)

    return metrics


def save_calibration_outputs(
    per_class_params: Dict[int, Dict[str, Any]],
    routing_config: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """保存校准输出: routing_config.json 和 calibration_stats.json

    参数:
        per_class_params: 每类校准参数
        routing_config: 路由配置
        metrics: 评估指标
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 routing_config.json
    routing_path = output_dir / "routing_config.json"
    with open(routing_path, "w", encoding="utf-8") as f:
        json.dump(routing_config, f, indent=2, ensure_ascii=False)
    logger.info(f"routing_config.json 已保存: {routing_path}")

    # 保存 calibration_stats.json (详细校准统计)
    stats = {
        "mode": routing_config.get("routing_mode", "unknown"),
        "num_classes": routing_config.get("num_classes", 0),
        "overall_metrics": {
            "R2_raw": metrics.get("overall_R2_raw", -999),
            "MAE_raw": metrics.get("overall_MAE_raw", -1),
            "RMSE_raw": metrics.get("overall_RMSE_raw", -1),
            "R2_cal": metrics.get("overall_R2_cal", -999),
            "MAE_cal": metrics.get("overall_MAE_cal", -1),
            "RMSE_cal": metrics.get("overall_RMSE_cal", -1),
        },
        "per_class_params": {str(k): v for k, v in sorted(per_class_params.items())},
        "per_class_metrics": metrics.get("per_class", []),
    }
    stats_path = output_dir / "calibration_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"calibration_stats.json 已保存: {stats_path}")

    # 输出摘要
    logger.info("=" * 60)
    logger.info("校准拟合结果摘要")
    logger.info("=" * 60)
    logger.info(f"总体 R² (原始):  {stats['overall_metrics']['R2_raw']:.4f}")
    logger.info(f"总体 MAE (原始): {stats['overall_metrics']['MAE_raw']:.4f}")
    if metrics.get("overall_R2_cal") is not None:
        logger.info(f"总体 R² (校准):  {stats['overall_metrics']['R2_cal']:.4f}")
        logger.info(f"总体 MAE (校准): {stats['overall_metrics']['MAE_cal']:.4f}")
    for per in stats["per_class_metrics"]:
        cls_id = per["class"]
        gas = per["gas"]
        n = per["n_samples"]
        r2 = per.get("R2", -999)
        mae = per.get("MAE", -1)
        r2_c = per.get("R2_cal")
        mae_c = per.get("MAE_cal")
        cal_str = f", 校准 R²={r2_c:.4f}, 校准 MAE={mae_c:.4f}" if r2_c is not None else ""
        logger.info(f"  {gas} (class={cls_id}): n={n}, R²={r2:.4f}, MAE={mae:.4f}{cal_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: 目标域校准参数拟合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # affine_only 模式 (推荐)
  python -m gaps_flower.calibration_fit \\
      --classifier_ckpt gaps_flower/outputs/classifier_global.pt \\
      --regression_ckpt gaps_flower/outputs/regression/regression_fedavg_global.pt \\
      --calib_data_dir gaps_flower/data/client_1 \\
      --output_dir gaps_flower/outputs/calibration \\
      --mode affine_only

  # bias_only 模式
  python -m gaps_flower.calibration_fit \\
      --classifier_ckpt gaps_flower/outputs/classifier_global.pt \\
      --regression_ckpt gaps_flower/outputs/regression/regression_fedavg_global.pt \\
      --calib_data_dir gaps_flower/data/client_1 \\
      --mode bias_only --cpu
        """,
    )
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="分类模型 checkpoint 路径 (Phase 4 输出)")
    parser.add_argument("--regression_ckpt", type=str, required=True,
                        help="回归模型 checkpoint 路径 (Phase 5 输出)")
    parser.add_argument("--calib_data_dir", type=str, required=True,
                        help="校准数据目录 (含 train_features.npy 等)")
    parser.add_argument("--output_dir", type=str,
                        default="gaps_flower/outputs/calibration",
                        help="输出目录")
    parser.add_argument("--mode", type=str, default="affine_only",
                        choices=["none", "bias_only", "affine_only"],
                        help="校准模式 (默认 affine_only)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--reg-output-mode", default="", choices=["", "sigmoid", "linear"],
                        help="回归输出模式; 默认从 regression_ckpt 的 model_config 推断")
    parser.add_argument("--reg-window-stats", action="store_true",
                        help="追加窗口响应统计特征到回归分支")
    parser.add_argument("--reg-window-stats-mode", default="", choices=["", "global", "per_channel"],
                        help="窗口统计模式; 默认从 checkpoint 推断")
    parser.add_argument("--reg-window-stats-dim", type=int, default=None,
                        help="窗口统计特征投影维度; 默认从 checkpoint 推断")
    parser.add_argument("--reg-response-branch", default="", choices=["", "none", "dct", "msconv"],
                        help="Regression response-shape branch; default inferred from checkpoint")
    parser.add_argument("--reg-dct-k", type=int, default=None,
                        help="DCT response branch k; default inferred from checkpoint")
    parser.add_argument("--reg-dct-gamma-init", type=float, default=None,
                        help="DCT response branch gamma init; default inferred from checkpoint")
    parser.add_argument("--reg-dct-dropout", type=float, default=None,
                        help="DCT response branch dropout; default inferred from checkpoint")
    parser.add_argument("--reg-msconv-channels", type=int, default=None,
                        help="msconv branch channels; default inferred from checkpoint")
    parser.add_argument("--reg-msconv-kernels", default="",
                        help="msconv branch kernels; default inferred from checkpoint")
    parser.add_argument("--reg-msconv-gamma-init", type=float, default=None,
                        help="msconv branch gamma init; default inferred from checkpoint")
    parser.add_argument("--reg-msconv-dropout", type=float, default=None,
                        help="msconv branch dropout; default inferred from checkpoint")
    parser.add_argument("--reg-tcn-adapter", action="store_true",
                        help="Enable regression TCN adapter; default inferred from checkpoint")
    parser.add_argument("--reg-tcn-adapter-kernel", type=int, default=None,
                        help="Regression TCN adapter kernel; default inferred from checkpoint")
    parser.add_argument("--reg-tcn-adapter-gamma-init", type=float, default=None,
                        help="Regression TCN adapter gamma init; default inferred from checkpoint")
    parser.add_argument("--reg-tcn-adapter-dropout", type=float, default=None,
                        help="Regression TCN adapter dropout; default inferred from checkpoint")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    args = parser.parse_args()

    # 设备选择
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"校准拟合: 使用设备={device}")
    logger.info(f"校准模式: {args.mode}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载分类模型 A
    logger.info("加载分类模型 A...")
    config = FLConfig()
    classifier_model = create_model_by_config(config, with_reg_head=False).to(device)
    ckpt = torch.load(args.classifier_ckpt, map_location=device, weights_only=False)
    classifier_state = ckpt.get("model_state", ckpt)
    classifier_model.load_state_dict(classifier_state, strict=False)
    classifier_model.eval()

    # 2. 加载回归模型 B
    logger.info("加载回归模型 B...")
    reg_ckpt = torch.load(args.regression_ckpt, map_location=device, weights_only=False)
    ckpt_model_config = reg_ckpt.get("model_config", {}) if isinstance(reg_ckpt, dict) else {}
    reg_output_mode = args.reg_output_mode or ckpt_model_config.get("reg_output_mode")
    reg_head_depth = ckpt_model_config.get("reg_head_depth")
    use_reg_window_stats = args.reg_window_stats or ckpt_model_config.get("reg_window_stats")
    reg_window_stats_mode = args.reg_window_stats_mode or ckpt_model_config.get("reg_window_stats_mode")
    reg_window_stats_dim = (
        args.reg_window_stats_dim
        if args.reg_window_stats_dim is not None
        else ckpt_model_config.get("reg_window_stats_dim")
    )
    reg_response_branch = args.reg_response_branch or ckpt_model_config.get("reg_response_branch")
    reg_dct_k = args.reg_dct_k if args.reg_dct_k is not None else ckpt_model_config.get("reg_dct_k")
    reg_dct_gamma_init = (
        args.reg_dct_gamma_init
        if args.reg_dct_gamma_init is not None
        else ckpt_model_config.get("reg_dct_gamma_init")
    )
    reg_dct_dropout = (
        args.reg_dct_dropout
        if args.reg_dct_dropout is not None
        else ckpt_model_config.get("reg_dct_dropout")
    )
    reg_msconv_channels = (
        args.reg_msconv_channels
        if args.reg_msconv_channels is not None
        else ckpt_model_config.get("reg_msconv_channels")
    )
    reg_msconv_kernels = args.reg_msconv_kernels or ckpt_model_config.get("reg_msconv_kernels")
    reg_msconv_gamma_init = (
        args.reg_msconv_gamma_init
        if args.reg_msconv_gamma_init is not None
        else ckpt_model_config.get("reg_msconv_gamma_init")
    )
    reg_msconv_dropout = (
        args.reg_msconv_dropout
        if args.reg_msconv_dropout is not None
        else ckpt_model_config.get("reg_msconv_dropout")
    )
    use_reg_tcn_adapter = (
        args.reg_tcn_adapter
        if args.reg_tcn_adapter
        else ckpt_model_config.get("reg_tcn_adapter")
    )
    reg_tcn_adapter_kernel = (
        args.reg_tcn_adapter_kernel
        if args.reg_tcn_adapter_kernel is not None
        else ckpt_model_config.get("reg_tcn_adapter_kernel")
    )
    reg_tcn_adapter_gamma_init = (
        args.reg_tcn_adapter_gamma_init
        if args.reg_tcn_adapter_gamma_init is not None
        else ckpt_model_config.get("reg_tcn_adapter_gamma_init")
    )
    reg_tcn_adapter_dropout = (
        args.reg_tcn_adapter_dropout
        if args.reg_tcn_adapter_dropout is not None
        else ckpt_model_config.get("reg_tcn_adapter_dropout")
    )
    reg_config = make_regression_config(
        device=device.type,
        batch_size=args.batch_size,
        reg_head_depth=reg_head_depth,
        reg_output_mode=reg_output_mode,
        use_reg_window_stats=use_reg_window_stats,
        reg_window_stats_mode=reg_window_stats_mode,
        reg_window_stats_dim=reg_window_stats_dim,
        reg_response_branch=reg_response_branch,
        reg_dct_k=reg_dct_k,
        reg_dct_gamma_init=reg_dct_gamma_init,
        reg_dct_dropout=reg_dct_dropout,
        reg_msconv_channels=reg_msconv_channels,
        reg_msconv_kernels=reg_msconv_kernels,
        reg_msconv_gamma_init=reg_msconv_gamma_init,
        reg_msconv_dropout=reg_msconv_dropout,
        use_reg_tcn_adapter=use_reg_tcn_adapter,
        reg_tcn_adapter_kernel=reg_tcn_adapter_kernel,
        reg_tcn_adapter_gamma_init=reg_tcn_adapter_gamma_init,
        reg_tcn_adapter_dropout=reg_tcn_adapter_dropout,
    )
    reg_model = create_regression_model(reg_config).to(device)
    reg_state = reg_ckpt.get("model_state", reg_ckpt)
    missing, unexpected = reg_model.load_state_dict(reg_state, strict=False)
    if missing:
        logger.info(f"回归模型: {len(missing)} 个键缺失 (正常)")
    if unexpected:
        logger.warning(f"回归模型: {len(unexpected)} 个多余键")
    reg_model.eval()

    # 3. 如果 mode 为 none, 只评估原始性能
    if args.mode == "none":
        logger.info("校准模式: none — 仅评估原始回归性能")
        metrics = compute_regression_metrics(args.calib_data_dir, reg_model, device,
                                              num_classes=4, batch_size=args.batch_size)
        # 创建 identity routing config
        per_class_params = {c: {"a": 1.0, "b": 0.0, "mode": "none", "n_samples": 0, "calib_r2": 0.0, "calib_mae": 0.0} for c in range(4)}
        routing_config = build_routing_config(per_class_params, mode="none")
        save_calibration_outputs(per_class_params, routing_config, metrics, output_dir)
        return

    # 4. 拟合 per-class affine/bias
    logger.info(f"开始拟合 per-class {args.mode} 校准参数...")
    per_class_params = fit_per_class_affine_params(
        data_dir=args.calib_data_dir,
        reg_model=reg_model,
        classifier_model=classifier_model,
        device=device,
        num_classes=4,
        mode=args.mode,
        batch_size=args.batch_size,
    )

    # 5. 构建 routing_config
    routing_config = build_routing_config(per_class_params, mode=args.mode)

    # 6. 评估校准前后性能
    metrics = compute_regression_metrics(
        args.calib_data_dir, reg_model, device,
        num_classes=4, batch_size=args.batch_size,
        affine_params=per_class_params,
    )

    # 7. 保存输出
    save_calibration_outputs(per_class_params, routing_config, metrics, output_dir)

    logger.info("Phase 6 校准拟合完成!")


if __name__ == "__main__":
    main()
