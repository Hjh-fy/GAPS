"""
文件级批量推理入口模块

提供从客户端数据文件 (.npy) 批量推理的入口，输出 CSV 格式的部署结果。

功能:
    1. 加载客户端 test/calibration 特征文件 (.npy)
    2. 批量推理: 分类 → 回归 → 校准 → QC 决策
    3. 输出 CSV: 包含 pred_gas, pred_ppm, calibrated_ppm, qc_status 等
    4. 生成汇总统计: accept/review/reject 比例, 各阶段指标

使用示例:
    python -m gaps_deploy.predict_client_file \
        --deploy-package deployment_package \
        --input "dataset/client_data_federated_window_fullgrid_src12_tgt345/client_3/test_features.npy" \
        --output results/deploy_output/client_3_deploy.csv \
        --client-id C3 \
        --device cpu

输出 CSV 列:
    - client_id: 客户端标识
    - sample_index: 样本索引
    - pred_gas: 预测气体名称
    - pred_class: 预测类别 ID
    - pred_ppm: 模型原始预测浓度
    - calibrated_ppm: 校准后浓度
    - qc_status: accept / review / reject
    - risk_score: 风险比
    - risk_reasons: 风险原因
    - confidence: 分类置信度
    - phase: 阶段标签
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .deploy_config import DeployConfig
from .calibration import DEFAULT_CONC_RANGES
from .inference import DeployPredictor, DeployResult

logger = logging.getLogger(__name__)


def read_csv(path: Path) -> List[Dict[str, str]]:
    """读取 CSV 文件"""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """写入 CSV 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_split_prefix(features_path: Path) -> str:
    """Infer sidecar label prefix from a features file name."""
    name = features_path.name
    suffix = "features.npy"
    if not name.endswith(suffix):
        return ""
    return name[: -len(suffix)]


def _load_sidecar_labels(features_path: Path) -> Dict[str, np.ndarray]:
    """Load optional labels next to `{split}_features.npy` for offline checks."""
    prefix = _infer_split_prefix(features_path)
    parent = features_path.parent
    paths = {
        "true_class": parent / f"{prefix}classification_labels.npy",
        "regression": parent / f"{prefix}regression_labels.npy",
        "phase": parent / f"{prefix}phase_labels.npy",
    }
    labels: Dict[str, np.ndarray] = {}
    for key, path in paths.items():
        if path.exists():
            labels[key] = np.load(path, allow_pickle=True)
            logger.info(f"加载{key}标签: {path}, shape={labels[key].shape}")
    return labels


def _true_ppm_at(regression_labels: np.ndarray, true_class: int, index: int) -> float:
    """Return the ppm label for the true gas class at one row."""
    values = np.asarray(regression_labels[index])
    if values.ndim == 0:
        return float(values)
    if 0 <= true_class < values.shape[0]:
        return float(values[true_class])
    return float("nan")


def predict_client_file(
    predictor: DeployPredictor,
    features_path: str,
    info_path: Optional[str] = None,
    client_id: str = "ALL",
    output_csv: Optional[str] = None,
    output_json: Optional[str] = None,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """对客户端数据文件进行批量推理

    参数:
        predictor: DeployPredictor 实例
        features_path: 特征文件路径 (.npy)
        info_path: 实验信息文件路径 (.json)，包含 phase 等元数据
        client_id: 客户端标识
        output_csv: 输出 CSV 路径
        output_json: 输出汇总 JSON 路径
        batch_size: 批处理大小

    返回:
        部署结果字典列表
    """
    # 加载特征
    feature_file = Path(features_path)
    features = np.load(feature_file).astype(np.float32)
    logger.info(f"加载特征: {features_path}, shape={features.shape}")
    sidecar_labels = _load_sidecar_labels(feature_file)

    # 加载元数据 (phase 信息)
    phase_map: Dict[int, int] = {}
    if "phase" in sidecar_labels:
        phase_labels = np.asarray(sidecar_labels["phase"], dtype=np.int64).reshape(-1)
        phase_map = {int(i): int(v) for i, v in enumerate(phase_labels)}
    if info_path and Path(info_path).exists():
        info = load_json(Path(info_path))
        if isinstance(info, list):
            for i, item in enumerate(info):
                if isinstance(item, dict) and "phase" in item and item["phase"] is not None:
                    phase_map[i] = int(item["phase"])
        logger.info(f"加载元数据: {info_path}, {len(phase_map)} 条记录")

    # 批量推理
    all_rows: List[Dict[str, Any]] = []
    total = len(features)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_features = features[start:end]

        batch_phases = np.asarray(
            [phase_map.get(i, -1) for i in range(start, end)],
            dtype=np.int64,
        )

        results = predictor.predict_batch(batch_features, client_id=client_id, phase=batch_phases)

        for i, result in enumerate(results):
            sample_index = start + i
            row = result.to_dict()
            row["sample_index"] = sample_index
            row["phase"] = int(batch_phases[i])
            if "true_class" in sidecar_labels:
                true_cls = int(np.asarray(sidecar_labels["true_class"]).reshape(-1)[sample_index])
                row["true_class"] = true_cls
                row["true_gas"] = (
                    predictor.config.gas_names[true_cls]
                    if 0 <= true_cls < len(predictor.config.gas_names)
                    else f"Class{true_cls}"
                )
                row["class_correct"] = int(int(row.get("pred_class", -1)) == true_cls)
                if "regression" in sidecar_labels:
                    true_ppm = _true_ppm_at(sidecar_labels["regression"], true_cls, sample_index)
                    row["true_ppm"] = true_ppm
                    try:
                        row["abs_error"] = abs(float(row["calibrated_ppm"]) - true_ppm)
                    except (TypeError, ValueError):
                        row["abs_error"] = ""
            all_rows.append(row)

        if (start // batch_size) % 10 == 0:
            logger.info(f"  推理进度: {end}/{total}")

    # 保存 CSV
    if output_csv:
        write_csv(Path(output_csv), all_rows)
        logger.info(f"保存部署结果: {output_csv} ({len(all_rows)} 行)")

    # 生成汇总
    summary = _build_summary(all_rows)
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"保存汇总: {output_json}")

    # 打印摘要
    _print_summary(summary)

    return all_rows


def _build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成推理汇总统计"""
    total = len(rows)

    # 按 QC 状态分组
    accepted = [r for r in rows if r.get("qc_status") == "accept"]
    reviewed = [r for r in rows if r.get("qc_status") == "review"]
    rejected = [r for r in rows if r.get("qc_status") == "reject"]

    # 按类别分组
    by_class: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        cls = int(r.get("pred_class", -1))
        if cls not in by_class:
            by_class[cls] = {"total": 0, "accepted": 0, "reviewed": 0, "rejected": 0}
        by_class[cls]["total"] += 1
        status = r.get("qc_status", "accept")
        if status == "accept":
            by_class[cls]["accepted"] += 1
        elif status == "review":
            by_class[cls]["reviewed"] += 1
        else:
            by_class[cls]["rejected"] += 1

    summary = {
        "total_windows": total,
        "accepted": len(accepted),
        "accepted_rate": len(accepted) / max(1, total),
        "reviewed": len(reviewed),
        "reviewed_rate": len(reviewed) / max(1, total),
        "rejected": len(rejected),
        "rejected_rate": len(rejected) / max(1, total),
        "by_class": {
            str(cls): {
                "total": info["total"],
                "accepted": info["accepted"],
                "accepted_rate": info["accepted"] / max(1, info["total"]),
                "reviewed": info["reviewed"],
                "rejected": info["rejected"],
            }
            for cls, info in sorted(by_class.items())
        },
        "avg_risk_score": float(np.mean([r.get("risk_score", 0) for r in rows])),
    }

    # 如果存在真实标签 (离线评估)，计算指标
    if rows and "true_ppm" in rows[0]:
        summary["evaluation"] = _compute_eval_metrics(rows)
        summary["per_true_class_evaluation"] = _compute_grouped_eval_metrics(
            rows,
            group_key="true_class",
            label_key="true_gas",
        )
    if rows and "class_correct" in rows[0]:
        correct = [int(r.get("class_correct", 0)) for r in rows]
        summary["classification_accuracy"] = float(np.mean(correct)) if correct else None

    return summary


def _compute_grouped_eval_metrics(
    rows: List[Dict[str, Any]],
    group_key: str,
    label_key: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Compute offline regression and classification metrics by group."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if group_key not in row:
            continue
        key = str(row.get(group_key))
        grouped.setdefault(key, []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for key, group_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        metrics = _compute_eval_metrics(group_rows)
        if label_key and group_rows:
            metrics["label"] = group_rows[0].get(label_key, "")
        if group_rows and "class_correct" in group_rows[0]:
            correct = [int(r.get("class_correct", 0)) for r in group_rows]
            metrics["classification_accuracy"] = float(np.mean(correct)) if correct else None
        out[key] = metrics
    return out


def _compute_eval_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算离线评估指标 (需要真实标签)"""
    def _fnum(v: Any, default: float = np.nan) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    y_true = np.array([_fnum(r.get("true_ppm")) for r in rows], dtype=np.float64)
    y_pred = np.array([_fnum(r.get("calibrated_ppm")) for r in rows], dtype=np.float64)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    if valid.sum() < 2:
        return {"n": int(valid.sum()), "R2": None, "MAE": None}

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    err = y_pred - y_true
    ae = np.abs(err)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ranges = []
    valid_rows = [row for row, keep in zip(rows, valid.tolist()) if keep]
    for row in valid_rows:
        cls_id = int(row.get("true_class", row.get("pred_class", -1)))
        lo, hi = DEFAULT_CONC_RANGES.get(cls_id, (float(np.min(y_true)), float(np.max(y_true))))
        ranges.append(max(float(hi) - float(lo), 1e-12))
    range_arr = np.asarray(ranges, dtype=np.float64)

    return {
        "n": int(valid.sum()),
        "R2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "MAE": float(np.mean(ae)),
        "RMSE": rmse,
        "NRMSE_range": float(np.sqrt(np.mean((err / range_arr) ** 2))),
        "MedAE": float(np.median(ae)),
        "P90AE": float(np.percentile(ae, 90)),
        "P95AE": float(np.percentile(ae, 95)),
        "Bias": float(np.mean(err)),
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    """打印汇总摘要"""
    print("\n" + "=" * 60)
    print("部署推理汇总")
    print("=" * 60)
    print(f"  总窗口数: {summary['total_windows']}")
    print(f"  Accept: {summary['accepted']} ({summary['accepted_rate']:.1%})")
    print(f"  Review: {summary['reviewed']} ({summary['reviewed_rate']:.1%})")
    print(f"  Reject: {summary['rejected']} ({summary['rejected_rate']:.1%})")
    print(f"  平均风险分数: {summary['avg_risk_score']:.3f}")

    if "evaluation" in summary:
        eval_ = summary["evaluation"]
        print(f"\n  离线评估指标 (基于真实标签):")
        print(f"    R2: {eval_.get('R2', 'N/A')}")
        print(f"    MAE: {eval_.get('MAE', 'N/A')}")
        print(f"    RMSE: {eval_.get('RMSE', 'N/A')}")
        print(f"    NRMSE_range: {eval_.get('NRMSE_range', 'N/A')}")
        print(f"    P90AE: {eval_.get('P90AE', 'N/A')}")
    if summary.get("classification_accuracy") is not None:
        print(f"    分类准确率: {summary['classification_accuracy']:.4f}")

    per_true = summary.get("per_true_class_evaluation", {})
    if per_true:
        print(f"\n  Per-gas offline metrics:")
        for cls, eval_ in sorted(per_true.items(), key=lambda x: int(x[0])):
            label = eval_.get("label") or f"Class{cls}"
            print(
                f"    {label} (class={cls}): "
                f"n={eval_.get('n')}, R2={eval_.get('R2')}, "
                f"MAE={eval_.get('MAE')}, RMSE={eval_.get('RMSE')}, "
                f"NRMSE_range={eval_.get('NRMSE_range')}, P90AE={eval_.get('P90AE')}, "
                f"cls_acc={eval_.get('classification_accuracy')}"
            )

    print(f"\n  各类别分布:")
    for cls, info in sorted(summary.get("by_class", {}).items(), key=lambda x: int(x[0])):
        print(f"    Class {cls}: total={info['total']}, "
              f"accept={info['accepted']} ({info['accepted_rate']:.1%}), "
              f"review={info['reviewed']}, reject={info['rejected']}")
    print("=" * 60 + "\n")


def main() -> None:
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="文件级批量部署推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m gaps_deploy.predict_client_file \\
        --deploy-package deployment_package \\
        --input "dataset/client_data_federated_window_fullgrid_src12_tgt345/client_3/test_features.npy" \\
        --output results/deploy_output/client_3_deploy.csv \\
        --client-id C3
        """,
    )
    parser.add_argument("--deploy-package", default="deployment_package",
                        help="部署包目录路径")
    parser.add_argument("--config", default=None,
                        help="部署配置 JSON 文件路径 (替代 --deploy-package)")
    parser.add_argument("--input", required=True,
                        help="输入特征文件路径 (.npy)")
    parser.add_argument("--info", default=None,
                        help="实验信息 JSON 文件路径 (含 phase 标签)")
    parser.add_argument("--output", default="results/deploy_output/deploy_result.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--output-json", default=None,
                        help="输出汇总 JSON 路径")
    parser.add_argument("--client-id", default="ALL",
                        help="客户端标识 (如 C3)")
    parser.add_argument("--device", default="cpu",
                        help="推理设备 (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批处理大小")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 加载预测器
    if args.config:
        config = DeployConfig.from_json(args.config)
        predictor = DeployPredictor.from_config(config)
    else:
        predictor = DeployPredictor.from_package(args.deploy_package, device=args.device)

    logger.info(f"模型版本: {predictor.model_version}")
    logger.info(f"推理设备: {predictor.device}")

    # 执行推理
    output_json = args.output_json or args.output.replace(".csv", "_summary.json")
    predict_client_file(
        predictor=predictor,
        features_path=args.input,
        info_path=args.info,
        client_id=args.client_id,
        output_csv=args.output,
        output_json=output_json,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
