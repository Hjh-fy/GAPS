"""
gaps_deploy: 联邦学习气体传感器漂移补偿部署推理接口

本包将已验证的离线链路（分类 → 回归 → 校准 → QC → 决策输出）
封装为统一的部署推理接口，支持单窗口和批量窗口输入。

主要模块:
    - inference: 统一推理入口 DeployPredictor
    - calibration: 回归校准层 (affine/bias/phase_affine/specialist)
    - qc_policy: QC 质量控制层 (风险计算 + 双阈值决策)
    - response_anchoring: 路由感知响应锚定
    - predict_client_file: 文件级批量推理入口
    - deploy_config: 部署配置管理

使用示例:
    >>> from gaps_deploy import DeployPredictor
    >>> predictor = DeployPredictor.from_package("deployment_package")
    >>> result = predictor.predict_single(features)
    >>> print(result.pred_gas, result.calibrated_ppm, result.qc_status)
"""

from .inference import DeployPredictor, DeployResult
from .deploy_config import DeployConfig

__all__ = ["DeployPredictor", "DeployResult", "DeployConfig"]