"""Phase 5: 源域回归联邦服务端 (独立脚本)

使用 FedAvg 方法聚合多个源域客户端的回归训练结果,
模拟联邦学习中的服务端聚合过程。
由于是单机模拟, 服务端脚本负责:
    1. 收集所有客户端本地训练的 checkpoint
    2. 执行 FedAvg 加权聚合
    3. 更新全局回归模型
    4. 保存聚合后的全局 checkpoint

用法:
    python -m gaps_flower.regression_server \
        --classifier_ckpt gaps_flower/outputs/classifier_global.pt \
        --client_ckpts_dir gaps_flower/outputs/regression \
        --data_root gaps_flower/data \
        --client_ids 1,2 \
        --output_dir gaps_flower/outputs/regression \
        [--cpu]

聚合流程:
    - 收集各客户端本地 state_dict
    - 按样本数加权平均 (FedAvg)
    - 输出全局回归模型 checkpoint
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch

# 将项目根目录加入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gaps_flower.regression_task import (
    build_source_regression_loaders,
    create_regression_model,
    fedavg_regression_states,
    get_regression_state_keys,
    init_regression_branch_from_classifier,
    load_classifier_weights,
    make_regression_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("regression_server")


def aggregate_regression_checkpoints(
    classifier_ckpt: str,
    client_ckpt_dir: str,
    data_root: str,
    client_ids: list[int],
    device: torch.device,
    batch_size: int = 32,
    reg_head_depth: int | None = None,
    reg_output_mode: str | None = None,
    use_reg_window_stats: bool | None = None,
    reg_window_stats_mode: str | None = None,
    reg_window_stats_dim: int | None = None,
    reg_response_branch: str | None = None,
    reg_dct_k: int | None = None,
    reg_dct_gamma_init: float | None = None,
    reg_dct_dropout: float | None = None,
    reg_msconv_channels: int | None = None,
    reg_msconv_kernels: str | None = None,
    reg_msconv_gamma_init: float | None = None,
    reg_msconv_dropout: float | None = None,
    use_reg_tcn_adapter: bool | None = None,
    reg_tcn_adapter_kernel: int | None = None,
    reg_tcn_adapter_gamma_init: float | None = None,
    reg_tcn_adapter_dropout: float | None = None,
    use_reg_shared_trunk: bool | None = None,
    reg_shared_trunk_dim: int | None = None,
    reg_gas_emb_dim: int | None = None,
    reg_residual_head_depth: int | None = None,
    use_reg_ratio_branch: bool | None = None,
    reg_ratio_gamma_init: float | None = None,
    reg_ratio_dropout: float | None = None,
    output_dir: str = "gaps_flower/outputs/regression",
) -> str:
    """收集客户端 checkpoint 并执行 FedAvg 聚合

    参数:
        classifier_ckpt: 分类模型 checkpoint 路径
        client_ckpt_dir: 客户端 checkpoint 目录
        data_root: 数据根目录
        client_ids: 源域客户端 ID 列表
        device: 计算设备
        batch_size: 批次大小
        reg_head_depth: 回归头深度; None 表示沿用 FLConfig 默认值
        reg_output_mode: 回归输出模式; None 表示沿用 FLConfig 默认值
        use_reg_window_stats: 是否追加窗口响应统计特征
        reg_window_stats_mode: global 或 per_channel
        reg_window_stats_dim: 统计特征投影维度
        output_dir: 输出目录

    返回:
        全局聚合 checkpoint 路径
    """
    # 1. 创建全局模型 (加载共享编码器)
    config = make_regression_config(
        device=device.type,
        batch_size=batch_size,
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
        use_reg_shared_trunk=use_reg_shared_trunk,
        reg_shared_trunk_dim=reg_shared_trunk_dim,
        reg_gas_emb_dim=reg_gas_emb_dim,
        reg_residual_head_depth=reg_residual_head_depth,
        use_reg_ratio_branch=use_reg_ratio_branch,
        reg_ratio_gamma_init=reg_ratio_gamma_init,
        reg_ratio_dropout=reg_ratio_dropout,
    )
    logger.info(
        "回归 FedAvg: REG_HEAD_DEPTH=%s, REG_OUTPUT_MODE=%s, REG_WINDOW_STATS=%s",
        config.REG_HEAD_DEPTH,
        config.REG_OUTPUT_MODE,
        config.REG_WINDOW_STATS,
    )
    global_model = create_regression_model(config).to(device)
    load_classifier_weights(global_model, classifier_ckpt)
    init_regression_branch_from_classifier(global_model)

    # 2. 获取样本数
    _, sample_counts = build_source_regression_loaders(
        data_root=data_root, client_ids=client_ids, batch_size=batch_size,
    )
    logger.info(f"回归 FedAvg: 客户端样本数={sample_counts}")

    # 3. 加载各客户端本地 state_dict
    ckpt_dir = Path(client_ckpt_dir)
    local_states: Dict[int, Dict[str, torch.Tensor]] = {}
    for cid in client_ids:
        ckpt_path = ckpt_dir / f"regression_source_client{cid}_local.pth"
        if not ckpt_path.exists():
            logger.warning(f"回归 FedAvg: 客户端 {cid} checkpoint 不存在: {ckpt_path}, 跳过")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_depth = ckpt.get("model_config", {}).get("reg_head_depth")
        if ckpt_depth is not None and int(ckpt_depth) != int(config.REG_HEAD_DEPTH):
            raise ValueError(
                f"客户端 {cid} 回归头深度不匹配: checkpoint={ckpt_depth}, "
                f"server={config.REG_HEAD_DEPTH}. 请在 client/server 使用相同 --reg-head-depth。"
            )
        ckpt_mode = ckpt.get("model_config", {}).get("reg_output_mode")
        if ckpt_mode is not None and str(ckpt_mode) != str(config.REG_OUTPUT_MODE):
            raise ValueError(
                f"客户端 {cid} 回归输出模式不匹配: checkpoint={ckpt_mode}, "
                f"server={config.REG_OUTPUT_MODE}. 请在 client/server 使用相同 --reg-output-mode。"
            )
        ckpt_stats = ckpt.get("model_config", {}).get("reg_window_stats")
        if ckpt_stats is not None and bool(ckpt_stats) != bool(config.REG_WINDOW_STATS):
            raise ValueError(
                f"客户端 {cid} 回归窗口统计配置不匹配: checkpoint={ckpt_stats}, "
                f"server={config.REG_WINDOW_STATS}. 请在 client/server 使用相同 --reg-window-stats。"
            )
        if bool(config.REG_WINDOW_STATS):
            ckpt_stats_mode = ckpt.get("model_config", {}).get("reg_window_stats_mode")
            ckpt_stats_dim = ckpt.get("model_config", {}).get("reg_window_stats_dim")
            if ckpt_stats_mode is not None and str(ckpt_stats_mode) != str(config.REG_WINDOW_STATS_MODE):
                raise ValueError(
                    f"客户端 {cid} 窗口统计模式不匹配: checkpoint={ckpt_stats_mode}, "
                    f"server={config.REG_WINDOW_STATS_MODE}."
                )
            if ckpt_stats_dim is not None and int(ckpt_stats_dim) != int(config.REG_WINDOW_STATS_DIM):
                raise ValueError(
                    f"客户端 {cid} 窗口统计维度不匹配: checkpoint={ckpt_stats_dim}, "
                    f"server={config.REG_WINDOW_STATS_DIM}."
                )
        ckpt_branch = ckpt.get("model_config", {}).get("reg_response_branch")
        if ckpt_branch is not None and str(ckpt_branch) != str(config.REG_RESPONSE_BRANCH):
            raise ValueError(
                f"瀹㈡埛绔?{cid} response branch 涓嶅尮閰? checkpoint={ckpt_branch}, "
                f"server={config.REG_RESPONSE_BRANCH}."
            )
        if str(config.REG_RESPONSE_BRANCH) == "dct":
            ckpt_dct_k = ckpt.get("model_config", {}).get("reg_dct_k")
            if ckpt_dct_k is not None and int(ckpt_dct_k) != int(config.REG_DCT_K):
                raise ValueError(
                    f"瀹㈡埛绔?{cid} DCT k 涓嶅尮閰? checkpoint={ckpt_dct_k}, "
                    f"server={config.REG_DCT_K}."
                )
        if str(config.REG_RESPONSE_BRANCH) == "msconv":
            ckpt_channels = ckpt.get("model_config", {}).get("reg_msconv_channels")
            ckpt_kernels = ckpt.get("model_config", {}).get("reg_msconv_kernels")
            if ckpt_channels is not None and int(ckpt_channels) != int(config.REG_MSCONV_CHANNELS):
                raise ValueError(
                    f"瀹㈡埛绔?{cid} msconv channels 涓嶅尮閰? checkpoint={ckpt_channels}, "
                    f"server={config.REG_MSCONV_CHANNELS}."
                )
            if ckpt_kernels is not None and str(ckpt_kernels) != str(config.REG_MSCONV_KERNELS):
                raise ValueError(
                    f"瀹㈡埛绔?{cid} msconv kernels 涓嶅尮閰? checkpoint={ckpt_kernels}, "
                    f"server={config.REG_MSCONV_KERNELS}."
                )
        ckpt_tcn_adapter = ckpt.get("model_config", {}).get("reg_tcn_adapter")
        if ckpt_tcn_adapter is not None and bool(ckpt_tcn_adapter) != bool(config.REG_TCN_ADAPTER):
            raise ValueError(
                f"瀹㈡埛绔?{cid} reg_tcn_adapter 涓嶅尮閰? checkpoint={ckpt_tcn_adapter}, "
                f"server={config.REG_TCN_ADAPTER}."
            )
        if bool(config.REG_TCN_ADAPTER):
            ckpt_kernel = ckpt.get("model_config", {}).get("reg_tcn_adapter_kernel")
            if ckpt_kernel is not None and int(ckpt_kernel) != int(config.REG_TCN_ADAPTER_KERNEL):
                raise ValueError(
                    f"客户端 {cid} reg_tcn_adapter_kernel 不匹配: checkpoint={ckpt_kernel}, "
                    f"server={config.REG_TCN_ADAPTER_KERNEL}."
                )
        # 共享主干一致性检查
        ckpt_shared_trunk = ckpt.get("model_config", {}).get("reg_use_shared_trunk")
        if ckpt_shared_trunk is not None and bool(ckpt_shared_trunk) != bool(config.REG_USE_SHARED_TRUNK):
            raise ValueError(
                f"客户端 {cid} reg_use_shared_trunk 不匹配: checkpoint={ckpt_shared_trunk}, "
                f"server={config.REG_USE_SHARED_TRUNK}."
            )
        if bool(config.REG_USE_SHARED_TRUNK):
            for dim_field, config_attr in (
                ("reg_shared_trunk_dim", "REG_SHARED_TRUNK_DIM"),
                ("reg_gas_emb_dim", "REG_GAS_EMB_DIM"),
                ("reg_residual_head_depth", "REG_RESIDUAL_HEAD_DEPTH"),
            ):
                ckpt_val = ckpt.get("model_config", {}).get(dim_field)
                server_val = int(getattr(config, config_attr, 0))
                if ckpt_val is not None and int(ckpt_val) != server_val:
                    raise ValueError(
                        f"客户端 {cid} {dim_field} 不匹配: checkpoint={ckpt_val}, "
                        f"server={server_val}."
                    )
        # 跨通道比率分支一致性检查
        ckpt_ratio = ckpt.get("model_config", {}).get("use_reg_ratio_branch")
        if ckpt_ratio is not None and bool(ckpt_ratio) != bool(config.USE_REG_RATIO_BRANCH):
            raise ValueError(
                f"客户端 {cid} use_reg_ratio_branch 不匹配: checkpoint={ckpt_ratio}, "
                f"server={config.USE_REG_RATIO_BRANCH}."
            )
        if bool(config.USE_REG_RATIO_BRANCH):
            for field, config_attr in (
                ("reg_ratio_gamma_init", "REG_RATIO_GAMMA_INIT"),
                ("reg_ratio_dropout", "REG_RATIO_DROPOUT"),
            ):
                ckpt_val = ckpt.get("model_config", {}).get(field)
                server_val = float(getattr(config, config_attr, 0.0))
                if ckpt_val is not None and abs(float(ckpt_val) - server_val) > 1e-8:
                    raise ValueError(
                        f"客户端 {cid} {field} 不匹配: checkpoint={ckpt_val}, "
                        f"server={server_val}."
                    )
        local_states[cid] = ckpt.get("model_state", ckpt)
        logger.info(f"回归 FedAvg: 加载客户端 {cid} 本地参数")

    if len(local_states) < 2:
        logger.warning(
            f"回归 FedAvg: 仅有 {len(local_states)} 个客户端, 聚合效果有限。"
            "至少需要 2 个客户端才能体现多源域协同优势。"
        )

    # 4. FedAvg 聚合
    state_keys = get_regression_state_keys(global_model)
    aggregated = fedavg_regression_states(
        local_states, sample_counts, state_keys, device
    )

    # 5. 更新全局模型
    new_state = {key: value.clone() for key, value in global_model.state_dict().items()}
    for key, value in aggregated.items():
        ref = new_state[key]
        new_state[key] = value.to(ref.device).type_as(ref)

    global_model.load_state_dict(new_state, strict=True)

    # 6. 保存全局 checkpoint
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    global_ckpt_path = output / "regression_fedavg_global.pt"
    torch.save(
        {
            "model_state": global_model.state_dict(),
            "aggregated_clients": list(local_states.keys()),
            "sample_counts": sample_counts,
            "model_config": {
                "reg_head_depth": int(config.REG_HEAD_DEPTH),
                "reg_output_mode": str(config.REG_OUTPUT_MODE),
                "reg_window_stats": bool(config.REG_WINDOW_STATS),
                "reg_window_stats_mode": str(config.REG_WINDOW_STATS_MODE),
                "reg_window_stats_dim": int(config.REG_WINDOW_STATS_DIM),
                "reg_response_branch": str(config.REG_RESPONSE_BRANCH),
                "reg_dct_k": int(config.REG_DCT_K),
                "reg_dct_gamma_init": float(config.REG_DCT_GAMMA_INIT),
                "reg_dct_dropout": float(config.REG_DCT_DROPOUT),
                "reg_msconv_channels": int(config.REG_MSCONV_CHANNELS),
                "reg_msconv_kernels": str(config.REG_MSCONV_KERNELS),
                "reg_msconv_gamma_init": float(config.REG_MSCONV_GAMMA_INIT),
                "reg_msconv_dropout": float(config.REG_MSCONV_DROPOUT),
                "reg_tcn_adapter": bool(config.REG_TCN_ADAPTER),
                "reg_tcn_adapter_kernel": int(config.REG_TCN_ADAPTER_KERNEL),
                "reg_tcn_adapter_gamma_init": float(config.REG_TCN_ADAPTER_GAMMA_INIT),
                "reg_tcn_adapter_dropout": float(config.REG_TCN_ADAPTER_DROPOUT),
                "reg_use_shared_trunk": bool(config.REG_USE_SHARED_TRUNK),
                "reg_shared_trunk_dim": int(config.REG_SHARED_TRUNK_DIM),
                "reg_gas_emb_dim": int(config.REG_GAS_EMB_DIM),
                "reg_residual_head_depth": int(config.REG_RESIDUAL_HEAD_DEPTH),
                "use_reg_ratio_branch": bool(config.USE_REG_RATIO_BRANCH),
                "reg_ratio_gamma_init": float(config.REG_RATIO_GAMMA_INIT),
                "reg_ratio_dropout": float(config.REG_RATIO_DROPOUT),
            },
        },
        global_ckpt_path,
    )
    logger.info(f"回归 FedAvg 全局模型已保存: {global_ckpt_path}")

    # 7. 使全局模型也保存为 regressor.pt (兼容 inference.py 路径)
    regressor_path = output / "regressor.pt"
    torch.save(
        {
            "model_state": global_model.state_dict(),
            "model_config": {
                "reg_head_depth": int(config.REG_HEAD_DEPTH),
                "reg_output_mode": str(config.REG_OUTPUT_MODE),
                "reg_window_stats": bool(config.REG_WINDOW_STATS),
                "reg_window_stats_mode": str(config.REG_WINDOW_STATS_MODE),
                "reg_window_stats_dim": int(config.REG_WINDOW_STATS_DIM),
                "reg_response_branch": str(config.REG_RESPONSE_BRANCH),
                "reg_dct_k": int(config.REG_DCT_K),
                "reg_dct_gamma_init": float(config.REG_DCT_GAMMA_INIT),
                "reg_dct_dropout": float(config.REG_DCT_DROPOUT),
                "reg_msconv_channels": int(config.REG_MSCONV_CHANNELS),
                "reg_msconv_kernels": str(config.REG_MSCONV_KERNELS),
                "reg_msconv_gamma_init": float(config.REG_MSCONV_GAMMA_INIT),
                "reg_msconv_dropout": float(config.REG_MSCONV_DROPOUT),
                "reg_tcn_adapter": bool(config.REG_TCN_ADAPTER),
                "reg_tcn_adapter_kernel": int(config.REG_TCN_ADAPTER_KERNEL),
                "reg_tcn_adapter_gamma_init": float(config.REG_TCN_ADAPTER_GAMMA_INIT),
                "reg_tcn_adapter_dropout": float(config.REG_TCN_ADAPTER_DROPOUT),
                "reg_use_shared_trunk": bool(config.REG_USE_SHARED_TRUNK),
                "reg_shared_trunk_dim": int(config.REG_SHARED_TRUNK_DIM),
                "reg_gas_emb_dim": int(config.REG_GAS_EMB_DIM),
                "reg_residual_head_depth": int(config.REG_RESIDUAL_HEAD_DEPTH),
                "use_reg_ratio_branch": bool(config.USE_REG_RATIO_BRANCH),
                "reg_ratio_gamma_init": float(config.REG_RATIO_GAMMA_INIT),
                "reg_ratio_dropout": float(config.REG_RATIO_DROPOUT),
            },
        },
        regressor_path,
    )
    logger.info(f"回归模型已保存为 regressor.pt: {regressor_path}")

    return str(global_ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="源域回归联邦 FedAvg 聚合")
    parser.add_argument("--classifier_ckpt", type=str, required=True,
                        help="分类模型 checkpoint 路径 (Phase 4 输出)")
    parser.add_argument("--client_ckpts_dir", type=str,
                        default="gaps_flower/outputs/regression",
                        help="客户端 checkpoint 目录")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据根目录")
    parser.add_argument("--client_ids", type=str, default="1,2",
                        help="客户端 ID 列表, 逗号分隔, 如 '1,2'")
    parser.add_argument("--output_dir", type=str,
                        default="gaps_flower/outputs/regression",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--reg-head-depth", type=int, default=None,
                        help="回归头深度; 默认沿用 config.py 中的 REG_HEAD_DEPTH")
    parser.add_argument("--reg-output-mode", default=None, choices=["sigmoid", "linear"],
                        help="回归输出模式; 默认沿用 config.py 中的 REG_OUTPUT_MODE")
    parser.add_argument("--reg-window-stats", action="store_true",
                        help="追加窗口响应统计特征到回归分支")
    parser.add_argument("--reg-window-stats-mode", default="global", choices=["global", "per_channel"],
                        help="窗口统计模式")
    parser.add_argument("--reg-window-stats-dim", type=int, default=8,
                        help="窗口统计特征投影维度")
    parser.add_argument("--reg-response-branch", default=None, choices=["none", "dct", "msconv"],
                        help="Regression response-shape branch")
    parser.add_argument("--reg-dct-k", type=int, default=None,
                        help="Number of low-frequency DCT coefficients")
    parser.add_argument("--reg-dct-gamma-init", type=float, default=None,
                        help="Initial residual scale for DCT branch")
    parser.add_argument("--reg-dct-dropout", type=float, default=None,
                        help="Dropout inside DCT branch")
    parser.add_argument("--reg-msconv-channels", type=int, default=None,
                        help="Hidden channels per temporal scale for msconv branch")
    parser.add_argument("--reg-msconv-kernels", default=None,
                        help="Comma-separated odd kernels for msconv branch")
    parser.add_argument("--reg-msconv-gamma-init", type=float, default=None,
                        help="Initial residual scale for msconv branch")
    parser.add_argument("--reg-msconv-dropout", type=float, default=None,
                        help="Dropout inside msconv branch")
    parser.add_argument("--reg-tcn-adapter", action="store_true",
                        help="Enable regression-specific residual adapter on shared TCN features")
    parser.add_argument("--reg-tcn-adapter-kernel", type=int, default=None,
                        help="Odd kernel size for regression TCN adapter")
    parser.add_argument("--reg-tcn-adapter-gamma-init", type=float, default=None,
                        help="Initial residual scale for regression TCN adapter")
    parser.add_argument("--reg-tcn-adapter-dropout", type=float, default=None,
                        help="Dropout for regression TCN adapter")
    parser.add_argument("--reg-use-shared-trunk", action="store_true",
                        help="use shared concentration trunk + class residual heads")
    parser.add_argument("--reg-shared-trunk-dim", type=int, default=None,
                        help="shared trunk hidden dim; default 128")
    parser.add_argument("--reg-gas-emb-dim", type=int, default=None,
                        help="gas embedding dim; default 16")
    parser.add_argument("--reg-residual-head-depth", type=int, default=None,
                        help="residual head depth; default 2 (1/2/3)")
    parser.add_argument("--use-reg-ratio-branch", action="store_true",
                        help="enable cross-channel ratio response features")
    parser.add_argument("--reg-ratio-gamma-init", type=float, default=None,
                        help="ratio branch gamma init (default 0.0)")
    parser.add_argument("--reg-ratio-dropout", type=float, default=None,
                        help="ratio branch dropout (default 0.05)")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用 CPU")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"回归服务端: 使用设备={device}")

    client_ids = [int(x.strip()) for x in args.client_ids.split(",") if x.strip()]
    logger.info(f"回归服务端: 聚合客户端={client_ids}")

    global_ckpt = aggregate_regression_checkpoints(
        classifier_ckpt=args.classifier_ckpt,
        client_ckpt_dir=args.client_ckpts_dir,
        data_root=args.data_root,
        client_ids=client_ids,
        device=device,
        batch_size=args.batch_size,
        reg_head_depth=args.reg_head_depth,
        reg_output_mode=args.reg_output_mode,
        use_reg_window_stats=args.reg_window_stats,
        reg_window_stats_mode=args.reg_window_stats_mode,
        reg_window_stats_dim=args.reg_window_stats_dim,
        reg_response_branch=args.reg_response_branch,
        reg_dct_k=args.reg_dct_k,
        reg_dct_gamma_init=args.reg_dct_gamma_init,
        reg_dct_dropout=args.reg_dct_dropout,
        reg_msconv_channels=args.reg_msconv_channels,
        reg_msconv_kernels=args.reg_msconv_kernels,
        reg_msconv_gamma_init=args.reg_msconv_gamma_init,
        reg_msconv_dropout=args.reg_msconv_dropout,
        use_reg_tcn_adapter=args.reg_tcn_adapter,
        reg_tcn_adapter_kernel=args.reg_tcn_adapter_kernel,
        reg_tcn_adapter_gamma_init=args.reg_tcn_adapter_gamma_init,
        reg_tcn_adapter_dropout=args.reg_tcn_adapter_dropout,
        use_reg_shared_trunk=args.reg_use_shared_trunk if args.reg_use_shared_trunk else None,
        reg_shared_trunk_dim=args.reg_shared_trunk_dim,
        reg_gas_emb_dim=args.reg_gas_emb_dim,
        reg_residual_head_depth=args.reg_residual_head_depth,
        use_reg_ratio_branch=args.use_reg_ratio_branch,
        reg_ratio_gamma_init=args.reg_ratio_gamma_init,
        reg_ratio_dropout=args.reg_ratio_dropout,
        output_dir=args.output_dir,
    )
    logger.info(f"回归服务端: 聚合完成, 全局模型={global_ckpt}")


if __name__ == "__main__":
    main()
