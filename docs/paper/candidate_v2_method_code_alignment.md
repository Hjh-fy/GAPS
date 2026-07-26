# GAPS IoT-J 中文投稿候选稿 v2：方法—代码对齐审计

## 审计结论

状态：`ALIGNED`

候选稿的方法公式、数据流和运行时身份均可追溯到冻结代码、协议 manifest 或性能账本。未发现候选稿引入代码中不存在的损失项、权重、特征维度、选参规则或运行时功能。审计只读取代码与既有结果，没有执行训练、推理、评估或 benchmark。

## 冻结边界

- v2 基于提交 `1d70917bfe0d78391a5c65433cba4f35efad8231` 上的 v1 派生，新建文件，没有覆盖 v1。
- 系统 portable-release 冻结版本仍为 `36ee62339c025064cb415bfd13c5e7139a954edc`。
- 证据冻结提交仍为 `b78e8bec989cc8a925698d682aff05efe859fcd2`。
- 正式模型、runtime、QC、threshold、calibration/test split 和结果目录均未修改。

## 逐项对齐

| 稿件方法项 | 冻结实现/配置 | 对齐结果 | 边界说明 |
|---|---|---|---|
| 输入 \(\mathbf{x}\in\mathbb{R}^{100\times8}\) | `preprocessor.py::PreprocessConfig`；dataset split arrays | `MATCH` | 10 Hz 下 100 点、8 通道 |
| 预处理：100→10 Hz、前 20 s、30 s baseline、相对电导、40–150 s、100/50 窗口 | `preprocessor.py::{remove_unstable_phase,downsample_data,calculate_relative_conductance,extract_response_region,create_time_windows}` | `MATCH` | 处理后的 40–150 s 对应原始约 60–170 s |
| 分类输入不额外 z-score | `scripts/prepare_iotj_b5_c5_bundle_inputs.py` 的 `input_normalization.enabled=false`；Flower loader `normalize=False` | `MATCH` | TCN InstanceNorm 是模型层，不是外部输入标准化 |
| TCN 8→32→48→48、dilation 1/2/4、InstanceNorm、4-head attention、64D、4-class head | `model.py::{TCNBlock,FedGasBaseModel}` | `MATCH` | MixStyle 关闭；训练时 feature noise 0.01、dropout 0.1 |
| 25 轮、5 local epochs、batch 32、Client Adam \(5\times10^{-4}\) | `results/iotj_b5_multiseed_20260724/protocol_manifest.json`；`gaps_flower/task.py::make_config` | `MATCH` | optimizer 每轮重建，无 scheduler |
| 本地目标：CE + 0.05 prototype contrast + 2.0 replay feature distillation | `gaps_flower/client.py::train_one_round`；`gaps_flower/config.py` | `MATCH` | classification-only；regression loss 关闭；可用性不足时辅助项为零 |
| FedAvg 样本量权重 | `gaps_flower/strategy.py::_aggregate_params_gaps` | `MATCH` | \(\sum_i w_i\theta_i\) |
| warmup 后的语义相似度缩放 | `gaps_flower/strategy.py::_compute_selective_weights` | `MATCH` | warmup=3，minimum scale=0.3；稿件仅作实现组件描述 |
| 服务器每轮 100 DA steps，适配后模型作为下一轮 global | `gaps_flower/strategy.py::aggregate_fit`；multi-seed protocol manifest | `MATCH` | 使用 C1/C2 source validation 与 C5 calibration |
| Server DA 的十个 active terms 与权重 | `gaps_flower/domain_adaptation.py::adapt_model`；protocol `server_domain_adaptation` | `MATCH` | CE(s)=1；CORAL=.5；global/class MMD2=.5/.5；stage=.2；Wasserstein-min=.5；anchor=.3；proto-fit=.05；consistency=2；residual=.1 |
| 关闭 target CE、prototype-pair MMD、legacy direct align | frozen `run_config.json` 与 protocol manifest | `MATCH` | 稿件没有把关闭项写进 active objective |
| H1 特征矩 \(n,\sum X,\sum X^2\) | `scripts/evaluate_iotj_h1_federated_ridge_equivalence.py::client_feature_moments` | `MATCH` | server API 不接收 raw rows/X/y |
| Global scaler \(\mu,\sigma\) | 同脚本 `server_aggregate_scaler` | `MATCH` | population variance；scale&lt;1e-9 置 1 |
| Normal equations \(A_i=D_i^TD_i,b_i=D_i^Ty_i\) | 同脚本 `client_normal_equations` | `MATCH` | 设计矩阵含截距 |
| Ridge 重构与截距不正则化 | 同脚本 `server_reconstruct_ridge` | `MATCH` | 使用 `np.linalg.pinv`；预测按 source min/max clip |
| H1 alpha grid 与 source-local validation | 同脚本 `fit_federated_h1`；`RIDGE_ALPHAS` | `MATCH` | `{0,.01,.1,1,10,100,1000}`；仅 SSE/count 离开客户端 |
| 104D feature contract | `gaps_deploy/rich_residual.py::_target_ridge_features` | `MATCH` | 64+19+10+3+4+4=104 |
| 105D target Ridge | `scripts/evaluate_iotj_h1_federated_ridge_equivalence.py::fit_target_ridge_h1` | `MATCH` | 104D 加 1D H1；每气体 60 fit/20 validation，选参后 80-row refit |
| Predicted-route inference | `gaps_deploy/c5_federated_source_ridge_runtime.py::infer` | `MATCH` | B5 argmax→同 route H1→同 route target Ridge；不是 oracle route |
| H1 420、target Ridge 424、total 844 | runtime frozen assets；`docs/system/GAPS_PERFORMANCE_LEDGER_20260726.zh.md` | `MATCH` | classifier 22,765 单独计数 |
| Accuracy/Macro-F1/NLL/ECE | `scripts/summarize_iotj_classification_ablation.py::classification_metrics`；`gaps_flower/evaluate_checkpoint.py::expected_calibration_error` | `MATCH` | ECE 为 15 个等宽 top-label bins |
| RMSE/MAE/NRMSE、S_CC/S_ALL | `scripts/evaluate_iotj_b5_regression_multiseed.py::metric_block` | `MATCH` | NRMSE 逐行除以对应气体量程后计算 RMS |
| Runtime benchmark 设置 | `docs/system/benchmark_command_manifest_20260725.json` | `MATCH` | CPU、batch 1、1 thread、warmup 50、runs 500、固定 1360 行、无稳态磁盘 I/O |
| 正式 QC 与 portable core 分离 | `docs/system/GAPS_PERFORMANCE_LEDGER_20260726.zh.md`；runtime manifests | `MATCH` | v4 accepted error 不与 v5 core 844 参数组成同一运行对象 |

## Active server-adaptation objective 核验

候选稿中的目标为：

\[
\mathcal L_{\rm DA}=
\mathcal L_{\rm CE}^{s}
+0.5\mathcal L_{\rm CORAL}^{cc}
+0.5\mathcal L_{\rm MMD,g}^{2}
+0.5\mathcal L_{\rm MMD,c}^{2}
+0.2\mathcal L_{\rm stage}^{2}
+0.5\mathcal L_{\rm W,min}
+0.3\mathcal L_{\rm anchor}
+0.05\mathcal L_{\rm proto-fit}
+2.0\mathcal L_{\rm consistency}
+0.1\mathcal L_{\rm residual}.
\]

这些是冻结配置中系数非零的项。Prototype、class/phase matching 等项仍受当轮所需统计是否可用的条件控制。`lambda_target_ce=0`、`lambda_proto_mmd=0`，legacy direct alignment 关闭。候选稿将 C5 标签用途表述为 calibration-assisted adaptation，没有写成 zero-shot 或 UDA。

## 104D 特征维度核验

| Code block | Count |
|---|---:|
| 8 channels × [mean, std, min, max, amplitude, slope, mean/max absolute difference] | 64 |
| Global, channel-order and ratio statistics | 19 |
| Window timing and quality metadata | 10 |
| Response-phase one-hot | 3 |
| Phase-label one-hot | 4 |
| Phase-id one-hot | 4 |
| **Total** | **104** |

未从变量名猜测维度；计数直接来自 `gaps_deploy/rich_residual.py::_target_ridge_features` 的追加顺序。

## 需要人工留意但不构成阻塞的表述

1. Server DA 的 prototype 相关项是“配置 active、数据可用时产生贡献”，不应在英文稿中简化为每一步必定非零。
2. H1 的 `PRACTICAL_EQUIVALENCE` 是数值重构审计，不是隐私证明，也不是完整 Flower regression network closure。
3. 五路回归变异只来自五个 frozen classifier routes；source heads 固定，target Ridge 按 route 重建。
4. Portable core 是工程可移植性证据；正式选择性输出性能仍属于另一运行对象。

最终判定：`METHOD_CODE_ALIGNMENT_PASS`
