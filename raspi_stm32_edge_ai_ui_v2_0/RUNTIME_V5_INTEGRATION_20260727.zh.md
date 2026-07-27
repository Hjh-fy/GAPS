# 上位机 Runtime v5 正式链路接入说明

日期：2026-07-27

## 当前结论

上位机已经支持两个相互独立的模型后端：

- `torchscript`：保留原 schema v1/v2 行为；
- `gaps_runtime_v5`：schema v3，加载当前正式
  `Final B5 → Federated H1 → C5 105D target Ridge → Runtime v5 core`。

本次接入没有将 Runtime v5 近似改写成单一 TorchScript，也没有修改模型资产、
正式测试集、冻结结果或性能账本。

## 正式公共数据回放包

当前通过验证的包：

```text
deployment_packages/
└── gaps_runtime_v5_public_c5_10hz_replay_20260727_r2/
    ├── manifest.json
    ├── runtime_v5_core/
    │   ├── portable_binding.json
    │   ├── assets/
    │   └── lineage/
    └── runtime_code/
        ├── code_manifest.json
        ├── model.py
        └── gaps_deploy/
```

关键身份：

- package fingerprint：
  `8029dada9a95fccd4db9838d5004bef4e5a2a4493f178f66b79e23ac95dad5ce`
- release id：`gaps_runtime_v5_core_20260726`
- B5 classifier：
  `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`
- Federated H1：
  `1ca10939f09e744fdddc0dce6f5fd959ccf769e9b78142030aa7e50aa6b2f3d4`
- C5 target Ridge：
  `2039d049776e7dfe0e8c4e6405dff2ae56a6e09b63f60ff2627ac0975aa075de`

加载时会执行：

1. UI manifest 和 portable binding 路径边界检查；
2. binding、模型资产、校准 lineage 的 bytes/SHA-256 检查；
3. 包内 Runtime Python 文件集合、bytes/SHA-256 和来源 commit 检查；
4. 输入维度、采样率、phase、数值有限性和输出 schema 检查；
5. QC 状态和 `auto_output_ppm` 的正式合同检查。

模型资产或运行代码发生单字节变化时，加载会 fail-closed。

## 输入合同

公共数据原始文件约为 100 Hz，但正式预处理已经 time-aware 降采样到 10 Hz。
Runtime v5 接收：

```text
shape: 100 × 8
rate: 10 Hz
physical window: about 10 s
dtype at runtime: float32
feature: precomputed relative conductance
extra Z-score: disabled
C5 drift phase: phase_id=2, phase_label=late
```

本包的 `feature_mode=precomputed`，仅用于公共数据的模型等价回放。
每一行必须显式带 `_model_input_precomputed=True`；普通串口解析帧没有该标志，
因此会被拒绝，防止把实验室原始 ADC 数值误送入公共数据模型。

这意味着：

- 可以验证 UI 窗口、采样率门禁、后台推理、输出映射和审计链；
- 不允许把该包声明为当前实验室 STM32 板的在线模型；
- 未来 1 Hz、不同传感器的实验室数据必须使用独立模型和独立 manifest。

## 输出映射

| Runtime v5 字段 | 上位机字段 |
|---|---|
| `pred_class` | `predicted_class` / `predicted_gas` |
| `source_h1_ppm` | `ppm_base_prediction` |
| `prediction_ppm` | `ppm_full_prediction` |
| `max_probability` | `confidence` |
| `qc_status` | `decision` |
| `auto_output_ppm` | `ppm_auto_output` |

Runtime v5 core 的正式 QC 状态为
`disabled_pending_dependency_audit`，因此：

- UI 可以展示完整浓度预测；
- `ppm_auto_output` 必须为空；
- UI 明确显示 `QC disabled; no auto output`；
- 不允许使用 UI 自己的置信度阈值把结果改写为 `accept`。

## 构建命令

构建器拒绝覆盖既有目录，并把正式 portable release 和匹配的运行代码快照
复制进新包：

```powershell
cd raspi_stm32_edge_ai_ui_v2_0
python build_runtime_v5_ui_package.py `
  --portable-release ..\.worktrees\iotj-confirmation-observability\release\gaps_runtime_v5_core_20260726 `
  --runtime-python-root ..\.worktrees\iotj-confirmation-observability `
  --output-dir deployment_packages\gaps_runtime_v5_public_c5_10hz_replay_20260727_r2
```

## 本机流式回放

回放器不加载 calibration metadata、分类标签或浓度标签；C5 phase 合同来自包内
manifest，冻结 parity CSV 只作为部署输出 oracle。

```powershell
python runtime_v5_stream_replay.py `
  --package-dir deployment_packages\gaps_runtime_v5_public_c5_10hz_replay_20260727_r2 `
  --features ..\dataset\client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid\client_5\calibration_features.npy `
  --reference ..\.worktrees\iotj-confirmation-observability\results\iotj_b5_c5_runtime_v5_candidate_20260724\parity\calibration_parity_rows.csv `
  --output ..\results\edge_ai_public_runtime_v5_calibration_replay_20260727\ui_local_stream_replay_summary.json
```

## 树莓派运行

当前验证目录：

```text
/home/gaps/edge_ui_runtime_v5_validation_20260727_r2
```

启动真实界面：

```bash
cd /home/gaps/edge_ui_runtime_v5_validation_20260727_r2
EDGE_UI_PYTHON=/home/gaps/GAPS/gaps_rpi_env/bin/python \
EDGE_UI_EXTRA_PYTHONPATH=/usr/lib/python3/dist-packages \
./run_edge_ui.sh \
  --ai-package deployment_packages/gaps_runtime_v5_public_c5_10hz_replay_20260727_r2
```

该包已经自带并校验 Runtime v5 代码，不依赖
`GAPS_RUNTIME_V5_PYTHONPATH`。环境变量仅保留给不含 code bundle 的兼容包。

## 已完成验证

- 原有 parser、UI logic、schema-v2 package 和 TorchScript runtime 自测：通过；
- schema-v3 package 完整加载：Windows、Raspberry Pi 均通过；
- 320 个 C5 calibration 窗口的逐点 10 Hz UI 流式回放：通过；
- 路由不一致：0；
- QC 状态不一致：0；
- 非法自动输出：0；
- Windows 最终浓度/H1 最大差：0；
- Pi 最终浓度最大差：`6.6791e-13 ppm`；
- Pi H1 最大差：`8.8107e-13 ppm`；
- Pi 概率最大差：`8.3447e-07`；
- 包内模型资产、运行代码和输入合同篡改测试：全部 fail-closed；
- replay-only 原始串口门禁：通过，未标记的 raw STM32 帧被拒绝；
- Raspberry Pi offscreen Qt 加载：通过；
- Raspberry Pi 真实 Wayland 会话 Qt 加载：通过。
- Raspberry Pi 真实 Wayland 下完成首个窗口
  `Qt signal → worker → Runtime v5 → UI card/result label`：
  路由一致、最终浓度差 `2.8422e-14 ppm`、H1 差 0、概率差
  `1.1921e-07`，并正确显示 `QC disabled; no auto output`。

这些是工程部署一致性证据，不是新的 Accuracy、F1、RMSE 或正式性能 benchmark。

## 尚未完成

- 公共预计算窗口尚未伪装成真实串口；当前设计刻意拒绝这种混淆；
- 实验室 1 Hz 传感器顺序、物理单位、负载电阻和缺失规则尚待硬件冻结；
- 实验室数据模型尚待采集、划分、训练和独立评估；
- Runtime v5 core 的 QC 仍未启用，v5 QC2 也未晋级；
- HC-04 当前仍未完成实际配对/串口节点验证。
