# GAPS Raspberry Pi STM32 Edge AI UI v2.2

面向 STM32 气体传感器阵列的树莓派上位机与可审计边缘 AI 运行框架。

它既可以作为纯采集上位机运行，也可以加载由 GAPS 主仓库导出的部署包，在后台完成：

- 真实串口 / HC-04 数据接收；
- 43 字节、20 字段协议解析；
- 16 通道单曲线查看；
- 连续 `raw.csv` 保存与实验事件标记；
- 基线构建、滑窗、归一化；
- TorchScript 或正式 Runtime v5 分类与浓度回归；
- package 声明的校准与 QC 语义；
- `accept / review / reject`，或 Runtime v5 core 的显式 `QC disabled` 状态；
- `edge_ai_predictions.csv` 推理审计记录。

没有模拟波形模式。界面中移动的曲线来自真实解析帧。

v2.1 的系统边界、双数据集模型路线和验收门见
[`EDGE_AI_SYSTEM_V2_1.md`](EDGE_AI_SYSTEM_V2_1.md)。

当前正式 Runtime v5 的 schema-v3 接入、公共数据流式回放命令和树莓派验证结果见
[`RUNTIME_V5_INTEGRATION_20260727.zh.md`](RUNTIME_V5_INTEGRATION_20260727.zh.md)。

v2.2 的 800×480 / 大屏响应式界面重构和实机截图证据见
[`UI_RESPONSIVE_REDESIGN_20260727.zh.md`](UI_RESPONSIVE_REDESIGN_20260727.zh.md)。

## v2.2 界面优化

1. 800×480 下隐藏非关键全局工具，保留连接、新建实验、记录和数据目录入口。
2. Edge AI 六张结果卡不再被内部长状态撑宽，禁止横向滚动。
3. Runtime QC 内部状态仍完整写入审计数据，主界面改为简短的可读状态。
4. 增加模型身份、阶段、基线、窗口、输入频率徽标和窗口进度。
5. 增加语义化 ready / warning / error 配色与主要操作按钮层级。
6. 大屏增加最近浓度推理趋势图，小屏优先显示当前推理结果。

## v2.1 边缘系统扩展

1. 部署包 schema v2 使用秒定义稳定期、基线、窗口和步长，支持 1 Hz 与 10 Hz 模型。
2. 公共数据集和实验室数据集使用独立 `dataset_profile/device_profile`，不复用传感器语义。
3. `normalization.enabled` 明确控制是否应用 mean/std，不再强制归一化。
4. Baseline / Exposure / Recovery 事件真正驱动 AI 状态机。
5. 断线、异常帧、时间倒退和超长帧间隔会清空模型窗口。
6. 加载时校验模型 SHA-256、配置范围、通道唯一性和物理负载电阻。
7. 每个预测记录窗口起止帧、连接、阶段、包指纹和原始记录覆盖状态。
8. 异步 AI 结果只有在整个窗口属于同一 `raw.csv` 会话时才写入预测审计文件。
9. schema v1/v2 保持兼容；正式 Runtime v5 使用带 portable binding 和代码身份锁的 schema v3。

## v2.0 主要修复

1. HC-04 默认使用 Linux 设备节点 `/dev/rfcomm0`，不再依赖 pyserial 不支持的 `rfcomm://` URL。
2. 串口线程停止时会主动 `cancel_read/close`，避免窗口关闭时出现 `QThread: Destroyed while thread is still running`。
3. 重连后保留全局 `stream_frame_index/stream_elapsed_s`，同时记录 `connection_id` 与连接内序号。
4. `raw.csv` 额外记录 `experiment_frame_index/experiment_elapsed_s`，每个实验从 0 开始。
5. 断线时自动暂停记录，不会在用户不知情时继续假装处于 Recording ON。
6. 新实验会建立干净的显示、片段和 AI 基线边界，防止旧实验数据混入新实验。
7. CSV 写入失败后不会被后续 LIVE 状态覆盖。
8. 磁盘空间检查针对当前实际写入目录；低于 0.2 GB 时自动停止记录。
9. 有限片段缓存达到上限时，JSON 会明确标记截断行数；连续 `raw.csv` 不受影响。
10. 结构有效帧与数值合理帧分开统计，ADC 超出 12-bit 范围会被标记但不会静默丢弃。
11. 增加 Baseline / Exposure / Recovery / Note 事件标记，写入 `event_log.csv`。
12. YAML/JSON 配置文件现在真正生效。

## 目录结构

```text
raspi_stm32_edge_ai_ui_v2_0/
├── edge_ui_app.py                 # PyQt 主界面
├── serial_worker.py               # 安全串口线程
├── frame_parser_v20.py            # 流式协议解析
├── csv_writer.py                  # raw/event/AI 输出写入
├── data_buffer.py                 # 绘图环形缓存
├── edge_ai_runtime.py             # 部署包、预处理、推理、校准、QC
├── edge_ai_worker.py              # AI 后台线程
├── build_edge_ai_package.py       # 通用部署包组装器
├── deployment_package_example/    # 部署包 schema 示例
├── bind_hc04.sh                   # HC-04 -> /dev/rfcomm0
├── config.example.yaml
└── *_self_test.py
```

## 安装与自测试

```bash
cd ~/raspi_stm32_edge_ai_ui_v2_0
python3 -m pip install -r requirements.txt
python3 -m py_compile *.py
python3 parser_self_test.py
python3 ui_logic_self_test.py
python3 edge_ai_package_self_test.py
python3 edge_ai_runtime_self_test.py
```

PyTorch 是可选依赖。只进行采集和保存时不需要安装；加载 AI 部署包时才需要安装与树莓派系统架构匹配的 PyTorch。

## HC-04 连接

先配对 HC-04，然后绑定为设备节点：

```bash
./bind_hc04.sh
```

默认 MAC 为旧项目使用的 `04:25:01:09:0A:EB`，可覆盖：

```bash
HC04_MAC=XX:XX:XX:XX:XX:XX ./bind_hc04.sh
```

启动：

```bash
./run_edge_ui_bluetooth.sh
```

如果 PyTorch 安装在项目虚拟环境中，而树莓派的串口、PyQt5 和
pyqtgraph 来自 Debian 系统包，可显式组合两套依赖：

```bash
EDGE_UI_PYTHON="$HOME/GAPS/gaps_rpi_env/bin/python" \
EDGE_UI_EXTRA_PYTHONPATH="/usr/lib/python3/dist-packages" \
./run_edge_ui.sh --ai-package /absolute/path/to/deployment_package
```

`run_edge_ui_bluetooth.sh` 也支持这两个环境变量，并统一调用
`run_edge_ui.sh`，避免加载 AI 包时误用不含 PyTorch 的系统 Python。

USB 串口示例：

```bash
python3 edge_ui_app.py --port /dev/ttyUSB0 --baudrate 115200 --fullscreen
```

## 使用配置文件

```bash
cp config.example.yaml config.yaml
python3 edge_ui_app.py --config config.yaml
```

命令行参数会覆盖对应的常规运行选择；配置项见 `config.example.yaml`。

## 实验目录

```text
~/GAPS_data/experiments/YYYYMMDD_HHMMSS_experiment_name/
├── raw.csv
├── meta.json
├── event_log.csv
├── edge_ai_predictions.csv        # 仅产生 AI 结果后创建
├── saved_segment_*.csv
└── saved_segment_*.json
```

`raw.csv` 中的时间和序号：

- `stream_*`：本次应用运行期间跨断线连续；
- `connection_*`：每次串口连接单独计数；
- `experiment_*`：每个实验的实际写入行从 0 开始；
- 旧字段 `frame_index/elapsed_s` 保留，并映射为连续 stream 口径。

## 加载 GAPS 部署包

界面 Edge AI 页点击 **Load Package**，或：

```bash
python3 edge_ui_app.py \
  --port /dev/rfcomm0 \
  --ai-package /home/gaps/GAPS_deploy/c12_to_c345_auto_v2_qc \
  --fullscreen
```

部署包至少包含：

```text
deployment_package/
├── manifest.json
├── model.ts
├── norm_stats.npz     # 仅 normalization.enabled=true 时需要
└── calibration.json   # 可选
```

TorchScript 模型输出支持：

```python
(logits, ppm)
(logits, ppm, risk_score)
```

或字典：

```python
{"logits": logits, "ppm": ppm, "risk_score": risk_score}
```

其中 `ppm` 可以是单一浓度，也可以是 4 类浓度向量，运行时会按预测类别路由。

## 重要一致性保护

GAPS 当前训练口径通常为：

- 8 个模型输入通道；
- 10 Hz；
- 30 s 基线；
- 100 点窗口；
- 50 点步长；
- `(G-G0)/G0`；
- 与冻结训练合同一致的归一化开关和统计量。

部署包必须明确这些参数。运行时会检查实际输入帧率；若实测约 1 Hz 而模型要求 10 Hz，默认阻止推理并显示 `rate_mismatch`，避免把不一致输入包装成“模型已成功部署”。

若使用 `relative_conductance`，`manifest.json` 必须提供全部模型输入通道的负载电阻。当前旧上位机只硬编码了前三路电阻，不能直接推断其余通道的真实电导。

## 构建部署包

先在 GAPS 主仓库中把最终模型导出为 `model.ts`，再组装：

```bash
python3 build_edge_ai_package.py \
  --model-ts /path/to/model.ts \
  --calibration-json /path/to/calibration.json \
  --output-dir /path/to/deployment_package \
  --package-name lab_stm32_1hz_v1 \
  --dataset-profile laboratory_stm32_1hz_v1 \
  --device-profile stm32_gas_board_v1 \
  --sensor-fields adc_ch0_pa0,adc_ch1_pa1,adc_ch2_pa2,adc_ch3_pa3,adc_ch4_pa4,adc_ch5_pa5,adc_ch6_pa6,adc_ch7_pa7 \
  --raw-sample-hz 1 \
  --target-sample-hz 1 \
  --unstable-duration-s 20 \
  --baseline-duration-s 60 \
  --window-duration-s 60 \
  --stride-duration-s 15 \
  --feature-mode relative_conductance \
  --rload-ohm R0,R1,R2,R3,R4,R5,R6,R7 \
  --normalization disabled \
  --phase-mode event_driven
```

上例中的 `R0...R7` 必须替换为经硬件确认的正数负载电阻。若冻结模型需要
mean/std，改用 `--normalization enabled --norm-stats /path/to/norm_stats.npz`。
公共 10 Hz 与实验室 1 Hz 设计模板位于 `profile_examples/`。

不要直接把训练 `.pth` 扔给上位机。通用模型应导出冻结后的 TorchScript；
当前正式 Runtime v5 则必须使用 schema-v3 portable binding 和经过 SHA 锁定的
运行代码包，不能绕过合同直接加载 checkpoint。

## 尚未声称完成的部分

- 已附带正式公共 C5 Runtime v5 的预计算 10 Hz 回放包，但它不是实验室原始
  STM32 ADC 的在线模型。
- 实验室 1 Hz 模型尚未采集、训练或封装。
- Runtime v5 core 的 QC 正式状态仍为
  `disabled_pending_dependency_audit`；v5 QC2 未晋级，因此不产生自动浓度输出。
- Flower Client、在线少样本校准上传和云端 OTA 尚未直接塞入 UI 主进程，建议作为独立服务与 UI 解耦。
