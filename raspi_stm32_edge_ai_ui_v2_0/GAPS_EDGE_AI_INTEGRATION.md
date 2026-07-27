# GAPS 系统与上位机结合：边缘 AI 应用包装方案

> v2.1 更新：公共 time-aware 数据模型与未来实验室 1 Hz 模型采用独立
> profile 和独立权重。最新主线是否归一化、目标客户端和 QC 策略必须以冻结
> package manifest 为准；本文中早期 C12→C345 和强制 mean/std 表述不再作为
> 运行时默认值。详细合同见 `EDGE_AI_SYSTEM_V2_1.md`。

## 1. 应用定位

建议项目名称：**GAPS EdgeSense：面向跨设备气体传感器的云–边–端智能感知终端**。

它不是单纯“画曲线的上位机”，而是一套可演示的边缘智能闭环：

```text
STM32 传感器阵列
  → HC-04 / USB 串口
  → 树莓派实时采集与协议校验
  → 基线与滑窗预处理
  → 气体分类 + 浓度回归
  → 目标设备少样本校准
  → QC 风险门控
  → 自动输出 / 复测 / 人工确认
  → 本地审计记录
  → 可选 Flower 联邦更新与云端模型下发
```

## 2. 四层系统故事

### 感知与采集层

- STM32 采集 16 路 MOS/辅助传感器；
- 43 字节固定帧协议；
- 树莓派做流式解析、重同步、帧质量统计；
- 原始数据、实验事件和环境量可追溯保存。

### 边缘智能层

- 从 16 路中映射模型所需 8 路；
- 构建基线和 100 点时序窗口；
- TCN 分类主干识别气体；
- 回归分支预测浓度；
- per-class auto_v2 校准缓解目标设备偏差；
- QC 只对低风险窗口开放静默自动浓度输出。

### 云–边协同层

- 云端 Flower Server 聚合多个设备的分类模型；
- 服务端使用小规模目标校准数据进行域适应；
- 输出冻结的边缘部署包；
- 树莓派可作为 Flower Client 或仅作为推理节点；
- 后续可上传统计量、校准摘要和异常窗口，而不是上传全部原始数据。

### 安全与审计层

- 每次预测都保留完整 ppm、置信度、风险分数和 QC 决策；
- `review/reject` 窗口不填 `ppm_auto_output`；
- 断线、低磁盘、帧异常、实验阶段和人工备注均进入事件日志；
- 训练、校准、QC 和部署包版本写入 manifest，便于论文复现和系统演示。

## 3. 现有 GAPS 资产如何映射

| GAPS 资产 | 边缘端落点 |
|---|---|
| adapted classification checkpoint | 导出到 `model.ts` 分类分支 |
| R3aK16 / 最终回归头 | 导出到 `model.ts` 回归分支 |
| `norm_stats.npz` | 仅冻结训练合同启用归一化时复用 |
| sensor channel mapping | `manifest.input.sensor_fields` |
| 100/50 滑窗 | schema v1；schema v2 使用 `window_duration_s/stride_duration_s` |
| 30 s baseline、相对电导率 | schema v2 使用 `baseline_duration_s/feature_mode/rload_ohm` |
| auto_v2 bias/affine/piecewise | `calibration.json` |
| full neural specialist | 冻结到 TorchScript 或额外 specialist 模型 |
| QC v2/two-threshold policy | `risk_score + accept/reject thresholds` |
| deployment output package | UI 的 `edge_ai_predictions.csv` 口径 |

## 4. 当前最关键的工程一致性问题

### 输入频率必须对齐

论文模型使用 10 Hz、100 点窗口，即约 10 s。若当前 HC-04 实测只有约 1 fps，则 100 点代表约 100 s，输入语义已经改变。

正式演示有三种选择：

1. 提升 STM32→树莓派链路到稳定 10 Hz；
2. 使用 USB 串口代替低速/不稳定 Bluetooth；
3. 重新按实际 1 Hz 训练模型。

不能只把窗口长度仍设为 100 就声称部署一致。v2.0 的 rate guard 会阻止这种错误推理。

### 特征物理口径必须对齐

训练使用 `(G-G0)/G0`，边缘端必须知道 8 路负载电阻和 ADC 电路关系，才能从 ADC 恢复电导。若这些参数尚未确认，可以先用 `relative_adc` 做工程联调，但论文正式部署实验必须说明它是否与训练特征等价。

### 目标校准不能偷看测试标签

树莓派上的 calibration buffer 必须来自独立少样本校准实验。正式 test 流只用于推理与评估，不能再次调整 affine/specialist/QC 阈值。

## 5. 建议的真实部署包

```text
GAPS_deployment_package_v1/
├── manifest.json
├── model.ts
├── norm_stats.npz
├── calibration.json
├── qc_policy.json                 # 后续可拆分
├── package_metrics.json           # 离线一致性、延迟、内存、版本
└── README.md
```

`manifest.json` 还应增加：

- Git commit；
- 训练数据 split id；
- source/target clients；
- adapted checkpoint hash；
- calibration protocol；
- QC workpoint；
- expected sample rate；
- sensor mapping和负载电阻；
- torch/OS/architecture；
- 导出日期。

## 6. 推荐推进顺序

### P0：做出可证明一致的单机边缘推理

1. 在 GAPS 主仓库增加最终模型导出脚本；
2. 用同一批 `.npy` 窗口分别跑原 PyTorch 与 `model.ts`；
3. 要求分类 logits、路由类别、ppm 和 QC 决策逐样本一致；
4. 输出 `offline_replay_equivalence.csv`；
5. 再部署到树莓派测单窗口延迟和内存。

### P0：修通 10 Hz 真实输入

- 验证 STM32 实际发送频率；
- 记录 30 分钟 good/bad/drop/fps；
- 确认 100 点真实对应 10 s；
- 若 HC-04 不稳定，正式部署使用 USB，Bluetooth 作为演示备用链路。

### P1：接入真实 auto_v2 与 QC v2

- 先支持 bias/affine/piecewise 参数化校准；
- full specialist 冻结到 TorchScript；
- 风险特征与阈值只从 calibration split 固定；
- UI 展示 `full prediction` 与 `auto output` 的差异。

### P1：形成论文系统实验

至少补充：

- PC 与树莓派逐窗口输出一致性；
- 边缘推理 P50/P90/P95 延迟；
- 预处理、推理、校准和 QC 分项耗时；
- 模型大小、峰值内存；
- 6 小时连续运行 good/bad/drop；
- 断线重连、磁盘不足和异常帧测试；
- accept/review/reject 的 coverage 与误差。

### P2：云边联邦闭环

将 Flower Client 作为独立进程：

```text
edge_ui_app.py          实时采集、推理、展示
edge_agent_service.py   calibration buffer、部署包管理、统计上传
flower_client.py        计划任务或人工触发的联邦更新
```

不要让 Flower 网络通信阻塞 PyQt 主线程。UI 只读取 agent 状态与最新部署包版本。

## 7. 简历与面试包装

可以表述为：

> 设计并实现基于 STM32 与 Raspberry Pi 的多通道气体传感器边缘 AI 终端，完成串口协议解析、长时数据采集、实验事件追踪、TCN 分类与浓度回归的滑窗推理、目标设备少样本校准和风险门控；通过 Flower 构建云端聚合与部署包下发链路，并针对断线重连、磁盘不足、异常帧和低可信预测设计安全降级与审计机制。

注意区分：

- **已经完成**：采集 UI、协议解析、真实保存、异常保护、部署包接口和 AI 后台线程；
- **需要用真实模型验收**：TorchScript 导出、逐窗口等价性、真实 auto_v2/QC 包、树莓派性能；
- **后续扩展**：Flower Client 常驻服务、在线少样本校准、OTA 与统计上传。
