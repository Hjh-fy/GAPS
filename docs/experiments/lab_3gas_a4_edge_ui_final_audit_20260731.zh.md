# A4 稳定段三分类上位机接入实验审计

## 审计范围与预期结论

本审计只判断以下结论是否成立：

> 冻结的 P2 → P3、REC-A4-STABLE150、第 25 轮三分类模型，可以在上位机 v2.2 与树莓派运行时中，对同一冻结的 P3 稳定段输入复现原模型类别，并以 classification-only 方式 fail-closed 运行。

“真实 STM32 原始串口已完成在线三气体识别”不属于本次可批准结论。

## 被比较实验

| 实验 ID | Split | 模型 | Checkpoint | DA | Calibration | QC | Seeds | 来源 |
|---|---|---|---|---|---|---|---|---|
| A4-EDGE-P1A | P2 train → P3 stable test | proto_replay, 3-class | round 25，SHA `ad51…1e76` | A4 既有流程 | P2-train-only z-score | none | 单个冻结 run | r3 `parity_summary.json` |
| A4-EDGE-P2A | 同一冻结输入 | r3 TorchScript + UI runtime | SHA `2a7f…cee5` | 不变 | 同一 norm，SHA `e321…f15` | Unavailable | 不适用 | PC replay JSON |
| A4-EDGE-P3A | 同一冻结输入 | 同一 r3 包 | 包指纹 `88a4…0877` | 不变 | 同一 norm | Unavailable | 不适用 | Pi replay/Qt/Wayland JSON |

## 审计发现

| ID | 严重度 | 检查项 | 证据 | 影响 | 要求 | 状态 |
|---|---|---|---|---|---|---|
| F-01 | informational | 模型导出一致性 | 360/360 类别一致；logit 最大差 1.9073×10⁻⁶ | 不改变稳定段预测类别 | 保留 hash 与 parity JSON | closed |
| F-02 | informational | 跨平台一致性 | PC/Pi 均 359/360；概率差 0 | 支持同包回放一致性 | 仅限冻结输入 | closed |
| F-03 | informational | Qt worker 稳定性 | r3 树莓派生产入口 5/5；Wayland PASS | 修复了首次后台 Torch 调用崩溃 | 保留主线程预热与标准 MHA 图 | closed |
| F-04 | major | 样本覆盖 | 稳定段 360/420，覆盖 85.71% | 99.72% 不能外推到全暴露过程 | 所有展示注明 stable-only | open-boundary |
| F-05 | major | 独立重复与不确定性 | 单个冻结 run；重叠时间窗 | 不能报告跨 seed 不确定性或把窗口当独立重复 | 后续补多日/多次测试 | open-boundary |
| F-06 | blocking-for-live | 原始硬件输入契约 | CH1/2/4/6/8/9 帧映射及六路 ADC→电阻参数未冻结 | 不能批准真实串口在线推理 | 先完成物理输入 parity | blocked |
| F-07 | major | 分类 QC | 没有目标验证的 QC 阈值 | 不能自动 Accepted/Rejected | 保持 `Unavailable` | closed-by-fail-closed |
| F-08 | informational | 浓度输出 | manifest `has_concentration=false`，ppm 为 null | 防止把三分类误报为浓度回归 | 保持 classification-only | closed |

## 泄漏评估

- r3 没有重新训练、调参或选择 checkpoint，只验证冻结模型到运行时的等价性。
- P3 稳定测试输入参与的是部署回放核对；因此这组数据不能再被描述成一次新的独立泛化验证。
- 本审计不把 359/360 用作“A4 优于其他方案”的因果比较，也不新增基于 P3 test 的阈值或早停选择。
- P2-only 归一化统计量 hash 与模型包绑定，没有观察到将 P3 test 统计量写入归一化的证据。

## 完整性与可复现性

- checkpoint、normalization、TorchScript、验证特征、验证标签、源码提交和最终包均有不可变 SHA/指纹。
- PC 与 Pi 使用相同包指纹 `88a4a4d75c6bd2f1a917a1e771712a0568e9c284bee599f4f83ab26257fb0877`。
- 回放、UI 逻辑、Qt worker、Runtime v5 回归和 Wayland 实机证据齐全。
- 缺少多 seed、独立新日期/新暴露以及原始串口电阻 parity；这些不阻塞“冻结输入部署等价性”，但阻塞更强的泛化与真实在线结论。

## Verdict: approved

批准范围仅为：

> r3 是可复现的 classification-only 稳定段回放候选，能够在 PC 与树莓派上复现冻结 A4 模型输出，并保持 ppm/QC fail-closed。

以下状态保持 `blocked`：

> 真实 STM32 串口到六通道电阻、在线基线、平滑、稳定段门控再到自动三气体输出的端到端验收。

## 未知项与交接

1. 确认 STM32 帧中的六个物理通道索引；
2. 冻结每路分压电路参数与 ADC→电阻公式；
3. 使用同一段真实采集同时生成离线电阻与在线电阻，逐通道做误差验收；
4. parity 通过后再实现 300 s baseline、1 Hz 重采样、中心 MA5 的在线延迟版本；
5. 使用新的独立暴露日验证全链路，不能复用当前 360 窗口作为新泛化测试。
