# A4 稳定段三分类接入上位机结果分析

## 输入契约与来源

- 实验标识：`A4-EDGE-P1A`、`A4-EDGE-P2A`、`A4-EDGE-P3A`
- 模型：P2 → P3、REC-A4-STABLE150、第 25 轮
- 模型输入：`[B,100,6]`，通道 `CH1/2/4/6/8/9`
- 源域归一化：仅使用 P2 train
- checkpoint SHA256：`ad51c370d9d79f90a385e3c2898df90184e79289eb67694b42ca1b5559ab1e76`
- norm SHA256：`e32143b410e217e0e581720f8c1aba72371049b10c3c223491116782aa5e5f15`
- 最终 TorchScript SHA256：`2a7fd4b5cc1dae1000141a9d287de77de6d32d62a1d535a56de936675e06cee5`
- 最终包指纹：`88a4a4d75c6bd2f1a917a1e771712a0568e9c284bee599f4f83ab26257fb0877`
- 源码提交：`6c483ed41e21b5f9d52ef5243d8d4c734f1e6057`
- 验证特征 SHA256：`fe3a6a9c002adb1f4c79b058cb72f92d824e7bc6b45c6435467a80682b69e0ad`
- 验证标签 SHA256：`073013f0be8aa25c5c1cf81bf9aeb85a07b501335493af2481799921e3286539`

下列数值均为从机器可读 JSON 复制的 `reported` 值；本报告没有重算或替换实验结果。

## 描述性结果

| 指标 | PC | 树莓派 | 样本范围 |
|---|---:|---:|---|
| Runtime 正确数 | 359/360 | 359/360 | P3 稳定段窗口 |
| Runtime Accuracy | 99.7222% | 99.7222% | P3 稳定段窗口 |
| 原模型/Runtime 类别一致 | 360/360 | 360/360 | 同一冻结输入 |
| 最大概率绝对差 | 0 | 0 | Runtime 与直接 TorchScript |
| 归一化往返最大绝对差 | 2.3842×10⁻⁷ | 2.3842×10⁻⁷ | 6 通道 |
| 单窗口延迟中位数 | 2.123 ms | 2.269 ms | 单次 360 窗口回放 |
| 单窗口延迟 P95 | 14.035 ms | 2.324 ms | 单次 360 窗口回放 |
| 浓度字段为空 | 是 | 是 | 分类模式 |
| QC 状态 | Unavailable | Unavailable | 未验证分类 QC |

稳定段覆盖为 `360/420 = 85.71%`。因此 99.72% 只描述 A4 稳定段窗口，不代表暴露全过程或原始连续串口流的准确率。

窗口来自同一块目标板、同一组暴露序列，并且相邻 100 s 窗口存在 50 s 重叠，不能把 360 个窗口当作 360 个独立实验重复。当前只有一个冻结 checkpoint/seed，故不报告跨 seed 标准差、置信区间、显著性检验或效应量。

## 运行稳定性与异常

- 最终 r3 包在树莓派真实生产入口的 offscreen Qt worker 连续 `5/5` 通过。
- 同一 r3 包在真实 Wayland 会话通过，800×480 窗口内容为 800×457，六张 AI 卡片均可见。
- 修复前，TorchScript 首次在 Qt 后台线程调用融合 MHA 算子时存在间歇性原生崩溃。
- 修复包含两层：导出时禁用融合 MHA fastpath，并在 UI 主线程预热 TorchScript 后再启动 worker。
- 导出后的标准图与修复前原模型在 360 个窗口上类别 `360/360` 一致，最大 logit 差 `1.9073×10⁻⁶`，最大概率差 `2.0750×10⁻¹²`。

PC 与树莓派的延迟来自不同操作系统和调度环境，只能作部署可行性描述，不能据此宣称平台性能优劣。

## 三机同步复核

同一源码归档、r3 模型包和冻结验证输入已同步到正式三机的独立目录。云服务器 A 与 B 均再次得到 `359/360`、运行时/直接模型类别 `360/360` 一致、最大概率差 `0`。描述性延迟如下：

| 节点 | Python/Torch | Median | P95 | 数值结果 |
|---|---|---:|---:|---|
| 云服务器 A | `/root/gaps_env/bin/python`, Torch 2.12.0+cu130 | 10.863 ms | 23.104 ms | PASS |
| 云服务器 B | `/root/gaps_c2_cpu_env/bin/python`, Torch 2.12.0+cpu | 3.664 ms | 5.423 ms | PASS |
| 树莓派 | `/home/gaps/GAPS/gaps_rpi_env/bin/python`, Torch 2.12.1+cpu | 2.269 ms | 2.324 ms | PASS |

三机的源码归档、模型和冻结特征 SHA256 完全一致。延迟仍只作单次部署描述，不作硬件优劣比较。

## 结论边界

当前证据支持：

1. A4 第 25 轮模型可无类别变化地导出为三分类 TorchScript；
2. 上位机 v2.2 能正确展示窗口类别、暴露共识类别和置信度；
3. 分类模式不会伪造 ppm，未验证 QC 会明确显示 `Unavailable`；
4. 同一模型包在 PC 和树莓派对冻结输入数值一致；
5. 树莓派真实 Wayland UI 与生产 worker 路径可运行。

当前证据不支持：

1. 原始 STM32 ADC 帧已能直接驱动 A4；
2. 六个通道的 ADC→电阻换算与实验室离线电阻完全一致；
3. 99.72% 适用于 0–150 s 早期段、恢复段或全时间覆盖；
4. 当前模型可输出浓度，或已有经过目标域验证的自动 QC。

## 机器可读来源

- `results/lab_3gas_a4_edge_ui_integration_20260731/p1_model_parity/replay_summary_r3_pc.json`
- `results/lab_3gas_a4_edge_ui_integration_20260731/p2_ui_logic/qt_worker_production_prewarm_pc_r3.json`
- `results/lab_3gas_a4_edge_ui_integration_20260731/p3_pi_replay/pi_replay_summary_r3.json`
- `results/lab_3gas_a4_edge_ui_integration_20260731/p3_pi_replay/pi_qt_worker_r3_stability1.json` 至 `stability5.json`
- `results/lab_3gas_a4_edge_ui_integration_20260731/p3_pi_replay/pi_qt_wayland_worker_r3.json`
- `results/lab_3gas_a4_edge_ui_integration_20260731/p4_cloud_sync/server_a_replay_summary.json`
- `results/lab_3gas_a4_edge_ui_integration_20260731/p4_cloud_sync/server_b_replay_summary.json`
