# GAPS IoT-J 最终系统 benchmark 结果（2026-07-25）

## 结论

固定资产与两平台实测均通过 fail-closed 审计。Runtime v4 继续作为正式 selective-output baseline；Federated-H1 Runtime v5 regression core 是最终简化回归实现；v5 QC2 仍为 `VALID_CANDIDATE_NOT_PROMOTED`，没有因本次 benchmark 改变算法或 QC 决策。当前已具备进入另行授权 low-calibration 阶段的工程条件，但本阶段未启动 low-calibration。

## 协议与环境

- 固定 C5 test 行宇宙：1360；benchmark 按 canonical 顺序取前 500 行，batch=1、warm-up=50、500 次、CPU 单线程、`torch.inference_mode()`。
- steady-state 不含磁盘读取；cold start 独立测量 Python child launch→runtime ready→first inference。
- PC：Intel64 Family 6 Model 158 Stepping 10, GenuineIntel，Windows 11，PyTorch 2.5.1，RAM 16 GiB。
- Pi：Raspberry Pi 5 Model B Rev 1.1，aarch64，PyTorch 2.6.0+cpu，RAM 8 GiB；全部三次 `throttled=0x0`。

## A–C. 延迟、吞吐与资源

| runtime | classifier_params | regression_params | QC_params_assets | bundle_size_bytes | PC_p50_ms | PC_p95_ms | Pi_p50_ms | Pi_p95_ms | Pi_peak_RSS_MiB | Pi_peak_temperature_C | Pi_throughput_windows_per_s | deployment_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RUNTIME_V4_FULL | 22765 | 28737 | non-trainable frozen references/policy | 2971538 | 5.4337 | 13.9448 | 4.57137 | 4.62118 | 237.891 | 54 | 209.109 | FORMAL_BASELINE |
| RUNTIME_V5_REGRESSION_CORE | 22765 | 844 | none | 289916 | 4.0716 | 14.7863 | 3.72454 | 3.76499 | 234.234 | 52.35 | 254.242 | FINAL_SIMPLIFIED_REGRESSION |
| RUNTIME_V5_QC2_CANDIDATE | 22765 | 844 | non-trainable frozen references/policy | 1065632 | 5.47655 | 19.3115 | 4.52254 | 4.57112 | 235.141 | 54.55 | 211.412 | VALID_CANDIDATE_NOT_PROMOTED |

分阶段明细见 `results/iotj_final_system_benchmark_20260725/benchmarks/latency_breakdown.csv`；B5 classification 是三个对象的主要 steady-state 延迟组成。PC 观测存在较大的系统调度尾部，故同时报告 p50/p95/p99，不用均值替代尾延迟。Pi 峰值温度为 54.55°C，未观测 throttling。

## D. 包大小与参数

系统表已同时报告 classifier 参数、regression 参数和 bundle 字节。v5 core 不含 QC；v5 QC2 的 reference/policy 是非训练参数资产。便携 Pi 合同仅重定位路径，模型与 policy 字节 SHA 不变。

## E. B5 FL communication

正式 B5 seed42 真实拓扑 25-round measured application payload 为 17,572,650 bytes；其中 downlink/uplink 为 8,764,300/8,808,350 bytes。理论模型 tensor payload 与 measured serialized application payload 已分列，transport bytes 未采集，不能把 application bytes 称为链路层流量。

## F. Federated H1 与 C5 target Ridge 构建成本

Federated H1 是一次性 sufficient-statistics exchange，C1/C2 的 moments、normal equations、clipped validation SSE/count 及 server 返回资产逐项见 `system_metrics/federated_h1_communication_summary.csv`；没有传输 raw source rows/X/y，也没有宣称 secure aggregation。C5 target Ridge 使用 320 calibration rows、105D 输入、424 个 target-head 参数。原正式构建未分阶段采集 wall time，因此 feature/alpha/refit/serialization 时间保持 `unknown`，没有事后伪造计时。

## G. QC quality–coverage

- v4 HC95：yield 97.28%，accepted RMSE 18.8520 ppm。
- v5 QC2 HC95：yield 93.75%，accepted RMSE 13.9178 ppm。
- v5 的 accepted RMSE 更低，但 yield 也更低；HC90 CO yield 与 accepted-RMSE promotion guard 失败，不能宣称 v5 QC 全局更优。

论文表格位于 `paper_tables/`，论文图位于 `figures/`。图只展示冻结 HC95/HC90 四个工作点，没有拟合曲线、增加阈值或重新打开 test。

## H. 异常与审计

Pi 首次两次尝试均在正式计时前 fail-closed，原因分别为便携包 import/reference 闭包和 v5 lineage 绝对路径；均通过测试后仅修复路径封装，不修改 runtime、模型、policy 或阈值。正式六个 PC/Pi 对象均 `PASS`，分阶段复算与普通 runtime 预测/decision 一致。

## I. Evidence boundary

- Runtime v4 是正式 selective-output baseline。
- v5 QC2 是有效但未晋级的 candidate；accepted RMSE 的降低必须和更低 yield 一起解释。
- v5 Federated-H1 regression 保持最终简化回归选择。
- filename grouping 仅适用于 calibration OOF folds；历史 calibration/test split 是 window-level，不宣称 original-file level 完全独立。
- test 打开后没有修改 candidate、组件、scale、ECDF 或 threshold。
- 本阶段没有启动 low-calibration、新 QC、训练或 runtime v5 promotion。
