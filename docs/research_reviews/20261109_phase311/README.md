# FedOnGas Phase 3.11 · Raspberry Pi 5 离线联合推理

在 Raspberry Pi 5 Model B Rev 1.1 上完成六组冻结模型的 5,040 条 Day22 单窗推理。相对 Phase 3.10 PC 单窗参考，类别、选头和状态全部一致，71D 特征一致，浓度最大绝对差为 3.13e-13 ppm。30 条模型—窗口记录的分类 logits 超出预注册容差，完整数值验收为 **BLOCKED_NUMERIC_LOGITS**。

| 门禁 | 状态 | 实际证据 |
| --- | --- | --- |
| ARM64 硬件及隔离依赖 | PASS | Pi 5、原权重严格加载、六分类端点与四 SVR 头验证通过 |
| Gate A 标定与合成 | PASS | 16 次 Day18 标定暴露的 32 窗和 4 个合成窗；216 条联合对照及 144 次回归头直接对照 |
| 重复、重启、重载与异常恢复 | PASS | 216 条向量在同进程重复、完整重载和独立进程中相同，故障后保留健康快照 |
| Gate B 实机重放完整性 | PASS_EXECUTION | 40 次历史暴露、840 窗、六固定视图，各推理阶段 5,040 次调用 |
| Gate B 完整数值验收 | BLOCKED_NUMERIC_LOGITS | 30 条视图记录、49 个分量超容差，涉及 27 窗、14 次暴露 |
| 分段延迟与资源目标 | NOT_RUN | 完整数值门禁未通过，未启动独立性能测试 |
| 真实传感器在线联调 | NOT_READY | 完整数值门禁与实机性能尚未验收，在线采集、通信和故障时序尚无实测证据 |

PC 依据为 Phase 3.10 固定提交 ae2a614249443edaeae78d421851a3bdc57a92d6，采用 Phase 3.9 的 PC-F32-CPU-SINGLE-v1。Pi 合同 PI5-ARM64-FP32-SINGLE-v1 在第一条 ARM 数值输出前锁定。原 Phase 3.8 batch128 的 BLOCKED 和 Phase 3.10 PC 单窗结果分别保留。

Day22 为已观察数据的工程重放。B0–B5 的权重、回归头、scaler 与支持范围均沿用冻结端点。本阶段共完成一次 Day22 实机重放。

[数值与故障证据](NUMERICAL_AND_RECOVERY.md) · [执行与恢复](EXECUTION_AND_RECOVERY.md) · [最终核验](FINAL_VERIFICATION.json)
