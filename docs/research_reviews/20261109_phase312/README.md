# FedOnGas Phase 3.12 · PC–Pi 数值诊断与 Pi 5 性能表征

| 证据 | 状态 | 结论 |
| --- | --- | --- |
| Phase 3.8 历史 batch128 | BLOCKED，保留 | 历史门禁不变 |
| Phase 3.10 PC 单窗 | PASS，保留 | 既有 PC 结果不变 |
| Phase 3.11 PC–Pi 完整数值验收 | BLOCKED，保留 | 5,040 条中 30 条 logits 超差 |
| 历史超差静态审计 | 完成 | 49 个分量、27 窗、14 次暴露，类别一致 |
| 32 标定窗 + 4 合成窗 PC–Pi 逐层对照 | 通过最终 logits 合同 | 216 组均在容差内，首处可观察字节差异均在 TCN 第一块 pointwise 卷积输出 |
| 独立 Pi 5 FP32 性能表征 | 四项参考目标全部达到 | 联合 P99 3.936–3.992 ms，60 秒持续 253.0 窗/秒 |
| 真实传感器在线联调 | 未执行 | 仍需独立合同与授权 |

本阶段依照 Phase 3.11 固定提交开展。历史 Day22 只静态分析封存输出。模型、scaler、类别和特征身份保持冻结，新增 Day22 推理为 0。[Phase 3.11 固定报告](https://github.com/Hjh-fy/GAPS/blob/33dff2d8b34e033d6bd69458d4401719ad1fc577/docs/research_reviews/20261109_phase311/README.md)

[历史静态审计](HISTORICAL_AUDIT.md) · [逐层分析](LAYER_DIAGNOSIS.md) · [性能表征](PERFORMANCE.md) · [证据索引](PUBLICATION_MANIFEST.json)
