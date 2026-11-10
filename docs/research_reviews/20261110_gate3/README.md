# FedOnGas Gate 3 · 独立开发暴露上的类别风险预测

**决定 REDESIGN_REQUIRED。** 第 1 轮有可复核的局部预警，第 5 轮存在标定符号稳定但开发方向相反的结果，后两轮类别均值均落入判定不确定。A/B 相对同范数缩放各有改善与恶化。

| 门禁 | 状态 | 证据 |
| --- | --- | --- |
| Gate 3A | PASS | 16 次标定与 24 次 C5 Day18 开发物理暴露互斥，开发没有进入 Phase 3.5 训练输入。候选方向和风险在开发开放前封存 |
| Gate 3B | COMPLETE | 4 轮 × 8 候选 × 24 暴露，共 768 条暴露评分，16,128 个窗口前向 |
| Gate 3C | REDESIGN_REQUIRED | 33 条方向一致、19 条漏报、0 条误报、76 条不确定。有效独立暴露数仍为 24 |
| 源保护与 C3 | 未评价 | 独立源开发需审计，C3 仅提交扩展方案 |

该实验为 C5 同设备同日标定外验证。Phase 0.5 曾使用开发暴露核对原始身份和窗口质量，已审计的 Phase 1/1.5 未用目标 Development 选参，Phase 3.1/3.5 未将其用于当前训练。旧版更大标定预算的执行范围与样本归属仍有历史不确定性，因此本轮不声称首次盲测。

[风险转移分析](RELIABILITY_TRANSFER_ANALYSIS.md) · [最终决定](FINAL_DECISION.md) · [源保护缺口](SOURCE_VALIDATION_GAP.md) · [C3 扩展方案](C3_EXTENSION_PROPOSAL.md) · [证据摘要与 SHA](PUBLICATION_MANIFEST.json)
