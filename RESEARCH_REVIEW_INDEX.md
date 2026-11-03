# GAPS 研究审阅入口

审阅分支：`codex/research-reviews`，每轮交付另附固定commit链接。

## 最新：2026-11-02 Phase1.5 少样本公平性与瓶颈

[主报告](docs/research_reviews/20261102_phase15/README.md) · [公平性审计](docs/research_reviews/20261102_phase15/FAIRNESS_AUDIT.md) · [特征误差诊断](docs/research_reviews/20261102_phase15/FEATURE_ERROR_DIAGNOSIS.md) · [研究决策](docs/research_reviews/20261102_phase15/RESEARCH_DECISIONS.md) · [审阅范围](docs/research_reviews/20261102_phase15/REVIEW_SCOPE.md)

30新配置/120头/25200逐窗预测完成并恢复核对。Ridge正则尺度修正六配置均改善，宏平均27.995→20.921%；F公平流程SVR仍6/6方向领先但4/6条件区间包含0；KRR4/6改善但宏平均仅0.210个百分点，停止扩网格。端点裁剪可掩盖raw超界，三个困难气体不能统一归于缺少浓度点或物理老化。所有新结果回顾性，下一步优先71D源知识合法合同与保守复用，不进入复杂模型/部署。

## 历史固定阶段

- [Phase1严格基线](https://github.com/Hjh-fy/GAPS/blob/772c0e9087435b9deb8ea4ac50b7c24814f6b345/docs/research_reviews/20261102_phase1/README.md)：原12配置保存，alpha1000限制由Phase1.5进一步审计，旧结果不覆盖。
- [Phase0.5严格协议](docs/research_reviews/20261102_phase05/README.md)：暴露角色、合法Day7源和两日期替代限制。
- [发布约定](docs/research_reviews/PUBLISHING_POLICY.md)：仅研究审阅分支，原始信号/特征/权重/v5/ZIP保留本地，不自动合并main。
