# GAPS 研究审阅入口

审阅分支：`codex/research-reviews`。

## 最新：2026-11-02 Phase1 严格回归强基线

[审阅范围](docs/research_reviews/20261102_phase1/REVIEW_SCOPE.md) → [主报告](docs/research_reviews/20261102_phase1/README.md) → [创新预判与停止条件](docs/research_reviews/20261102_phase1/INNOVATION_AND_STOP_GO.md) → [审计](docs/research_reviews/20261102_phase1/EXPERIMENT_AUDIT.md)。

12配置/48头/10080逐窗结果完成，仅Oracle。SVR在固定source-only选择流程下6/6低于Ridge；必须同时看Ridge alpha1000上界和source→fewshot正则强度不匹配限制。C4 A4 SVR略差，负结果保留。C3三日期，C4/C5两日期替代；均值融合不是单窗部署性能。

## 历史阶段

- [Phase0.5协议审计](docs/research_reviews/20261102_phase05/README.md)：源Day7队列、暴露互斥及C4/C5三日期阻塞。
- [发布约定](docs/research_reviews/PUBLISHING_POLICY.md)：每轮报告和精简证据持续发布；原始信号/权重/v5/ZIP保留本地，旧窗口结果不当独立暴露证据，不自动合并main。
