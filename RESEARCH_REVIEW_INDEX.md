# GAPS 研究审阅入口

当前审阅分支：`codex/research-reviews`。供外部GPT agent阅读的最新阶段为[2026-11-02 Phase0.5协议审计](docs/research_reviews/20261102_phase05/README.md)。

阅读顺序：[审阅范围](docs/research_reviews/20261102_phase05/REVIEW_SCOPE.md) → [主报告](docs/research_reviews/20261102_phase05/README.md) → [最小实验计划](docs/research_reviews/20261102_phase05/EXPERIMENT_PLAN.md) → [矩阵](docs/research_reviews/20261102_phase05/EXPERIMENT_MATRIX.csv) → [Flower静态审计](docs/research_reviews/20261102_phase05/FLOWER_AUDIT.md)。

结论：C3可三日期隔离；C4/C5仅暴露隔离和跨日test，三日期阻塞。仅完成数据/配置审计，0新模型训练。历史窗口结果和新严格角色不能混为同等证据。

本分支的审阅文档有自己的提交历史；原冻结v5/数据/权重/ZIP保留本地。[发布约定](docs/research_reviews/PUBLISHING_POLICY.md)长期适用。
