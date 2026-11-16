# 外部技术与场景复用调研

2026-11-16。核验11个GitHub仓库及官网/社区来源，静态检查3个源模块、4份许可证和1份依赖表。没有安装、训练、推理、重评分或Pi操作。

最值得借鉴的是光谱仪少标准样本校准、Calibration内的主动选样接口、任务分别监控，以及候选/在用模型版本管理。它们提供成熟基线与工程部件，不自动构成原创算法。

[主要技术与适用边界](TECHNICAL_FINDINGS.md) · [11项复用矩阵](REUSE_CANDIDATES.csv) · [固定上游版本](UPSTREAM_LOCK.json) · [源码检查](CODE_REVIEW_FINDINGS.md) · [社区经验](COMMUNITY_EVIDENCE.md) · [创新边界与下一步](NOVELTY_AND_NEXT_STEPS.md) · [来源登记](SOURCE_REGISTER.csv) · [检索范围](SEARCH_METHOD_AND_LIMITS.md)

建议只保留两类低成本前置审计：代表性选窗、低自由度源知识校准。系统侧借鉴River/MLflow的监控与版本模式，继续复用现有Flower、模型包、Pi和GUI。未经新注册授权，不启动任何候选实验。
