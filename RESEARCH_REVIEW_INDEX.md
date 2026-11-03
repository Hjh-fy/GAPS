# GAPS 研究审阅入口

审阅分支：`codex/research-reviews`。每阶段交付提供固定commit，旧版本保留。

## 最新：2026-11-03 Phase2A 71D历史浓度知识迁移资产审计

[主报告](docs/research_reviews/20261103_phase2a/README.md) · [复用审计](docs/research_reviews/20261103_phase2a/REUSE_AUDIT.md) · [源输入/时间合同](docs/research_reviews/20261103_phase2a/SOURCE71_INPUT_AND_TIME_CONTRACT.md) · [最小矩阵](docs/research_reviews/20261103_phase2a/EXPERIMENT_PLAN.md) · [创新与负迁移](docs/research_reviews/20261103_phase2a/INNOVATION_AND_NEGATIVE_TRANSFER.md) · [Phase2B门禁](docs/research_reviews/20261103_phase2a/PHASE2B_DECISION.md)

只读审计/7项检查/24次合成fit完成，真实源和目标fit均0。旧源银行与残差83/104权重不能直用；120暴露/2520窗合法71D输入闭合，原数值部件可复用。rho0合成探针严格相同但源NaN仍会破坏旧预测接口；生产兼容层和真实新源模型未验收。仅继续源侧构建与合同验收，不释放目标迁移训练。640暴露均有历史元数据成员关系，独立确认集尚未建立。

## 历史固定阶段

- [Phase1.5 4988abb](https://github.com/Hjh-fy/GAPS/blob/4988abb3160c0b0b9230a48a817f2d7baed137d2/docs/research_reviews/20261102_phase15/README.md)：正则迁移公平性和三个困难气体，KRR停止扩网格。
- [Phase1 772c0e9](https://github.com/Hjh-fy/GAPS/blob/772c0e9087435b9deb8ea4ac50b7c24814f6b345/docs/research_reviews/20261102_phase1/README.md)：12配置严格Oracle回归基线。
- [Phase0.5](docs/research_reviews/20261102_phase05/README.md)：暴露角色、Day7源和两日期替代限制。
- [发布约定](docs/research_reviews/PUBLISHING_POLICY.md)：审阅分支，不自动合并main；原始数据/特征/权重/v5/ZIP保留本地。
