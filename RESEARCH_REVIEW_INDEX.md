# GAPS 研究审阅入口

审阅分支`codex/research-reviews`，每轮提供固定commit。

最新：[Phase3.3 SOURCE启动失败审计](docs/research_reviews/20261105_phase33/README.md) · [根因](docs/research_reviews/20261105_phase33/ROOT_CAUSE.md) · [执行证据](docs/research_reviews/20261105_phase33/EXECUTION_AUDIT.md) · [停止/下一步](docs/research_reviews/20261105_phase33/NEXT_DECISION.md)。

唯一SOURCE启动在第1轮fit分发前失败：冻结configure_fit参数名不兼容Flower实际关键字调度；客户端另遇默认沙箱localhost连接权限10013。0梯度更新、0完成轮、0checkpoint、无Sstar、无Test数值读取、无重试或pilot。此前Phase3.2局部预检漏测原生调度，不能继续据此宣布完整入口就绪。需要新注册兼容修复/无真实数据网络预检并另行批准恢复SOURCE。

历史：[Phase3.2固定b0c7a5c](https://github.com/Hjh-fy/GAPS/blob/b0c7a5cabfec3a41b40cdf67fc0802b67103a3ed/docs/research_reviews/20261105_phase32/README.md) · [Phase3.1固定b05c139](https://github.com/Hjh-fy/GAPS/blob/b05c139d757a79994f8fc2fc983f1475052bdd45/docs/research_reviews/20261105_phase31/README.md) · [Phase3](docs/research_reviews/20261104_phase3/README.md) · [Phase2B](docs/research_reviews/20261104_phase2b/README.md) · [Phase2A](docs/research_reviews/20261103_phase2a/README.md) · [Phase1.5](docs/research_reviews/20261102_phase15/README.md) · [Phase1](docs/research_reviews/20261102_phase1/README.md) · [Phase0.5](docs/research_reviews/20261102_phase05/README.md)。

[发布约定](docs/research_reviews/PUBLISHING_POLICY.md)：不自动合并main，不上传原始输入/模型权重/v5/ZIP。
