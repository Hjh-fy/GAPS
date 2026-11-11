# 创新性审阅

最近邻技术矩阵见NOVELTY_NEAREST_NEIGHBOR_MATRIX.csv。检索来源只支持其明确列出的技术范围；论文摘要不能支持未读的实现细节。FedLAG与充分统计量Ridge来源为作者预印本，不提升为高质量期刊验证。

原型语义、协作加权、目标梯度投影、层级冲突控制、充分统计量Ridge和气体阵列联邦更新都已有先行。FedOnGas的服务器聚合→目标适配→跨轮反馈组合可以作为本项目实现方案描述，不能从组合存在推出算法首创或稳定性能优势。

A全参数投影相对FedGP的可核验差异是约束服务器适配位移而非其原聚合梯度，并采用源/目标八类CE约束。B相对FedLAG的差异是固定参数组的连续缩放QP而非冲突层个性化。两者只通过数值/KKT和固定模型局部测试，没有形成经过独立验证的算法保护机制。更新范数保留不等于知识保留，B可行域包含于A的数学关系也不是经验性能优势。

本研究更有区别的观察角度是物理暴露等权、Calibration风险与独立开发暴露有限步CE对照，以及把标定代表性与有限位移余项明确分开。这可以标为分析问题的DIFFERENTIATED，不是宣称首次提出梯度可靠性，也不是已证实新算法。若写方法论文，仍需完整最近邻复现、可靠约束规则及独立证据，当前均未闭合。

本轮不通过新增论文叙事将已有成熟模块改名为创新算法，也不以文献不同数据集为由忽略机制重叠。


可核验主来源：

- [FedProto](https://ojs.aaai.org/index.php/AAAI/article/view/20819)，peer-reviewed primary abstract。
- [FedPAC](https://arxiv.org/abs/2306.11867)，author primary paper。
- [FedGP](https://proceedings.iclr.cc/paper_files/paper/2024/file/d649dd586dba64672a96beff24178125-Paper-Conference.pdf)，peer-reviewed full paper。
- [FedLAG](https://arxiv.org/abs/2410.02845)，author preprint; no peer-review status inferred。
- [STP-FCL](https://pmc.ncbi.nlm.nih.gov/articles/PMC13667711/)，peer-reviewed full primary article。
- [One-Shot Federated Ridge Regression: Exact Recovery via Sufficient Statistic Aggregation](https://arxiv.org/abs/2601.08216)，author preprint; not journal evidence。
- [FedGSSU](https://globals.ieice.org/en_transactions/electronics/10.1587/transele.2024ECP5007/_f)，publisher primary article。
- [Feature disentanglement and adaptive collaboration for multi-source domain adaptation in gas recognition](https://www.sciencedirect.com/science/article/abs/pii/S0925400526006787)，publisher primary abstract; full details not independently checked。

FedProto出版商页面本次有机器人访问限制，机制通过[作者原文](https://arxiv.org/abs/2105.00243)核验，不把受限页面当作已读取全文。DAFD/PGAC采用出版商摘要及公开引言片段，没有声称已核验付费全文实现。
