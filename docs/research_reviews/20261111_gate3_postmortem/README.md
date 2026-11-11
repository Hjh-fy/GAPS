# Gate 3 之后 · 失效机制复盘

**研究决定 INSUFFICIENT_EVIDENCE_STOP。** 标定代表性与有限更新非线性均有支持性观察，现有封存评分无法识别两者各自对风险反转的贡献。

第 5 轮 Ethanol 的开发 CE 在 0.25、0.5、1.0 倍反馈下均上升，且曲线非单调。Ethylene 的上升主要由一次 25 ppm 暴露贡献，Methane 的上升主要由 200、225 ppm 两次暴露贡献。即使只看开发中的原标定时间锚窗，类别平均反转仍存在，幅度和暴露分布会改变。以上均为关联或有限步几何描述。

依据 Gate 3 固定提交 93c2f9fe445b0c8079cba8d52bae8f6279dd089e。只读取封存 CE、标定风险、候选结果及身份元数据，新增原始信号读取、模型运算与 Day22 评价均为 0。全部原始暴露身份和详细表保存在本地，公开副本仅含摘要、方法与 SHA。

[案例审计](FAILURE_CASE_AUDIT.md) · [固定缩放曲线](FIXED_SCALE_ANALYSIS.md) · [浓度支持](CONCENTRATION_SUPPORT_MATRIX.md) · [可识别性](IDENTIFIABILITY.md) · [数据资产](ASSET_AUDIT.md) · [新增访问提案](NEW_ACCESS_PROPOSAL.md) · [最终决定](FINAL_DECISION.md)
