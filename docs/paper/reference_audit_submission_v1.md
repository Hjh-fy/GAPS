# GAPS IoT-J 投稿候选稿 v1 参考文献审计

审计对象：`docs/paper/GAPS_IoTJ_submission_candidate_v1.zh.html`

上游清单：`docs/submission_preparation/reference_inventory.csv`、`reference_audit.md`
方法：只读复用既有核验结果；本轮未联网补文献、未猜测 DOI。

## 收口结论

- 候选稿保留 19 条学术/数据来源，文内均有引用。
- 原 Ref. 9 的 claim mismatch 已修正：Refs. 5–8 支持 distribution alignment/prototype 语义；Ref. 9 只支持 historical experience-replay 机制，不再用于支持域适应或原型学习。
- 原 Refs. 13/14 已在方法中用于说明 residual block 与 multi-head attention 的通用架构来源，并明确这些骨干组件不是本文贡献。
- 原 Ref. 15 为 IEEE IoT-J author guideline，属于投稿规则而非科学 claim evidence，已从候选稿参考文献表移除；后续只在投稿检查清单中使用。
- 原 Refs. 16–20 因上述删除，在候选稿中顺次编号为 Refs. 15–19；元数据本身不变。
- 既有参考文献核验未发现需要本轮自行替换的条目。冻结条目没有 DOI 时保持空缺，不生成推测 DOI。

## 逐条 claim–reference 映射

| Candidate ref. | 元数据/DOI状态 | 正文位置 | 支持的 claim | 处理结论 |
|---|---|---|---|---|
| 1 | VERIFIED；DOI `10.1016/j.snb.2012.01.074` | 引言、相关工作 | MOS gas-sensor drift and classifier-based compensation | 保留 |
| 2 | VERIFIED；DOI `10.1016/j.dib.2015.01.003` | 引言、相关工作 | chemical gas-sensor array dataset and drift context | 保留 |
| 3 | VERIFIED；冻结条目未记录 DOI | 引言、相关工作 | FedAvg/decentralized model aggregation foundation | 保留 |
| 4 | VERIFIED；冻结条目未记录 DOI | 引言、相关工作 | federated optimization under heterogeneity | 保留；不新增 FedProx experiment |
| 5 | VERIFIED；冻结条目未记录 DOI | 相关工作 | MMD/two-sample distribution discrepancy | 保留 |
| 6 | VERIFIED；DOI `10.1007/978-3-319-49409-8_35` | 相关工作 | CORAL-based distribution alignment | 保留 |
| 7 | VERIFIED；冻结条目未记录 DOI | 相关工作 | domain-adversarial alignment | 保留 |
| 8 | VERIFIED；冻结条目未记录 DOI | 相关工作 | prototype-based semantic representation | 保留 |
| 9 | VERIFIED；冻结条目未记录 DOI | 相关工作 | experience replay for continual learning；仅对应 legacy replay semantics | 已修正 CLAIM_MISMATCH |
| 10 | VERIFIED；冻结条目未记录 DOI | 相关工作 | neural-network calibration context | 保留 |
| 11 | VERIFIED；冻结条目未记录 DOI | 相关工作 | confidence-based error/OOD baseline context | 保留 |
| 12 | VERIFIED；冻结条目未记录 DOI | 相关工作 | selective prediction/reject-option context | 保留 |
| 13 | VERIFIED；冻结条目未记录 DOI | 方法 IV-A | residual learning as generic backbone component | 由“未引用”修正为精确引用 |
| 14 | VERIFIED；冻结条目未记录 DOI | 方法 IV-A | attention architecture as generic backbone component | 由“未引用”修正为精确引用 |
| 15 | VERIFIED；DOI `10.24432/C5MW3K` | 相关工作 | Twin Gas Sensor Arrays dataset identity | 原 Ref. 16；顺次重编号 |
| 16 | VERIFIED；DOI `10.26599/TST.2024.9010251` | 引言、相关工作 | personalized federated learning in edge–cloud sensing | 原 Ref. 17；顺次重编号 |
| 17 | VERIFIED；DOI `10.1109/JIOT.2021.3053055` | 引言、相关工作 | federated sensing and edge–cloud collaboration | 原 Ref. 18；顺次重编号 |
| 18 | VERIFIED；DOI `10.1587/transele.2024ECP5007` | 引言、相关工作 | federated model update for gas-sensor replacement | 原 Ref. 19；顺次重编号 |
| 19 | VERIFIED；DOI `10.1016/j.snb.2016.05.089` | 引言、相关工作 | calibration transfer and drift counteraction | 原 Ref. 20；顺次重编号 |

## 投稿阶段保留的人工检查

1. 在 IEEEtran/BibTeX 转换时，再核对缺 DOI 条目的 Crossref/出版社字段；查不到就保留无 DOI，不猜测。
2. 将原 Ref. 15 author guideline 放入 submission checklist 的格式来源，不纳入 `.bib`。
3. 英文稿生成后逐句复核 Refs. 5–9 的 claim scope，避免再次把 replay、alignment 和 prototype 混写。
4. 正式 LaTeX 排版时核对 en dash、卷期、页码与会议名称缩写；不得改变论文事实或引用对象。
