# GAPS IoT-J 中文投稿候选稿 v1 图表规划

状态：`PAPER_EVIDENCE_READY`

候选稿：`docs/paper/GAPS_IoTJ_submission_candidate_v1.zh.html`
原则：只复用冻结图源；本轮不重绘、不重新计算、不改变任何数据。

## 正文图

### Fig. 1 — System/data-flow

- Message：真实三机联邦分类、源端充分统计量聚合、C5 calibration/personalization 与部署输出构成一条证据边界清晰的数据流；C1/C2 原始训练行不上传。
- Required data：设备角色、模型/统计量流向、C5 calibration/test 角色；不需要实验数值。
- Source asset：`docs/paper_evidence_freeze/figures/fig1_architecture.svg`。
- Caption：Cloud–edge–device system and data flow. C1/C2 retain raw source rows locally; the server performs FedAvg-based model aggregation, semantic-statistics collection, calibration-assisted adaptation, and H1 sufficient-statistics aggregation; C5 supplies calibration and held-out windows.
- Placement：正文，III. 系统与数据协议。
- Duplicate table to remove：不删除 Table I；Fig. 1 表达流向，Table I 只保留设备、角色与 split identity，不重复通信量或性能数字。

### Fig. 2 — Single-window inference architecture

- Message：单窗口先由 frozen B5 产生 gas route，再将对应 Federated H1 prediction 与 104D rich features 拼接为 105D 输入，交给 per-gas target Ridge；target Ridge 不属于网络 FedAvg。
- Required data：分类 route、H1 source prediction、104D→105D 拼接、per-gas Ridge、ppm output；不需要 test 数值。
- Source asset：双面板复用 `fig2_federated_classification.svg` 与 `fig3_h1_target_personalization.svg`。
- Caption：Single-window inference architecture. The frozen B5 classifier selects the gas route; a one-dimensional Federated H1 prediction augments 104D rich features before the per-gas target Ridge.
- Placement：正文，IV-C 目标个性化。
- Duplicate table to remove：无；不再保留旧稿中独立的 classification Fig. 2 与 personalization Fig. 3。

### Fig. 3 — Paired five-route regression comparison

- Message：在同一组 seeds 42–46 frozen B5 routes 下，Federated H1 相对 all-prior 满足 1% simplification-noninferiority，但 all-prior 在 5/5 routes 上具有更低 S_CC。
- Required data：RG0/RG1/RG2 的五条 paired route S_CC RMSE。
- Source asset：`docs/paper_evidence_freeze/figures/fig4_regression_five_seed.png`；可打印版本为同名 PDF。
- Caption：Paired five-route correct-route RMSE. Federated H1 meets the preregistered 1% simplification-noninferiority rule; all-prior retains the lower absolute S_CC value in every route.
- Placement：正文，VI-B RQ2。
- Duplicate table to remove：删除正文 per-seed regression table；Table III 仅保留 mean±sample-SD、input dependency 与 decision。

### Fig. 4 — QC quality–coverage

- Message：v5 QC2 的较低 accepted RMSE 与较低 yield 必须联合解释；HC90 CO promotion guard 未满足，故不晋级。
- Required data：v4/v5 的 HC95/HC90 accepted yield 与 accepted RMSE frozen workpoints。
- Source asset：`docs/paper_evidence_freeze/figures/fig5_qc_quality_coverage.png`；可打印版本为同名 PDF。
- Caption：Frozen HC95/HC90 quality–coverage workpoints. Lower accepted error is interpreted jointly with accepted yield; no curve or threshold is refitted.
- Placement：正文，VI-C RQ3。
- Duplicate table to remove：删除旧正文 QC summary table；完整 A/R/R 与 CO diagnostics 只保留在 Table A5。正文图不添加逐点数值标签。

### Fig. 5 — Calibration sensitivity

- Message：group-aware 描述性分析中，calibration budget 减少会明显提高 S_CC RMSE；该轨迹不能替代 historical seed42 frozen result。
- Required data：320/160/80/40 calibration budgets 的 group-aware S_CC mean 与 registered variability。
- Source asset：`docs/paper_evidence_freeze/figures/fig7_groupaware_calibration.png`；可打印版本为同名 PDF。
- Caption：Group-aware calibration-budget sensitivity. Error bars retain registered fold/subset variability, and lines only connect observed budgets.
- Placement：正文，VI-C RQ3。
- Duplicate table to remove：删除旧正文 Table VI；Table A6 仅保留 historical/group-aware harmonization 与 protocol delta，不重复 S_ALL。

## 正文表

| Table | 内容 | 正文功能 | 与图的边界 |
|---|---|---|---|
| I | dataset/devices/protocol | 固定 C1/C2/server/C5 角色与 held-out-window identity | Fig. 1 只画流向 |
| II | B5 five-seed classification | 报告 Accuracy/Macro-F1/NLL/ECE 的 mean/SD/range | 无重复主图 |
| III | regression personalization | 报告 RG0/RG1/RG2 summary 与非劣选择 | Fig. 3 展示 paired routes |
| IV | runtime efficiency and functional identity | 区分 v4 formal QC、v5 core、v5 QC2；报告参数、bundle、PC/Pi latency | 不保留旧 Fig. 6 efficiency |

## 附录表

- Table A1：legacy classification，必须保留 `historical mechanism semantics`、`corrected single-seed screening`、`final five-seed evidence` 三种证据身份。
- Table A2：per-gas RG1 与 seed42 CO-high；显式区分 five-route mean 与 seed42 subset。
- Table A3：H1 practical equivalence。
- Table A4：application communication 与 H1 theoretical serialized exchange；标注 transport bytes 未采集。
- Table A5：full QC diagnostics；正文不再放置同一数值表。
- Table A6：calibration harmonization；historical 与 group-aware 不合并为一条方法排名。

## 双栏预审结论

- Fig. 1、Fig. 2 建议双栏宽图；Fig. 2 保留双面板。
- Fig. 3–5 可单栏，最终排版时优先使用 PDF 矢量/可打印源。
- Table I、III、IV 建议双栏；Table II 可单栏。
- Table A1、A4、A5、A6 建议附录双栏。
- 本轮不更改任何图内文字、分辨率或冻结数据；最终重绘需另行人工批准，并以冻结 CSV 为唯一数值源。
