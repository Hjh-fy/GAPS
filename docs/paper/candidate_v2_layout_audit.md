# GAPS IoT-J 中文投稿候选稿 v2：IEEE 双栏布局审计

## 结论

状态：`LAYOUT_PLAN_READY`

当前 HTML 是内容审阅稿，不是最终 IEEEtran 版。五幅正文图、四组正文表与六张附录表均已有明确版面角色。Table IV 已拆为 IV(a) 功能身份/规模和 IV(b) 延迟，最终建议作为一个跨双栏 `table*`，避免在单栏中压缩七列。

## 正文图规划

| Figure | 核心信息 | 冻结 source asset | 最终宽度 | 英文稿处理 | 重复信息处置 |
|---|---|---|---|---|---|
| Fig. 1 | C1/C2→server→C5 系统角色、原始源行本地性与两类交换对象 | `docs/paper_evidence_freeze/figures/fig1_architecture.svg` | 双栏 `figure*`，页顶 | `REDRAW_FOR_ENGLISH_V1`：保留拓扑与数据边界，统一箭头方向、字体≥8 pt | Table I 只保留数据/角色，不重复画流程 |
| Fig. 2 | 单窗口分类路由与 H1→105D target Ridge | `fig2_federated_classification.svg` + `fig3_h1_target_personalization.svg` | 双栏 `figure*`，两 panel | `REDRAW_FOR_ENGLISH_V1`：删除 “B5” 内部代号，改为 semantic labels；清晰区分 network FL 和 offline target fit | 方法公式解释算法；图不重复超参数表 |
| Fig. 3 | 五条 frozen route 的三种回归输入配对比较 | `fig4_regression_five_seed.png` | 单栏；若标签拥挤则双栏 | `REEXPORT_FOR_ENGLISH`：用 semantic method names，色盲安全；不改变点和数值 | Table III 保留均值±SD；图只呈现配对方向 |
| Fig. 4 | 正式 baseline 与 candidate 的 quality–coverage workpoints | `fig5_qc_quality_coverage.png` | 单栏 | `REEXPORT_FOR_ENGLISH`：标注 workpoints 而非拟合 curve，统一 accept/yield 术语 | 完整 A/R/R、CO 指标移至 Appendix Table A5 |
| Fig. 5 | Group-aware calibration-budget sensitivity | `fig7_groupaware_calibration.png` | 单栏 | `REEXPORT_FOR_ENGLISH`：保持 historical/group-aware 身份分离，图注说明 descriptive | Appendix Table A6 保留精确数值，正文不重复全表 |

所有英文重绘只能改变视觉编码、字体和标签，不得改变数据、阈值、工作点或数值。

## 正文表规划

| Table | 内容 | 推荐环境 | 风险与处理 |
|---|---|---|---|
| Table I | dataset、预处理、设备角色、split identity | `table*` 或精简后单栏 | 当前文本较长；IEEE 版优先转为 3 列，协议边界保留在正文 |
| Method Table 1 | 104D feature groups | 单栏 `table` | 6 行可读；physical meaning 列需压缩为短语；总维度 104 保留 |
| Table II | five-seed classification | 单栏 `table` | 5 列可放单栏；数字保留 6 位小数 |
| Table III | regression personalization | `table*` | 方法/输入/两类 RMSE/身份共 5 列；semantic names 较长，不建议强塞单栏 |
| Table IV(a)+(b) | runtime identity/complexity + latency | 一个 `table*`，上下两个 panel | 不把 v4 accepted RMSE 放入表；classifier 22,765 在表注单独计数 |

## 附录与 supplementary material 分配

### 论文附录

- Table A1：legacy classification mechanism context。必须保留 evidence identity，且明确不是 final classifier 的严格消融。
- Table A2：Federated-H1 per-gas 与 CO-high。
- Table A3：H1 practical-equivalence 核心摘要。
- Table A5：完整 QC accept/review/reject、yield 与 CO guards。
- Table A6：historical/group-aware calibration harmonization。

### Supplementary material

- Table A4 的逐方向应用层通信明细及完整通信 manifest。
- H1 每气体 alpha、scaler/coefficient equivalence 的全量审计行。
- 五种子 confusion matrices、per-class precision/recall/F1。
- QC 每组件、每气体、每工作点的完整 diagnostics。
- Calibration harmonization 的 fold/subset 级记录。
- Runtime benchmark 的环境 manifest、冷启动记录、row timing 和 SHA index。

### 不进入论文性能图表的工程材料

- Portable archive SHA、clean-checkout synthetic smoke receipt、CLI usage。
- Runtime contract 的资产路径和内部 schema 字段。
- 训练控制器、preflight/postflight 和远端命令记录。

这些内容可放代码仓库 release 文档，但不作为算法性能证据。

## 双栏放置顺序

1. Fig. 1：Introduction 末或 System Model 开始后的首个页顶 `figure*`。
2. Table I：Problem Definition/Data Protocol 后。
3. Fig. 2：Method 总览开始处，双栏页顶。
4. Method Table 1：104D contract 小节内单栏。
5. Table II：RQ1 结果段首。
6. Table III：RQ2 结果页顶 `table*`。
7. Fig. 3：紧随 Table III 后的单栏配对图。
8. Fig. 4：RQ3 QC 段单栏。
9. Table IV(a)/(b)：RQ3 efficiency 段页顶 `table*`。
10. Fig. 5：Calibration boundary 段单栏。

## 版面风险

1. **方法密度：中等。** Server adaptation 十项 loss 若全部保留在正文可能占用约半栏；IEEE 版可保留总式和 active/disabled 边界，将逐项定义压缩为一段。
2. **Table I：中等。** 当前长文本适合 HTML 审阅，不适合原样进入双栏；需在不删除 held-out-window boundary 的前提下压缩。
3. **Table III：高。** semantic method names 长，必须使用 `table*` 或缩短为首处定义后的名称。
4. **Table IV：已缓解。** 拆为两个 panel，建议共用一个 caption 和 table notes；不得重新合并为七列单栏。
5. **Fig. 2：中等。** 两个原始 SVG 横向总宽较大；英文版需统一坐标、字号和 panel 标签 `(a)/(b)`。
6. **附录长度：中等。** 若期刊页数受限，优先把通信全量和 fold-level diagnostics 移入 supplementary，而不是删除不利结果或边界。

## 最终视觉 QA 门槛

- 双栏正文最小图中文字不低于 8 pt。
- 灰度打印仍能区分三种 regression methods 和正式/candidate QC。
- 图注能够独立说明 seed/scope/metric，不依赖内部 run ID。
- 表中 `S_CC`、`S_ALL`、accepted RMSE、yield 和 parameter count 不跨运行对象合并。
- 所有 PNG 按最终版面宽度检查有效分辨率；SVG 字体嵌入或转路径策略在 LaTeX 阶段统一。
- Caption 不复述正文全部数字，正文也不再次列出附录全表。

最终判定：`IEEE_LAYOUT_PREAUDIT_PASS`
