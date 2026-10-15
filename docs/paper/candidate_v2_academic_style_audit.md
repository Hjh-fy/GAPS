# GAPS IoT-J 中文投稿候选稿 v2：学术写作审计

## 结论

状态：`PASS_FOR_HUMAN_CHINESE_REVIEW`

v2 已从“证据汇总稿”改写为按研究问题组织的学术论文候选稿。标题、摘要、引言、相关工作、问题定义、方法、实验设置、结果、讨论和结论形成完整论证链；正式数字和 claim boundary 未改变。本轮没有生成英文稿或 LaTeX。

## P1–P10 检查

| 要求 | 完成情况 | 审计说明 |
|---|---|---|
| 精简标题 | `PASS` | 使用“GAPS：面向跨设备气体感知的联邦分类与轻量回归个性化” |
| 摘要采用问题→方法→三个结果→意义/边界 | `PASS` | 保留 five-seed classification、Federated H1、28,737→844、Pi latency 与 held-out-window boundary；摘要未写 seed42、PC p95、HC90 guard 或多组 calibration 数字 |
| 五段式引言 | `PASS` | 应用动机、集中/迁移局限、FL gap、分类—回归—部署 gap、GAPS/RQ/贡献 |
| 三分相关工作 | `PASS` | Cross-device calibration；federated sensing/personalization；DA/lightweight regression/selective prediction；各节均以 remaining gap 收束 |
| 问题定义与符号 | `PASS` | 定义 C1/C2、C5、\(\mathbf{x}\in R^{100\times8}\)、类别/浓度、calibration/test、predicted route 和回归输出 |
| 方法公式与实现对齐 | `PASS` | FedAvg/语义缩放、本地目标、server DA active losses、H1 scaler/normal equations/alpha selection、105D Ridge、predicted-route inference |
| 104D 特征表 | `PASS` | 64+19+10+3+4+4=104；来自冻结 feature extractor |
| 完整实验设置 | `PASS` | 数据、气体/浓度、采样/预处理、窗口/步长、normalization、真实拓扑、超参数、baselines、metric formulas、seed scope、benchmark 设置均已补充 |
| 内部代号降噪 | `PASS` | 摘要与正文主表使用语义方法名；B5/RG/v4/v5/QC2 等只在必要的证据身份或 legacy appendix 中保留 |
| Discussion 机制顺序 | `PASS` | 先讨论 classification vs regression、H1、all-prior marginal gain、candidate QC，再讨论 evaluation/privacy/seed/runtime limitations |

## Claim-strength 检查

- “五种子稳定”限定于当前 C1/C2→C5、数据与超参数协议。
- H1 的选择写为“满足预先设定的 1% performance-retention tolerance”，并明确这不是形式化统计检验。
- All-prior 在 5/5 routes 的绝对 \(S_{\mathrm{CC}}\) 更低被完整保留，没有把 H1 写成精度优越。
- H1 机制解释使用“与观察结果相符”，没有写成未经独立因果实验验证的确定结论。
- 候选 QC 的较低 accepted error 与较低 coverage、CO guard failure 同时报告。
- Pi 延迟只限定于指定平台和 benchmark 设置；正文明确 PC p95 没有改善。
- Portable release 只被描述为工程可移植性证据，不是性能证据。
- “raw source rows remain local”没有扩张为 secure aggregation、DP 或形式化隐私保证。

## 术语审计

| 术语/风险 | 结果 |
|---|---|
| calibrated-target held-out-window evaluation | 首次定义并贯穿正文 |
| calibration-assisted adaptation | 使用正确；未写成 UDA 或 zero-shot |
| original-file/session independent | 只用于否定性边界和 future work |
| “预注册 1% 非劣” | 已移除 |
| “statistically non-inferior” | 未使用 |
| “97.1% model reduction” | 已替换为 28,737→844 |
| “更强 baseline” | 未使用 |
| “所有平台/所有 percentile 延迟均改善” | 未使用 |
| selective aggregation 作为核心贡献 | 未使用，仅作为 implementation component |
| v4 accepted RMSE 与 v5 core 参数量混为一体 | 已显式禁止 |

## 文献表述修正

- Ref. 9 仅用于支持持续学习中 replay 的一般机制；正文明确本项目的 replay feature distillation 是实现特定组件，不用 Ref. 9 支持域适配或原型主张。
- Refs. 13/14 已用于说明残差与注意力是通用结构组件，不作为 GAPS 独立算法贡献。
- Ref. 15 用于数据来源/数据集上下文，不再承担投稿指南身份。
- 参考文献总数保持 19；本轮没有添加未经核验的新文献或元数据。

## 仍需人工中文审阅的非阻塞项

1. 检查中英文术语混排密度，决定 `rich features`、`performance-retention tolerance`、`accepted yield` 等在最终中文稿中的保留形式。
2. 检查服务器适配十项损失是否需要在正文缩写、完整式移至附录；任何缩写不得改变 active/disabled 身份。
3. 确认“明显响应差异”“降低 RMSE”等描述性措辞符合导师偏好，避免被理解为统计显著性声明。
4. 最终英文稿应重新核验冠词、时态和 `calibration`/`adaptation` 的角色，不应逐字直译内部协议名称。
5. 文献格式仍需按 IEEEtran/BibTeX 做最终统一，本轮只保持已核验元数据。

最终判定：`ACADEMIC_STYLE_PASS`
