# GAPS IoT-J 中文投稿候选稿 v3.1 修订审计

## 结论

状态：`READY_TO_FREEZE_CHINESE_CONTENT_FOR_ENGLISH_TRANSLATION`

冻结动作状态：`PENDING_HUMAN_SIGNOFF`

v3.1 是在 v3 上完成的纯文字局部修订稿。论文结构、正式数字、模型与 QC 决策、图表数据和 claim–evidence 关系均未改变；本轮未运行训练、推理、评估或 benchmark，也未访问正式 test 资产。

在作者或导师确认标题、三项贡献、目标校准依赖、附录/补充材料分配及作者公开信息之前，本文件只能作为中文内容冻结候选，不应标记为不可变中文冻结稿。

## 版本边界

- 父稿：`docs/paper/GAPS_IoTJ_submission_candidate_v3.zh.html`
- 父稿 SHA256：`83f665d0442cbcef2aa0f9b931c9efd758b25926dbca2dcef1dbd7fb3f9c1422`
- v3.1：`docs/paper/GAPS_IoTJ_submission_candidate_v3_1.zh.html`
- v3.1 SHA256：`5fdf5f5b57ca72c969c3684dc2fda795b3727dd8caaff3dbee5c4c1fc4ad5f3e`
- 父稿 v1/v2/v3、protocol-closed、evidence-frozen 稿均未覆盖。

## 六项必改问题

| 问题 | 状态 | v3.1 处理 |
|---|---|---|
| 源数据本地性作用域过宽 | `PASS` | 区分本地分类训练窗口、H1 统计交换路径和服务器端源侧适配参考集；不再声称所有源域窗口始终留在客户端 |
| 运行对象数量逻辑错误 | `PASS` | 明确区分正式选择性输出基线、简化回归核心、独立 QC 候选三个运行对象 |
| Fig. 1 “最终评估”与 test 历史不一致 | `PASS` | 改为模型/超参数/阈值锁定后评价，不参与拟合或选择；同时限定冻结 SVG 中 broad locality label 的含义 |
| “机器精度范围内”过强 | `PASS` | 正文与 Table A3 改为“数值上高度一致”，并保留最大绝对预测差 \(6.2532\times10^{-8}\) ppm |
| 摘要延迟比较功能身份不等价 | `PASS` | 摘要只报告不承载正式选择性输出语义的简化核心 Pi p50=3.725 ms；完整对象比较留在 Table V |
| MAE/NRMSE 定义与展示不一致 | `PASS` | 回归主要报告 RMSE；MAE/NRMSE 明确为已计算的冻结补充诊断指标，不参与模型选择或主结论 |

## 数据流 claim 边界

v3.1 采用以下限定：

1. C1/C2 的本地分类训练窗口不通过联邦训练链路上传。
2. H1 重构接口交换特征矩、正规方程、聚合验证统计，不传输原始 \(X/y\) 行或逐样本预测。
3. 服务器端分类适配仍使用预先配置的 C1/C2 源侧适配参考集。
4. 因此只能主张受限定的数据流分离，不能主张所有源域原始窗口始终留在客户端。
5. 当前系统未实现 secure aggregation、差分隐私或形式化攻击分析。

该修订来自用户确认的系统数据流边界，并与 H1 真实拓扑结果中“统计交换不传输原始 \(X/y\)”的已审计证据共同使用；它没有扩张为形式化隐私 claim。

## 术语锁定

| 概念 | v3.1 中文固定写法 | 英文稿建议 |
|---|---|---|
| 104D feature contract | 统计增强特征（104D） | rich statistical features |
| source regression reference | 源域回归参考 | source regression reference |
| predicted class route | 预测类别路由 | predicted class route |
| three-reference configuration | 三参考源先验个性化 | three-reference source-prior personalization |
| target calibration set | 目标校准集 | target calibration set |
| acceptance coverage | 接受覆盖率 | acceptance coverage |
| runtime benchmark | 运行时基准测试 | runtime benchmark |
| performance-retention rule | 预先设定的 1% 性能保持容差 | prespecified 1% performance-retention tolerance |

`all-prior` 仅在首次定义时作为实验记录标签保留一次，不再用作论文语义方法名。

## 三个预期审稿问题的回答边界

### Q1. 为什么没有最终语义下的严格适配组件消融？

本文定位为真实设备分类—个性化回归—边缘可靠输出的系统链路研究。Appendix A1 只提供不同开发阶段的机制背景，不构成最终配置的严格逐项因果消融。

### Q2. 已有 320 个带标签目标窗口，联邦学习的价值在哪里？

源设备本地分类训练窗口不集中汇集；目标校准集只用于服务器适配、轻量目标 Ridge 与 QC 校准；目标端不重训完整分类模型；源回归信息由 420 参数 H1 压缩并与 424 参数目标 Ridge 组合。校准预算敏感性仍作为明确限制保留。

### Q3. 为什么最终回归核心与正式 QC 来自不同运行对象？

回归依赖简化与选择性输出晋级是两个独立决策。简化核心完成 844 参数回归链路和可移植加载闭环；独立 QC 候选未通过覆盖率与 CO 条件，因此正式选择性输出基线没有被替换。

## 非阻塞版面事项

- 当前正文保留 Table I–V 和 Fig. 1–5。
- Appendix A/B 的证据不删除；IEEE 排版时可将通信明细、完整 harmonization 表和服务器十项配置移至 supplementary。
- 冻结 Fig. 1 SVG 内的 `Raw C1/C2 rows stay local` 标签范围过宽；v3.1 图注已限定其含义，最终英文重绘必须同步替换图内文字。
- 近年文献覆盖应在英文稿前单独审计，但不得在缺少核验时自行补 DOI 或新增 unsupported claim。

最终判定：`READY_TO_FREEZE_CHINESE_CONTENT_FOR_ENGLISH_TRANSLATION`
