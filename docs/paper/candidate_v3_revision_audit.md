# GAPS IoT-J 中文投稿候选稿 v3 修订审计

## 结论

状态：`READY_FOR_HUMAN_REVIEW_NOT_YET_FROZEN_FOR_ENGLISH`

v3 在 v2 的章节结构和证据链上完成定向学术化收口，没有重写论文结构，也没有改变正式数字、模型选择、QC 决策或 claim–evidence 边界。该版本可进入导师/作者中文内容审阅，但在人工确认术语、篇幅和文献呈现前，不应冻结为英文翻译唯一源稿。

## 版本边界

- v2 源稿：`docs/paper/GAPS_IoTJ_submission_candidate_v2.zh.html`
- v2 SHA256：`57eef73898122c73eb1c5ec4727c3f578d34f5ee2e8b09f627d16675c3bb9860`
- v3 候选稿：`docs/paper/GAPS_IoTJ_submission_candidate_v3.zh.html`
- v2 保持未修改；v1、protocol-closed、evidence-frozen 稿均未覆盖。
- 本轮未运行训练、推理、评估或 benchmark，也未打开正式 test 资产。

## 八项定向修订

| 审阅意见 | 状态 | v3 处理 |
|---|---|---|
| 统一 source validation 身份 | `PASS` | 全文改为“源侧适配参考集”或“源侧适配/参考窗口”，不再把该 320 行子集称为 validation/calibration |
| 修正 test 访问历史 | `PASS` | 明确冻结的 1360 行 test universe 曾用于多个已完成阶段；每一阶段均在模型、超参数或阈值锁定后评价，test labels 不参与 fitting、selection 或 reselection |
| 定义 all-prior 的 H1/H2/H3 | `PASS` | H1=104D 分气体 Ridge；H2=104D 分气体 16-unit ReLU MLP；H3=104D+4D predicted-route one-hot 的共享 16-unit ReLU MLP；三项预测形成 107D target Ridge 输入 |
| 压缩服务器适配主文 | `PASS` | 主文按监督保持、分布匹配、语义约束和正则化四组描述；十项非零目标、权重、零权重与可用性条件移至 Appendix B |
| 降低贡献 2 的理论新颖性语气 | `PASS` | 使用“构建并验证……源参考机制”，不把标准 Ridge 充分统计量重构本身表述为新理论 |
| 工程状态自然语言化 | `PASS` | 主表与附录用“正式选择性输出基线”“选定轻量回归核心”“已评价，未采用”等论文语言替代内部状态 token |
| 正式五表编号 | `PASS` | Table I 数据/协议；Table II 104D 特征；Table III 分类；Table IV 回归；Table V(a)/(b) 运行身份、规模与延迟 |
| 摘要和讨论局部降噪 | `PASS` | 摘要保留一个分类稳定性指标、H1 回归、28,737→844、Pi p50 与 held-out-window 边界；将“分类只需”等绝对句式限定为当前数据和模型下的机制解释 |

## 方法—实现对齐

### All-prior source references

- H1：四个分气体 Ridge 头，输入 104D rich features；使用每气体冻结的 alpha 和源侧标准化统计。
- H2：四个分气体 `MLPRegressor` 头，输入 104D；单隐藏层 16、ReLU、LBFGS、`max_iter=800`；冻结资产中每气体 alpha 为 0.01。
- H3：一个共享 `MLPRegressor`，输入 104D rich features 与 4D predicted-route one-hot；单隐藏层 16、ReLU、LBFGS、`max_iter=800`；冻结 alpha 为 0.1。
- All-prior 将三项 source prediction 与 104D 特征拼接为 107D target Ridge 输入。五路回归比较中这些 source heads 固定复用，不进行五次独立重训。
- 冻结策略来源：`results/iotj_b5_c5_deployment_p1_20260722/h8_no_rescue/r4_policy.json`，SHA256 `18b6c14373018474807eec2bd19a0b508b75adfbf994b0821a786a11def9c263`。
- 训练语义与序列化映射分别核对 `run_source_lightweight_regression_head_ablation.py`、`run_source_augmented_target_ridge_eval.py` 和 `gaps_deploy/c5_h8_runtime.py`；v3 没有调用或修改这些实现。

### 服务器适配

- 主文四组只是表达层分组，不改变冻结目标或权重。
- Appendix B 保留源侧 CE、类条件 CORAL、全局/类条件/类阶段 MMD²、Wasserstein-min、原型锚定、原型拟合、一致性和残差项。
- 目标监督 CE 与 prototype-pair MMD 的权重为 0，legacy direct-alignment 关闭。
- 该复合目标用于描述最终冻结分类实现，不把十个子项分别列为独立贡献。

## Claim-strength 审计

- 分类稳定性限定在 C1/C2→C5、当前数据与超参数协议。
- H1 相对 rich-only 的改善按五条冻结路由的描述性均值报告；未使用“统计显著”。
- All-prior 在 5/5 路由上的绝对 \(S_{\mathrm{CC}}\) 更低仍被保留；H1 的最终选择只依据预先设定的 1% 性能保持容差和依赖简化。
- 文件级相关性与 test-label leakage 保持区分；评价身份仍为 `calibrated-target held-out-window evaluation`。
- 源原始行保持本地不扩张为 secure aggregation、差分隐私或形式化隐私保证。
- 正式选择性输出基线与 844 参数的简化可移植核心保持为不同运行对象。
- Pi 延迟结论限定于已冻结平台与 benchmark 设置；PC p95 未改善的反例仍保留。

## 结构与版面

- 正文逻辑表共五张，其中 Table V 分为功能/规模和延迟两个子表，建议 IEEE 双栏排版时合并为一个 `table*`。
- 正文图保持 Fig. 1–Fig. 5，未重绘、未替换数据。
- 支撑性细节保留在 Appendix A；服务器适配完整式移至 Appendix B。
- 19 条参考文献保持不变；本轮未添加未经核验的文献或 DOI。

## 人工审阅前的非阻塞项

1. 决定中英文混排术语（如 rich features、source reference、accept/review/reject）在最终中文稿中的统一程度。
2. 检查 Appendix B 的篇幅是否放入论文附录或 supplementary material；无论位置如何，不能改变 active/disabled 项身份。
3. 按目标页数检查 Table V 的双栏宽度和 Appendix A 的压缩方式。
4. 在冻结英文源稿前完成 IEEEtran/BibTeX 的最终元数据与引用格式核验。

最终判定：`READY_FOR_HUMAN_REVIEW_NOT_YET_FROZEN_FOR_ENGLISH`
