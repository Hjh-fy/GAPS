# Canonical-v1 Q1-v2 结果状态

状态：`completed / audit_pass / authoritative`。

最终决策：`CONFIDENCE_QC_FINAL`。

正式结果位于：
`results/iotj_canonical_v1_final/canonical_q1_conformal_qc_v2_20260812/`。

| Scope | Confidence NRMSE-AURC | Equal-mean CDF NRMSE-AURC | 相对改善 |
|---|---:|---:|---:|
| C5 | 0.07935383 | 0.09154944 | -15.3686% |
| C3/C4/C5 pooled | 0.05681674 | 0.06430592 | -13.1813% |

C5 的固定 90% 名义 empirical prediction interval 实际覆盖率为
92.1324%，但 interval-width 与 confidence 的固定 equal-mean 组合没有改善
选择性输出的风险排序。预注册的 C5 与 pooled 双 5% 门槛均失败，因此保留
confidence-only QC，不调整权重、阈值或方法。

Q1-v1 因 scalar-only interval helper 无法处理合法的逐样本 routed radius 而在
正式评估中止，只留下 partial root；它是 `invalid/superseded`，不得引用。Q1-v2
是唯一 authoritative Q1 证据。正式 v2 结果及 SHA256 索引未被本次闭环修改。

本阶段至此停止所有实验，不再启动新的 QC、回归或分类方法。
