# GAPS IoT-J 稿件协议与实验范围锁定（2026-07-26）

- 状态：`NO_FURTHER_EXPERIMENTS_REQUIRED_FOR_CURRENT_SCOPE`
- 正式协议名称：`calibrated-target held-out-window evaluation`
- evidence-frozen 基线：`docs/paper/GAPS_IoTJ_evidence_frozen_20260726.zh.html`
- protocol-closed 稿件：`docs/paper/GAPS_IoTJ_protocol_closed_20260726.zh.html`
- 本次实验运行数：0
- 冻结实验结果修改数：0

## 1. 数据协议

目标设备先从原始 measurement files 构造 100×8 windows，再按 gas class 与
concentration 分层划分 calibration/test。C5 calibration 为 320 windows，test
为 1360 windows。同一个具体 window/sample row 不跨 subset；同一原始文件的
不同 windows 可以分别进入 calibration 与 test。

该协议用于评价目标校准后的 held-out-window performance，不提供原始文件或
measurement session 互斥保证。

## 2. Test-access boundary

Test labels 不用于：

- model fitting；
- hyperparameter selection；
- alpha selection；
- QC threshold selection；
- checkpoint selection。

Evidence freeze 后不再进行 method selection 或 reselection。Filename overlap
反映 grouping dependence 和外推范围，不等同于 test-label leakage。

## 3. Legacy ablation evidence

论文简化表保留 A0、A0T、A5、A6、B1–B5 和 final B5。A 系列标记 historical
mechanism semantics，B1–B5 标记 corrected single-seed screening。旧 A7 不进入
表格，也不与 final B5 比较。该表只提供历史方法上下文，不构成 canonical final
B5 的严格逐组件消融。

## 4. Limitation

> The target-device evaluation uses a class- and concentration-stratified
> window-level split. Different windows from the same measurement run may
> occur in both calibration and test. The reported results therefore
> characterize post-calibration held-out-window performance rather than
> generalization to entirely unseen measurement runs.

摘要不展开 filename overlap 数字。

## 5. 取消的投稿前实验

以下项目不再属于 active experiment plan 或 manuscript TODO：

- original-file-level retraining；
- FedProx、FedAdam、SCAFFOLD；
- multi-target reruns；
- new QC；
- new regression heads；
- component ablation reruns。

历史计划和审计记录继续作为 provenance 保留，但不得解释为仍待执行的投稿阻塞项。

## 6. Future Work

仅保留：

- unseen-session 或 original-file-independent evaluation；
- additional physical target devices；
- stronger privacy protection for sufficient statistics。

## 7. 后续工作

后续只进行中文叙事修订、英文翻译、图重绘、表格压缩、参考文献核验、
IEEE LaTeX 转换和导师审阅。不得重新打开实验或 test。
