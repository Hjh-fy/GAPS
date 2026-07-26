# GAPS IoT-J 投稿候选稿 v1 Claim 审计

审计对象：`docs/paper/GAPS_IoTJ_submission_candidate_v1.zh.html`

主证据源：`docs/paper_evidence_freeze/claim_evidence_matrix.json`
规则：claim 强度不得高于冻结 evidence；所有条件、scope、comparison 和 limitation 必须随 claim 保留。

## RQ 与贡献身份

- RQ1：real-device federated classification with calibration-assisted server adaptation。
- RQ2：sufficient-statistics Federated H1 and 105D target personalization。
- RQ3：edge efficiency, selective-output trade-off and calibration boundary。
- 三项核心贡献分别对应真实设备分类、充分统计量 H1 + 目标个性化、系统级效果/效率/可靠性评估。
- Selective aggregation、portable release、QC2 均未作为独立核心算法贡献。

## Approved claims

| ID | Candidate claim | Strength | Condition/scope | Frozen evidence | Required limitation | Audit |
|---|---|---|---|---|---|---|
| SUB-C01 | B5 在 seeds 42–46 下取得 `0.989118 ± 0.005983` accuracy。 | Direct | C1/C2→C5；每 seed 1360 windows | approved C1；Table II | 只支持当前设备角色与五个种子，不外推任意设备/种子 | PASS |
| SUB-C02 | 真实拓扑分类由 C1、C2 与 Alibaba ECS server 完成，目标 labels 用于 calibration-assisted server adaptation。 | Direct/qualified | 25 rounds；5 local epochs；100 server DA steps/round | frozen B5 run manifests | 不称 zero-shot 或 UDA | PASS |
| SUB-C03 | Federated H1 通过 sufficient statistics 达到 pooled H1 的 practical equivalence。 | Direct | 4 gases；registered real topology | approved C2；Table A3 | `PRACTICAL_EQUIVALENCE` 不是数学恒等或隐私证明 | PASS |
| SUB-C04 | H1 交换过程中 raw source rows remain local。 | Direct/qualified | moments、normal equations、clipped SSE/count exchange | approved C3；Table A4 | 没有 secure aggregation 或 DP，不主张形式化隐私 | PASS |
| SUB-C05 | Federated H1 相对 all-prior 的 S_CC 退化为 `0.981637%`，通过预注册 1% 简化非劣门槛。 | Direct | five frozen B5 routes；paired comparison | approved C4；Table III/Fig. 3 | all-prior 在 5/5 routes 上绝对 S_CC 更低；选择是简化非劣，不是精度优越 | PASS |
| SUB-C06 | 105D target personalization 由 104D rich features 与 1D H1 prediction 构成。 | Direct | per-gas target Ridge；C5 calibration-only selection | frozen v5 contract/ledger | 不声称 target Ridge 经网络 FedAvg 训练 | PASS |
| SUB-C07 | 回归参数由 `28,737` 减至 `844`，其中 H1 为 420、target Ridge 为 424；classifier 22,765 单列。 | Direct | regression-core parameter identity | approved C5；Table IV | 不写成百分比，不将 classifier 混入 844 | PASS |
| SUB-C08 | v5 core 的 Pi p50 低于 v4，但性能改善不覆盖所有平台和 percentile。 | Direct/qualified | frozen PC/Pi benchmark objects | approved C5；Table IV | PC p95 由 13.945 ms 变为 14.786 ms；不作普遍加速 claim | PASS |
| SUB-C09 | v5 QC2 在较低 accepted yield 下获得较低 accepted RMSE，但 promotion guards 未满足，因此 v4 保留 formal baseline。 | Direct | HC95/HC90 frozen workpoints | approved C6/C7；Fig. 4/Table A5 | v4 formal QC 与 v5 core 是不同 runtime roles；HC90 CO guard 失败 | PASS |
| SUB-C10 | calibration budget 从 320 减至 160 windows 时，group-aware mean S_CC 从 10.8724 增至 23.9156 ppm。 | Direct/qualified | post-freeze descriptive group-aware protocol | approved C8/C9；Fig. 5/Table A6 | 不替代 historical 11.3416；不作 confirmatory method selection | PASS |
| SUB-C11 | 正式评价是 calibrated-target held-out-window evaluation。 | Direct | 320 calibration / 1360 test；gas/concentration stratified | approved protocol claim | 同一 file 的不同 windows 可跨 subset；不代表 original-file/session independence | PASS |
| SUB-C12 | test labels 未用于 fitting、hyperparameter/alpha/QC-threshold/checkpoint selection 或 evidence-freeze 后的方法选择。 | Direct/qualified | frozen test-access boundary | protocol closeout evidence | filename overlap 与 test-label leakage 分开陈述 | PASS |
| SUB-C13 | 五种子回归评价改变 B5 routes，而不是全部回归 heads 的五次独立训练。 | Direct | frozen source heads and registered target protocol | regression multi-seed manifest/ledger | 不写成 end-to-end regression five-seed retraining | PASS |
| SUB-C14 | portable v5 release 完成 clean-checkout core load/synthetic smoke。 | Qualified engineering claim | synthetic input only；formal test not accessed | clean-checkout receipt at frozen system commit | 不是性能证据，不等于 full v4/v5 QC reproduction | PASS |

## Claim boundaries scan

- `calibrated-target held-out-window evaluation`：已在摘要、数据协议、实验协议和局限中保留。
- `different windows from the same file may cross subsets`：已明确。
- `not original-file/session independent`：已明确。
- `calibration-assisted, not zero-shot/UDA`：已明确。
- `raw source rows remain local` 与 `no secure aggregation/DP`：成对出现。
- `H1 simplification-noninferiority`：已保留 all-prior 绝对更低的反向证据。
- `five-seed regression varies B5 routes`：已在实验协议与局限中限定。
- `v5 core ≠ v4 formal QC`：已在方法、Table IV 与局限中限定。
- `portable release = engineering evidence`：未用于性能或算法优越性 claim。

## Prohibited/overstated wording result

- 未使用 “97.1% model reduction”。
- 未使用 “显著简化” 作为无统计检验结论。
- 未将 v4 称为 “更强 baseline”；改为 “promotion guards 未满足，因此保留 formal baseline”。
- 未宣称所有平台或所有 latency percentile 均改善。
- 未将 selective aggregation、QC2 或 portable release 提升为主要创新。
- 未宣称 secure aggregation、differential privacy、zero-shot target transfer 或 original-file-independent evaluation。

## Final audit result

`PASS_WITH_DECLARED_LIMITATIONS`

候选稿的主要 claims 均可追溯到 approved frozen evidence；没有需要新训练、重新打开 test 或补充实验才能成立的当前 scope claim。
