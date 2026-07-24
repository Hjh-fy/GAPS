# IoT-J B5 seed44 正式完成报告

- run：`c12_to_c5__b5__s44`
- attempt：`c12_to_c5__b5__s44__a001`
- 状态：`canonical / validator_accepted`
- 训练：25/25 rounds，C1/C2 每轮参与，0 fit/eval failure
- Server DA：100 steps/round，共 2500 steps
- attempt wall：6303.639 s（1.751 h）
- round-25 adapted checkpoint SHA256：`cc1b07da93a45165f3acb47b86ed98caf263f7f0c8625ceb1ecca751067553f3`

## 冻结 C5 test 分类评估

- N：1360，唯一 row key：1360
- Accuracy：0.994852941176
- Macro-F1：0.994851695672
- NLL：0.054088350133
- ECE：0.005050430666
- Error count：7

## 审计结论

POSTFLIGHT_PASS。checkpoint 已由严格 round-25 加载路径完成推理验证；训练、DA、拓扑、seed、row key、有限数值与冻结 runtime/HC95/HC90 检查全部通过。C5 test 未参与训练、停止或 checkpoint 选择。

本结果只构成 B5 classification multi-seed 的单 seed 证据，不形成回归、QC、runtime v5 或 Pi benchmark 结论。
