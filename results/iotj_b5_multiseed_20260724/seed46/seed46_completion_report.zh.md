# IoT-J B5 seed46 正式完成报告

- run：`c12_to_c5__b5__s46`
- attempt：`c12_to_c5__b5__s46__a001`
- 状态：`canonical / validator_accepted`
- 训练：25/25 rounds，C1/C2 每轮参与，0 fit/eval failure
- Server DA：100 steps/round，共 2500 steps
- attempt wall：6110.450 s（1.697 h）
- round-25 adapted checkpoint SHA256：`26bcc33066a10268ce21ac7011ba636982c9073a645658e96c5d454b69608913`

## 冻结 C5 test 分类评估

- N：1360，唯一 row key：1360
- Accuracy：0.992647058824
- Macro-F1：0.992661386988
- NLL：0.088364180024
- ECE：0.007307823967
- Error count：10

## 审计结论

POSTFLIGHT_PASS。checkpoint 已由严格 round-25 加载路径完成推理验证；训练、DA、拓扑、seed、row key、有限数值与冻结 runtime/HC95/HC90 检查全部通过。C5 test 未参与训练、停止或 checkpoint 选择。

本结果只构成 B5 classification multi-seed 的单 seed 证据，不形成回归、QC、runtime v5 或 Pi benchmark 结论。
