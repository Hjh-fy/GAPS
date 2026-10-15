# IoT-J B5 seed42 metric lineage reconciliation

## 唯一 canonical 结论

当前 five-seed 与后续 regression multi-seed 必须使用：

- Accuracy：`0.9801470588235294`（1333/1360）；
- checkpoint：
  `results/iotj_ecs_c2_representative_20260720/raw/c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/training/server_round_025_adapted.pth`；
- checkpoint SHA256：
  `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`；
- training commit：
  `2ef7aea77b9dfabdd09da4f38742907a37c58c30`；
- classification result commit：
  `9ca4a70e5e17c600b058b1f156d35935052496e5`；
- dataset：
  `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`；
- C5 frozen test：1360 rows；
- route：
  `results/iotj_b5_multiseed_20260724/seed42_reference/classification_evaluation/seed42_test_predictions.csv`；
- row-map contract SHA256：
  `54a42bb9f622c441a889a36fb1e585cb437e04c11128eb0578cfef6fd7711c3c`。

历史 `0.9889705882352942` 不属于上述 checkpoint，不得进入 five-seed
或 regression multi-seed。

## 两条 lineage

| 字段 | 历史 screening | 当前 canonical |
|---|---|---|
| Accuracy | 0.9889705882352942（1345/1360） | 0.9801470588235294（1333/1360） |
| 角色 | v3 B1–B5 单 seed screening | 正式三机 B5 seed42 canonical |
| checkpoint SHA256 | `d497bba27f1a217b83ab1ff212fc5d0ab13fea17b6efc7ab7fd8ba308d73445f` | `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c` |
| checkpoint path | `/root/GAPS/results/iotj_reg_checkpoints_20260713/B5.pth` | 当前 canonical path，见上 |
| 数据/测试范围 | frozen C5 N=1360 | 同一正式 C5 N=1360 |
| evaluation | `summarize_iotj_classification_ablation` 的 v3 screening stream | five-seed frozen evaluator + v4 row map |
| source/result commit | `62f77bad` 记录 screening evidence | training `2ef7aea`；result `9ca4a70` |
| 可用于当前 regression | 否 | 是 |

## 差异判定

差异的首要且充分解释是**不同 checkpoint identity**，不是把同一 checkpoint
算出了两个 Accuracy。仓库已有
`results/iotj_b5_c5_deployment_p1_20260721/classifier_identity_conflict.json`
记录：

- historical formal regression classifier：`d497bba2…`；
- current canonical classifier：`9b268f65…`；
- 旧资产因 classifier identity 不同且缺少完整序列化资产，不得成为当前部署
  bundle。

未发现把 B2 误标为 B5 的证据：v3 summary 中 B2 为
`0.9926470588`，B5 为 `0.9889705882`。`0.9889705882` 确为旧 B5，
但它仍然不是当前 canonical B5。

## Route closure

five-seed 的 seed42 route：

- 由 `9b268f65…` round-25 adapted checkpoint 严格加载产生；
- test N=1360；
- 1360 个 row key 全部唯一；
- predicted route 范围为 0–3；
- 绑定 v4 `runtime_contract.json` 与 `row_map_1360.json`；
- 与 runtime v4/HC95/HC90 使用的 classifier identity 一致。

因此后续 RG0/RG1/RG2 必须使用该 route。不得使用 v3 screening prediction
stream、`d497bba2…` checkpoint，或把历史 `0.9889705882` 写入当前
five-seed 表。

## Audit verdict

`PASS_FOR_REGRESSION_PROTOCOL_FREEZE`

不存在未解释的 metric conflict。Evidence boundary：本报告只解决 classifier
metric/checkpoint/route lineage，不预先支持任何 regression variant 结论。
