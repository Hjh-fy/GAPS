# B5 final multi-seed seed43 canary 实验记录

## 身份与状态

- experiment ID：`MS-B5-SEED43-FORMAL`
- run / attempt：`c12_to_c5__b5__s43` / `c12_to_c5__b5__s43__a001`
- M0 control commit：`d3ac8b90c87d50a7a0be1fd985d883db15456703`
- training commit：`2ef7aea77b9dfabdd09da4f38742907a37c58c30`
- topology：Alibaba Cloud ECS server/DA + Pi C1 + ECS C2
- status：`canonical / validator_accepted`
- audit SHA256：`41c30d0bf4e494cd08e560b7e2d87fd8eb711a96062edfa4721aa89dc6d6168e`

## 已完成检查

- live three-host preflight：PASS；
- 25/25 rounds、C1/C2 每轮参与、DA 100 steps/round：PASS；
- final 与 round-25 adapted checkpoints：PASS；
- NaN/Inf、silent skip、seed/run identity：PASS；
- 训练后内容寻址代码/数据复检：PASS；
- runtime v4、HC95、HC90 六个冻结文件：SHA256 unchanged；
- C5 test prediction stream：1360 行、1360 unique row keys；
- postflight：PASS。

## 分类结果

seed43 C5 test：Accuracy `0.9860294118`、Macro-F1 `0.9860150907`、NLL `0.1256369339`、ECE `0.0133756879`。这些值是正式 multi-seed 的一个样本，不用于 classifier 重选。

## Evidence boundary

本记录只证明 seed43 分类训练在冻结正式三机拓扑上 canonical 完成，并提供单 seed 泛化指标。它不构成五种子稳定性结论，不包含 RG0/RG1/RG2，不支持 H1 vs all-prior gate，也不改变 runtime v4/QC。

正式明细见：

- `results/iotj_b5_multiseed_20260724/seed43/seed43_completion_report.zh.md`
- `results/iotj_b5_multiseed_20260724/seed43/seed43_postflight.json`
- `results/iotj_b5_multiseed_20260724/seed43/classification_evaluation/seed43_classification_metrics.json`
