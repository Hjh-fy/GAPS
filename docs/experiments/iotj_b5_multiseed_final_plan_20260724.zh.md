# IoT-J B5 five种子最终确认计划（M0–M5）

## Research brief and scope

- 最终 classifier 已固定为 B5，不再进行 B2/B5 选择。
- 种子集合固定为 42、43、44、45、46。
- seed42 复用正式 checkpoint；seed43–46 在与 seed42 相同的真实 Flower
  拓扑和算法协议下新训练。
- 本阶段最终回答 B5 分类稳定性，以及 `Federated H1` 与 `All-prior`
  在五个 classifier seeds 下的回归稳定性。
- 不创建 runtime v5，不修改 runtime v4/QC，不运行 Pi benchmark 或
  low-calibration。

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Acceptance criterion |
|---|---|---|---|---|---|
| MS-B5-CLS | B5 在五个独立训练种子上保持可接受的分类稳定性 | seed42 | seeds43–46 | Accuracy、Macro-F1、NLL、ECE | 报告 mean、sample SD、min、max；不进行 classifier 重新选择 |
| MS-RG-H1 | sufficient-statistics H1 在不同 B5 route 下可保持 All-prior 的 S_CC 性能 | RG2_ALL_PRIOR | RG1_FEDERATED_H1 | 五种子 paired S_CC RMSE | `(mean_H1-mean_ALL)/mean_ALL <= 1%` 且通过辅助退化门 |
| MS-COMPONENT-FREEZE | source H1/H2/H3 与 classifier training seed 解耦 | 每 seed 重训 source heads | 固定复用 source heads | source asset hash | 所有 seed 使用完全相同的 source-head hash |

## STAGE M0 — freeze seed42 and prepare 43–46

seed42 正式基线：

- run/attempt：`c12_to_c5__b5__s42 / c12_to_c5__b5__s42__a001`
- checkpoint：
  `results/iotj_ecs_c2_representative_20260720/raw/c12_to_c5__b5__s42/c12_to_c5__b5__s42__a001/raw/ecs/training/server_round_025_adapted.pth`
- checkpoint SHA256：
  `9b268f659c60a1d3b9bb789d89e82b5cedae56b92173daca616caef247371e5c`
- algorithm/training commit：
  `2ef7aea77b9dfabdd09da4f38742907a37c58c30`
- frozen source archive SHA256：
  `52bdbf96568014cc363f0ce3c666026be29f5f0279c7a130b41458d42a0d0c68`
- dataset：
  `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`
- dataset manifest SHA256：
  `fb8946da138bea5aa829dd1f5b733561a443083beb77a873e7173cbc95fcd430`
- source/target：C1/C2 → C5；split seed 固定为 42；
- 25 rounds；每轮每客户端 5 local epochs；batch size 32；
- client optimizer：PyTorch Adam，LR `5e-4`，默认 betas/eps、
  weight decay 0；每轮重新创建 optimizer；该路径无 scheduler；
- B5 profile：`proto_replay`，prototype alignment、replay distillation、
  semantic/device residual decoupling、selective aggregation 启用；
- selective aggregation：warmup 3，minimum scale 0.3；
- server semantic prototype EMA alpha：0.8；
- target supervised CE 关闭；
- server DA：每轮 100 steps、warmup 0、CPU、strict calibration split、
  adapted checkpoint 作为下一轮 global。

真实执行拓扑固定为：

`Alibaba ECS server/DA -> reverse tunnel -> Pi C1 + ECS C2`

不得使用 command manifest 中历史保留的 `client_c2_pc` 标签推断实际拓扑。
正式 seed42 的 execution topology manifest 与 controller launch manifest
证明 C2 实际位于 `root@114.55.171.63` 的 ECS-C2。

seeds43–46 的冻结 command manifests 与 seed42 的数值算法配置一致。允许变化
仅为 training seed，以及由 seed 必然派生的 run ID、run name、output path 和
algorithm-config hash。`execution_stage=v3_confirmation` 相对 seed42 的
`v3_correction_screening` 只是 provenance 标签，不进入训练命令或数值算法。

M0 gate：

1. 三台主机 preflight 全部通过；
2. source archive、dataset、topology、command manifest hash 匹配；
3. 没有残留 Flower、tunnel 或 resource sampler；
4. 目标 seed 输出目录不存在或为空；
5. 只有上述 gate 通过才允许启动一个 seed。

## STAGE M1 — train seeds43–46

四个 seed 必须顺序运行，禁止并行占用同一 server/端口/设备。每个 seed 均保存：

- controller stdout/stderr；
- 三端原始 events/resource/training logs；
- 25 个 adapted checkpoints；
- attempt audit/status/provenance；
- 最终 checkpoint SHA256；
- 实际 wall time 和 validator 状态。

任一 seed 失败时保留失败证据并停止队列，不得覆盖或跳过后继续合并统计。
seed42 不重训。

完整命令已分别冻结在：

- `results/iotj_b5_multiseed_20260724/commands/launch_seed43.cmd`
- `results/iotj_b5_multiseed_20260724/commands/launch_seed44.cmd`
- `results/iotj_b5_multiseed_20260724/commands/launch_seed45.cmd`
- `results/iotj_b5_multiseed_20260724/commands/launch_seed46.cmd`

seed42 实测 25-round training wall 为 `5932.34 s = 1.65 h`。预算按每 seed
约 2 h（训练、preflight、回收、validator），四个串行约 8 h；考虑网络重连和
证据回收，建议预留 10 h。controller fail-safe timeout 仍为 48 h，它不是预计
耗时。

## STAGE M2 — B5 classification stability

只使用每 seed 的最终 adapted checkpoint，在同一 frozen C5 1360-row test、
canonical logits route 下计算：

- Accuracy；
- Macro-F1（由四类 confusion matrix 计算）；
- NLL；
- top-label ECE，固定 15 个等宽 confidence bins。

输出：

- `per_seed_b5_classification_metrics.csv`
- `b5_classification_multiseed_summary.csv`

汇总对五个 seed 报告 mean、sample SD（`ddof=1`）、min、max。test 只评估，
不用于 checkpoint、seed 或 classifier 选择。

## STAGE M3 — regression multi-seed

每个 seed 使用自身 frozen B5 route。固定比较：

- `RG0_RICH_ONLY`：104D rich → C5 per-gas Ridge；
- `RG1_FEDERATED_H1`：104D rich + fixed sufficient-statistics H1
  → C5 per-gas Ridge；
- `RG2_ALL_PRIOR`：104D rich + fixed pooled H1/H2/H3
  → C5 per-gas Ridge。

source components 冻结规则：

- RG1 H1 固定复用
  `results/iotj_h1_federated_ridge_equivalence_20260724/federated_h1_manifest.json`
  所绑定的模型，SHA256
  `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc`；
- RG2 H1/H2/H3 固定复用 runtime-v4 R4 source heads；R4 policy SHA256
  `18b6c14373018474807eec2bd19a0b508b75adfbf994b0821a786a11def9c263`；
- H1/H2/H3 source fit 都只读取固定 C1/C2 train/calibration，不读取
  classifier checkpoint 或 classifier seed，因此不重复训练五次；
- 随 classifier seed 变化的是 C5 calibration/test 的 B5 route；
- C5 target Ridge 的 calibration-validation alpha 选择必须按各 seed route
  独立执行，随后使用该 seed 的完整 calibration refit。test 不参与选择。

主指标为 S_CC RMSE；同时输出 S_ALL RMSE/MAE/NRMSE、per-gas RMSE、CO 和
CO-high RMSE。RG0/RG1/RG2 在同一 seed 内严格共享 route 和 S_CC subset。

## STAGE M4 — final regression gate

首先计算五个 seed 的 paired RG1–RG2 结果。选择
`SELECT_B5_FEDERATED_H1` 必须同时满足：

1. `(mean_S_CC_H1 - mean_S_CC_ALL) / mean_S_CC_ALL <= 0.01`；
2. RG1 S_ALL mean 不高于 RG2 的 `1.01x`；
3. RG1 CO mean 和 CO-high mean 均不高于 RG2 的 `1.01x`；
4. 对任一气体，若 RG1 相对 RG2 退化超过 5% 的 seed 数达到 3/5，则失败；
5. CO 或 CO-high 若相对退化超过 5% 的 seed 数达到 3/5，则失败。

任何一项失败则选择 `SELECT_B5_ALL_PRIOR`。不得用 test 排名改换 classifier
或剔除 seed。

## STAGE M5 — stop

完成 M4 后停止，只报告：

1. seed42 正式配置；
2. seeds43–46 完成状态；
3. 每 seed 分类指标与五种子 mean±SD；
4. 每 seed RG0/RG1/RG2 S_CC；
5. RG1 vs RG2 paired 结果；
6. regression recommendation；
7. 是否值得另行设计 runtime v5。

本计划不授权创建 runtime v5、修改 runtime v4/QC、Pi benchmark 或
low-calibration。

## Risks, unknowns, conflicts, and stopping rules

- 训练启动前的三机在线状态目前为 `unknown`，只能由每 seed preflight 解决；
- seed42 是单次 canonical 系统运行，耗时估计不是 SLA；
- sufficient-statistics H1 的既有结论是 `PRACTICAL_EQUIVALENCE`，不是
  coefficient-level exact equivalence；
- 任一 checkpoint 缺失、hash 不匹配、validator rejected、数据/topology
  drift 或非 seed 算法字段变化，立即停止。
