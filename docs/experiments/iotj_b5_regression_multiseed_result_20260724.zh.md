# IoT-J B5 回归五种子最终确认（2026-07-24）

## 1. 正式结论

本次正式运行绑定 commit
`99cd23e8b4a5f2f103170f1d8a110d6d85febd5e`，运行时 local HEAD 与
origin HEAD 一致。最终协议判定为：

`SELECT_B5_FEDERATED_H1`

下一步建议为：

`BUILD_RUNTIME_V5_CANDIDATE`

这只是构建候选 runtime v5 的建议。本阶段没有创建或修改 runtime v5，
没有修改 runtime v4 或 QC，也没有启动 Pi benchmark、low-calibration
或其他训练。

## 2. Evidence boundary 与选择顺序

- classifier 固定为五个已冻结 B5 checkpoint（seeds 42–46），每个 seed
  使用自己的预测 route；
- H1 固定为经 `PRACTICAL_EQUIVALENCE` 审计的 sufficient-statistics
  federated per-gas Ridge；
- H2/H3 固定为 R4 中的 pooled-source per-gas MLP 与 shared MLP；
- source heads 不随 classifier seed 重训；
- 每个 seed、gas、variant 的 Ridge alpha 只使用 C5 calibration 内部
  240 fit / 80 validation 选择，随后在完整 320 calibration 上 refit；
- 五个 calibration selection CSV 全部写入后，先持久化
  `calibration_selection_lock.json`，当时
  `test_opened_after_lock=false`；之后才一次性打开 1360 行 test；
- test 不参与 alpha fit/select/refit，但按预注册 gate 用于最终五种子
  回归配置确认。

seed42 的 `0.980147` 与历史约 `0.988971` 已完成谱系对账：两者来自不同
SHA256 的 B5 checkpoint。本轮使用五种子协议绑定的 canonical seed42
checkpoint `9b268f659c60...`，不存在指标混用。

## 3. 三个回归配置

| Variant | 目标输入 | 目标 head |
|---|---:|---|
| RG0_RICH_ONLY | 104D rich | C5 per-gas Ridge |
| RG1_FEDERATED_H1 | 104D rich + federated H1 | C5 per-gas Ridge |
| RG2_ALL_PRIOR | 104D rich + H1/H2/H3 | C5 per-gas Ridge |

## 4. 每个 seed 的主指标 S_CC RMSE

S_CC 仅包含该 seed 下 B5 分类正确的 test 行；三个 variant 在同一 seed
内严格共享 route 和 S_CC mask。

| seed | S_CC N | RG0 | RG1 | RG2 | RG1−RG2 |
|---:|---:|---:|---:|---:|---:|
| 42 | 1333 | 14.201930 | 11.341599 | 11.148364 | +0.193235 |
| 43 | 1341 | 14.755509 | 11.991560 | 11.827744 | +0.163816 |
| 44 | 1353 | 14.275528 | 11.416515 | 11.379096 | +0.037419 |
| 45 | 1349 | 14.719783 | 11.957551 | 11.871197 | +0.086354 |
| 46 | 1350 | 14.320405 | 11.462067 | 11.377430 | +0.084637 |

五种子汇总：

| Variant | S_CC RMSE mean ± sample std | min–max | S_ALL RMSE mean ± sample std |
|---|---:|---:|---:|
| RG0 | 14.454631 ± 0.262100 | 14.201930–14.755509 | 20.466785 ± 3.457738 |
| RG1 | 11.633858 ± 0.314203 | 11.341599–11.991560 | 18.508025 ± 4.321091 |
| RG2 | 11.520766 ± 0.314776 | 11.148364–11.871197 | 18.598173 ± 4.419164 |

RG1−RG2 的 paired mean 为 `+0.113092 ppm`，sample std 为
`0.063731 ppm`，描述性 t 95% CI 为 `[+0.033959, +0.192225] ppm`。
RG2 在 5/5 seed 的 S_CC RMSE 都更低。因此这里不能表述为“RG1 精度
优于 RG2”；正式判定依据是 RG1 相对 RG2 的 mean 退化
`0.981637%`，仍在预注册的 `≤1%` 非劣门槛内。

## 5. Guard 结果

| Guard | 观测值 | 阈值 | 结果 |
|---|---:|---:|---|
| mean S_CC：RG1 相对 RG2 | +0.981637% | ≤1% | PASS |
| mean S_ALL：RG1−RG2 | −0.090148 ppm | ≤+0.5 ppm | PASS |
| mean CO 相对退化 | +0.152960% | ≤5% | PASS |
| mean CO-high 相对退化 | −0.006409% | ≤5% | PASS |
| 任一 gas >10% 退化达到多数 seed | 0/4 gases | 禁止 ≥3/5 | PASS |
| COMMON_CORRECT 相对退化 | +0.991758% | ≤1% | PASS |

COMMON_CORRECT 是五个 classifier-correct 集合的交集，共 1314 个唯一行：

- RG0 RMSE：14.209791；
- RG1 RMSE：11.309719；
- RG2 RMSE：11.198655。

分气体 mean RMSE（RG1 / RG2）：

- Ethanol：17.240755 / 17.766071，RG1 更优；
- CO：18.425012 / 18.396872，RG2 略优；
- Ethylene：22.506446 / 22.172270，RG2 更优；
- Methane：13.419255 / 13.785120，RG1 更优；
- CO-high：24.149245 / 24.150793，RG1 略优。

没有任何气体出现 RG1 相对 RG2 退化超过 10% 且持续至少 3/5 seed。

## 6. 审计结果与异常

- 五个 checkpoint SHA256 均与 classification five-seed manifest 一致；
- 每个 test route 均为 1360 行、sample index 唯一且覆盖完整 universe；
- 正式重放与每个 seed 已冻结 route CSV 逐行一致；
- 三个 variant 在每个 seed 内使用完全相同 route、真实标签和 mask；
- federated H1 manifest SHA256 为
  `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc`；
- 所有 target 输出均为 finite，没有 NaN/Inf 或缺失预测；
- runtime v4、HC95、HC90 六个冻结文件的运行前后 SHA256 一致；
- 没有生成或修改任何 runtime/QC 资产。

未发现协议、拓扑、数据或资产异常。需要保留的统计解释是：RG1 的简化
优势通过了预注册非劣 gate，但精度方向在 S_CC 上一致偏向 RG2，且两个
1% guard 的余量都很小。因此 runtime v5 应先作为 candidate 构建并执行
独立 parity/部署验证，不能把本结果扩写成 RG1 的绝对性能优势。

## 7. 停止位置

本阶段到此停止。未自动创建 runtime v5，未启动任何后续实验或部署任务。

