# Real-route vs Oracle-route Gap 与低 Calibration Stress

本轮补充两个实验：

1. `real-route vs oracle-route gap`：量化实际分类/route 错误对回归 full 指标的污染。
2. `low calibration stress`：在 H2.3+ 已生成的 validation predictions 上按客户端下采样，重新选择 per-client blend weight，观察低 calibration 预算下权重和 test RMSE/NRMSE 是否稳定。

## 实验产物

- gap 审计脚本：`run_route_gap_audit.py`
- low-cal stress 脚本：`run_low_calibration_blend_stress.py`
- gap 输出：`results/real_vs_oracle_gap_audit_20260630/`
- stress 输出：`results/low_calibration_blend_stress_20260630/`

## Gap 结论

full 指标里 real-route 与 oracle-route 的 gap 非常大，尤其集中在 C5。

| profile | scope | real RMSE / NRMSE | oracle RMSE / NRMSE | gap RMSE / NRMSE | gap RMSE 占 real |
|---|---|---:|---:|---:|---:|
| H2.3 | ALL | 22.94 / 0.1789 | 10.57 / 0.0564 | 12.37 / 0.1224 | 53.9% |
| H2.3 | C5 | 39.48 / 0.3207 | 13.23 / 0.0643 | 26.25 / 0.2563 | 66.5% |
| H2.3+ | ALL | 22.43 / 0.1743 | 9.86 / 0.0515 | 12.57 / 0.1228 | 56.0% |
| H2.3+ | C5 | 39.27 / 0.3190 | 12.18 / 0.0596 | 27.09 / 0.2594 | 69.0% |
| H8+C4 | ALL | 22.44 / 0.1786 | 9.10 / 0.0511 | 13.34 / 0.1274 | 59.4% |
| H8+C4 | C5 | 39.14 / 0.3250 | 9.57 / 0.0499 | 29.56 / 0.2751 | 75.5% |
| client selector | ALL | 22.38 / 0.1771 | 9.11 / 0.0489 | 13.27 / 0.1283 | 59.3% |
| client selector | C5 | 39.14 / 0.3250 | 9.57 / 0.0499 | 29.56 / 0.2751 | 75.5% |

解释：

- full real-route 指标主要不是在评价回归 profile 本身，而是在混合评价“分类/route 是否正确 + 回归是否准”。
- C5 是污染最大的客户端；client selector 在 real-route full 下仍被 C5 route 错误拖住。
- 这支持老师的意见：论文主表应该重点报告 classification-correct / oracle-route 下的回归性能。

Accepted+Review 的 gap 基本为 0。唯一明显的残余是 H2.3+ 的 C5：real-route Accepted+Review `9.99 / 0.0500`，oracle-route `9.53 / 0.0478`，gap 为 `0.46 / 0.0022`。这不是 QC 覆盖导致的主污染，而是 oracle-route 下 C5 blend weight 重新选择后更合适。

## Low Calibration Stress 口径

本轮 stress 只测试 H2.3+ blend lambda 的低样本稳定性：

```text
blend_ppm = (1 - w_client) * H2.3_anchor_ppm + w_client * regfeat_candidate_ppm
```

每个 route 口径分别使用已有的：

- `fusion_profile_validation_predictions.csv`
- `fusion_profile_predictions.csv`

然后对每个客户端 validation rows 做预算下采样：`12 / 24 / 48 / 96` 条，每档 `20` 次重复。每次只用采样后的 validation rows 选择 `w_client`，再应用到固定 test predictions。

这不是重新训练回归头的低样本实验，也不是 H2.3+ vs H8+C4 的 profile-choice stress。H8+C4 当前没有逐样本 validation prediction 文件，只有 gate candidate calibration scores 和 test predictions；要做 profile-choice stress，需要下一步先补 H8+C4 validation replay 输出。

## Stress 结果：Oracle-route

oracle-route 下，全量 H2.3+ 权重是 `C3=0.5、C4=0.5、C5=0.25`。

| budget/client | ALL RMSE mean ± std | C3 RMSE mean ± std | C4 RMSE mean ± std | C5 RMSE mean ± std |
|---:|---:|---:|---:|---:|
| 12 | 10.303 ± 0.368 | 9.742 ± 0.771 | 8.767 ± 0.296 | 12.513 ± 0.588 |
| 24 | 10.067 ± 0.407 | 9.444 ± 0.764 | 8.768 ± 0.319 | 12.208 ± 0.356 |
| 48 | 9.939 ± 0.178 | 9.335 ± 0.390 | 8.557 ± 0.086 | 12.127 ± 0.181 |
| 96 | 9.884 ± 0.098 | 9.211 ± 0.206 | 8.504 ± 0.000 | 12.179 ± 0.000 |

| budget/client | C3 weight mode | C4 weight mode | C5 weight mode |
|---:|---:|---:|---:|
| 12 | 0.75, rate 35% | 0.50, rate 40% | 0.25, rate 35% |
| 24 | 0.50, rate 60% | 0.50, rate 40% | 0.25, rate 40% |
| 48 | 0.50, rate 70% | 0.50, rate 60% | 0.25, rate 65% |
| 96 | 0.50, rate 95% | 0.50, rate 100% | 0.25, rate 100% |

解释：

- 12/24 条每客户端 validation 太少，C3/C4/C5 的权重都还会明显抖动。
- 到 48 条时，主模态已经回到全量选择附近，但仍有一定波动。
- 到 96 条时，C4/C5 完全稳定，C3 也 95% 稳定到 `0.5`。
- oracle-route 下 C5 稳定支持 `w=0.25`，说明分类正确时 C5 的 H2.3+ candidate 仍有小幅可用增益。

## Stress 结果：Real-route

real-route 下，全量 H2.3+ 权重是 `C3=0.5、C4=0.5、C5=0.0`。

| budget/client | ALL RMSE mean ± std | C3 RMSE mean ± std | C4 RMSE mean ± std | C5 RMSE mean ± std |
|---:|---:|---:|---:|---:|
| 12 | 22.671 ± 0.209 | 12.813 ± 0.638 | 12.968 ± 0.296 | 39.348 ± 0.271 |
| 24 | 22.591 ± 0.235 | 12.564 ± 0.673 | 12.975 ± 0.320 | 39.322 ± 0.302 |
| 48 | 22.585 ± 0.128 | 12.385 ± 0.296 | 12.768 ± 0.091 | 39.495 ± 0.186 |
| 96 | 22.537 ± 0.045 | 12.260 ± 0.164 | 12.725 ± 0.000 | 39.478 ± 0.000 |

| budget/client | C3 weight mode | C4 weight mode | C5 weight mode |
|---:|---:|---:|---:|
| 12 | 0.75, rate 35% | 0.50, rate 40% | 0.00, rate 30% |
| 24 | 0.50, rate 60% | 0.50, rate 40% | 0.00, rate 35% |
| 48 | 0.50, rate 70% | 0.50, rate 60% | 0.00, rate 55% |
| 96 | 0.50, rate 95% | 0.50, rate 100% | 0.00, rate 100% |

解释：

- C3/C4 与 oracle-route 的稳定性几乎一致，说明这两个客户端的 blend lambda 对 route 口径不太敏感。
- C5 完全不同：real-route 最终稳定到 `w=0.0`，oracle-route 稳定到 `w=0.25`。
- 这再次证明 C5 的回归 calibration 不能脱离 route 质量解释；如果 route 仍有错误，C5 更适合保守地不引入 H2.3+ candidate；如果分类正确，C5 可以接受弱 blend。

## 对后续计划的影响

1. 主论文/汇报应把 classification-correct oracle-route full 指标作为回归主表，real-route full 放到 route-error impact / deployment appendix。
2. C3/C4 的 H2.3+ `w=0.5` 在 48-96 条 validation 后比较稳定，可以作为 balanced profile 的主要证据。
3. C5 不能用一个固定 blend weight 横跨 real-route 和 oracle-route；要在方法里强调“每个目标域客户端使用 calibration-validation 重新选 lambda”。
4. 若要正式验证 `C3/C4 -> H2.3+、C5 -> H8+C4` 的低样本稳定性，下一步需要补 H8+C4 的 validation predictions，然后做 profile-choice stress，而不是用 test set 反推 profile 选择。
5. 当前最值得继续推进的不是再盲目增强全局 profile，而是把 route-correct / route-error 两条口径分开，并针对 C5 做 route rescue 或 classification-correct 条件下的 profile calibration。
