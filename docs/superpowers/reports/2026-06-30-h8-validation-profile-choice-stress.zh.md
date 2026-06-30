# H8+C4 Validation Replay 与 Profile-choice Low-cal Stress

本轮继续推进上一轮留下的问题：要正式验证 `C3/C4 -> H2.3+、C5 -> H8+C4`，不能只看 test 或 Accepted+Review 结果，必须先给 H8+C4 补出 validation predictions，再用 validation-only 的低 calibration 子样本做 profile choice。

## 新增产物

- source-augmented target ridge 新增 holdout validation 输出：
  - `target_validation_plus_source_preds.csv`
  - `target_validation_plus_source_preds_plus_c4_rescue.csv`
- H8 validation replay：
  - `results/f6_fixed_da_strong_r25_profile_replay_20260630/co_only_h8_validation_replay/`
  - `results/f6_fixed_da_strong_r25_profile_oracle_route_20260630/co_only_h8_validation_replay/`
- formal H8+C4 validation predictions：
  - `formal_c4_route_rescue_validation_predictions.csv`
- profile-choice stress：
  - `results/low_calibration_profile_choice_stress_20260630/`

## 方法口径

每个 repeat 中：

1. 从 validation predictions 里按客户端抽 `12 / 24 / 48 / 96` 条。
2. 对每个客户端分别比较 H2.3+ 与 H8+C4 的 validation RMSE。
3. validation RMSE 更低者成为该客户端 profile。
4. 将选出的 per-client profile 应用到固定 test predictions。

重要边界：

- profile 选择只看 validation，不看 test。
- 本轮 profile-choice stress 使用已固定的 H2.3+ profile 与已固定的 H8+C4 profile；H2.3+ blend weight 的低样本稳定性已经在上一轮单独测试。
- 这里的选择目标是 full RMSE，不是 Accepted+Review，也不是 QC-filtered validation。

## Profile 选择稳定性

### Oracle-route

| budget/client | C3 mode | C4 mode | C5 mode |
|---:|---|---|---|
| 12 | H2.3+ 60% | H2.3+ 60% | H2.3+ 60% |
| 24 | H2.3+ 75% | H8+C4 60% | H8+C4 60% |
| 48 | H2.3+ 60% | H8+C4 90% | H8+C4 65% |
| 96 | H2.3+ 60% | H8+C4 100% | H8+C4 100% |

oracle-route 下，高预算时 validation-only selector 支持：

```text
C3 -> H2.3+
C4 -> H8+C4
C5 -> H8+C4
```

这和我们之前根据 test / Accepted+Review 得到的 `C3/C4 -> H2.3+、C5 -> H8+C4` 不完全一致。关键分歧在 C4：validation full RMSE 认为 H8+C4 略好，但 oracle test full 上 H2.3+ 是 `8.50 / 0.0482`，H8+C4 是 `8.55 / 0.0502`，H2.3+ 反而略好。

### Real-route

| budget/client | C3 mode | C4 mode | C5 mode |
|---:|---|---|---|
| 12 | H2.3+ 60% | H2.3+ 60% | H2.3+ 75% |
| 24 | H2.3+ 75% | H8+C4 60% | H2.3+ 50% |
| 48 | H2.3+ 60% | H8+C4 90% | H2.3+ 65% |
| 96 | H2.3+ 60% | H8+C4 100% | H2.3+ 100% |

real-route 下，高预算时 validation-only selector 支持：

```text
C3 -> H2.3+
C4 -> H8+C4
C5 -> H2.3+
```

C5 的方向和 oracle-route 相反。这不是矛盾，而是再次说明：C5 的 profile choice 强依赖 route 质量和评价目标。real-route full RMSE 被 route 错误主导，validation full RMSE 更倾向保守地留在 H2.3+；oracle-route 消除分类错误后，C5 才稳定转向 H8+C4。

## Test 指标反馈

| route | budget/client | ALL RMSE mean ± std | ALL NRMSE mean ± std |
|---|---:|---:|---:|
| oracle-route | 12 | 9.558 ± 0.366 | 0.0513 ± 0.0017 |
| oracle-route | 24 | 9.413 ± 0.365 | 0.0506 ± 0.0018 |
| oracle-route | 48 | 9.376 ± 0.355 | 0.0509 ± 0.0016 |
| oracle-route | 96 | 9.114 ± 0.008 | 0.0501 ± 0.0009 |
| real-route | 12 | 22.445 ± 0.036 | 0.1756 ± 0.0011 |
| real-route | 24 | 22.437 ± 0.044 | 0.1763 ± 0.0013 |
| real-route | 48 | 22.462 ± 0.030 | 0.1762 ± 0.0012 |
| real-route | 96 | 22.487 ± 0.008 | 0.1753 ± 0.0005 |

oracle-route 高预算下，profile-choice stress 的 ALL RMSE 接近 H8+C4 全局 profile，但 NRMSE 不如前面人工 client selector：

- previous oracle client selector `C3/C4 H2.3+ / C5 H8+C4`: `9.11 / 0.0489`
- validation-only high-budget selector: approximately `9.11 / 0.0501`

差别主要来自 C4 被 validation-only selector 切到 H8+C4，而 test 归一化指标上 C4 更适合 H2.3+。

real-route 下，所有 full RMSE 仍然被 route error 主导，profile-choice 对 ALL full 的改善空间很小。这个结果不应该用来否定 H8+C4 的 deployment 价值，因为 Accepted+Review 上 C5 的 H8+C4 仍然明显更好。

## 关键反馈

第一，validation-only full RMSE selector 暴露了一个系统遗漏：profile choice 的目标函数必须和报告目标一致。

- 如果老师要求 classification-correct 下的回归性能，应该用 oracle-route full 作为主表。
- 如果讨论部署可用性，应该看 Accepted+Review 或 QC-filtered 目标。
- 如果用 real-route full 做 profile choice，C5 会被 route error 污染，倾向选择保守 profile。

第二，C4 不能简单切到 H8+C4。虽然 validation full RMSE 在高预算下偏向 H8+C4，但 test full 和 NRMSE 不支持强切换。C4 更适合使用带 margin 的守门规则：只有当 H8+C4 在 validation 上有足够大的、跨预算稳定的优势时才切换；否则保留 H2.3+。

第三，C5 的结论必须分口径：

- oracle-route / classification-correct：C5 稳定支持 H8+C4。
- real-route full：C5 更偏 H2.3+，因为 route error 污染太大。
- deployment Accepted+Review：C5 仍应优先看 H8+C4，因为该子集上 H8+C4 明显降低误差。

## 下一步建议

1. 不要直接采用“validation RMSE 最低即切换”的裸 selector。
2. 增加 profile-choice guardrail：例如 H8+C4 必须在 validation RMSE 上至少领先一个 margin，并且在 nonCO / NRMSE 上不恶化，才允许替代 H2.3+。
3. 为 deployment 报告另做 QC-filtered validation 或 calibration-review proxy；否则 full validation selector 和 Accepted+Review 目标会错位。
4. 论文主线继续使用 classification-correct oracle-route full；deployment appendix 再报告 Accepted+Review。
5. 当前最稳的表述应是：C3/C4 的 balanced H2.3+ 是默认稳态，C5 在 classification-correct 和 post-QC 部署子集下支持 CO-priority H8+C4；C4 的 H8+C4 只可作为候选，不应直接升为默认。
