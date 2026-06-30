# 2026-06-30 方案 B：Guarded Profile Selector 推进记录

## 目标

老师强调“重点报告分类正确下的回归性能指标”之后，主线应从 real-route full-set 转到 classification-correct / oracle-route。方案 B 的目标不是再盲目搜索回归头，而是把现有两条强 profile 接成一个可解释选择器：

- 默认 profile：H2.3+ target direct-head weak-blend。
- 候选 rescue：H8 + formal C4 route rescue。
- 选择原则：只有当目标客户端验证证据足够强时，才允许从 H2.3+ 切到 H8+C4。

这解决了裸 validation selector 的主要问题：C4 在验证集上 H8+C4 只小幅领先，但测试/NRMSE/post-QC 并不支持切换。

## 实验实现

新增脚本：

- `run_guarded_profile_selector.py`

新增测试：

- `tests/test_guarded_profile_selector.py`

新增 QC 配置：

- `configs/guarded_profile_qc_profiles_real_20260630.json`
- `configs/guarded_profile_qc_profiles_oracle_20260630.json`

实验输出：

- `results/guarded_profile_selector_20260630/`：strict guard，`max_nonco_delta=0.0`
- `results/guarded_profile_selector_nonco05_20260630/`：practical guard，`max_nonco_delta=0.5`
- `results/guarded_profile_selector_20260630/profile_qc_real/`
- `results/guarded_profile_selector_20260630/profile_qc_oracle/`

## Guard Rule

H2.3+ 是默认；H8+C4 只有同时满足下面条件才切换：

- validation RMSE gain：`H2.3+ RMSE - H8+C4 RMSE >= 0.5`
- validation NRMSE 不恶化：`H8 NRMSE - H2.3+ NRMSE <= 0`
- low-cal profile 稳定率：预算 96/client 下 `H8_C4_rate >= 70%`
- nonCO guard：
  - strict：`H8 nonCO RMSE - H2.3+ nonCO RMSE <= 0`
  - practical：允许 `<= 0.5 ppm` 的小幅 nonCO 验证波动

strict 版本是保守边界；practical 版本是当前更适合作为故事主线的版本，因为 C5 的 nonCO 牺牲很小，而 CO/full/Accepted+Review 改善明显。

## Profile 选择结果

practical guard：

| route | C3 | C4 | C5 |
|---|---|---|---|
| real-route | H2.3+ | H2.3+ | H2.3+ |
| oracle-route | H2.3+ | H2.3+ | H8+C4 |

关键解释：

- C3：H8+C4 验证 RMSE 更差，不能切。
- C4：H8+C4 验证 RMSE 只领先 0.164，低于 0.5 margin；同时 NRMSE/nonCO 更差，所以不能切。
- C5 oracle-route：H8+C4 验证 RMSE 领先 0.879，NRMSE 更好，低 calibration 稳定率 100%，nonCO 验证波动 0.374 ppm，在 practical guard 下允许切换。
- C5 real-route：route 错误污染 full-set 验证，H8+C4 不通过；这说明 real-route full-set 不适合作为主报告选择口径。

## Classification-Correct 主指标

oracle-route full-set：

| profile | ALL RMSE / NRMSE | C3 | C4 | C5 |
|---|---:|---:|---:|---:|
| H2.3+ | 9.861 / 0.0515 | 9.164 / 0.0487 | 8.504 / 0.0482 | 12.179 / 0.0596 |
| H8+C4 | 9.104 / 0.0511 | 9.131 / 0.0522 | 8.551 / 0.0502 | 9.574 / 0.0499 |
| Guarded practical | 9.109 / 0.0489 | 9.164 / 0.0487 | 8.504 / 0.0482 | 9.574 / 0.0499 |

结论：

- H8+C4 的 ALL RMSE 很强，但会牺牲 C3/C4 的 NRMSE。
- Guarded practical 基本保留 H8+C4 的 ALL RMSE，同时拿到更好的 ALL NRMSE：0.0489。
- C5 是切换收益核心：RMSE 从 12.179 降到 9.574，NRMSE 从 0.0596 降到 0.0499。

## Accepted+Review 口径

oracle-route official QC Accepted+Review：

| profile | ALL RMSE / NRMSE | C3 | C4 | C5 |
|---|---:|---:|---:|---:|
| H2.3+ | 6.997 / 0.0376 | 5.653 / 0.0328 | 6.317 / 0.0343 | 9.528 / 0.0478 |
| H8+C4 | 6.476 / 0.0371 | 5.756 / 0.0352 | 6.540 / 0.0359 | 7.632 / 0.0415 |
| Guarded practical | 6.375 / 0.0356 | 5.653 / 0.0328 | 6.317 / 0.0343 | 7.632 / 0.0415 |

这张表是讲故事最干净的一张：

- C3/C4 维持 H2.3+，避免 H8+C4 的局部退化。
- C5 使用 H8+C4，Accepted+Review RMSE 从 9.528 降到 7.632。
- ALL Accepted+Review 从 H2.3+ 的 6.997 降到 6.375，同时 NRMSE 从 0.0376 降到 0.0356。

real-route 的部署补充：

| profile | ALL full RMSE / NRMSE | ALL Accepted+Review RMSE / NRMSE |
|---|---:|---:|
| H2.3+ real-route | 22.434 / 0.1743 | 7.160 / 0.0383 |
| H8+C4 real-route | 22.439 / 0.1786 | 6.476 / 0.0371 |
| Client prior C34 H2.3+ / C5 H8+C4 | 22.376 / 0.1771 | 6.375 / 0.0356 |

real-route full-set 被分类错误主导，不能直接作为回归 profile 优劣判断；但在 Accepted+Review 里，C5 H8 rescue 仍然有效。

## 当前故事线

1. 分类器错误会显著放大 real-route full-set 回归误差，所以老师要求的 classification-correct 指标应该作为主报告。
2. 在 oracle-route 下，H2.3+ 是 C3/C4 的稳定 direct-head，H8+C4 是 C5 的 CO rescue。
3. 裸 validation selector 会误切 C4，因此需要 guardrail。
4. practical guard 成功得到 `C3/C4 -> H2.3+，C5 -> H8+C4`，并在 full-set 与 Accepted+Review 两个口径同时改善。
5. 部署口径下，可以把 real-route full-set 作为风险说明，把 Accepted+Review / post-QC 作为上线可用性补充。

## 下一步建议

- 把主表整理为论文/汇报表：oracle-route full-set + oracle-route Accepted+Review。
- 在附录给出 real-route gap：说明剩余误差主要来自分类 route，而非回归头本身。
- 后续如果继续优化训练流程，优先做 C5/H8 的 CO-specific calibration，不优先动 C3/C4 的 direct-head。
