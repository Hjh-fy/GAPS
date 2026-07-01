# 当前基座回归主线故事

> Generated from `run_current_base_regression_story.py` on 2026-07-01 using frozen current-base CSV artifacts.

## Frozen Base And R3aK16

当前阶段冻结 F6 分类基座、backbone features、H2.3+/H8 profile predictions 和 QC records。R3aK16/auto_v2 保留为 baseline、fallback 和 gate context，不再作为每轮回归优化都要重训的主线回归头。

## Oracle-route Full

| profile | ALL RMSE/NRMSE | C3 | C4 | C5 |
|---|---:|---:|---:|---:|
| H2.3 oracle-route | 10.574 / 0.0564 | 9.627 / 0.0540 | 9.284 / 0.0527 | 13.232 / 0.0643 |
| H2.3+ oracle-route weak-blend | 9.861 / 0.0515 | 9.164 / 0.0487 | 8.504 / 0.0482 | 12.179 / 0.0596 |
| H8+C4 oracle-route | 9.104 / 0.0511 | 9.131 / 0.0522 | 8.551 / 0.0502 | 9.574 / 0.0499 |
| Guarded practical oracle-route | 9.109 / 0.0489 | 9.164 / 0.0487 | 8.504 / 0.0482 | 9.574 / 0.0499 |

## Accepted+Review

| profile | ALL RMSE/NRMSE | C3 | C4 | C5 |
|---|---:|---:|---:|---:|
| H2.3+ oracle-route weak-blend | 6.997 / 0.0376 | 5.653 / 0.0328 | 6.317 / 0.0343 | 9.528 / 0.0478 |
| H8+C4 oracle-route | 6.476 / 0.0371 | 5.756 / 0.0352 | 6.540 / 0.0359 | 7.632 / 0.0415 |
| Guarded practical oracle-route | 6.375 / 0.0356 | 5.653 / 0.0328 | 6.317 / 0.0343 | 7.632 / 0.0415 |
| Client prior C34 H2.3+ / C5 H8+C4 oracle-route | 6.375 / 0.0356 | 5.653 / 0.0328 | 6.317 / 0.0343 | 7.632 / 0.0415 |

## Route Gap

| family | scope | N | gap RMSE | gap NRMSE | gap RMSE / real |
|---|---|---:|---:|---:|---:|
| H2.3 | ALL | 5400 | 12.370 | 0.1224 | 53.9% |
| H2.3 | C5 | 1360 | 26.247 | 0.2563 | 66.5% |
| H2.3+ | ALL | 5400 | 12.573 | 0.1228 | 56.0% |
| H2.3+ | C5 | 1360 | 27.090 | 0.2594 | 69.0% |
| H8+C4 | ALL | 5400 | 13.335 | 0.1274 | 59.4% |
| H8+C4 | C5 | 1360 | 29.562 | 0.2751 | 75.5% |
| client_selector | ALL | 5400 | 13.267 | 0.1283 | 59.3% |
| client_selector | C5 | 1360 | 29.562 | 0.2751 | 75.5% |

## Low Calibration Stability

| route | client | budget | profile mode | mode rate | H8+C4 rate | blend weight mode | weight mode rate |
|---|---|---:|---|---:|---:|---:|---:|
| oracle-route | C3 | 96 | H2.3+ | 60.0% | 40.0% | 0.50 | 95.0% |
| oracle-route | C4 | 96 | H8+C4 | 100.0% | 100.0% | 0.50 | 100.0% |
| oracle-route | C5 | 96 | H8+C4 | 100.0% | 100.0% | 0.25 | 100.0% |
| real-route | C3 | 96 | H2.3+ | 60.0% | 40.0% | 0.50 | 95.0% |
| real-route | C4 | 96 | H8+C4 | 100.0% | 100.0% | 0.50 | 100.0% |
| real-route | C5 | 96 | H2.3+ | 100.0% | 0.0% | 0.00 | 100.0% |

## Reading

- 主报告使用 oracle-route full-set 回答分类正确下的回归能力。
- Accepted+Review 是部署补充，不替代 oracle-route 主指标。
- real-route full-set 的大 gap 说明主要污染来自 classification/route error，尤其是 C5。
- 当前基座内的后续优化应集中到 C5 CO-priority calibration/rescue。
