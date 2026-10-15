# IoT-J Runtime v5 独立 QC 正式结果

## 决策

最终状态为 `RUNTIME_V5_QC_VALID_BUT_NOT_SUPERIOR`。Runtime v5 的 QC2 风险方向、
test tail enrichment 和 320/1360 runtime parity 有效，但没有通过全部相对 runtime
v4 的预注册 promotion guards。因此 runtime v4 继续作为正式部署基线；不得根据
本次 test 调整 v5 candidate、ECDF、MAD scale 或阈值。

## Calibration OOF 与选择

protocol amendment v2 使用 320 行、80 个 variable-size filename groups。五个 folds
均为 64 行、16 groups，filename 跨 fold 数为 0。各 fold 的 target Ridge、B5
prototype/support、ECDF 和 regression-consistency MAD scale 均只由其余四 folds
构建。

| Candidate | Spearman | lowest/highest risk decile RMSE | HC95 accepted RMSE | HC90 accepted RMSE | OOF gate |
|---|---:|---:|---:|---:|---|
| QC1 | 0.1430 | 71.0148 / 34.7558 | 28.7241 | 27.9479 | fail risk direction |
| QC2 | 0.1888 | 6.4335 / 35.2088 | 28.1532 | 27.8848 | pass; selected |
| QC3 | 0.1966 | 5.6471 / 19.7936 | 29.1778 | 29.5258 | fail tail enrichment |

QC2 是唯一通过完整 OOF gate 的候选。锁在 test 前持久化，SHA-256 为
`64877b7676bc4497074a2619282e0d2fedd658db76c2d1868b91699edba6d518`。

## 1360 行 test

| Workpoint | A/R/R | Yield | Accepted RMSE | Reject RMSE | CO-high yield / RMSE |
|---|---:|---:|---:|---:|---:|
| HC95 | 1275 / 41 / 44 | 93.75% | 13.9178 ppm | 109.7344 ppm | 78.43% / 30.4522 ppm |
| HC90 | 1183 / 113 / 64 | 86.99% | 12.7723 ppm | 95.8997 ppm | 39.22% / 40.0014 ppm |

最高风险 decile RMSE 为 69.6317 ppm，最低风险 decile 为 5.6433 ppm；两个工作点
的 reject RMSE 均高于 accepted RMSE。27 个错路由中，HC95 accept/review/reject
为 6/5/16，HC90 为 3/5/19。

## 为什么未晋级

自动加载的冻结 v4 基线表明：

- HC95 v5 yield 93.75%，低于 v4 97.28% 约 3.53 个百分点，超过允许的 2 个百分点；
- HC90 v5 yield 86.99%，低于 v4 90.81% 约 3.82 个百分点；
- HC90 CO yield 从 v4 84.71% 降到 v5 64.12%，下降 20.59 个百分点，同时 CO
  accepted RMSE 从 19.7243 ppm 恶化到 20.8318 ppm。

虽然 v5 accepted RMSE 整体优于该冻结 v4 runtime reference，但 coverage 和 CO
guard 不满足，因此只能判定 QC 有效但不优于正式 v4。

## Runtime parity 与异常

calibration 320 行与 test 1360 行均同时验证 HC95/HC90：class、row key、decision、
auto output mismatch 全部为 0；prediction、raw components、normalized risk 和
deployment risk 的最大差均为 0。

test 首次打开后，v4 baseline adapter 因遗漏已有 `pred_class` 字段而停止。该异常
发生在 candidate/threshold 锁定之后，且在 v4 guards 和 parity 之前；失败记录被
保留。修复只补充 v4 CSV 字段映射，35 项相关测试通过后从同一 immutable QC lock
继续，未更改任何风险资产。

## Evidence boundary

filename grouping 仅适用于 calibration OOF folds。历史 calibration/test split 仍为
window-level，不能声称两者在 original-file level 完全独立。本次结果不构成 Pi/PC
benchmark、runtime v5 晋级、低校准或新回归模型证据。
