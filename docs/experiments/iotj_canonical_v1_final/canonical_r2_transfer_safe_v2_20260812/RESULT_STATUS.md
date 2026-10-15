# R2-v2 结果状态

状态：`completed / audit_pass / authoritative`。

最终决策：`RETAIN_R84_DEVICE_DEPENDENT`。

| 方法 | pooled S_ALL RMSE (ppm) | 相对 R84 | Bootstrap RMSE delta 95% CI | 保留 |
|---|---:|---:|---:|---|
| R84 | 13.990912 | 基线 | — | 是 |
| Residual transfer | 25.446645 | +81.88% | [9.690392, 13.364003] | 否 |
| Shrinkage transfer | 15.663414 | +11.95% | [-0.464799, 4.430575] | 否 |

两个候选均未满足预注册的 pooled 改善、per-gas 安全和 bootstrap CI 三重门槛。因此冻结 R84 为 device-dependent regression backend，不再扩展 transfer-safe regression 搜索。

旧结果根 `canonical_r2_transfer_safe_20260812` 因执行代码身份不匹配及 OOF held-out label-bound leakage 被标记为 `invalid/superseded`；本 v2 结果是唯一 authoritative R2 证据。
