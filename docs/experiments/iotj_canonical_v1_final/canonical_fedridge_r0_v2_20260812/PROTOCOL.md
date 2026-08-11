# Canonical FedRidge R0-v2 frozen protocol

Status: `DESIGN_FREEZE_READY_FORMAL_NOT_STARTED`. Study ID:
`CAN-V1-FEDRIDGE-R0V2-20260812`. `formal_execution_started=false`.

This is a new source-only numerical-reconstruction protocol. It does not
supersede or reinterpret C0=`V1_INTERLEAVED_RETAINED` or the original
R0=`R0_EXACT_RECOVERY_NOT_ESTABLISHED`.

## Provenance and immutable inputs

- Approved design: `docs/superpowers/specs/2026-08-12-canonical-fedridge-r0-v2-design.md`
  at `b41fee1d5bd64a19d6fefcad5fde610183856202`.
- Task base and numerical implementation: `6668dc5db83428a2d957d962d6a5fa4bb5dc2430`.
- Dataset: `dataset/iotj_canonical_v1`; aggregate SHA256
  `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Preprocessing: `HZ5_MEAN_W10S`, 5 Hz, 10-second windows, 5-second stride,
  `50x8` tensors.
- Ordered source clients: `C1;C2`. Target-client set: empty. Every target
  calibration/test feature and label access field is false.
- Per source client: train 2360, calibration 320, test 680. Per gas, the
  train-plus-calibration refit has 670 rows per client/1340 pooled; source
  test has 170 rows per client/340 pooled.
- Features: 83 sensor coordinates and 104 H1 coordinates. The extractor file
  SHA256 is `7627b72ee4e1823d24c374d41a6c931f66b5efedd6eaf4a839c62e7b5b1fa72a`;
  ordered H1 and sensor-name hashes are respectively
  `df696d3cfbe43eff40b515f6f1a7bb51c9cd11900dba93e231a3ded0755c3259`
  and `4cb9e621b39cf726b18d0102d2ec395ba11b109b6ffcabb381c729dd44f26248`.
  Exact per-split source-array, phase, and metadata hashes are frozen in
  `protocol_manifest.json`.

## Fixed split, model, and checkpoint policy

For each gas, source train fits each candidate alpha; distributed C1/C2 source
calibration SSE/count selects the first minimum on the registered grid. Source
train plus calibration is then refit. Only after both source alpha and model
locks exist may the shared C1/C2 source test rows and labels be opened for the
functional-equivalence gates. DA, post-hoc calibration, and QC modes are
`none`.

The model is four per-gas 104D `CanonicalRidgeModelV2` reconstructions. No
checkpoint exists pre-run. A separately authorized run must create immutable
`source_alpha_lock.json` and `model_lock.json` beneath
`results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/`.
The authorized freeze commit supplied to the future CLI must equal current Git
HEAD. This bundle creates no result directory.

## Frozen numerical protocol

All computation is float64. Each client emits central `(n, mean, M2, min,
max)` records. C1 is merged before C2 with the Chan merge equations. Population
variance is `max(M2/n, 0)`. The safe scale becomes 1.0 only when
`raw_scale < 1e-9`; equality at `1e-9` retains the raw scale.

The alpha grid is `(0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)`, with the
first-in-grid tie break. Both paths use `numpy.linalg.pinv`; the intercept is
unregularized. Coefficient discrepancy and its condition-aware envelope are
diagnostic only.

The exact formula-derived gates are `tau_moment=64*gamma(1340)`,
`tau_residual=128*gamma(105)`, functional tolerance `1e-6 ppm`, and finite
`kappa` with `kappa*epsilon < 1`. The final conjunction contains exact alpha,
scaler, safe-scale-mask, normal-equation, condition, both residual, raw and
clipped prediction, RMSE, MAE, and finite-value gates for exactly gases 0-3.

## Decision and stop boundary

The only R0-v2 decisions are:

```text
FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED
R0_V2_FAILED
```

Any missing, nonfinite, mismatched, prematurely accessed, or failed hard-gate
evidence produces `R0_V2_FAILED`. Partial evidence is preserved; there is no
automatic resume, rerun, threshold change, solver change, alpha change,
feature change, or aggregation-order change. A PASS establishes algebraic
identity in exact arithmetic and numerical equivalence within the registered
tolerances; it does not establish bitwise identity.

Formal execution remains blocked pending a separately named freeze commit.
The planner-to-registry handoff records `execution_resource_budget` and future
execution environment/BLAS metadata as `unknown`; there are no conflicts.
