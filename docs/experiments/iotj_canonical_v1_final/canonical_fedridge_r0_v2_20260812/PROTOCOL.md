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

### Strengthened provenance contract

Preflight authenticates the whole `dataset_sha256.json` file at
`4aa511a59e62cf878a1b230b637591f5509728da149a7dff9876fa8f303e1486`
and the canonical preprocessing manifest at
`6c33f0a1586653b2bfa5a43f43ab502c5bdaa3474c24ac03015e36ddd40c2c41`.
It checks that the index's stored aggregate identity remains the frozen
canonical-v1 aggregate, then verifies the exact 32-file C1/C2 source subset
against both the pinned index and `canonical_source_artifact_sha256`. Only the
C1 and C2 directories are enumerated; missing, extra, linked, wrong-type, or
digest-drifted source files fail closed. C3/C4/C5 directories and artifacts
are never enumerated, hashed, opened, or used to recompute a target-inclusive
aggregate during this source-only preflight.

The prerequisite indexes are independently pinned before any indexed record
is trusted. `C0/C0_SHA256_INDEX.json` has whole-file SHA256
`18d6fa01352be80273460439e6c3a77196d8d93df53e3ea967f0e9ebdf335da0`;
`R0/R0_SHA256_INDEX.json` has whole-file SHA256
`0f9a4ed854df5b87acad2d6801fa1e5607ac8df58d6e21e5138b6e1401bfc242`.
The anchored C0 decision remains exactly `V1_INTERLEAVED_RETAINED`; the
anchored original R0 remains `FAIL_CLOSED` at
`R0.4_CANONICAL_FEDRIDGE_EXACT_RECOVERY`, with practical fallback, threshold
relaxation, rerun, downstream release, and R1 release all false.

Authorization binds the working bytes of this runner, the v2 numerical
module, the versioned cache module, and this machine-readable protocol to the
authorized Git HEAD. Unrelated logs and temporary directories do not affect
that critical-path check.

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

All future evidence files are immutable publications: bytes are written to a
same-directory temporary file and linked exclusively to the official name.
An idempotent retry may accept only byte-identical official evidence. The
completion marker is last and requires exact access, pre-test locks, every
hard gate, no blocking finding, exact artifact types, complete hash coverage,
and semantic agreement across diagnostics, locks, access records, decision,
and audits. Audit-only mode reads and cross-checks existing evidence; it does
not regenerate metrics or alter evidence. Reserved index and completion names
are excluded only at the evidence root, so identically named nested files
remain hash-indexed.

A PASS audit requires exactly 416 H1 feature-audit rows (104 coordinates for
each of gases 0-3), plus exactly one row for each gas in every other diagnostic
family; `NO_ROWS` is never valid PASS evidence. A partial FAIL may retain a
stage-valid `NO_ROWS` sentinel, but every family already present is validated.
When gas rows are favorable, a stored FAIL is auditable only when the complete
recomputed access, lock, artifact, or execution-provenance context contains a
blocking defect and the stored blocking-finding list matches it exactly; an
exception string is not required for a provenance-only failure.

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

## Explicit reconciliation check

The following read-only checks were performed during the Task 5 review fix.
They compare sources; they do not generate experimental evidence.

| Source | Read-only command or artifact | Checked fields | Outcome |
|---|---|---|---|
| Approved design commit | `git show b41fee1d5bd64a19d6fefcad5fde610183856202:docs/superpowers/specs/2026-08-12-canonical-fedridge-r0-v2-design.md` | study ID; C1/C2 order; split roles/counts; target/QC boundary; float64, M2/n, scale floor, alpha, solver, intercept, tolerance formulas, decisions | No traceable disagreement found |
| Task 4 code/constants | Read-only import of `CLIENT_ORDER`, `RIDGE_ALPHAS`, `SCALE_FLOOR`, and `registered_tolerances_v2()` from `gaps_flower/canonical_fedridge_v2.py` at `6668dc5db83428a2d957d962d6a5fa4bb5dc2430` | C1/C2; alpha grid; `1e-9`; epsilon, dimensions, `tau_moment`, `tau_residual`, functional ppm | No traceable disagreement found |
| Canonical manifests/data roles | Parsed `dataset/iotj_canonical_v1/dataset_sha256.json` and C1/C2 `stats.json`; recomputed the manifest-listed C1/C2 feature/phase/metadata SHA256 values | aggregate SHA; source roles; train/calibration/test counts; per-file hashes | No traceable disagreement found |
| Planner/registry records | Parsed both CSVs with `csv.DictReader` and compared the canonical fields with this manifest and the sources above | one experiment ID; ordered sources; empty targets; split/model/checkpoint policy; none controls; seed; result/metric/status/provenance fields | No traceable disagreement found |

Reconciliation result: `conflict_fields=[]`. The two non-configuration unknowns
remain the future execution resource budget and future execution
environment/BLAS metadata.

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
