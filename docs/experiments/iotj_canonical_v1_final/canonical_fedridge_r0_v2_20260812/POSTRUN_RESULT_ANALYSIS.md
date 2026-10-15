# R0-v2 post-run result analysis

## Scope and labels

This closure records the already sealed source-only R0-v2 evidence at
`results/iotj_canonical_v1_final/canonical_fedridge_r0_v2_20260812/`.  It does
not rerun the controller, audit, or numerical computation, and it does not
modify any result artifact.

Reported execution label: `FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED`.
Reported audit label: `PASS; Evidence eligible`.  Documentation-level
cross-checking finds the same decision in `R0_V2_DECISION.json`,
`DATA_ACCESS_AUDIT.md`, and `fixed_endpoint_complete.json`; no independent
numerical recomputation was performed in this closure.  The controller audit
was run exactly once and returned `PASS` with `blocking_findings=[]`; its
before/after 34-file hashes were identical.

## Registered gates and numerical result

All twelve registered hard booleans are `true` for each of gases 0, 1, 2, and
3: alpha equality, scaler, safe-scale mask, normal equations, condition,
federated residual, pooled residual, raw prediction, clipped prediction, RMSE
parity, MAE parity, and finite values.  Thus all four per-gas hard gates pass.

The registered raw-prediction maximum-difference threshold is `1e-6 ppm`.
Observed maximum absolute raw prediction differences were:

| Gas ID | Maximum absolute raw prediction difference (ppm) |
|---:|---:|
| 0 | 1.2889245226688217e-09 |
| 1 | 2.224234663117386e-08 |
| 2 | 8.081229907475063e-09 |
| 3 | 1.0743463008111576e-08 |

The observed range is `[1.2889245226688217e-09, 2.224234663117386e-08] ppm`,
well within the registered functional-parity threshold.

## Locks, access, integrity, and completion

`source_alpha_lock.json` and `model_lock.json` existed before source-test
access.  The access audit records C1/C2 only, source-test exclusion from
selection, and source-test opening only after the locks.  The completion marker
states `R1_released=true` and `downstream_launched=false`: R1 is released but
has not been executed.

The immutable result root is ignored/untracked by Git and contains 34 files
totalling 7,453,988 bytes.  `sha256_index.json` contains 32 indexed evidence
files; the index itself and `fixed_endpoint_complete.json` are the two reserved
unindexed completion artifacts.  The completion marker records the index hash,
which matches the index file hash.

Core result hashes:

| File | SHA256 |
|---|---|
| `R0_V2_DECISION.json` | `dec11094ea15f3484f13ddf7b3fb28102975141f253c307bafe1c04df9798935` |
| `R0_V2_EXPERIMENT_AUDIT.md` | `dbba6c4d57e95aef1d46ba92da52c57d75d1b4c8a980edd3b8de05a13b74030a` |
| `DATA_ACCESS_AUDIT.md` | `325d633c17605fd96c1f0ff6b26d4cc96d6560694b646c2a4923a27f41b68938` |
| `source_alpha_lock.json` | `8263be2e8f3bd669e37e593fc07129eb43dfad278aa9b119ed1e19cfd94e5390` |
| `model_lock.json` | `40c06848f19d211920b200946328e6e95cae9656a17ae0d18036951b7c5d67f6` |
| `r0_v2_scaler_diagnostics.csv` | `1277e9746ee209cd7e940c4a5ea14596cf8651555985fe67bff23d2b75bc7e09` |
| `r0_v2_normal_equation_diagnostics.csv` | `a41df59fad5654a20a087d08dae498d77b465cf16313dcc7d3f9b7099484ad9f` |
| `r0_v2_system_diagnostics.csv` | `385d53a52c73f890cd8416806af95701a1b55755b8ee926aaa5f9be8a46064a2` |
| `r0_v2_functional_equivalence.csv` | `89f75d48a935904008ea4a559da4648e450e1aed72520f5f6f88b6859dbe4708` |
| `sha256_index.json` | `04e62b0a2b363fcb79cbeb60ad6c2c6c9a3dedef380b2b0f7c2aa808732d4d82` |
| `fixed_endpoint_complete.json` | `cfa279c71c871ed9d41a46e044b0714a6923f8ec33357073d3807851dcb90b53` |

## Interpretation limits

This establishes numerical/algebraic equivalence of the federated
sufficient-statistics and pooled Ridge constructions on the registered rows.
It does not establish bitwise identity, target/client performance, accuracy
improvement, target-transfer benefit, or statistical inference.  Deterministic
seed 42 is unused for this reconstruction, and the resource budget remains
unreported.  The original R0 result remains
`R0_EXACT_RECOVERY_NOT_ESTABLISHED` / fail-closed and is neither overwritten
nor reinterpreted by this independent R0-v2 result.
