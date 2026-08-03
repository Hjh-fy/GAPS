# Ablation Plan — target-feature metadata

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| H-FMETA-01 | Metadata profile | M83_SENSOR; M104_FULL | B5 seed route, H1, calibration rows, alpha grid, target head | `S_CC_NRMSE` | Paired five-seed delta | Same route/mask within seed | Stop on row/schema/hash mismatch |
| H-FMETA-02 | Causal feature availability | M91_ONLINE_SAFE; M104_FULL | Same as above | `S_CC_NRMSE` | Relative degradation and per-gas persistence gate | Online-safe allowlist is frozen before test | Stop if excluded fields enter M91 |
| H-FMETA-03 | Source-prior interaction | no H1; + frozen H1 | Metadata profile and all protocol controls | `S_CC_NRMSE` | 3x2 matrix | H1 model and predictions fixed across profiles | Stop if source head is retrained |

## Required baselines

- M83 sensor-only target Ridge without H1.
- M83 sensor-only target Ridge with frozen federated H1.
- M91 online-safe target Ridge without H1.
- M91 online-safe target Ridge with frozen federated H1.
- M104 full target Ridge without H1.
- M104 full target Ridge with frozen federated H1.

## Resource budget and execution order

1. Validate feature-key partitions and dimensions.
2. Open calibration only; select and lock all per-seed/per-gas alphas.
3. Open test once after the persisted lock.
4. Produce per-seed, per-gas, common-correct, paired-profile, and summary artifacts.
5. Audit outputs before manuscript use.

## Unknown or conflicting protocol fields

- Physical-file-independent performance: not assessable under the frozen window-level split.
- Exact streaming parity of offline `t_onset`: unknown; this experiment treats the value as controller/detector-available but does not validate the detector online.

