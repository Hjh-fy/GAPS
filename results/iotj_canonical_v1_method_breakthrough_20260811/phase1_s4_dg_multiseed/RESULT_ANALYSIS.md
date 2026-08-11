# Phase-1 S4 DG multi-seed result analysis

## Confirmed scope

- Metric: C5 zero-shot classification Macro-F1; higher is better.
- Paired seeds: 41, 42, and 43.
- Seed42 values are reported from immutable Gate A reuse; seed41/43 values were calculated by the fixed Phase-1 runner after all new round25 endpoints were locked.
- C5 contains 1,360 sealed-test windows per endpoint. Source-pooled evaluation contains 2,358 windows per endpoint.

## Descriptive result

| Method | C5 Macro-F1 mean | Across-seed SD |
|---|---:|---:|
| FedAvg | 0.534020 | 0.167252 |
| GAPS-DG-P | 0.490290 | 0.171499 |

Paired DG-P minus FedAvg gains were -0.164730, +0.074990, and -0.041451 for seeds 41, 42, and 43. The mean paired gain was -0.043730 (sample SD 0.119876; 95% t interval -0.341519 to +0.254058; paired standardized mean change -0.364795).

## Stability and anomaly check

- The favorable seed42 result did not replicate: both new seeds reversed direction.
- Seed41 is the largest reversal (-16.47 percentage points), but it is retained without deletion or replacement.
- Source-pooled Macro-F1 remained between 0.995330 and 0.998727 for all endpoints, so the C5 instability is not explained by failed basic source optimization.
- The three-seed sample is intentionally small and no inferential superiority claim is supported.

## Registered decision

`SOURCE_DG_NOT_CONFIRMED`

The result rejects a stable multi-seed source-DG advantage under this frozen C5 design. It does not prohibit the already registered Phase-2 test of whether initialization differences survive fixed target commissioning.

