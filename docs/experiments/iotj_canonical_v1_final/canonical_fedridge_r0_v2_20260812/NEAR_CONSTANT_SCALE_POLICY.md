# Near-constant scale policy

Status: `DESIGN_FREEZE_READY_FORMAL_NOT_STARTED`; no formal evidence exists.
The prior decisions remain C0=`V1_INTERLEAVED_RETAINED` and original
R0=`R0_EXACT_RECOVERY_NOT_ESTABLISHED`.

The canonical safe-scale threshold `1e-9` is reused from the original frozen
protocol. It is not selected or tuned from feature values, regression
performance, original-R0 discrepancies, source-test metrics, or target data.
It is not an R0-v2 hyperparameter.

For every H1 coordinate, float64 population variance and scale are:

```text
variance = max(M2 / n, 0)
raw_scale = sqrt(variance)
canonical_scale = 1.0 if raw_scale < 1e-9 else raw_scale
```

The comparison is strict. `raw_scale < 1e-9` applies scale 1.0;
`raw_scale == 1e-9` retains the raw scale. Every one of the 104 coordinates,
including the named `window_len_s` row, must retain minimum, maximum, mean,
population variance, raw scale, dynamic range, floor-applied flag, and final
canonical scale in the future numerical audit. No feature is deleted or
redefined because it is constant or near constant.

The federated and pooled safe-scale masks must be exactly identical. A mismatch
is a hard failure; it does not authorize threshold adjustment.

The only possible R0-v2 decisions are
`FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED` and
`R0_V2_FAILED`. Formal execution remains blocked pending a separately named
freeze commit.
