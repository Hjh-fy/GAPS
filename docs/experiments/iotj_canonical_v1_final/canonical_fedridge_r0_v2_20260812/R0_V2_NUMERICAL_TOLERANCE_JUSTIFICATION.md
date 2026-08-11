# R0-v2 numerical tolerance justification

Status: `DESIGN_FREEZE_READY_FORMAL_NOT_STARTED`. C0 remains
`V1_INTERLEAVED_RETAINED`; original R0 remains
`R0_EXACT_RECOVERY_NOT_ESTABLISHED`.

## Formula-derived constants

For IEEE-754 float64,

```text
epsilon = 2.220446049250313e-16
gamma(m) = m*epsilon / (1 - m*epsilon)
n_max = 1340
d = 104
p = 105
tau_moment = 64*gamma(1340) = 1.9042545318376352e-11
tau_residual = 128*gamma(105) = 2.9842794901924903e-12
tau_functional = 1e-6 ppm
```

`n_max` is the pooled per-gas source train-plus-calibration refit count, `d`
is the H1 feature dimension, and `p` is the intercept-augmented design
dimension. The multipliers and dimensions were frozen in the approved design
before formal output. These tolerances are not selected or adjusted from observed R0-v2 results.

## Coordinate and equation gates

For coordinate `j`,

```text
S_j = max(1, max_abs_j, dynamic_range_j, abs(mean_pool_j), scale_pool_j)
abs(mean_fed_j - mean_pool_j) <= tau_moment*S_j
abs(scale_fed_j - scale_pool_j) <= tau_moment*S_j
```

The safe-scale masks must be exactly identical. Normal equations require
Frobenius-relative `A` and L2-relative `b` discrepancies no greater than
`tau_moment`; zero or nonfinite denominators fail closed.

For `M=A+alpha*P`, where the intercept entry of `P` is zero, `kappa=cond(M)`
must be finite and `kappa*epsilon < 1`. Both federated and pooled systems use

```text
relative_residual = ||M*beta-b||_2 / (||M||_2*||beta||_2 + ||b||_2)
```

and must not exceed `tau_residual`. The coefficient forward envelope
`kappa*(2*tau_moment + tau_residual)` is diagnostic only.

After the alpha/model locks, identical C1/C2 source-test rows require raw
prediction, clipped prediction, clipped RMSE, and clipped MAE differences no
greater than `1e-6 ppm`.

## Complete hard conjunction

For exactly one row for each gas 0,1,2,3, all of these built-in Boolean fields
must be true:

- `alpha_equal`
- `scaler_pass`
- `safe_scale_mask_equal`
- `normal_equations_pass`
- `condition_pass`
- `fed_residual_pass`
- `pooled_residual_pass`
- `raw_prediction_pass`
- `clipped_prediction_pass`
- `rmse_parity_pass`
- `mae_parity_pass`
- `finite_pass`

Only a complete pass returns
`FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED`; any other state
returns `R0_V2_FAILED`. No coefficient-only diagnostic can override the hard
conjunction. Formal execution remains blocked pending a separately named
freeze commit.
