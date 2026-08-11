# R0 numerical failure analysis

Status: **FAIL CLOSED at R0.4 exact recovery**

The canonical feature operator and source-only alpha path ran from commit `1b16f1e`. Four source train/calibration caches were newly computed from canonical-v1 5 Hz / 50x8 arrays; no legacy feature, scaler, alpha, coefficient, or QC asset was loaded. The run stopped before source test labels, target caches, target labels, or R1 were opened.

## What agreed

- Pooled and sufficient-statistics selection chose the same alpha for every gas: Ethanol 0.1; CO, Ethylene, and Methane 0.01.
- All scaler differences except the Methane `window_len_s` feature were at floating-point roundoff scale.
- Maximum prediction difference was `2.5330e-08 ppm`, below the frozen `1e-6 ppm` prediction tolerance.
- Ethanol passed every strict tolerance.

## Why the strict gate failed

CO and Ethylene coefficient differences were `6.1002e-08` and `2.0023e-08`, exceeding the frozen `1e-8` coefficient tolerance even though their predictions were numerically indistinguishable at the registered prediction scale.

For Methane, canonical `window_len_s` is physically constant at 10 seconds up to approximately `1.42e-14` floating representation spread. The pooled two-pass standard deviation falls below the `1e-9` scale floor and becomes 1.0. The registered sufficient-statistics expression `sum_x2/n - mean^2` suffers cancellation and yields a spurious positive variance whose square root is `1.6859e-07`, above the scale floor. This creates a scaler difference of `0.9999998314` and a coefficient difference of `0.00184619`, while predictions still differ by only `2.5330e-08 ppm`.

This is a numerical-reconstruction failure, not evidence of regression-performance inferiority. Nevertheless, the frozen protocol explicitly requires scaler, coefficient, and prediction tolerances simultaneously and forbids a practical-equivalence fallback. No threshold was relaxed and no alternative solver, variance estimator, feature definition, or rerun was introduced.

## Scientific decision

`R0_EXACT_RECOVERY_NOT_ESTABLISHED`. R1, R2, Q0, and Q1 remain unopened. A future continuation would require an explicit protocol amendment for numerically stable distributed variance/normal-equation reconstruction; it cannot be silently treated as the current canonical experiment.
