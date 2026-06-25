# Lightweight Fair Matrix Summary

Date: 2026-06-25

Scope: C12 -> C345 target test, no-QC full-set.

## Summary

| candidate | status | ALL | C3-CO | C4-CO | C5-CO | C3-CO_high_200_250 | C4-CO_high_200_250 | C5-CO_high_200_250 | nonCO_ALL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 H8 CO-specialist candidate | CO-specialist candidate | 18.47 | 14.97 | 19.76 | 23.69 | 19.93 | 32.22 | 27.54 | 18.38 |
| B1 H2.3 target direct-head mainline | current mainline | 18.62 | 16.15 | 22.02 | 26.85 | 20.02 | 34.24 | 34.82 | 17.83 |
| Diagnostic source shared MLP + forced piecewise | diagnostic | 21.96 | 14.27 | 53.02 | 27.79 | 19.42 | 95.73 | 34.88 | 17.56 |
| L1 source Ridge + full residual auto_v2 | L1 fair | 22.62 | 17.04 | 48.35 | 25.83 | 21.41 | 85.87 | 29.53 | 19.54 |
| L1 source shared MLP + full residual auto_v2 | L1 fair | 22.63 | 14.27 | 53.14 | 27.25 | 19.42 | 95.38 | 29.73 | 18.70 |
| L1 source per-gas MLP + full residual auto_v2 | L1 fair | 22.76 | 15.73 | 53.71 | 24.87 | 20.59 | 96.37 | 27.62 | 18.85 |
| Diagnostic source per-gas MLP + forced ridge_phase | diagnostic | 22.83 | 15.23 | 53.69 | 24.87 | 19.99 | 96.33 | 27.62 | 19.04 |
| Diagnostic source Ridge + forced ridge_phase | diagnostic | 22.91 | 15.53 | 53.94 | 26.84 | 21.50 | 96.35 | 29.53 | 18.85 |
| Diagnostic source shared MLP + forced ridge_phase | diagnostic | 22.92 | 15.26 | 53.60 | 27.25 | 20.88 | 96.22 | 29.73 | 18.93 |
| Diagnostic source per-gas MLP + forced piecewise | diagnostic | 23.32 | 15.63 | 53.42 | 25.20 | 20.59 | 95.83 | 28.54 | 19.79 |
| Diagnostic source Ridge + forced piecewise | diagnostic | 23.57 | 17.04 | 48.35 | 31.14 | 21.41 | 85.87 | 38.56 | 20.36 |
| B0 R3aK16 + original auto_v2 | baseline | 27.34 | 33.70 | 56.59 | 46.12 | 41.70 | 95.32 | 60.00 | 19.00 |
| L0 source shared MLP + target affine | affine | 36.87 | 40.83 | 60.33 | 52.08 | 54.65 | 102.46 | 59.87 | 31.69 |
| L0 source per-gas MLP + target affine | affine | 39.41 | 50.59 | 73.29 | 59.68 | 65.03 | 113.31 | 78.35 | 29.96 |
| L0 source Ridge + target affine | affine | 40.17 | 46.23 | 82.34 | 44.23 | 59.10 | 119.14 | 61.75 | 32.63 |
| L0 source shared MLP direct | direct-transfer | 63.89 | 79.10 | 73.68 | 64.12 | 32.22 | 119.75 | 49.15 | 60.05 |
| L0 source Ridge direct | direct-transfer | 65.42 | 89.18 | 130.12 | 46.91 | 33.00 | 193.91 | 62.91 | 52.77 |
| L0 source per-gas MLP direct | direct-transfer | 71.28 | 97.83 | 118.55 | 107.91 | 32.27 | 90.61 | 37.94 | 55.07 |

## Reading

- Direct source-lightweight transfer still collapses, confirming that source fit quality alone is not enough.
- Target affine calibration rescues some error, but remains worse than B0.
- Full residual auto_v2 changes the conclusion materially: L1 source-lightweight heads improve from about 36-71 ALL RMSE down to about 22-23 ALL RMSE.
- Best formal L1 result: L1 source Ridge + full residual auto_v2 with ALL RMSE 22.62.
- Best diagnostic forced result: Diagnostic source shared MLP + forced piecewise with ALL RMSE 21.96.
- L1 now beats the original B0 baseline on ALL RMSE, C3 CO, C5 CO, and nonCO is roughly comparable.
- L1 still does not reach H2.3/H8 performance, especially on C4 CO and C4 high-CO.

## Decision

Lightweight source heads should not replace the H2.3/H8 performance mainline yet.

They are now credible deployment-lite candidates, because full target residual auto_v2 makes them much stronger than the earlier affine-only result. The next useful checks are parameter count, artifact size, and runtime latency. A unified L2 selector is worth keeping as a follow-up, but current evidence says it should treat lightweight heads as optional candidates, not the default route.
