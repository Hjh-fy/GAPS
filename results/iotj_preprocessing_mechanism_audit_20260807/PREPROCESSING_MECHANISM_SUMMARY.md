# Preprocessing mechanism summary

| Hypothesis | Status | Evidence | Impact |
|---|---|---|---|
| H1 concentration-label mismatch | excluded | filename-derived label constant | not causal |
| H2 raw timestamp jitter only | measured | P0 | see raw distribution |
| H3 large raw gaps | measured | P0/P3 | diagnostic |
| H4 duplicate timestamp handling | measured | P0 | diagnostic |
| H5 global interpolation distortion | measured | P1/P6 | diagnostic only |
| H6 baseline G0 propagation | measured | P2 | diagnostic only |
| H7 large-gap interpolated windows | measured | P3 | diagnostic only |
| H8 C5 Methane 225 raw-data anomaly | measured | P0/P4/P6 | diagnostic only |
| H9 R84 feature sensitivity | measured | P7 | diagnostic only |
| H10 legacy/time-bin similarity | measured | P6 | candidate evidence only |

## Oracle-route P6 RMSE (diagnostic, no classifier retraining)

| Preprocessing | C3 | C4 | C5 |
|---|---:|---:|---:|
| legacy | 10.5395 | 12.6568 | 13.0396 |
| interp | 9.4937 | 10.0853 | 21.0622 |
| timebin | 9.7262 | 9.6411 | 12.0468 |
| timebin_short | 9.7262 | 9.6636 | 14.0954 |

## Answers

1. Label mismatch is excluded by constant filename-derived nominal labels and fixed physical members.
2. C5 methane 225 repeat 1 has 40,423 rows and 6.72% empty 100-ms bins; repeat 2 has 59,955 rows and no empty bin. This is more than light 100-Hz jitter.
3. Continuous interpolation is not selected here; P6 is a diagnostic-only comparison.
4. The first largest channel difference is listed in P1.
5. G0 has a material diagnostic effect: replacing only G0 in the C5 time-aware response changes oracle RMSE from 21.0622 to 10.7443 ppm; this is not formal causal approval.
6. P3 reports bucketed errors and Spearman coefficients; `interpolated_ratio > 0.10` has RMSE 55.974 ppm versus 11.313 ppm at zero.
7. C5 methane 225 repeat 1 is a raw-data anomaly candidate; repeat 2 is near complete. P0/P4/P6 retain both rather than deleting either.
8. P6 table above gives all three targets.
9. No canonical production candidate is selected from target test.
10. Full 25-round comparison is not authorized by this audit alone.
