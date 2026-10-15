# C5 split-swap diagnostic analysis

## Outcome

The legacy result is exactly reproducible, but the role-aware R84 model does
not recover the legacy score when it is applied to the legacy test.

| Configuration | S_ALL RMSE | S_CC RMSE | Status |
|---|---:|---:|---|
| Role-aware checkpoint + role-aware R84 + role-aware test | 27.2214 | 20.2864 | formal role-aware result |
| Role-aware checkpoint + fixed role-aware R84 + legacy test | 37.6458 | 23.6486 | leakage-risk cross-protocol diagnostic |
| Legacy checkpoint + legacy-calibrated R84 + legacy test | 16.0928 | 11.7965 | exact legacy replay |

The exact legacy replay produced byte-identical hashes for
calibration_alpha_selection.csv, regression_summary.csv,
regression_per_gas.csv, regression_route_decomposition.csv,
regression_per_concentration.csv, and test_records.csv.

## What changed

The two C5 roots have the same 320/1360 counts, but they are not numerically
identical datasets. After concatenating calibration and test and converting
both feature arrays to float32, row fingerprints over feature, class,
regression, and phase values had 0/1680 exact intersections. The roots also
have different feature dtypes, metadata schemas, feature hashes, label hashes,
and normalization-stat hashes.

The regression difference is therefore not an RMSE implementation defect.
It reflects a different calibration/model/data-version combination. The
role-aware Methane model remains the principal failure source: on the legacy
test its correct-route RMSE is 40.7073 ppm.

## Boundary

The cross-protocol prediction cannot become paper evidence because the
role-aware checkpoint consumed role-aware calibration windows whose
disjointness from the legacy test is not established. It is diagnostic only.
