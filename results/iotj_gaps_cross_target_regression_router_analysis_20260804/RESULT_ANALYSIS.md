# Cross-target regression capability analysis

This analysis reuses three frozen GAPS classification routes and one frozen Federated-H1 source regression head. No target regression model was fitted and no calibration or test-based selection was performed.

| Target | Classification acc. | Macro-F1 | Routed RMSE | Route-correct RMSE | Oracle-route RMSE | Route penalty | Routed NRMSE | Misroute rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C3 | 0.9897 | 0.9897 | 67.735 | 66.374 | 66.443 | +1.293 | 0.3857 | 0.0103 |
| C4 | 0.9906 | 0.9906 | 74.709 | 73.903 | 73.742 | +0.967 | 0.4145 | 0.0094 |
| C5 | 0.9846 | 0.9845 | 69.464 | 69.347 | 69.061 | +0.403 | 0.4798 | 0.0154 |

## Findings

- Classification routing is already strong on all targets (98.46%–99.06% accuracy), but source-only H1 routed RMSE remains 67.74–74.71 ppm and overall R2 is negative for all three targets.
- Replacing predicted routes with post-hoc true-class routes changes RMSE by only 0.40–1.29 ppm (0.6%–1.9% of routed RMSE). Thus the dominant observed limitation is source-to-target concentration mapping, not classification routing.
- CO is the largest absolute-RMSE slice on C3 and C4. On C5, CO has the largest absolute RMSE, while ethanol and ethylene have the largest class-range-normalized errors.
- The separately reported C5 A4+R84 personalized result is 12.855 ppm versus 69.464 ppm for GAPS+source-only H1 (81.5% lower descriptively). This is not a single-factor effect because both router identity and target-calibrated regression protocol differ.

Interpretation boundary: the routed value is deployable under the frozen source-only H1 path; the oracle-route value uses the true class only after evaluation and is diagnostic. Cross-target differences combine sensor-domain shift and concentration-distribution differences, so they are descriptive rather than a causal ranking of devices.

The existing C5 A4+R84 target-personalized result is preserved in a separate reference CSV and must not be pooled with the no-fit H1 rows.
