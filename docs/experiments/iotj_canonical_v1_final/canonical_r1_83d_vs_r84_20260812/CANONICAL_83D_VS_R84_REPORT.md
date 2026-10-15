# Canonical 83D versus R84 report

Status: `completed`; audit: `PASS`; registered decision: `CANONICAL_R84_DEVICE_DEPENDENT`.

On the routed all-sample (`S_ALL`) endpoint, R84 reduced RMSE relative to target-only 83D on C3 from 10.795 to 9.445 ppm (delta -1.350 ppm; grouped-bootstrap 95% CI [-1.956, -0.849]) and on C4 from 13.544 to 13.281 ppm (delta -0.263 ppm; 95% CI [-0.397, -0.122]). On C5, RMSE changed from 21.320 to 20.611 ppm (delta -0.709 ppm), but its 95% CI [-1.862, 0.676] crossed zero. The pooled RMSE delta was -0.793 ppm (95% CI [-1.241, -0.317]); the pooled NRMSE-range interval crossed zero, and gas-specific effects were heterogeneous. These results support a device-dependent benefit, not a universal R84 improvement.

No registered severe-collapse event was present. The dedicated C5 Methane 225 ppm repeat-1 slice was retained: S_ALL RMSE was 113.921 ppm for 83D and 104.162 ppm for R84, with `n=17`; this small diagnostic slice is descriptive rather than a standalone generalization claim.

Evidence root: `results/iotj_canonical_v1_final/canonical_r1_83d_vs_r84_20260812`. It contains 36 files totaling 11,173,967 bytes. `sha256_index.json` indexes 34 evidence files and a fresh post-run check found zero mismatches; the index SHA256 is `778af3e0f9d57b95211fee012145e8be87c103b1005647e1b2ab5a455e444aa7`. The two intentionally unindexed retained files are the index itself and `COMPLETE.json` (SHA256 `4bc5fa370da9814523cf090624baab1775166862cee16bc07acbf25f2b135d0d`).

Limitation: calibration and test windows may share raw-file/time neighborhoods under the frozen split. R2 was triggered by the registered device-dependent outcome, but no R2 result is reported here.
