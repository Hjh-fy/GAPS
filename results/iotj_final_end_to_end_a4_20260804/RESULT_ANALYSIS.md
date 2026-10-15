# Result analysis

## Confirmed endpoint

The final router is the frozen server-centric A4 round-25 checkpoint (seed 42, LE1). Its one-time C5 test result is accuracy 99.338% and macro-F1 99.339%. C3 and C4 remain blocked because same-protocol A4 endpoints are unavailable; full-GAPS endpoints were not substituted.

## Concentration estimation

- 83-D sensor only: end-to-end RMSE 16.020 ppm, NRMSE 0.10102.
- 84-D + federated H1 (proposed): end-to-end RMSE 12.855 ppm, NRMSE 0.08535; route-correct RMSE 11.462 ppm.
- 86-D + H1/H2/H3 (diagnostic): end-to-end RMSE 12.696 ppm, NRMSE 0.08313. It is slightly better descriptively but is not promoted over the simpler federated-H1 design.
  - Ethanol: RMSE 10.397 ppm, NRMSE 0.09242.
  - CO: RMSE 16.089 ppm, NRMSE 0.07151.
  - Ethylene: RMSE 12.780 ppm, NRMSE 0.11360.
  - Methane: RMSE 11.433 ppm, NRMSE 0.05081.

These are fixed-endpoint seed-42 descriptive results, not cross-seed inferential estimates.

## Label-free QC

The final QC score uses three label-free components normalized by calibration-only p95 scales. HC90 targets 90% calibration coverage and realizes 87.87% test coverage with NRMSE 0.06765; HC95 realizes 94.78% with NRMSE 0.06762. Full coverage NRMSE is 0.08535. The matched random-reference HC90 mean NRMSE is 0.08527. HC90/HC95 denote targeted auto-output coverage, not accuracy.

The initial maximum-component QC attempt was excluded after a calibration-only tie audit showed that clipped source-prior outputs collapsed most high-coverage thresholds. Its artifacts remain under `qc_attempt1_degenerate` for traceability and are not manuscript evidence.

## Deployment boundary

The three-machine Flower run completed 25/25 rounds in 77.4 min with target test closed during training. Raspberry Pi 5 measurements are independent deployment benchmarks. Flower bytes are measured application payload; federated-H1 bytes are theoretical serialized exchange, so Fig. 8 does not present them as transport-controlled measurements.
