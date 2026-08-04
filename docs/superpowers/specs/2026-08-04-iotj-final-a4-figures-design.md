# IoT-J Final A4 Regression, QC, and Deployment Figures Design

## Scope

Produce the final manuscript Fig. 5--Fig. 8 from frozen, traceable evidence. Existing classification results and source Flower checkpoints are read-only. The only classifier replay permitted is inference from the fixed round-25 server-centric A4 endpoint; no classifier training or loss/weight search is permitted.

## Evidence boundary

- C5 has a complete A4 round-25 endpoint in `results/iotj_final_classification_le1_20260804/FCL-E4-A4` and can proceed through regression, QC, deployment packaging, and figure generation.
- No separately identifiable C3/C4 A4 endpoint is present locally or in the audited server result root. C3/C4 entries therefore remain `unknown`/`blocked`; full-GAPS C3/C4 endpoints must not be relabeled as A4.
- Fig. 5--Fig. 8 are C5 end-to-end figures. Cross-target classification remains confined to the already generated classification evidence used by Fig. 3.
- Existing result directories, checkpoints, calibration locks, and benchmark outputs remain read-only. New files are written only under `results/iotj_final_end_to_end_a4_20260804`.

## Phase 1: classifier freeze

Create a final classifier manifest with exact protocol fields, the whole-file SHA-256 for provenance, and an ordered state-content fingerprint for checkpoint identity. Record C5 as complete and C3/C4 as blocked without substituting another method. Copy no checkpoint and perform no training.

## Phase 2: final regression replay

Use the C5 A4 checkpoint independently on calibration and test windows. Reuse the frozen source prior models and the existing 83-D sensor-statistic feature builder. Fit three target Ridge variants using calibration only:

1. `R83_TARGET_ONLY`: 83-D sensor statistics;
2. `R84_FED_H1`: 83-D sensor statistics plus the fixed Federated-H1 source prior;
3. `R86_ALL_PRIORS`: 83-D sensor statistics plus fixed H1/H2/H3 source priors.

Alpha selection uses only the frozen calibration internal 60/20 per-gas split and the existing fixed grid. All models are refit on the 80 calibration windows per gas after selection. The fixed test endpoint is opened only after the calibration lock is persisted. Test predictions are evaluated both end-to-end (`S_ALL`) and on the route-correct subset (`S_CC`).

Window records include client/sample identity, true and predicted gas, route correctness, true and predicted ppm, absolute and squared error, classifier confidence/margin/entropy, all three regression outputs, and label-free QC risk features. Class probability fields come only from A4 inference.

## Phase 3: final QC

Use `R84_FED_H1` as the proposed regression output. Define one fixed label-free risk score before test evaluation as the maximum of normalized classifier uncertainty, normalized 83-D versus 84-D prediction disagreement, and normalized H1/H2/H3 source-prior disagreement. Calibration labels may evaluate the risk score but do not change its formula.

For coverages 70%, 72.5%, ..., 100%, derive per-client risk quantiles from calibration only and apply them unchanged to test. At every coverage, compare QC risk-ranked retention with 1,000 random selections using fixed seed 20260804. Report accepted RMSE/NRMSE/MAE and the capture rates for misroutes, errors at least 40 ppm, and the top 10% largest errors. HC90 and HC95 are reliability operating points targeting auto-output coverage, not classification accuracy.

## Phase 4: deployment and system evidence

Build HC90 and HC95 deployment output tables only after QC is frozen. The runtime manifest contains classifier, target Ridge, source-prior, normalization/routing, QC threshold, schema, and hash provenance. System panels reuse the audited communication and Raspberry Pi 5 benchmark tables; no model training or new optimizer search is allowed. Real-device validation is represented by the completed three-machine Flower run and fixed endpoint audit, not by an unverified hardware photograph.

## Figures

- Fig. 5: overall S_ALL/S_CC concentration error plus per-gas regression results for the three variants.
- Fig. 6: 83-D source-prior ablation plus the frozen calibration-budget curve. Different split protocols are visibly separated and not treated as a single-factor causal comparison.
- Fig. 7: QC coverage--NRMSE curve, 1,000-run random reference band, and HC90/HC95 operating-point summary.
- Fig. 8: communication payload, Raspberry Pi 5 latency/throughput/RSS, and real-device execution/final package validation.

All figures use IEEE double-column width, vector PDF plus 600-DPI PNG, sans-serif typography, lowercase panel labels, Okabe--Ito colors, redundant line styles/markers, and source-data CSV files.

## Fail-closed checks

- Reject output reuse when checkpoint, dataset, row universe, or schema identity differs.
- Reject overwriting a non-empty final output directory.
- Reject non-finite predictions, duplicate/misaligned sample IDs, or missing class probabilities.
- Reject any QC threshold or method selection that reads test labels.
- Preserve all blocked/unknown provenance rather than substituting a convenient historical result.

