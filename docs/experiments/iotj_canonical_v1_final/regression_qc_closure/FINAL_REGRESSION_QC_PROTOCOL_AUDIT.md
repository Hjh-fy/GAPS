# Final regression/QC protocol audit

## Phase-0 preliminary verdict

`HARD_FAIL_LEGACY_CANONICAL_MIX`

The final post-hoc lifecycle chain is formally available for C5 only. C3 and C4
have historical canonical-v1 interleaved-A4 regression/QC assets, but no formal
final post-hoc classification endpoint. Those assets are not interchangeable.
In addition, the frozen H1 manifest points to
`client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`, whose source
arrays are `100x8`. The canonical-v1 arrays are `50x8`. The task explicitly
defines this legacy/canonical mix as a hard fail. Consequently, the C5
post-hoc classification endpoint remains valid, but the derived R84 and QC
numbers are diagnostic only and cannot close the final canonical pipeline.
C3/C4 and pooled cross-target conclusions are also fail-closed.

## Audited C5 chain

| Item | Frozen value |
|---|---|
| Final post-hoc checkpoint | `posthoc_a0t_full_c5.pth` |
| Checkpoint SHA256 | `857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c` |
| Ordered state fingerprint | `4c7f13115e8a16c26e615f0f501cbd08a990e733bac2ac368c25b5533f53c2ef` |
| Source checkpoint SHA256 | `2d114a8ae23fcdea574d1e7c64e638620f60e49560da594397187bd5de1505fa` |
| Source state fingerprint | `cad6726ec29fb574314a5f2a45ed9800d1d90906b81cbd3ba8f8efb48a0df5d7` |
| Adaptation endpoint | fixed step 100, Adam 5e-4, seed 42 |
| Canonical dataset aggregate | `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6` |
| C5 calibration / test | 320 / 1360 windows |
| H1 manifest SHA256 | `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc` |
| H1 training input | historical 10 Hz, `100x8` |
| canonical-v1 input | 5 Hz, `50x8` |
| C5 R84 alphas | gas 0: 1.0; gas 1: 0.01; gas 2: 10.0; gas 3: 0.1 |
| Alpha selection in this stage | none |
| Target test used for selection | false |

The existing numerical endpoint is
`results/iotj_canonical_v1_method_breakthrough_20260811/phase3_posthoc_argmax/retry3`.
It was locked after calibration and before sealed-test evaluation. Its reported
C5 endpoint is Accuracy `0.9764705882`, Macro-F1 `0.9765440505`, and diagnostic
R84 S_ALL RMSE `28.0574962` ppm. Only the classification metrics are unaffected
by the H1 preprocessing conflict.

## QC lifecycle ruling

The historical deployment QC lock is tied to the historical interleaved-A4
classifier risk distribution and therefore must not be copied numerically to
the final post-hoc endpoint. The *formula and registered coverage policy* remain
frozen. Phase 1 may compute new component p95 scales and coverage quantiles from
the exact final post-hoc C5 calibration predictions only. This is the normal
calibration commissioning operation specified by the frozen policy, not QC
formula or target-test threshold tuning.

## Leakage and reproducibility gates

- Target-test labels may appear only in final metric computation and event
  capture after all model and QC locks exist.
- Ridge fitting and alpha selection must not use target test. No alpha search is
  authorized.
- QC scales and thresholds use calibration risk fields only; no calibration
  concentration label is required for the label-free risk score.
- Ordered state-content fingerprint establishes checkpoint identity; whole-file
  SHA256 is retained for provenance.
- Calibration/test sample identities and counts must match canonical manifests.
- Every derived prediction and manifest receives a SHA256 index.

## Deployment-package impact

The currently marked `FINAL_DEPLOYED_RUNTIME` package contains the historical
interleaved A4 classifier. It remains valid evidence for that historical package
but is not the final post-hoc lifecycle runtime package. The final audit must
report this mismatch; it must not silently relabel the historical package.

## Producer-commit limitation

The canonical preprocessing, H1, and Pi5 package manifests record producer
commits. The source-classifier, post-hoc checkpoint, R83, and Phase-3 R84
manifests preserve experiment IDs and hashes but do not embed a producer-commit
field. They are recorded as `UNRECORDED_IN_ASSET` in
`phase0_provenance/artifact_provenance.csv`; the audit does not fabricate a
commit value. This is a provenance-metadata limitation, not a numerical hash
failure.
