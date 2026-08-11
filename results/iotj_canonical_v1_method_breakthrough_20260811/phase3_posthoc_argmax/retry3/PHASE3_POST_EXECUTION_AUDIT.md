# Phase 3 post-execution audit

## Verdict: PASS

- Dataset aggregate SHA256 matches canonical-v1:
  `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Classifier is exact immutable I0+B20 step100 checkpoint SHA256
  `857f3954003bffad1af716002a1bd2915923389faec31b69f5c72e563aaa212c`.
- H1 manifest SHA256 is unchanged:
  `d32217a30f491ba46be436f3baf469b764b54a08d4d542b4eb71dbc007338ecc`.
- C5 Ridge alphas remain `{0:1.0, 1:0.01, 2:10.0, 3:0.1}`; no search ran.
- Calibration count is 320 and R84 model SHA256 is locked before test opening.
- Endpoint manifest records `target_test_used_for_selection=false`.
- Four required scopes and all per-gas/per-concentration tables are present.
- Recovery after the report renderer defect was artifact-only; classifier
  inference, R84 fitting, and C5 target-test evaluation were not rerun.
- Failed attempts remain separately preserved and are not evidence endpoints.

Phase 3 is approved as the immutable `POSTHOC_ARGMAX_BASELINE` input to the
already-registered Phase 4 test.
