# GAPS canonical-v1 next-method validation decision

## Final decision: Story E — `STOP_NEW_ALGORITHMS`

| Phase | Registered question | Decision | Key evidence |
|---|---|---|---|
| 1 | Is S4 DG-P stable across seeds 41/42/43? | `SOURCE_DG_NOT_CONFIRMED` | Paired Macro-F1 gain mean `-0.04373`, SD `0.11988`; two of three seeds reverse. |
| 2 | Does DG initialization retain value after Full A0T commissioning? | `DG_TO_COMMISSIONING_NOT_SUPPORTED` | I2-I1 is `+0.01690` at B20 but `-0.01179` at B05. |
| 3 | Can the preregistered post-hoc identity establish a frozen R84 baseline? | `POSTHOC_ARGMAX_BASELINE_ESTABLISHED` | I0+B20 C5 Macro-F1 `0.976544`; S_ALL RMSE `28.05750` ppm. |
| 4 | Does calibration-only expected-cost routing improve the fixed baseline? | `COST_AWARE_ROUTING_NOT_SUPPORTED` | RMSE worsens by `0.31309` ppm (`-1.1159%` relative improvement), Macro-F1 drops `0.003019`, bootstrap P(ΔRMSE<0) `0.4245`. |

## Story A–E audit

- Story A: false — multi-seed DG and DG commissioning criteria both fail.
- Story B: false — the registered cost router is not supported.
- Story C: false — the registered cost router is not supported.
- Story D: false — DG is not confirmed and the cost router is not supported.
- Story E: true — DG is unstable/not confirmed and cost routing is not supported.

## Scientific interpretation

The seed42 DG improvement is not stable evidence of a general algorithmic
advantage. It also does not survive the commissioning bridge consistently.
Although Phase 3 confirms that misrouting can carry large quantitative error,
the preregistered parameter-free calibration cost matrix does not turn that
observation into a beneficial router on sealed C5 test data. High-concentration
RMSE worsens by `1.91649` ppm and high-concentration SSE rises by
`102569.88 ppm²`.

Accordingly, no MME integration, extra seed, target expansion, uncertainty
router, S4 FedRidge, or new algorithm search is authorized. The supported
paper direction remains the already validated lifecycle/evidence framework,
with the negative DG and cost-routing studies retained as limitations rather
than promoted as method components.

Final action: `STOP_NEW_ALGORITHMS`.
