# GAPS Canonical-v1 Next Method Validation Plan

## Status and routing

- Current stage: complete; final Story E audit approved.
- Largest evidence gap disposition: multi-seed source-DG was not confirmed and the fixed expected-cost router was not supported.
- Executed order: Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Story E audit.
- Existing Gate A/B/C artifacts and every canonical-v1 dataset/checkpoint are read-only.

## Alternatives considered

1. **Sequential gated execution (selected):** preserves the Phase 3 method-identity rule and calibration/test boundary.
2. Run all phases in parallel: rejected because Phase 3 depends on the registered Phase 2 decision.
3. Run cost-aware routing first: rejected because it would skip the required DG stability and commissioning bridge evidence.

## Global frozen controls

- Dataset: `dataset/iotj_canonical_v1`, 5 Hz, 50x8, canonical aggregate SHA256 `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Source role view for S4: the already frozen C1+C2+C3+C4 role view used by Gate A.
- Final target: C5. C5 X, Y, statistics, calibration, and test are unavailable to every source-DG training API.
- Source training: 25 rounds, LE=1, batch 32, Adam 5e-4, fixed round25, no checkpoint selection.
- DG-P: exact Gate A prototype mechanism and `lambda_proto=0.05`; no search.
- Post-hoc Full A0T: 100 steps, Adam 5e-4, batch 32, fixed step100.
- Calibration budgets: frozen nested B20=320 windows (8/stratum) and B05=80 windows (2/stratum), all 40 class-by-concentration strata covered.
- R84: `R84_FED_H1_fixed_alpha`; C5 alphas `{0:1.0,1:0.01,2:10.0,3:0.1}`; original canonical H1/FedRidge source pool remains unchanged.
- No target-test tuning, no new optimizer/lambda/rank/alpha/threshold search, no C3/C4 expansion, no MME integration in this run.

## Phase 1: S4 DG multi-seed confirmation

Hypothesis `H-P1-DG-STABILITY`: under matched S4 source-only training, DG-P improves C5 zero-shot Macro-F1 consistently across seeds 41, 42, and 43.

- Reuse seed42 Gate A endpoints; execute only seed41/43 x FedAvg/DG-P.
- Report source-pooled and C5 Accuracy, Macro-F1, NLL, ECE, per-class recall/F1.
- Paired difference is `DG-P - FedAvg` within seed.
- `SOURCE_DG_SUPPORTED`: all three paired differences >0, mean difference >=0.03, and paired-difference sample SD <=0.05.
- `SOURCE_DG_UNSTABLE`: mean difference >0 but any seed reverses or paired-difference SD >0.05.
- Otherwise `SOURCE_DG_NOT_CONFIRMED`.
- Stop after exactly seeds 41/42/43; never add seeds in response to the result.

## Phase 2: DG-to-commissioning bridge

Hypothesis `H-P2-DG-COMMISSIONING`: an S4 DG-P source initialization retains value after fixed Full A0T commissioning, especially under B05.

- I0: S2 FedAvg; I1: S4 FedAvg; I2: S4 DG-P.
- B20/B05 x I0/I1/I2. Reuse only endpoints whose source fingerprint, budget manifest, adaptation protocol, and fixed step match exactly.
- Evaluate the sealed C5 test only after every new endpoint is locked.
- A post-hoc gain of >=0.01 Macro-F1 is “meaningful”; equality band is absolute difference <0.01.
- `DG_TO_COMMISSIONING_SUPPORTED`: I2-I1 >=0.01 at both B20 and B05.
- `DG_LOW_BUDGET_VALUE_SUPPORTED`: I2-I1 >=0.01 at B05 and <0.01 at B20.
- `DG_ZERO_SHOT_ONLY`: both post-hoc I2-I1 differences have absolute value <0.01 while the frozen zero-shot DG gain remains >=0.01.
- `SOURCE_DIVERSITY_ONLY`: I1 materially exceeds I0 in at least one budget and I2 adds <0.01 in both.
- Otherwise `DG_TO_COMMISSIONING_NOT_SUPPORTED`.

## Phase 3: post-hoc R84 baseline

Hypothesis `H-P3-BASELINE`: the registered source-method identity can produce one immutable Posthoc Argmax R84 baseline without changing the quantitative source prior.

- If Phase 2 is `DG_TO_COMMISSIONING_SUPPORTED` or `DG_LOW_BUDGET_VALUE_SUPPORTED`, use I2+B20.
- Otherwise select the simplest effective fixed source identity in order I0 then I1, where “effective” means its B20 post-hoc Macro-F1 is within 0.01 of the best B20 method.
- This rule selects a method identity, never a round/step/checkpoint using target test.
- Fit R84 only on C5 B20 calibration with frozen alphas and unchanged canonical H1.
- Produce `POSTHOC_ARGMAX_BASELINE` with classification and S_ALL/S_CC/Oracle_ALL/Oracle_CC regression metrics.

## Phase 4: cost-aware routing direct test

Hypothesis `H-P4-COST-ROUTER`: the parameter-free expected downstream-cost router improves S_ALL regression without materially degrading gas classification.

- Build `C(c,j)` only from B20 calibration by forcing all four R84 routes.
- Primary cost: `max(0, mean(SE_forced_j - SE_correct_c))`, diagonal zero.
- Router: `argmin_j sum_c p(c|x) C(c,j)`; no lambda and no threshold.
- Lock and hash the matrix before any test labels/errors enter the routing evaluation.
- Grouped bootstrap by raw filename, 2000 replicates, seed42.
- `COST_AWARE_ROUTING_SUPPORTED`: relative RMSE improvement >=5%, Macro-F1 drop <=0.005, and bootstrap probability of negative RMSE difference >0.5. CI crossing zero is reported but is not an additional hard gate.
- `COST_AWARE_ROUTING_MODEST`: relative RMSE improvement in [2%,5%), Macro-F1 drop <=0.005.
- `QUANTITATIVE_GAIN_WITH_CLASSIFICATION_COST`: relative RMSE improvement >=2% and Macro-F1 drop >0.005; create proposal only.
- Otherwise `COST_AWARE_ROUTING_NOT_SUPPORTED` and stop the direction.

## Final story and stop rule

- Story A: multi-seed DG supported, DG commissioning value supported, and cost routing supported.
- Story B: source diversity explains commissioning but cost routing is supported.
- Story C: DG is zero-shot only and cost routing is supported.
- Story D: DG is supported but cost routing is not.
- Story E: DG unstable/not confirmed and cost routing not supported.
- `GO_MME_INTEGRATION` only for Story A and only as a future proposal; do not execute it here.
- `FREEZE_METHOD` for Stories B/C/D.
- `STOP_NEW_ALGORITHMS` for Story E.

## Required verification

Each phase requires relevant pytest, `python -m compileall`, checkpoint/data/manifest SHA checks, leakage audit, one commit, and push to `codex/iotj-final-classification-le1`.
