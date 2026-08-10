# Gate 1 Experiment Audit

## Audit scope and intended claim

Audit whether Gate 1 supports a real new-node post-hoc commissioning lifecycle and whether A4 or Target-head adds value beyond supervised A0T-full.

## Compared experiments

| Experiment ID | Split | Model | Source state | DA | Calibration | Seed | Provenance |
|---|---|---|---|---|---|---|---|
| CAN-V1-MR-G1-SOURCE | canonical C1/C2/C5 test | canonical TCN | `cad672...f5d7` | none | none | 42 | source run manifest |
| CAN-V1-MR-G1-A0T-FULL | same tests | same | `cad672...f5d7` | target CE | C5 calibration 320 | 42 | endpoint manifest and hash |
| CAN-V1-MR-G1-A4 | same tests | same | `cad672...f5d7` | registered A4 | same C5 320 | 42 | endpoint manifest, activity audit, hash |
| CAN-V1-MR-G1-HEAD | same tests | same | `cad672...f5d7` | target CE | same C5 320 | 42 | endpoint manifest and hash |

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Required action | Status |
|---|---|---|---|---|---|---|
| G1-F01 | informational | Same initialization | all manifests contain the same source state fingerprint | intended single mechanism comparison | none | closed |
| G1-F02 | informational | Fixed budget | all adapted endpoints are step 100, Adam 5e-4, batch 32, seed 42 | no result-driven budget change | none | closed |
| G1-F03 | informational | Test gate | `SEALED_TEST_OPEN.json` was created only after all three completion locks | no test-driven checkpoint selection | preserve marker | closed |
| G1-F04 | informational | C5 identities | calibration=320 and test=1360 canonical identities | same commissioning/test population across methods | preserve hashes | closed |
| G1-F05 | major | A4 completeness | several registered losses lack interleaved prototype/residual inputs | A4 result tests true post-hoc availability, not full interleaved state | label inactive terms and avoid claiming intrinsic loss inferiority | closed with limitation |
| G1-F06 | informational | Source retention semantics | adapted checkpoints were evaluated on source, while source global remains immutable | retention is diagnostic, not operational damage | keep distinction in claims | closed |
| G1-F07 | major | Uncertainty | seed42 only | no across-seed CI or significance claim | report descriptive fixed-seed evidence | closed with limitation |

## Leakage assessment

Adaptation received C5 calibration X/class; A4 also received calibration phase. No concentration was used. The adaptation request type has no target-test loader and rejects any target-test manifest. Test arrays were loaded only after all fixed endpoints existed. No target-test metric selected a hyperparameter, step, or checkpoint.

## Baseline, completeness, and reproducibility assessment

The comparison includes the required source-only, full target CE, registered A4, and lightweight head endpoints. Checkpoint, calibration manifest, test manifest, predictions, and result files are hashed. All endpoints are reproducible at the frozen seed and configuration. The study is not a multi-seed superiority test.

## Verdict: approved with stated limitations

Gate 1 supports the narrow post-hoc lifecycle claim and the pre-registered method decisions. It does not support A4 superiority, Target-head competitiveness, strict non-overlap robustness, or population-level uncertainty claims.

