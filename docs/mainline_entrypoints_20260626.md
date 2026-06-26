# GAPS Mainline Entrypoints and Role Map — 2026-06-26

This document is the current entrypoint map for the GAPS codebase. It intentionally separates **algorithm training**, **target-side calibration/selection**, **deployment export**, and **runtime validation** so future Codex/local work does not depend on outdated README commands.

## 0. Current Mainline Roles

| Component | Current role | Status |
|---|---|---|
| R3aK16 | Source-domain neural regression base model and baseline | keep as baseline/fallback |
| H2.3 | Balanced target-side performance mainline | default mainline |
| H8 | pred-CO source-augmented target Ridge specialist | CO-specialist candidate |
| H8 + formal C4 route rescue | CO/high-CO deployable specialist candidate | v2 runtime/export/schema/guardrail/parity validated |
| L1 | lightweight source head + full residual auto_v2 | deployment-lite candidate |
| L2/L3 | selector/lightweight diagnostics | diagnostic, not default mainline |
| QC | accept/review/reject deployment reliability layer | post-hoc only, not model capability metric |

## 1. Flower Classification Entrypoints

Use these for federated classification and fixed-DA classification checkpoint generation.

| File | Purpose |
|---|---|
| `gaps_flower/task.py` | Flower config, dataset loader helpers, classifier construction |
| `gaps_flower/client_app.py` | Flower client-side local classification training |
| `gaps_flower/server_app.py` | Flower server entrypoint |
| `gaps_flower/strategy.py` | GAPS strategy, prototype/stat aggregation, server-side adaptation hooks |
| `gaps_flower/domain_adaptation.py` | server-side fixed/adapted classifier DA logic |

### Expected classification output

- adapted classifier checkpoint, usually `server_latest_adapted.pth`
- class logits/top1 route used by regression/deployment
- classification metrics should be reported separately from regression metrics

## 2. R3aK16 Regression Base Entrypoints

| File | Purpose |
|---|---|
| `gaps_flower/regression_task.py` | source regression config, R3aK16 initialization, source-client regression FedAvg |
| `model.py` | TCN backbone, regression-only Transformer, DCT16 response branch, per-gas regression heads |
| `utils.py` | concentration normalization, model factory, legacy evaluation utilities |
| `run_regression_head_ablation.py` | regression head matrix feature helpers and ablation support |

### Current R3aK16 interpretation

R3aK16 is not the final best ppm output. It is the source-domain neural regression base:

```text
fixed-DA classifier route
-> R3aK16 source regression base
-> target calibration/direct-head/specialist
-> runtime/QC
```

## 3. H2.3 / Target Direct-Head Entrypoints

| File | Purpose |
|---|---|
| `run_formal_target_ridge_auto_v2_eval.py` | formal target Ridge direct-head candidate with calibration-fit/validation/refit |
| `run_formal_target_mlp_auto_v2_eval.py` | formal target MLP direct-head candidate |
| `summarize_target_direct_head_mainline.py` | summarize H2.3/H8/H8+C4 mainline report |
| `select_target_profile.py` | profile selector scaffold/current direction-level profile selection |

### H2.3 role

H2.3 remains the **balanced default mainline** unless a specialist mode is explicitly requested and passes guardrails.

## 4. H8 / H8+C4 Deployment Candidate Entrypoints

| File | Purpose |
|---|---|
| `run_formal_c4_route_rescue_selector.py` | calibration-only C4 route-rescue gate selection |
| `export_h8_source_aug_deployment_candidate.py` | export original H8 source-aug deployment candidate |
| `export_h8_formal_c4_rescue_deployment_candidate.py` | export H8 + formal C4 rescue candidate; v2 preserves `max_conf_margin` |
| `validate_rich_residual_runtime_candidate.py` | validate runtime output against target test splits |
| `compare_h8_runtime_equivalence.py` | compare analysis prediction CSV and runtime output equivalence |
| `audit_h8_c4_guardrails.py` | audit formal C4 route-rescue hits and nonCO guardrails |
| `validate_feature_schema.py` | validate feature/runtime schema contract for deployment bundles |
| `benchmark_runtime_profiles.py` | benchmark exported bundle size and CPU latency |
| `select_target_profile.py` | emit formal balanced/co-priority/deployment-lite profile selection record |

### Current H8+C4 v2 validation status

The current H8+C4 v2 bundle has been re-exported and validated:

- feature/runtime schema: pass
- guardrail audit: pass
- runtime vs analysis parity: 5400 rows, 0 mismatch
- public runtime output includes `auto_output_ppm`

See `results/h8_c4_deployable_specialist_validation_20260626.md` for the compact evidence report.

## 5. Lightweight / Deployment-Lite Entrypoints

| File | Purpose |
|---|---|
| `run_source_lightweight_full_auto_v2_eval.py` | L1 lightweight source heads + full residual auto_v2 |
| `run_lightweight_l2_unified_selector.py` | L2 unified selector diagnostic |
| `run_l3_lightweight_hybrid_matrix.py` | L3 lightweight/hybrid matrix expansion |

### Lightweight role

L1/L2/L3 should not replace H2.3/H8 on performance alone. They need runtime/size/latency evidence to justify the deployment-lite role.

## 6. Bidirectional and Selector Entrypoints

| File | Purpose |
|---|---|
| `run_c45_c123_optimal_config_analysis.py` | reverse direction C45 -> C123 optimal profile analysis |
| `summarize_bidirectional_profile_selection.py` | summarize C12->C345 and C45->C123 profile selection |
| `select_target_profile.py` | formal deployment-mode selector for balanced / co-priority / deployment-lite |

Current responsibility split:

- `select_target_profile.py` owns deployable profile modes and guardrail/schema/parity evidence.
- `summarize_bidirectional_profile_selection.py` owns direction-level C12->C345 / C45->C123 reporting.
- `results/gaps_final_mainline_summary_20260626.md` is the current single-page mainline conclusion.

## 7. Deployment Runtime Entrypoints

| File | Purpose |
|---|---|
| `gaps_deploy/inference.py` | base deployment predictor: classifier -> R3aK16 -> auto_v2 -> risk -> QC |
| `gaps_deploy/final_runtime.py` | public runtime wrapper and output schema |
| `gaps_deploy/rich_residual.py` | H2.3/H8/L1 rich residual/direct-head runtime policy |
| `gaps_deploy/qc_policy.py` | risk scoring and accept/review/reject decision logic |

## 8. Known README cleanup requirement

The README should avoid listing missing/legacy scripts as current mainline commands. Use this document as the source of truth until README is updated.

Legacy or possibly local-only names that should not be used as current GitHub mainline unless re-added:

- `preprocessor_time_aware.py`
- `run_time_aware_raw_calibrated_qc_eval.py`
- `scripts/build_final_deployment_package.py`
- `scripts/validate_final_deployment_bundle.py`

## 9. Minimum acceptance before calling a candidate deployable

A candidate is deployable only if all conditions hold:

1. The selector uses calibration or calibration-validation only.
2. Test data is not used to select profile/gate/hyperparameters.
3. Runtime artifact exists and loads with `FinalDeployRuntime`.
4. Runtime vs analysis parity is checked: expected mismatch = 0, max_abs_diff < 1e-8.
5. Output schema is stable and documented.
6. QC is reported as post-hoc reliability, not as model capability selection.
