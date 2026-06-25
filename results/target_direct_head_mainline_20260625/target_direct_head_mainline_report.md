# Target Direct-Head Mainline Confirmation

Date: 2026-06-25

Scope: C12 -> C345 target test, no-QC full-set. This report consolidates formal target Ridge, target MLP, hybrid profile selection, H8 CO-specialist, and H2.3 runtime parity evidence.

- Summary CSV: `results/target_direct_head_mainline_20260625/target_direct_head_mainline_summary.csv`
- H2.3 deployment bundle: `results/deployment_h2_3_mlp_ridge_candidate_20260624`
- H2.3 runtime validation dir: `results/runtime_validation_h2_3_mlp_ridge_candidate_20260624`
- H2.3 profile JSON: `results/target_direct_head_mainline_20260625/h2_3_profile.json`
- H2.3 profile export check artifact: `results/target_direct_head_mainline_20260625/c12_c345_h2_3_profile_export.json`
- H8 switch audit: `results/h8_switch_rule_audit_20260625/h8_switch_rule_audit_report.md`
- H8 calibration-only selector: `results/h8_calibration_selector_20260625/h8_calibration_selector_report.md`
- H8 analysis profile: `results/h8_calibration_selector_20260625/h8_pred_co_source_aug_selector_profile.json`
- H8 deployment bundle: `results/deployment_h8_source_aug_candidate_20260625`
- H8 runtime validation dir: `results/runtime_validation_h8_source_aug_candidate_20260625`
- H8 runtime equivalence: `results/equivalence_h8_source_aug_candidate_20260625/equivalence_summary.json`

## Main Metrics

| candidate | status | ALL | NRMSE | C3 CO | C4 CO | C5 CO | C3 high | C4 high | C5 high | nonCO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 baseline final | baseline | 27.34 | 0.1578 | 33.70 | 56.59 | 46.12 | 41.70 | 95.32 | 60.00 | 19.00 |
| H1 Ridge + C4 rescue | reference | 20.15 | 0.1405 | 18.70 | 22.02 | 30.67 | 24.49 | 34.24 | 31.66 | 19.07 |
| H2 MLP + C4 rescue | reference | 18.94 | 0.1342 | 16.15 | 22.93 | 31.03 | 20.02 | 32.11 | 39.41 | 17.61 |
| H2.2 MLP C3 + Ridge C4/C5 | deployment candidate archived | 19.89 | 0.1397 | 16.15 | 22.02 | 30.67 | 20.02 | 34.24 | 31.66 | 19.09 |
| H2.3 MLP C3 + Ridge C4 + C5-grid MLP | current mainline | 18.62 | 0.1326 | 16.15 | 22.02 | 26.85 | 20.02 | 34.24 | 34.82 | 17.83 |
| H8 pred-CO source-aug else H2.3 | CO-specialist candidate | 18.47 | 0.1354 | 14.97 | 19.76 | 23.69 | 19.93 | 32.22 | 27.54 | 18.38 |

## Delta vs Original Baseline

| candidate | ALL | NRMSE | C4 high | C5 high | nonCO |
| --- | --- | --- | --- | --- | --- |
| A0 baseline final | 0.00 | 0.0000 | 0.00 | 0.00 | 0.00 |
| H1 Ridge + C4 rescue | -7.19 | -0.0173 | -61.07 | -28.34 | 0.07 |
| H2 MLP + C4 rescue | -8.40 | -0.0236 | -63.21 | -20.60 | -1.39 |
| H2.2 MLP C3 + Ridge C4/C5 | -7.45 | -0.0182 | -61.07 | -28.34 | 0.09 |
| H2.3 MLP C3 + Ridge C4 + C5-grid MLP | -8.71 | -0.0253 | -61.07 | -25.18 | -1.17 |
| H8 pred-CO source-aug else H2.3 | -8.87 | -0.0224 | -63.10 | -32.46 | -0.62 |

## H8 vs H2.3

| metric | H8 - H2.3 |
| --- | --- |
| ALL RMSE | -0.15 |
| ALL NRMSE | 0.0029 |
| C4 CO | -2.26 |
| C5 CO | -3.16 |
| C5 CO high | -7.28 |
| nonCO ALL | 0.55 |

## Artifact Checklist

| artifact | path | status |
| --- | --- | --- |
| H2.3 deployment bundle | results/deployment_h2_3_mlp_ridge_candidate_20260624 | ok-dir |
| H2.3 runtime validation | results/runtime_validation_h2_3_mlp_ridge_candidate_20260624 | ok-dir |
| H2.3 runtime equivalence | results/equivalence_h2_3_mlp_ridge_candidate_20260624/equivalence_summary.json | ok-file |
| H2.3 profile JSON | results/target_direct_head_mainline_20260625/h2_3_profile.json | ok-file |
| H8 deployment bundle | results/deployment_h8_source_aug_candidate_20260625 | ok-dir |
| H8 runtime validation | results/runtime_validation_h8_source_aug_candidate_20260625 | ok-dir |
| H8 runtime equivalence | results/equivalence_h8_source_aug_candidate_20260625/equivalence_summary.json | ok-file |
| H8 selector profile | results/h8_calibration_selector_20260625/h8_pred_co_source_aug_selector_profile.json | ok-file |

## Reproduction Workflow

| step | purpose | command |
| --- | --- | --- |
| 1 | Target Ridge direct | python run_formal_target_ridge_auto_v2_eval.py |
| 2 | Target MLP direct | python run_formal_target_mlp_auto_v2_eval.py |
| 3 | Hybrid H2 profile selection | python run_hybrid_regression_head_selection.py |
| 4 | H2.3 deployment export | python export_hybrid_mlp_ridge_deployment_candidate.py --candidate h2_3 --output results/deployment_candidates_20260624/c12_c345_h2_3_mlp_ridge_candidate.json |
| 5 | H8 CO-specialist analysis | python run_co_only_source_aug_hybrid_eval.py --output-dir results/co_only_source_aug_hybrid_stratcalval_20260625 |
| 6 | H8 selector profile | python select_h8_profile_from_calibration.py |
| 7 | H8 deployment export | python export_h8_source_aug_deployment_candidate.py |
| 8 | Runtime validation | python validate_rich_residual_runtime_candidate.py --deployment-dir <deployment_dir> --output-dir <runtime_validation_dir> |
| 9 | Mainline summary | python summarize_target_direct_head_mainline.py |

## H8 Selector Status

- The H8 switch rule is deployment-visible: switch to the CO specialist when `pred_class == CO`.
- Calibration split audit supports the rule: overall precision 0.991, false-positive rate 0.009, CO recall 0.976, CO-high recall 0.960.
- A calibration-only selector was added and enables H8 for C3/C4/C5 because all three clients pass switch-support thresholds and source-augmented CO validation RMSE improves over rich-only target Ridge.
- H8 runtime/export support has been implemented and parity has been verified.

## Runtime Parity

H2.3:
- rows compared: 5400
- missing in analysis/runtime: 0 / 0
- mismatch rows: 0
- max abs diff: 1.1368683772161603e-13
- mean abs diff: 7.61761024629474e-15

H8:
- rows compared: 5400
- missing in analysis/runtime: 0 / 0
- mismatch rows: 0
- max abs diff: 1.1368683772161603e-13
- mean abs diff: 1.1339406779808525e-14

## Decision

- Promote H2.3 as the current balanced mainline: it gives a large gain over the original baseline and already has deployment/runtime equivalence.
- Keep H8 as a CO-specialist candidate, not the default mainline: it improves CO and high-CO, but worsens ALL NRMSE and nonCO versus H2.3.
- H8 now has calibration-only selector support and runtime parity, so it can be treated as a deployable CO-specialist candidate.
- Export/profile parameterization has started: `export_hybrid_mlp_ridge_deployment_candidate.py` now accepts `--profile-json` while preserving `--candidate h2_2/h2_3` compatibility.
- Mainline decision remains H2.3 vs H8: H8 improves CO/high-CO and ALL RMSE slightly, but worsens ALL NRMSE and nonCO versus H2.3.
