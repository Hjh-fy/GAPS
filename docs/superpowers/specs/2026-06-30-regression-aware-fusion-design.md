# Regression-Aware Fusion Follow-Up Design

## Context

The current regression story is strong but structurally exposed: the best H2.3/H8+C4 results still depend heavily on target calibration direct-heads and the B0-dependent route-rescue/profile layer. That makes the federated classification backbone look useful mainly for route prediction, not for continuous ppm estimation.

Recent local evidence sharpens this point:

- The 2026-06-26 selector audit keeps `H2_3_R3aK16_current_mainline` as the balanced C12_to_C345 profile and `H8_plus_formal_C4_rescue` as the CO-priority profile.
- The F6 fixed-DA strong classification run produced official final adapted checkpoint `server_latest_adapted.pth` at round 25 and a useful diagnostic checkpoint `server_round_019_adapted.pth`.
- Round 25 is the official reporting line. Round 19 may be used only for best-checkpoint or mechanism diagnosis, not as the main result unless a calibration-only checkpoint selector is formalized later.
- Existing H2.3 no-B0 ablations show that the current direct-head feature dictionary contains target-window rich statistics, not B0/R3aK16/QC-risk ppm features. Removing B0-dependent rescue/profile behavior degrades C4 high-CO sharply.

## Design Goal

Build a clean follow-up experiment track that tests whether the latest classification backbone can contribute directly to regression through exported route confidence and embedding features, then uses the result to build H2.3+ feature fusion and an optional H8+ CO risk-gated specialist.

## Non-Goals

- Do not retrain a new regression-aware encoder in this first pass.
- Do not promote round 19 as the official checkpoint.
- Do not select any profile or gate from test metrics.
- Do not merge QC accepted-only performance into model capability metrics.
- Do not replace H2.3/H8+C4 with a large MoE architecture before the feature ablation is understood.

## Architecture

The follow-up keeps the current pipeline boundaries:

1. Flower classification backbone produces logits, probabilities, confidence metrics, `cls_feat`, and `reg_feat`.
2. Existing auto_v2/R3aK16 runtime predictions provide B0/source prior ppm columns and route context.
3. Target calibration direct-heads train small per-client, per-gas Ridge or ElasticNet heads.
4. Evaluation reports no-QC full-set target test metrics first, with CO/high-CO/nonCO scopes separated.
5. Optional specialist gates use calibration-validation evidence only.

The method can be described as:

```text
window x
  -> official F6 r25 adapted classification backbone
  -> route confidence + cls_feat + reg_feat
  -> existing B0/source ppm priors + rich response statistics
  -> target profile adapter
      -> H2.3+ balanced fusion head
      -> optional H8+ CO risk-gated specialist
  -> no-QC ppm report
  -> QC report only as deployment reliability
```

## Experiment P0: Official Backbone Feature Export

Create a feature export step for the official F6 final adapted checkpoint:

- Checkpoint: `results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/server_latest_adapted.pth`.
- Data root: `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`.
- Clients: C3, C4, C5.
- Splits: calibration and test.
- Output key: `(client, split, sample_index)`.

Exported columns:

- `pred_class_f6_r25`
- `prob_0` through `prob_3`
- `confidence`
- `margin`
- `entropy`
- `cls_feat_000` through `cls_feat_063`
- `reg_feat_000` through `reg_feat_063`

Also export the same schema for `server_round_019_adapted.pth` into a clearly labeled diagnostic directory. Round 19 outputs must include `diagnostic_only=true` in their manifest.

Acceptance:

- Export row counts match the calibration and test split sizes for C3/C4/C5.
- Keys align exactly with existing target prediction CSV rows.
- The manifest records checkpoint path, round, adaptive flag, clients, splits, feature dimensions, and whether the export is official or diagnostic.

## Experiment P1: H2.3 Backbone Feature Ablation

Train and evaluate target direct-heads using the same calibration split policy as the formal H2.3/H1/H2 scripts. The only variable is the feature group.

Feature groups:

- `A0_rich_only`: current rich response statistics.
- `A1_rich_plus_confidence`: rich stats plus confidence, margin, entropy, predicted class one-hot, and probability vector.
- `A2_rich_plus_cls_feat`: rich stats plus `cls_feat`.
- `A3_rich_plus_reg_feat`: rich stats plus `reg_feat`.
- `A4_rich_plus_b0`: rich stats plus B0/final ppm prior columns from the aligned prediction CSV.
- `A5_rich_plus_source_priors`: rich stats plus available source/layer prediction priors.
- `A6_rich_plus_all_backbone`: rich stats plus confidence, probabilities, `cls_feat`, and `reg_feat`.
- `A7_rich_plus_all_priors`: rich stats plus confidence, probabilities, `cls_feat`, `reg_feat`, B0/final ppm, and source/layer priors.

Models:

- Ridge is the first required head.
- ElasticNet is optional only after Ridge output exists.
- Shallow MLP is not part of P1.

Metrics:

- ALL RMSE and NRMSE.
- Macro-client RMSE and NRMSE.
- C3/C4/C5 RMSE and NRMSE.
- C3/C4/C5 CO RMSE.
- C3/C4/C5 high-CO RMSE.
- nonCO_ALL RMSE.
- C5 nonCO wrong-route audit, especially nonCO predicted as CO.

Acceptance:

- If A2/A3/A6/A7 improve over A0 on ALL or macro-client NRMSE without increasing nonCO_ALL by more than 1.0 RMSE, the backbone has direct regression value.
- If only A4/A5/A7 improve, the useful signal is source prior ppm rather than backbone embedding.
- If none improve, the current classification backbone should be treated as route-only for the thesis and regression-aware encoder retraining becomes a separate follow-up item.

## Experiment P2: H2.3+ Balanced Fusion Profile

Use the best P1 feature group to define H2.3+.

Required candidates:

- `H2_3_current_r25`: existing formal H2.3 recomputed on official r25 predictions.
- `H2_3_plus_ridge_r25`: Ridge over the selected fused feature group.
- `H2_3_plus_elasticnet_r25`: ElasticNet over the same feature group if Ridge improves over A0 or if Ridge is tied but more stable by client.

Selection:

- Candidate selection uses calibration-validation only.
- Test metrics are final reporting only.
- H2.3+ can replace H2.3 balanced only if it improves ALL or macro-client NRMSE and does not materially hurt nonCO_ALL.

Acceptance:

- Strong success: ALL RMSE below 18.0 on official r25 while nonCO_ALL does not degrade by more than 1.0 RMSE versus H2.3 current.
- Moderate success: H2.3+ beats H2.3 direct-only and narrows the gap to H2.3 current, proving the fused features recover part of the B0/profile-layer benefit.
- Failure: H2.3+ is worse than current H2.3 and does not improve the direct-only head; keep H2.3 as balanced mainline and move to regression-aware encoder work later.

## Experiment P3: H8+ CO Risk-Gated Specialist

Replace the hard C4 rescue story with a learned calibration-validation gate for CO-risk cases.

Gate candidates:

- Logistic gate over risk features.
- Ridge-style linear score with threshold chosen on calibration-validation.

Gate input features:

- predicted class and CO probability.
- confidence, margin, entropy.
- B0/final ppm.
- H2.3 current ppm.
- H2.3+ ppm if available.
- H8 or CO-specialist ppm.
- disagreement between balanced and specialist ppm.
- response phase and phase id.
- client id.

Gate scope:

- First version may restrict to predicted CO windows.
- C4-specific behavior must be learned from calibration-validation evidence and recorded in the manifest.

Acceptance:

- CO/high-CO RMSE matches or improves H8+C4.
- `hit_nonCO_N` is zero or explicitly small and explained.
- nonCO_ALL RMSE does not degrade by more than 1.0 versus H2.3 current.
- The report labels the profile as CO-priority specialist, not balanced default, unless all guardrails pass robustly.

## Experiment P4: Calibration Size Curve

Use P1/P2 outputs to test whether fused features reduce target calibration cost.

Calibration modes:

- 20 percent.
- 10 percent.
- 5 percent.
- 2.5 percent.
- one shot per gas/concentration when the split supports it.
- two shots per gas/concentration when the split supports it.
- high-CO removed from calibration for stress testing.

Candidates:

- B0/R3aK16 baseline.
- H2.3 current.
- H2.3+ fused profile.
- H8+C4 current specialist.
- H8+ learned gate if P3 passes.
- no-backbone direct-head control.

Acceptance:

- At 10 percent calibration, H2.3 or H2.3+ should remain better than B0.
- At 5 percent and 2.5 percent, safe fallback is preferred over forcing a specialist.
- If H2.3+ remains stable with fewer calibration samples than rich-only direct-head, the thesis can claim that backbone/source-prior fusion reduces target calibration cost.

## Reporting Rules

- Official headline tables use F6 r25 final adapted checkpoint only.
- Round 19 can appear in an appendix or diagnostic table labeled `best-checkpoint diagnostic`.
- Model capability uses no-QC full-set target test metrics.
- QC accepted-only metrics are reported separately.
- Calibration-validation chooses alphas, features, profiles, and gates.
- Test metrics never choose features, profiles, gates, or checkpoints.

## Deliverables

- `results/f6_r25_backbone_feature_export_*/backbone_features_calibration.csv`
- `results/f6_r25_backbone_feature_export_*/backbone_features_test.csv`
- `results/h2_3_backbone_feature_ablation_*/feature_ablation_summary.csv`
- `results/h2_3_backbone_feature_ablation_*/feature_ablation_report.md`
- `results/h2_3_plus_fusion_profile_*/fusion_profile_summary.csv`
- `results/h2_3_plus_fusion_profile_*/fusion_profile_report.md`
- `results/h8_plus_co_risk_gate_*/co_gate_audit.csv`
- `results/h8_plus_co_risk_gate_*/co_gate_report.md`
- `results/calibration_size_curve_fusion_*/calibration_size_curve_report.md`

## Risks And Mitigations

- Risk: r25 classification accuracy is high but C5 nonCO regression worsens through CO wrong routes.
  Mitigation: report C5 nonCO wrong-route audit and do not let ALL RMSE hide route-specific failures.
- Risk: high-dimensional embeddings overfit small target calibration.
  Mitigation: start with Ridge, standardize features, keep alpha selection on calibration-validation, and compare against rich-only controls.
- Risk: source priors dominate the apparent backbone gain.
  Mitigation: separate embedding-only groups from B0/source-prior groups in P1.
- Risk: r19 looks better than r25 and tempts checkpoint cherry-picking.
  Mitigation: label r19 diagnostic-only until a calibration-only checkpoint selector is designed.
- Risk: learned CO gate becomes another opaque rule.
  Mitigation: export feature coefficients or threshold audit, hit counts, false-hit counts, and nonCO guard metrics.

## Approval

The official follow-up uses the final F6 r25 adapted checkpoint as the main line. Round 19 remains diagnostic only.
