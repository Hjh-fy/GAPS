# Experiment Plan — IoT-J target-feature metadata ablation

## Research brief and scope

- Brief source: v5.3 manuscript review found that the 104-D target Ridge input mixes 83 window-derived sensor statistics with 21 timing/protocol features, including non-causal `t_min` derivatives.
- Target venue/audience: IEEE Internet of Things Journal; cross-device gas sensing and federated edge intelligence.
- Resource budget: reuse the five frozen B5 classifier routes (seeds 42–46), frozen sufficient-statistics federated H1 source head, fixed C5 320-calibration/1360-test split, and per-gas Ridge target heads. No classifier, source head, runtime, or QC retraining.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| H-FMETA-01 | Removing all 21 metadata features changes target concentration performance under otherwise identical training. | 104-D full + H1 | 83-D sensor-only + H1 | Five-seed mean `S_CC_NRMSE` | Paired per-seed difference with identical routes and masks | Descriptive effect is reported regardless of direction; no equivalence claim unless relative mean degradation is within 5% |
| H-FMETA-02 | A causal online-safe subset can retain practically equivalent performance to the full 104-D profile. | 104-D full + H1 | 83-D + 8 online-safe fields + H1 (91-D base, 92-D with H1) | Five-seed mean `S_CC_NRMSE` | Paired per-seed safe-minus-full differences and `S_ALL_NRMSE` check | Practical equivalence only if mean `S_CC_NRMSE` relative degradation is <=5% and no gas is >10% worse in >=3/5 seeds |
| H-FMETA-03 | The effect of metadata is not merely an interaction with the federated H1 prior. | Each profile without H1 | Same profile with H1 | `S_CC_NRMSE` and `S_ALL_NRMSE` | Complete 3x2 profile/prior matrix | Report interaction descriptively; no causal H1 claim if routes, masks, or alpha-selection scopes differ |

## Fixed protocol

- Source clients: C1;C2.
- Target client: C5.
- Split protocol: frozen role-aware target 8:2 window-level protocol; C5 calibration=320 and test=1360. Existing protocol allows file overlap, so this ablation only isolates feature-profile dependence and does not repair physical-independence limitations.
- Model/checkpoint policy: five canonical B5 round-25 adapted checkpoints and their frozen routes; source H1 is the frozen sufficient-statistics federated per-gas Ridge.
- Seeds: classifier seeds 42, 43, 44, 45, 46. Each seed uses its own predicted route. Source H1 is shared and frozen.
- Target head: per-gas Ridge; alpha grid `{0,0.01,0.1,1,10,100,1000}`; within each seed/gas/variant use C5 calibration 60 fit / 20 validation, then refit on all 80 calibration rows.
- Metrics: primary `S_CC_NRMSE`; secondary `S_ALL_NRMSE`, RMSE, per-gas RMSE/NRMSE, route-error count, and common-correct-set metrics.
- DA / calibration / QC controls: classifier DA fixed to canonical B5; target calibration fixed; QC disabled for this ablation.

## Frozen feature profiles

- `M83_SENSOR`: the 83 features whose names are channel statistics or one of the 19 global/amplitude/slope summary features.
- `M91_ONLINE_SAFE`: `M83_SENSOR` plus `window_start_s`, `window_end_s`, `window_center_s`, `window_len_s`, `t_onset`, `center_minus_onset`, `interpolated_ratio`, and `max_gap_inside_window`.
- `M104_FULL`: the existing complete 104-D feature dictionary.
- Excluded from the online-safe profile: `t_min`, `center_minus_t_min`, all `response_phase_*`, `phase_label_*`, and `phase_id_*` fields.

## Risks, unknowns, conflicts, and stopping rules

- Known limitation: the frozen dataset manifest uses `split_level=window_level` and `allows_file_overlap=true`; results cannot establish generalization to unseen physical exposure files.
- `t_onset` is retained as conditionally online-safe because it can be emitted causally by a past-only detector or supplied by the experiment controller. The current offline preprocessing implementation must not be cited as proof of streaming parity.
- Stop on any checkpoint/hash mismatch, row misalignment, missing/non-finite prediction, schema dimension mismatch, target-test access before calibration alpha lock, or modification of frozen runtime/QC assets.
- Existing experiment assets are read-only. Results go only to the new dated result directory.

