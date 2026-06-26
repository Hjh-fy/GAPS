# GAPS Feature Schema and Runtime Contract — 2026-06-26

This document fixes the expected feature space and deployment runtime contract for the current GAPS pipeline. It is intended to prevent training/runtime mismatch when Codex or local scripts add new deployment bundles.

## 1. Window-level input contract

| Item | Value |
|---|---|
| window shape | `(100, 8)` for a single window, `(N, 100, 8)` for batch |
| time length | 100 points |
| sensor channels | 8 MOS channels |
| expected preprocessing | time-aware response window extraction, 60-170 s protocol unless explicitly stated |
| base feature definition | `(G - G0) / G0` or the already preprocessed feature used by the saved dataset |
| label shape | `classification_labels: (N,)`, `regression_labels: (N,4)`, `phase_labels: (N,)` |

## 2. Training dataset contract

For each client directory:

```text
train_features.npy
train_classification_labels.npy
train_regression_labels.npy
train_phase_labels.npy
calibration_features.npy
calibration_classification_labels.npy
calibration_regression_labels.npy
calibration_phase_labels.npy
test_features.npy
test_classification_labels.npy
test_regression_labels.npy
test_phase_labels.npy
*_experiment_info.json
```

The current `GasSensorWindowDataset` expects `features`, `classification_labels`, `regression_labels`, and `phase_labels`, and returns:

```text
x, y_cls, y_reg_full, y_phase
```

Regression always uses:

```text
y_true_ppm = y_reg_full[i, y_cls[i]]
```

## 3. Model input normalization contract

The codebase currently has two possible normalization stages:

1. dataset loader normalization via `GasSensorWindowDataset(normalize=True, mean, std)`;
2. deployment runtime normalization via `FinalDeployRuntime._prepare_features()` and `runtime_config["normalization"]`.

Current mainline expectation:

```json
{
  "training_loader_normalize": false,
  "runtime_normalization_control": "runtime_config.normalization.enabled",
  "runtime_norm_stats": "runtime_config.norm_stats",
  "rich_residual_feature_source": "raw input window before runtime model normalization"
}
```

This means:

- model predictor receives normalized features only when runtime config enables normalization;
- rich residual/direct-head feature extraction receives the raw pre-model window passed into `FinalDeployRuntime`;
- runtime parity must be checked whenever feature-space assumptions change.

## 4. Rich residual/direct-head feature contract

`gaps_deploy.rich_residual.RichResidualPolicy` uses window-level statistics such as:

- channel mean/std/min/max/amplitude/slope;
- global mean/std/amplitude;
- response phase one-hot fields;
- phase label one-hot fields;
- metadata fields such as `window_start_s`, `window_end_s`, `interpolated_ratio`;
- source augmented predictions for H8 when available.

Therefore, runtime metadata should preserve:

```text
response_phase
phase_label
window_start_s
window_end_s
window_center_s
t_onset
t_min
interpolated_ratio
max_gap_inside_window
filename
repeat_id
```

If metadata is missing, runtime can still execute, but phase-aware gates may not trigger and parity with analysis may fail.

## 5. FinalDeployRuntime public output contract

Current output fields:

```text
gas_class
gas_name
class_prob
base_r3ak16_raw_ppm
routed_pred_ppm
final_ppm
co_corrected_ppm
qc_decision
risk_score
```

Recommended next field:

```text
auto_output_ppm
```

Proposed semantics:

```text
final_ppm         = base calibrated/direct-head output before outer rich-residual correction
co_corrected_ppm  = final output after optional H8/C4/rich-residual correction
auto_output_ppm   = co_corrected_ppm only when qc_decision == accept, otherwise blank/null
qc_decision       = accept / review / reject
```

This keeps prediction audit values available while preventing review/reject rows from being interpreted as silent automatic output.

## 6. H8+C4 route-rescue v2 gate contract

The formal C4 route-rescue gate must preserve every selection condition used by the calibration selector:

```json
{
  "phase": "any|main_response|recovery",
  "risk_threshold": 0.0,
  "max_ppm": 50.0,
  "max_conf_margin": 1.0,
  "pred_classes": "0,2",
  "rescue_ppm": 250.0,
  "selection_source": "target_calibration_only"
}
```

Runtime must check:

```text
client_id == C4
response_phase matches gate phase unless phase == any
pred_class in pred_classes
final_ppm < max_ppm
risk_score >= risk_threshold
confidence_margin <= max_conf_margin
```

If any selected gate uses `max_conf_margin < 1.0`, failing to enforce this condition in runtime can create false hits.

## 7. Feature schema validation requirements

A future `validate_feature_schema.py` should check these items for every deployment bundle:

### Dataset checks

- `features.ndim == 3`;
- `features.shape[1:] == (100, 8)`;
- class/regression/phase label lengths match feature length;
- regression labels have either `(N,4)` or a documented compatible shape;
- metadata length equals feature length if metadata is required.

### Runtime checks

- `runtime_config.json` exists;
- `runtime_config.norm_stats` points to an existing npz;
- norm `mean` and `std` are finite;
- if normalization is enabled, `mean/std` broadcast against `(N,100,8)`;
- client package paths exist;
- rich residual artifact path exists when configured.

### Metadata checks

- `response_phase` values should be one of:
  - `main_response`
  - `recovery`
  - `pre_response`
  - `unknown`
- `phase_label` values should be one of:
  - `early`
  - `middle`
  - `late`
  - `unknown`
- missing phase metadata should be reported as warning, not silently ignored.

### Output

The validator should emit:

```text
feature_schema_validation.json
feature_schema_validation.md
```

Minimum JSON fields:

```json
{
  "bundle": "...",
  "data_root": "...",
  "clients": ["C3", "C4", "C5"],
  "status": "pass|fail",
  "checks": {
    "feature_shape": "pass",
    "label_length": "pass",
    "metadata_length": "pass",
    "norm_stats": "pass",
    "runtime_config": "pass"
  },
  "warnings": []
}
```

## 8. Failure policy

Do not accept runtime validation results if:

- feature schema validation fails;
- metadata lengths do not match feature lengths for phase-aware policies;
- runtime normalization is inconsistent with training/evaluation feature space;
- H8+C4 v2 gate does not preserve `max_conf_margin`;
- runtime-vs-analysis parity fails without a documented reason.

## 9. Integration checklist for STM32 / upper-computer deployment

Before connecting live data:

1. Confirm the live window builder outputs `(100,8)` in the same feature space.
2. Confirm G0/baseline handling matches offline preprocessing.
3. Confirm runtime normalization setting matches the bundle.
4. Confirm metadata availability; if unavailable, phase-aware gates should fallback safely.
5. Confirm `auto_output_ppm` is used for UI automatic display, not raw `final_ppm` or `co_corrected_ppm` for review/reject windows.
6. Log raw window, prepared model window, prediction row, and QC decision for each saved live test.
