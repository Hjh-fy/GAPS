# System Safety Hardening Design

**Status:** Approved for implementation on 2026-07-13.

## Objective

Remove silent fallback behavior from the production classification, regression, domain-adaptation, deployment, and QC paths without changing the frozen B1-B5 or formal R0-R7 experiment definitions. Invalid inputs and incomplete deployment bundles must fail before training or inference; runtime uncertainty must produce a reject decision instead of an automatic prediction.

## Scope And Compatibility Boundary

This hardening covers the production paths under `gaps_deploy`, the Flower server domain-adaptation path, and the legacy specialist-regression selection/evaluation path still referenced by system entry points. It does not retune losses, thresholds, model architectures, experiment budgets, or completed result files.

Historical experiment reproducibility remains explicit:

- The generic legacy MMD/stage/Wasserstein defaults are not changed. Frozen v3 B1-B5 manifests continue to select their corrected modes explicitly.
- Detached prototype pair-L2 remains disabled in the formal B suite, where its weight is already zero. This repair does not turn it into a new training objective.
- Existing reports and checkpoints are not rewritten. New validation determines whether an artifact is safe to execute, not whether an old result should be deleted.
- Direct research scripts may retain clearly named diagnostic fallbacks when they do not feed the production runtime. Production entry points do not inherit those fallbacks.

## Considered Approaches

### A. Strict Contracts At Every Production Boundary - Selected

Validate packages when they are built, validate them again when loaded, and make QC reject when required evidence is absent. Flower domain adaptation and regression evaluation validate their complete inputs before starting. This duplicates a small amount of checking but prevents callers from bypassing a standalone validator.

### B. Compatibility-First Warning And Fallback

Keep random/uninitialized models, default thresholds, root-level label fallback, and base-model fallback while emitting warnings. This preserves permissive behavior but cannot guarantee that a run or prediction corresponds to the declared method, so it is rejected for production paths.

### C. Standalone Validator Only

Strengthen `validate_deployment_packages.py` but leave runtime loading permissive. This is insufficient because callers can invoke `DeployPredictor.from_package` or `FinalDeployRuntime` directly, so it is rejected.

## 1. Deployment Package And Checkpoint Contract

`DeployPredictor.from_package` and `DeployPredictor.from_config` are always fail-fast production APIs. Before constructing a usable predictor they require the package/configuration and every selected asset:

- deployment and model configuration;
- classifier checkpoint;
- regression checkpoint(s) and routing configuration required by the configured regression mode;
- calibration statistics required by the selected calibration mode;
- a syntactically and semantically valid QC policy for final-runtime use.

Missing or malformed required assets raise a dedicated deployment-package validation exception with the asset path and reason. The runtime never substitutes a randomly initialized classifier/regressor.

There is no implicit compatibility flag that weakens these production APIs. Research diagnostics that need injected or partial components use the existing explicit constructor or a separately named diagnostic helper. `FinalDeployRuntime` always uses the strict production package API.

Checkpoint loading validates the serialized object, extracts the declared state dictionary, and compares its keys and tensor shapes with the configured model. Loading may inspect missing/unexpected keys internally, but a production predictor is returned only when the effective match is strict. Model configuration embedded in checkpoints must agree with the package configuration for architecture-defining fields.

`build_package.py` derives architecture-defining values from explicit source configuration/checkpoint metadata. Because current Flower classifier checkpoints do not contain a complete core model configuration, the builder requires a trusted explicit model-configuration JSON whenever checkpoint metadata is incomplete. It rejects missing values and explicit/checkpoint disagreements instead of silently writing hard-coded defaults. It also refuses to build a final-runtime package without a valid QC policy and the calibration assets selected by the configuration. Production callers propagate these required arguments rather than recreating defaults.

`validate_deployment_packages.py` performs the same content-level checks without running inference: JSON schema and cross-field checks, QC-policy parsing, finite threshold checks, checkpoint deserialization, state-dictionary key/shape compatibility, and required calibration/routing asset checks. The builder, validator, and runtime share validation helpers so their acceptance rules cannot drift.

Routing configuration is complete and explicit. `selected_modes` must contain exactly the integer classes `0..num_classes-1` after key normalization; the base route is written as `none`. Missing, duplicate-after-normalization, out-of-range, or unknown modes are errors. Every selected mode must have its complete matching parameter/model asset.

## 2. QC Fail-Closed Contract

`TwoThresholdDecider` has only three valid production outcomes: `accept`, `review`, or `reject`. It may accept only when a selected policy and all evidence named by that policy are available.

A valid policy has a non-empty `scores` list. Every score is registered and has exactly one finite denominator threshold greater than zero. The policy-level ratios are finite and satisfy `0 <= low_ratio < high_ratio`. Unknown scores, missing thresholds, duplicate/conflicting normalized keys, and thresholds for undeclared scores are validation errors.

The following conditions produce `reject`, `risk_ratio = None`, and a stable machine-readable reason rather than an exception or automatic prediction:

- no selected QC policy;
- the required score is missing or non-finite;
- a denominator threshold or policy ratio is missing or invalid;
- the selected score/threshold key is not defined by the policy.

Malformed policy files are rejected during package validation. The fail-closed decision remains a runtime guard for programmatic construction or corrupted in-memory state. For unavailable evidence, the in-memory and public `risk_ratio` is `None`/JSON `null`, with a stable reason such as `qc_policy_missing`, `qc_score_missing:<name>`, or `qc_threshold_invalid:<name>`; the runtime does not emit non-standard JSON `Infinity`. `FinalDeployRuntime` emits an automatic prediction only for `accept`; `review` and `reject` remain non-automatic outcomes.

Risk-score availability is explicit. Classifier-only scores require finite logits. Response-dependent scores, including response composites, require finite input features and valid response references for every routable class. Reference centers/scales/signatures must be non-empty, finite, dimensionally consistent, and have a finite positive normalization scale. An unavailable score is omitted/marked unavailable; it is never synthesized as zero, so a policy that requires it rejects.

## 3. Calibration And Unknown-Phase Contract

Calibration configuration is validated when loaded. Every named mode must supply the parameters or model artifact that mode requires; an unknown mode or missing parameters raises a configuration error. The runtime does not silently return raw predictions when a configured calibrator is incomplete.

Unknown phase uses one central normalization rule in every inference interface: only integer `-1` is the supported unknown sentinel. Preserve it as `phase_raw=-1` in diagnostics, but use `phase_model=0` before model encoding, calibration, and post-processing. Batch and generator inference therefore make the same prediction for identical rows. Non-integers, values below `-1`, and values greater than or equal to `num_phases` are input errors. This preserves the established batch-path compatibility behavior while removing the generator inconsistency.

## 4. Flower Domain-Adaptation Contract

Domain adaptation may be enabled only with the GAPS strategy and with explicit `--server-val-data` and `--server-calib-data` directory lists. The CLI/server rejects `--use-domain-adapt true` with plain FedAvg instead of silently ignoring it. Parsed source and target directory sets must be non-empty and disjoint.

Before Flower starts, every comma-separated directory must contain `calibration_features.npy`, `calibration_classification_labels.npy`, and `calibration_phase_labels.npy`. Root-level label fallback and synthesized `-1` phases are not allowed for strict domain-adaptation inputs. Features must match the model input rank/tail dimension and be finite; labels must be integer arrays; classes must be in `[0, num_classes)` and phases in `[0, num_phases)`. All three arrays must have the same non-zero first dimension.

Strategy construction repeats the contract check before creating adaptation loaders. A missing loader is therefore reported as an input error before the first round, never as an `iter(None)` failure inside adaptation.

For class-conditional adversarial alignment, a batch with no class shared by source and target has no valid critic comparison. In that case the critic step is skipped and the feature-alignment term is a differentiable zero connected to the current feature tensors. Backpropagation remains valid and no fabricated class match is introduced.

## 5. Specialist Regression And Aggregation Contract

Concentration-group calibration/validation splitting always yields disjoint index sets. When the dataset has at least two rows, calibration and validation are both non-empty, their intersection is empty, and their union is exactly the original indices. For one-group or very small data, the fallback splits row indices deterministically; it never duplicates a row. With fewer than two rows, the entry point raises a named insufficient-validation-data error before model selection rather than scoring reused rows.

A missing or non-finite validation metric cannot win specialist selection. Candidate priority is fixed as `none`, `bias_only`, `affine_only`, `phase_affine_only`, then `full`. A candidate replaces the current choice only when its finite score satisfies `score > best_score + min_delta`; ties retain the earlier/simpler mode.

Deployable gate selection and reported deployable metrics use a strictly loaded classifier to produce `route_class = argmax(logits)` on validation rows. The same `route_class` conditions the regression head, selects base/full/specialist models, selects calibration parameters, and groups route-specific candidate scores. Concentration truth is still read from the real class label. A route class with no predicted validation rows has an unavailable metric and retains `none`. True-class routing may be emitted only in explicitly named `oracle_*` diagnostics and cannot select the deployed gate. If a selected specialist is refit on all calibration rows, the stored selection decision and gate metrics remain the pre-refit independent-validation values; post-refit results are labeled as such and do not retroactively select the model.

Federated regression aggregation uses a positive integer `n_samples` stored in each accepted client checkpoint; missing, Boolean, non-integral, or non-positive values are errors. A separately enabled live-count consistency check may compare checkpoint counts with the data root and raises on disagreement. Live counts never replace checkpoint weights.

Evaluators require every asset named by the selected routing configuration. A missing selected specialist/full checkpoint or state-dictionary mismatch is an error. They never warn and substitute the base model, because that would evaluate a different system than the declared one.

## 6. Verification And Change-Control Contract

Every behavior change starts with a focused failing test that reproduces the reviewed unsafe behavior, followed by the smallest implementation that makes it pass. Tests cover both the negative contract and a valid-path control.

Verification proceeds from focused tests to subsystem suites and then the repository suite that is runnable in the current environment. Formal B1-B5/R0-R7 configuration-contract tests are included to prove that experiment definitions did not drift. Any unrelated pre-existing failure is recorded separately and is not masked.

Only files required by this hardening are staged. Existing local paper edits, result artifacts, temporary directories, and unrelated untracked research scripts remain untouched.

The Git-tracked production closure must be runnable from a clean checkout. Any mainline builder, validator, runtime module, or transitively imported dependency used by the repaired entry points is added deliberately and covered by an import/CLI smoke test. This includes the currently local-only deployment builder/validator and the `r4a_residual` module already imported by inference. Unrelated exploratory QC and regression scripts are not added merely because they are present locally.
