# C5 H8 Runtime-Parity Design

## Goal

Provide one versioned, fail-closed C1/C2->C5 runtime for the frozen B5/C5 deployment bundle.  It must execute the B5 classifier, fixed H8/R4 C5 Ridge expert, deployment-visible risk, and a manifest-selected HC95 or HC90 QC decision; it must also prove exact 1360-row parity against the frozen offline reference.  This work neither trains nor refits any asset.

## Scope and boundaries

The implementation adds `gaps_deploy/c5_h8_runtime.py`, `gaps_deploy/c5_h8_bundle.py`, and dedicated tests.  It reuses the existing fail-closed `gaps_deploy/qc_policy.py` and `gaps_deploy/package_contract.py`; it does not duplicate QC decisions, package primitives, or shared feature logic.  It does not modify `gaps_deploy/final_runtime.py`, which owns a historical C3/C4/C5, R3aK16/H8+C4-facing runtime.  The new modules must never load, route through, or fall back to C3, C4, R3aK16, H8+C4, or P4 assets.

The only accepted package is a directory whose `manifest.json` reports `schema_version == "iotj.b5_c5_deployment_bundle.v1"`, `status == "ready"`, the exact ten runtime asset roles in `scripts.iotj_b5_c5_bundle_contract.RUNTIME_ASSET_KEYS`, and the offline parity reference.  Every file named by that manifest is checked for existence and SHA-256 equality before it can be read.  Extra, missing, malformed, or forbidden asset roles make loading fail.

## Components and data flow

`C5H8Runtime.from_bundle(bundle_dir, device="cpu", workpoint=None)` is the only construction API.  It validates the manifest and loads immutable JSON assets plus the classifier checkpoint.  Its public `predict_batch(windows)` validates finite `float32` windows with the bundle-declared `(100, 8)` schema and returns one structured row per input with `pred_class`, H8/R4 ppm, risk score(s), QC decision, and `auto_output_ppm`.

The regression path is fixed: classifier -> predicted class -> fixed H8/R4 -> deployment-visible risk -> manifest-selected HC95 or HC90 -> accept/review/reject.  R4 is fixed H8: C1/C2 source Ridge, per-gas MLP, and shared MLP prediction-augmented C5 per-gas Ridge, with C4 rescue disabled.  HC95 is the formal default; HC90 is selectable only when it appears as a frozen workpoint in the manifest/policy.  The implementation may reuse existing model/risk primitives only when their inputs and outputs exactly match the frozen B5/C5 asset schemas; it must not delegate to the legacy `FinalDeployRuntime` or silently select any historical policy.  Malformed values, unknown class IDs, non-finite intermediate results, or missing required policy fields raise a dedicated runtime-contract error before a candidate prediction is emitted.

`verify_c5_h8_parity(runtime, input_rows, reference_csv, workpoint)` executes every supplied keyed input row, requires exactly the key set `0..1359`, and compares `sample_index`, `pred_class`, H8 ppm, risk, QC decision, and `auto_output_ppm` to the workpoint-matched reference.  Floating comparison uses documented absolute tolerances; categorical fields compare exactly.  Any row mismatch, duplicate/missing key, bad input, or output count mismatch raises an error and writes no success report.  A passing report records bundle manifest hash, reference hash, workpoint, row count, and tolerances only; it does not replace experimental results.

## Reuse and versioning

Code is reusable across future classifier bases only when a new bundle implements the same schema and runtime contract.  A compatible future bundle changes hashes and asset values, not runtime code.  A changed window schema, class map, H8/risk semantics, or schema version is rejected.  Supporting it requires a separately tested adapter or a new runtime schema version; it may not be accepted by loosening this contract.  B2 may be loaded only as an explicitly exploratory bundle and never replaces B5 as the predeclared formal deployment mainline.

## Tests and acceptance criteria

Tests are written before production code.  They cover: accepting the audited B5/C5 manifest; rejecting each hash/schema/forbidden/missing-role failure; rejecting invalid windows; preserving the fixed H8/R4 route with C4 rescue disabled; selecting HC95 by default and HC90 only explicitly; rejecting malformed policy/intermediate data; rejecting duplicate, missing, and mismatched parity rows; and accepting a complete valid 1360-row fixture.  The final verification runs the canonical replay verifier first, then the new runtime parity verifier against the frozen bundle and the workpoint-matched reference.  Success requires exactly 1360 compared rows and zero mismatches.

## Evidence boundary

The output is deployment-parity evidence only.  It does not create a new training result, change formal C5 regression/QC measurements, or promote the existing classifier->R4 Pi metrics beyond their preliminary/no-QC status.
