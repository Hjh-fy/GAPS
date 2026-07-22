# B5/C5 HC90 Runtime-Parity Design

## Goal

Provide one isolated, fail-closed runtime for the frozen B5/C5 deployment bundle.  It must execute the B5 classifier, the canonical H2.3 and R4 regression components, deployment risk, and the HC90 decision; it must also prove exact 1360-row parity against the frozen offline reference.  This work neither trains nor refits any asset.

## Scope and boundaries

The implementation adds `gaps_deploy/b5_c5_runtime.py` and dedicated tests.  It does not modify `gaps_deploy/final_runtime.py`, which owns a historical C12->C345, R3aK16/H8+C4-facing runtime.  The new module must never load, route through, or fall back to C3, C4, R3aK16, H8+C4, or P4 assets.

The only accepted package is a directory whose `manifest.json` reports `schema_version == "iotj.b5_c5_deployment_bundle.v1"`, `status == "ready"`, the exact ten runtime asset roles in `scripts.iotj_b5_c5_bundle_contract.RUNTIME_ASSET_KEYS`, and the offline parity reference.  Every file named by that manifest is checked for existence and SHA-256 equality before it can be read.  Extra, missing, malformed, or forbidden asset roles make loading fail.

## Components and data flow

`B5C5Runtime.from_bundle(bundle_dir, device="cpu")` is the only construction API.  It validates the manifest and loads immutable JSON assets plus the classifier checkpoint.  Its public `predict_batch(windows)` validates finite `float32` windows with the bundle-declared `(100, 8)` schema and returns one structured row per input with `pred_class`, H2.3 result, R4 result, risk score(s), HC90 decision, and selected final ppm.

The regression path is fixed: classifier -> H2.3 + R4 -> risk -> HC90.  The implementation may reuse existing model/risk primitives only when their inputs and outputs exactly match the frozen B5/C5 asset schemas; it must not delegate to the legacy `FinalDeployRuntime` or silently select any historical policy.  Malformed values, unknown class IDs, non-finite intermediate results, or missing required policy fields raise a dedicated runtime-contract error before a candidate prediction is emitted.

`verify_hc90_parity(runtime, input_rows, reference_csv)` executes every supplied keyed input row, requires exactly the key set `0..1359`, and compares `sample_index`, `pred_class`, selected profile, HC90 decision, and final ppm to the reference.  Floating ppm comparison uses the frozen reference representation plus a documented absolute tolerance; all categorical fields compare exactly.  Any row mismatch, duplicate/missing key, bad input, or output count mismatch raises an error and writes no success report.  A passing report records bundle manifest hash, reference hash, row count, and tolerance only; it does not replace experimental results.

## Reuse and versioning

Code is reusable across future classifier bases only when a new bundle implements the same schema and runtime contract.  A compatible future bundle changes hashes and asset values, not runtime code.  A changed window schema, class map, regression/risk semantics, or schema version is rejected.  Supporting it requires a separately tested adapter or a new runtime schema version; it may not be accepted by loosening this contract.

## Tests and acceptance criteria

Tests are written before production code.  They cover: accepting the audited B5/C5 manifest; rejecting each hash/schema/forbidden/missing-role failure; rejecting invalid windows; preserving the fixed route; rejecting malformed policy/intermediate data; rejecting duplicate, missing, and mismatched parity rows; and accepting a complete valid 1360-row fixture.  The final verification runs the canonical replay verifier first, then the new runtime parity verifier against the frozen bundle and HC90 reference.  Success requires exactly 1360 compared rows and zero mismatches.

## Evidence boundary

The output is deployment-parity evidence only.  It does not create a new training result, change formal C5 regression/QC measurements, or promote the existing classifier->R4 Pi metrics beyond their preliminary/no-QC status.
