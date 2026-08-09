# SCAFFOLD sanity audit

Status: `PASS_WITH_LIMITATION`.

The source-only numerical gate and every one of 25 runtime rounds pass finite-value, participation, failure-count, canonical SGD, fixed-LR, LE1, positive-step, no-Adam-state, and server-control-update checks. Server control norms, model-delta norms, loss, and accuracy are preserved in `scaffold_roundwise_diagnostics.csv`.

Both clients received the same server-parameter fingerprint in every round: **True**. Server control fingerprints and saved tensors change across rounds, confirming server-c updates.

Per-client control-variate persistence and the gradient correction `grad L + c - c_i` are enforced by the canonical implementation tests (`test_scaffold_client_control_variate_persists`, `test_scaffold_gradient_contains_control_variate_correction`, and related tests). Flower's aggregated history does not retain the clients' string-valued before/after control fingerprints, so per-client control norms cannot be reconstructed post hoc from the server bundle; this is the stated audit limitation.

No learning-rate search or target information was used.
