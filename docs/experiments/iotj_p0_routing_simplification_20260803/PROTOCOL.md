# Frozen P0 Protocol

Execution order is code/tests, compile and targeted tests, three-host preflight, P0-A source training, checkpoint/hash verification, source-only evaluation, simple Target-CE commissioning, full target-adapter commissioning, fixed round-25 comparison, figures, strict audit, analysis, final regression tests, commit, and push.

P0-A uses `profile=ce_only`, `LOCAL_EPOCHS=1`, `GLOBAL_ROUNDS=25`, `BATCH_SIZE=32`, `LR_CLIENT=5e-4`, `FEDPROX_MU=0`, and standard sample-weighted Flower FedAvg. All alignment, replay, decoupling, prototype upload, sensor augmentation, regression and target access are disabled.

Simple Target-CE updates the full classification model for 100 Adam steps at 5e-4 using only C5 calibration CE. Full DA reuses the registered R1-M2 matched-adapter configuration with target CE disabled. Because P0-A uploads no `ce_stats`, prototype, consistency, and device-residual terms requiring those statistics must be reported as `ZERO_NO_INPUT_STATISTICS`, never as active mechanisms.

The round-wise target-test curves are post-hoc diagnostics only. No output from C5 test may affect training, commissioning, checkpoint selection, early stopping, or hyperparameters. Round 25 is the sole formal comparison checkpoint.
