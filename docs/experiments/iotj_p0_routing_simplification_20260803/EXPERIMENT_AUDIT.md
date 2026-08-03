# P0 Routing Simplification Experiment Audit

## Verdict: approved for seed42 descriptive evidence

The strict audit passed all frozen checks: dataset and client roles, seed42, 25 rounds × one local epoch, CE-only sample-weighted FedAvg, no P0-A target access, 25 checkpoint hashes, independent checkpoint reload for every commissioning branch, 100 steps at 5e-4, simple-CE DA exclusion, locked full-adapter semantics, inactive-statistics disclosure, sealed C5 test role, and fixed round25 formal comparison.

The P0 evidence supports a descriptive comparison among `source_only`, `simple_target_ce`, and `full_target_adapter`. Round-wise C5 curves cannot select a checkpoint or tune any configuration. Single-seed evidence cannot support uncertainty, significance, or stability claims. The LE1 P0 protocol is not a single-factor comparison against historical five-local-epoch B5.

One postflight issue was preserved: whole-file SHA equality between two independently serialized but tensor-identical PyTorch checkpoints is invalid. Verification was corrected to require round identity, identical state keys, and exact tensor equality; no checkpoint or training result was modified.
