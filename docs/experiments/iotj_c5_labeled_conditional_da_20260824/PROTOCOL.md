# C5 Labeled and Class-Conditional DA Diagnostic Protocol

Study ID: `C5-LABELED-CONDITIONAL-DA-PROBE-20260824`.

This supplementary mechanism diagnostic is separate from the canonical x-only
E2 study and does not replace or mutate TABLE II. Every branch independently
reloads the P0A pure-FedAvg round25 source checkpoint with whole-file SHA256
`4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c`
and ordered state-content fingerprint
`c0379777c63c7a5b0910e5d2ab8a1c37869692db14fd940e0a86778c9f5a1769`.

Held constants are C1+C2 source calibration, the existing 320-window C5
calibration split, the sealed 1,360-window C5 test, batch size 32, Adam
5e-4, 100 steps, alignment weight 0.5, seed42, and fixed step100. There is
no early stopping, checkpoint selection, threshold selection, or
hyperparameter search.

The TCE variants use target class labels in target CE and global alignment.
The class-conditional variants use target class labels only for true-class
grouping and structurally omit target CE. CC-DANN uses one shared domain
classifier and averages per-shared-class GRL/binary-BCE losses. No phase,
concentration, pseudo label, Wasserstein critic, one-hot concatenation, or
per-class discriminator is used.

The target test may be opened only after all six fixed endpoint markers
exist. All comparisons are descriptive because seed42 is the only seed.

