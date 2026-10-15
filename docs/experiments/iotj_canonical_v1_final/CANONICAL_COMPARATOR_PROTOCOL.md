# Canonical-v1 comparator protocol

Status before execution: `FROZEN` at commit `3ba1285`; protocol hash `fe862a61a7f8f1090dbbbabb0f058c66d9410d24b0d800bee297e9ea0d74d64f`.

The minimal main-table matrix is FedAvg, FedProx, canonical SCAFFOLD, canonical x-only MMD, equal-label A0T, and GAPS/A4. All use canonical-v1, C1/C2 source roles, C3/C4/C5 target identities, HZ5_MEAN_W10S, the same classifier backbone, seed42, batch size 32, 25 source-FL rounds, local_epochs=1, and fixed endpoints. Target test is unavailable during training, adaptation, checkpoint selection, and hyperparameter selection.

FedAvg and FedProx use Adam 5e-4 under the frozen experimental system; FedProx uses μ=0.01. SCAFFOLD uses its canonical SGD control-variate update at the preregistered 5e-4 and no learning-rate search. Its source-only numerical gate passed finite-value, CE-decrease, gradient-norm, parameter-norm, source-discrimination, source-only, and no-search checks.

MMD independently adapts the exact canonical FedAvg round-25 checkpoint for C3/C4/C5 using target calibration x only: unconditional global MMD², 100 steps, Adam 5e-4, alignment weight 0.5, combined C1/C2 calibration batches, fixed step-100 endpoint, no labels, phases, pseudo-labels, or target CE.

A0T and GAPS use the same target calibration identities and class-label budget. A0T enables only target CE for 100 steps per round; GAPS/A4 uses its frozen conditional alignment/prototype/stage mechanisms with target CE weight zero. Consequently comparisons across different target-information regimes are algorithm-level reference comparisons, not optimizer-controlled or supervision-controlled single-factor ablations.

Required manuscript wording: “SCAFFOLD is implemented with its canonical SGD-style control-variate update, whereas FedAvg, FedProx, and GAPS use the frozen Adam optimizer adopted by the experimental system. Therefore, the comparison represents standard algorithm-level baselines rather than an optimizer-controlled single-factor ablation.”
