# Final Classification Protocol (Frozen Pre-run Specification)

## Scope and endpoint

This study contains 21 registered configurations: one E0 diagnostic, three E1 federated baselines, nine E2 canonical UDA branches, three E3 Full GAPS targets, and five new E4 ablation runs. Ten new federated runs use 25 rounds, one local epoch, batch size 32, and seed 42. Nine E2 branches independently reload the same P0A FedAvg round-25 ordered state and run exactly 100 adaptation steps. There is no hyperparameter search and no target-test-based checkpoint selection.

## Optimizer disclosure

| Method | optimizer | optimizer_lr | optimizer_note |
|---|---|---:|---|
| FedAvg | Adam | 5e-4 | frozen GAPS experimental protocol |
| FedProx | Adam | 5e-4 | frozen GAPS experimental protocol; mu=0.01 |
| SCAFFOLD | SGD | 5e-4 | canonical SCAFFOLD implementation; no momentum |
| CORAL/MMD/DANN | Adam | 5e-4 | fixed post-hoc model optimizer convention |
| GAPS | Adam | 5e-4 | proposed method protocol |

SCAFFOLD uses the canonical local update `w <- w - eta * (grad L + c - c_i)`, persistent per-client `c_i`, a persistent server `c`, the Option-II client control update, and the mean client control-delta server update. Its fixed SGD lr=5e-4 is preregistered and is not claimed to be equivalent in fairness or effective step size to Adam lr=5e-4. If the C1/C2-only numerical gate fails basic optimization, the run fails closed without trying another learning rate.

## Target information gate

Target test x/class are an absolute hard failure in every training, adaptation, early-stopping, threshold-selection, hyperparameter-selection, and checkpoint-selection path. They become available once per method-target only after its fixed endpoint completion marker exists.

Target calibration access is method-specific:

- E0 and E2 CORAL/MMD/DANN: x only.
- E1 FedAvg/FedProx/SCAFFOLD and E4 A0-A3: no target calibration fields.
- E3 Full GAPS and E4 A4-A6: x, class and phase; concentration is unavailable and target CE has configured weight zero.

## Canonical Domain-Adaptation Reference Study (E2)

CORAL is unconditional/global covariance alignment, MMD is unconditional/global MMD-squared, and DANN is an unconditional binary domain discriminator with GRL and BCE. Each uses the exact same imported FedAvg round-25 ordered state, the same C1+C2 source convention, the corresponding C3/C4/C5 calibration x, Adam lr=5e-4, coefficient 0.5, 100 steps, and seed 42. Conditional alignment, target CE, pseudo-labels, phase labels, target class labels, target concentration, hyperparameter search and target-test endpoint selection are unavailable.

## Full GAPS and ablation boundary

Full GAPS uses target calibration x/class/phase, no concentration, target CE weight zero, 100 server-DA steps per completed federated round, and adapted-as-global lineage. Selective aggregation warmup=5 means five complete sample-weighted FedAvg rounds: rounds 1 through 5 have final weights equal to base weights; round 6 onward uses selective scaling. Formal Full GAPS/A3/A6 fails closed if the registered semantic inputs are unavailable after warm-up. Logs record phase, base weights, similarities/scales and final weights using this exact boundary.

For C5 A0-A6, client and server records are aggregated to `ablation_loss_activity.csv` with `loss_name`, `configured_weight`, `input_available`, `active_steps`, `mean_raw_loss`, `mean_weighted_loss`, and `inactive_reason`. A4/A5 server losses are classified by actual observed inputs and steps rather than inferred from configuration names.

## Equality, evaluation, and cost

Checkpoint equality is established only by the ordered state-content fingerprint over key order, dtype, shape and tensor bytes. Whole-file SHA-256 is provenance only. E0 includes raw sensor-space per-channel mean, standard deviation, median, IQR, 5th/95th quantile, standardized mean difference and covariance-shift diagnostics. E1 reports the combined registered C1+C2 source-test macro-F1 once and `source_target_f1_gap = source_macro_f1 - target_macro_f1` for C3/C4/C5. Formal evaluation uses fixed four-class order, accuracy, macro-F1, per-class scores, NLL, 15-bin ECE, confusion matrices and per-window probabilities. Training/commissioning time and communication bytes are retained.

After `formal_training_started.lock` is written, the matrix and this protocol are immutable. A completed method-target has an immutable fixed-endpoint marker. Results never trigger tuning, matrix changes, checkpoint reselection, new seeds or additional unsupervised-method optimization.
