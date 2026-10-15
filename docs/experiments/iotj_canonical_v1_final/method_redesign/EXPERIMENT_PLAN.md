# Experiment Plan: canonical-v1 New-node Method Redesign

## Research questions

1. Can one-time post-hoc commissioning recover unseen C5 after immutable C1/C2 source-only FL?
2. Can source-only class/phase prototype alignment improve C5 zero-shot performance without C5 X/Y?
3. Can 80 labeled plus 240 label-inaccessible C5 calibration windows outperform 80-label target CE?

## Fixed comparisons

- G1: Source-only vs Posthoc A0T-full vs Posthoc A4 vs Posthoc Target-head.
- G2: Source-only FedAvg vs GAPS-DG-P, with C5 unavailable until fixed endpoint evaluation.
- G3: A0T-5L vs MME-5L15U vs GAPS-SSDA-5L15U, all from the same source checkpoint and with the same final adaptation budget.

## Metrics and decisions

- Classification: Accuracy, Macro-F1, NLL, ECE, per-class Precision/Recall/F1.
- Retention: adapted checkpoint Macro-F1 minus source checkpoint Macro-F1 on C1/C2/merged.
- Systems: trainable and total parameters, wall-clock, peak RSS, checkpoint size, relative parameter displacement.
- G2 representation: inter-client prototype distance, within-class C1-C2 distance, between-class margin.
- G3 diagnostics: pseudo-label acceptance, post-hoc hidden-label precision, per-class coverage, confidence distribution.
- No result is used to alter a frozen run. Decisions only determine whether the next gate is scientifically allowed and how claims are downgraded.

### Pre-registered Gate-1 decision thresholds

- `POSTHOC_LIFECYCLE_SUPPORTED`: best post-hoc C5 Macro-F1 is at least 0.95 and improves over zero-shot by at least 0.05.
- `POSTHOC_LIFECYCLE_WEAK`: supported threshold is not met, but either the best improvement is at least 0.05 or the best post-hoc result is within 0.05 of the corresponding historical interleaved endpoint.
- Otherwise: `POSTHOC_LIFECYCLE_FAILED` and the hard gate stops G2/G3.
- A4 is `KEEP` only if it exceeds A0T-full C5 Macro-F1 by at least 0.005, or stays within 0.005 while improving the mean C1/C2 retention delta by at least 0.01. Otherwise it is `RETIRE_AS_CORE`.
- Target-head is `PROMISING` only if it is within 0.005 of A0T-full on C5 while using fewer trainable parameters, retaining source performance at least as well on average, and adapting faster. Otherwise it is `NOT_COMPETITIVE`.
- `INTERLEAVED_DEPENDENCY_RISK` is raised when each comparable 100-step post-hoc endpoint is more than 0.05 below its historical interleaved reference. No update-budget experiment is launched automatically.

### Pre-registered Gate-2 decision threshold

- `SOURCE_DG_SUPPORTED` requires GAPS-DG-P to improve C5 zero-shot Macro-F1 over the same canonical FedAvg reference by at least 0.01 while reducing merged C1+C2 Macro-F1 by no more than 0.01.
- Otherwise the decision is `SOURCE_DG_NOT_SUPPORTED`; lambda, warm-up, prototype key, and training budget remain unchanged and no additional prototype-DG run is allowed.

### Frozen Gate-3 protocol and decision thresholds

- All final endpoints independently reload the Gate-1 canonical source-only FedAvg round25 state and perform exactly 100 Adam updates at 5e-4, batch size 32, seed42.
- `A0T-5L` uses only the frozen 80 labeled identities. `MME-compatible-5L15U` and `GAPS-SSDA-5L15U` additionally receive an X+identity-only loader for the fixed 240-identity complement.
- The MME comparator retains the registered biased linear head and applies the minimax entropy gradient direction with fixed weight 0.1. It is explicitly an MME-compatible implementation, not an exact reproduction of the paper's temperature-scaled cosine classifier.
- GAPS-SSDA uses EMA alpha 0.99 and class-only frozen C1/C2 prototypes with lambda 0.05. Target phase and concentration are unavailable to the adaptation API.
- Only GAPS-SSDA selects `tau` in {0.90, 0.95} and `lambda_u` in {0.25, 0.5, 1.0}. Selection uses the pre-registered deterministic two-fold split of the 80 labeled samples (one train and one validation identity per stratum), mean validation Macro-F1 descending, mean validation NLL ascending, then grid declaration order. The target test is unavailable.
- `SSDA_COMPONENT_SUPPORTED` requires GAPS-SSDA to exceed A0T-5L by at least 0.005 Macro-F1 and be no more than 0.005 below MME-compatible.
- `MME_DOMINATES` is assigned when MME-compatible exceeds GAPS-SSDA by at least 0.005, unless neither SSDA method improves A0T-5L by 0.005.
- `NO_SSDA_SPACE` is assigned when neither SSDA endpoint improves A0T-5L by 0.005. All other outcomes are `SSDA_COMPONENT_NOT_SUPPORTED`.
- The 240 hidden labels may open only after selection and all three final endpoints are locked, solely for a labeled `POST_HOC_DIAGNOSTIC_ONLY` pseudo-label audit.

## Leakage controls

- Adaptation code receives calibration-only manifests and rejects any test-role identity.
- C5 test opens only after every method endpoint in a Gate is complete and hashed.
- G2 training interfaces contain no target path or target tensor.
- G3 unlabeled samples expose only X and immutable identity to training; hidden Y is loaded only by the post-hoc diagnostic after training and selection are complete.
