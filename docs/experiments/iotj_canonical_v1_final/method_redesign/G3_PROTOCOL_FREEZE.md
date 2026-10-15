# Gate 3 Protocol Freeze

Status: `READY_FOR_EXECUTION`.

## Fixed scientific comparison

| Method | Labeled C5 | Unlabeled C5 | Final updates | Classifier identity |
|---|---:|---:|---:|---|
| A0T-5L | 80 | 0 | 100 | canonical linear head |
| MME-compatible-5L15U | 80 | 240 X-only | 100 | canonical linear head; minimax entropy gradient |
| GAPS-SSDA-5L15U | 80 | 240 X-only | 100 | canonical linear head; EMA pseudo-label + source class prototype |

All three endpoints start independently from source-only FedAvg round25 state fingerprint `cad6726ec29fb574314a5f2a45ed9800d1d90906b81cbd3ba8f8efb48a0df5d7` and use Adam 5e-4, batch size 32, seed42. No source Flower run is launched.

## Label-access contract

- The 80 labeled identities are the existing 5% nested budget: 2 per each of 40 class×concentration strata.
- The 240 unlabeled identities are the complement in the canonical 20% pool: 6 per stratum.
- Unlabeled datasets own only X and physical identity; no class, phase, concentration, regression label, or hidden-truth member exists.
- Target test is absent from adaptation and selection APIs and opens only after all final step100 checkpoints are hashed.
- Hidden unlabeled class truth opens only after endpoint locking for one post-hoc pseudo-label diagnostic.

## Fixed coefficients and selection

- MME-compatible entropy weight: 0.1, taken from the official implementation default; no search.
- GAPS-SSDA EMA alpha: 0.99; source prototype weight: 0.05; prototype scope: class-only.
- GAPS-SSDA bounded grid: tau {0.90, 0.95} × unlabeled CE weight {0.25, 0.5, 1.0}; exactly six candidates.
- Two deterministic labeled folds swap the two identities within every stratum. Selection ranks mean validation Macro-F1, then mean validation NLL, then declaration order.
- The selected configuration is frozen before final all-80L+240U training and before target-test access.

## MME identity

The ICCV MME paper and official code use a temperature-scaled cosine-similarity classifier. Replacing the registered GAPS head would change the architecture and source endpoint semantics. Gate 3 therefore retains the current linear head and implements the minimax entropy gradient direction; it is named `MME-compatible`, never `exact MME reproduction`.

## Stop rule

After C5 evaluation and the G2/G3 story gate, stop. G4 is forbidden because G2 is already `SOURCE_DG_NOT_SUPPORTED`; G5 is not automatically launched.
