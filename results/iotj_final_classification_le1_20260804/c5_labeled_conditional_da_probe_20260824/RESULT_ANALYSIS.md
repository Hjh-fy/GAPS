# C5 Labeled and Class-Conditional DA Diagnostic

## Scope

All six branches independently reload the same P0A round25 source checkpoint. This is a seed42 mechanism diagnostic and does not replace the x-only TABLE II rows.

## Fixed-endpoint results

| Method | Accuracy (%) | Macro-F1 (%) | NLL | ECE | Source F1 (%) | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| CORAL+TCE | 96.838 | 96.840 | 0.1437 | 0.0181 | 98.317 | 11.02 |
| MMD+TCE | 96.471 | 96.480 | 0.1744 | 0.0192 | 98.388 | 7.93 |
| DANN+TCE | 96.765 | 96.773 | 0.1556 | 0.0171 | 98.461 | 7.99 |
| CC-CORAL | 38.382 | 27.544 | 4.6570 | 0.6022 | 99.706 | 7.96 |
| Class-MMD | 88.382 | 88.514 | 0.5188 | 0.0673 | 98.900 | 8.37 |
| CC-DANN | 35.735 | 29.180 | 4.3158 | 0.6088 | 99.337 | 8.05 |

## Descriptive mechanism deltas

- **CORAL+TCE vs CORAL:** Accuracy +58.162 pp; Macro-F1 +69.053 pp (global alignment + target CE).
- **MMD+TCE vs MMD:** Accuracy +39.779 pp; Macro-F1 +41.364 pp (global alignment + target CE).
- **DANN+TCE vs DANN:** Accuracy +44.926 pp; Macro-F1 +55.114 pp (global alignment + target CE).
- **CC-CORAL vs CORAL:** Accuracy -0.294 pp; Macro-F1 -0.242 pp (true-class conditioning without target CE).
- **Class-MMD vs MMD:** Accuracy +31.691 pp; Macro-F1 +33.398 pp (true-class conditioning without target CE).
- **CC-DANN vs DANN:** Accuracy -16.103 pp; Macro-F1 -12.479 pp (true-class conditioning without target CE).

## Target-CE control

- CORAL+TCE minus Simple Target-CE: -0.809 pp Accuracy, -0.813 pp Macro-F1.
- MMD+TCE minus Simple Target-CE: -1.176 pp Accuracy, -1.173 pp Macro-F1.
- DANN+TCE minus Simple Target-CE: -0.882 pp Accuracy, -0.880 pp Macro-F1.

## Interpretation

The highest new single-seed C5 Macro-F1 is CORAL+TCE (96.840%). These values are descriptive; one seed cannot establish stability or statistical significance. Because target labels are used by all six diagnostics, they are not unsupervised baselines and must remain outside the canonical E2 rows.
