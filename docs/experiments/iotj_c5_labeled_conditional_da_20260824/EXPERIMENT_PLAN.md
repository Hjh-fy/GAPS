# Experiment Plan

## Research question

Does access to C5 calibration class labels improve canonical CORAL, MMD, and
DANN through (a) target CE or (b) true-class-conditional alignment without
target CE?

## Falsifiable hypotheses

- H1-TCE: each global alignment + TCE branch may improve C5 Accuracy/Macro-F1
  relative to its existing x-only E2 counterpart.
- H2-CC: each class-conditional no-TCE branch may improve C5 Accuracy/Macro-F1
  relative to its existing unconditional x-only counterpart.
- H3-Control: comparison with Simple Target-CE distinguishes the effect of
  alignment plus supervision from supervision alone.

## Evidence and stopping rule

Report Accuracy, Macro-F1, NLL, ECE, confusion matrix, per-class recall,
source retention, and commissioning seconds at fixed step100. Run exactly
six seed42 branches once and stop. A favorable single-seed result is
descriptive only and cannot replace the canonical unsupervised TABLE II rows.

