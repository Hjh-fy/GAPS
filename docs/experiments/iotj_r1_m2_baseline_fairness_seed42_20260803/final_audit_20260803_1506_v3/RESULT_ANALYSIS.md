# R1-M2 Seed42 Result Analysis

## Input contract and provenance

- Experiments: R1M2-TARGET-ONLY-S42, R1M2-CENTRAL-SOURCE-S42, R1M2-FEDPROX-SOURCE-S42, R1M2-FEDAVG-SAME-ADAPTER-S42, R1M2-DS-FEDAVG-S42 plus `IOTJ-B5-S42-REFERENCE`.
- Scope: C5 sealed test, 1,360 windows, final checkpoint, seed42 only.
- Values in `comparison_metrics.csv` are reported from experiment manifests; percentage-point deltas versus B5 are recomputed here.
- No standard deviation, confidence interval, p-value, or seed-level effect-size uncertainty is estimable from one seed.

## Descriptive comparison

| Experiment | Accuracy | Macro-F1 | Delta vs B5 (pp) | NLL | ECE |
|---|---:|---:|---:|---:|---:|
| R1M2-TARGET-ONLY-S42 | 0.988235 | 0.988251 | +0.803 | 0.052944 | 0.005192 |
| R1M2-CENTRAL-SOURCE-S42 | 0.329412 | 0.228841 | -75.138 | 9.731264 | 0.666554 |
| R1M2-FEDPROX-SOURCE-S42 | 0.325000 | 0.230814 | -74.941 | 4.829544 | 0.657320 |
| R1M2-FEDAVG-SAME-ADAPTER-S42 | 0.990441 | 0.990443 | +1.022 | 0.099726 | 0.009648 |
| R1M2-DS-FEDAVG-S42 | 0.945588 | 0.945362 | -3.486 | 0.612786 | 0.053160 |
| IOTJ-B5-S42-REFERENCE | 0.980147 | 0.980220 | +0.000 | 0.150294 | 0.019368 |


## Seed42 findings

- FedAvg+same target adapter exceeds B5 by +1.022 Macro-F1 percentage points. The result does not support attributing the seed42 gain to GAPS client alignment/replay/decoupling or selective aggregation.
- Target-only exceeds B5 by +0.803 points, but uses a stronger fully supervised target-CE objective and is therefore an upper/reference configuration.
- B5 exceeds DS by 3.486 points and substantially exceeds both no-target source baselines.

## Interpretation boundaries

- Target-only uses all 320 C5 calibration labels for 2,500 supervised CE steps. It is a strong target-supervised upper/reference configuration, not an equal-objective GAPS ablation.
- Centralized Source-only and FedProx use no target calibration labels; their differences from GAPS combine target adaptation and method effects.
- FedAvg+same target adapter is the closest mechanism comparator: it keeps C5 calibration and server distribution-adaptation settings while disabling client alignment/replay/decoupling and selective aggregation.
- DS uses calibration-only ridge selection over 34 matched strata and the hash-pinned historical A0 checkpoint.
- Communication accounting is exact for model payload bytes. The adapter-matched run also sends `ce_stats` JSON; those extra wire bytes were not instrumented and must be labeled as additional/unknown rather than folded into the exact model-byte total.
- All conclusions are seed42-specific.

## Proposed paper table

Use `comparison_metrics.csv`, report Accuracy/Macro-F1/NLL/ECE and the declared target-label access. Do not add significance markers.
