# Phase-2 DG-to-commissioning result analysis

## Confirmed comparison

All six rows use fixed Full A0T commissioning for 100 Adam steps at 5e-4 and seed42. The only experimental factors are the registered source identity (I0/I1/I2) and the frozen calibration budget (B20/B05). I0+B20 is exact immutable G1 reuse.

## C5 Macro-F1

| Source identity | B20 | B05 |
|---|---:|---:|
| I0 S2-FedAvg | 0.976544 | 0.951568 |
| I1 S4-FedAvg | 0.966918 | 0.969067 |
| I2 S4-DG-P | 0.983821 | 0.957275 |

- I2-I1: +0.016902 at B20 and -0.011792 at B05.
- I1-I0: -0.009626 at B20 and +0.017499 at B05.
- The direction of both the DG mechanism and source-diversity effect changes with calibration budget.

## Source-retention diagnostic

Hypothetical use of each personalized checkpoint on its source pool reduced Macro-F1 by 0.2794 to 0.3722. This does not alter the immutable operational source checkpoint, but it confirms that Full A0T produces target-personalized rather than globally retained models.

## Registered decision

`DG_TO_COMMISSIONING_NOT_SUPPORTED`

DG-P clears the one-point gate at B20 only and reverses at B05, so neither the all-budget nor low-budget DG decision is supported. The seed42 zero-shot benefit therefore does not establish a robust commissioning advantage.

## Phase-3 rule application

The best B20 row is I2 at 0.983821. I0 is only 0.007276 lower, within the pre-registered 0.01 effectiveness band; therefore the simplest effective identity is I0+B20. Phase 3 must use the immutable I0+B20 post-hoc A0T checkpoint, not select I2 by its higher target-test number.

