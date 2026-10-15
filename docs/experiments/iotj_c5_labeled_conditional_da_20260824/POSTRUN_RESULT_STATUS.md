# Post-run Result Status

## Completion

- Pre-run freeze commit: `0accdd2`
- Formal methods: 6/6 fixed step100 endpoints
- Source checkpoint SHA256: `4313c375a8fa2e929de9d65637a2196f6c0f0752c2dc78112020b8727351751c`
- C5 calibration/test: 320/1360
- Seed: 42 only
- Hash audit: 39 indexed artifacts, 0 mismatches
- Target-test policy: opened only after all six completion markers; no selection

## C5 results

| Method | Accuracy (%) | Macro-F1 (%) | Source Macro-F1 (%) |
|---|---:|---:|---:|
| CORAL+TCE | 96.838 | 96.840 | 98.317 |
| MMD+TCE | 96.471 | 96.480 | 98.388 |
| DANN+TCE | 96.765 | 96.773 | 98.461 |
| CC-CORAL | 38.382 | 27.544 | 99.706 |
| Class-MMD | 88.382 | 88.514 | 98.900 |
| CC-DANN | 35.735 | 29.180 | 99.337 |

## Scientific interpretation

Adding target CE raises all three global DA methods from the weak x-only C5
regime to 96.47--96.84% Accuracy. However, all three remain 0.81--1.18
percentage points below the existing Simple Target-CE control, so the large
gain is primarily attributable to target-label supervision; the added global
alignment does not provide an additional benefit under this frozen setting.

True-class conditioning without target CE is method dependent. Class-MMD gains
31.691 pp Accuracy and 33.398 pp Macro-F1 over unconditional MMD. CC-CORAL is
essentially unchanged/slightly worse, while CC-DANN is worse. Thus
class-conditional alignment is not a general solution; only the MMD
formulation is supported by this seed42 diagnostic.

The existing GAPS C5 result (98.456% Accuracy, 98.454% Macro-F1) remains above
all six new diagnostics. These labeled probes are supplementary mechanism
evidence and do not replace the canonical unsupervised TABLE II rows.

