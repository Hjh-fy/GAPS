# Gate 2 Source-only DG Result Analysis

## Input contract and provenance

- Baseline: canonical source-only FedAvg round25, SHA-256 `2d114a8ae23fcdea574d1e7c64e638620f60e49560da594397187bd5de1505fa`.
- GAPS-DG-P: C1/C2 only, 25 rounds, LE1, Adam 5e-4, seed42, class×phase prototype alignment weight 0.05, round25 SHA-256 `3a19f14e4aa77111b775d55eaeaee8b54dbeda90b789ddb721c5ccf9582a4063`.
- GAPS-DG-P source archive SHA-256: `462a4985ccd92b7bad5214cf9ef4dc465ae3b67c9ccf9e05f1d1e386ba51c5cb` from freeze commit `40bcf59f779e8c694b4f8a1d05fe68dadc361b13`.
- Evaluation scope: C1 test 680, C2 test 680, merged source 1360, C5 test 1360. Values are recomputed at the fixed seed; no across-seed uncertainty is claimed.

## Classification results

| Method | C1+C2 F1 | C5 Accuracy | C5 Macro-F1 | C5 NLL | C5 ECE |
|---|---:|---:|---:|---:|---:|
| FedAvg | 0.999265 | 0.478676 | 0.368610 | 4.082857 | 0.505797 |
| GAPS-DG-P | 0.999265 | 0.338235 | 0.316017 | 5.413318 | 0.648408 |

GAPS-DG-P changes C5 Macro-F1 by `-0.052592`, C5 Accuracy by `-0.140441`, NLL by `+1.330460`, and ECE by `+0.142611`. Merged source Macro-F1 changes by exactly `0.000000` at displayed precision.

## Mechanism activity and representation analysis

- Round 1: both clients received 0 global prototypes and semantic alignment had 0 active steps.
- Round 2 and round 25: both clients received all 12 class×phase prototypes; semantic alignment was active for all 74 local mini-batches, while replay and regression remained inactive.
- Mean source inter-client class×phase prototype distance decreases from 0.077069 to 0.063715 (`-17.33%`).
- Mean within-class C1-C2 centroid distance decreases from 0.071949 to 0.060371 (`-16.09%`).
- Mean between-class centroid margin decreases slightly from 1.624671 to 1.616528 (`-0.50%`).

The intended source-alignment mechanism is therefore active and measurably aligns C1/C2 representations. The failure is not an inactive-loss artifact: better source alignment does not extrapolate to C5 under this frozen protocol.

## Decision

`SOURCE_DG_NOT_SUPPORTED`.

The pre-registered criterion required at least +0.01 C5 Macro-F1 with no more than -0.01 merged-source loss. The observed C5 change is negative. Prototype-DG expansion, lambda search, warm-up search, and additional FDG implementations are stopped.

This result supports only a mechanistic statement about C1/C2 alignment, not a domain-generalization improvement claim.

