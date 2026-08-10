+# Gate 3 C5 SSDA Result Analysis

Decision: `MME_DOMINATES`.

## Fixed endpoint comparison

| Method | Accuracy | Macro-F1 | NLL | ECE | Adaptation s |
|---|---:|---:|---:|---:|---:|
| A0T-5L | 0.951471 | 0.951568 | 0.233038 | 0.036221 | 13.133 |
| MME-compatible-5L15U | 0.970588 | 0.970586 | 0.121314 | 0.017732 | 16.715 |
| GAPS-SSDA-5L15U | 0.913971 | 0.914147 | 0.348184 | 0.049464 | 19.545 |

- MME-compatible minus A0T: **+0.019018 Macro-F1** and +0.019118 accuracy; NLL decreases by 0.111723 and ECE by 0.018489.
- GAPS-SSDA minus A0T: **-0.037420 Macro-F1** and -0.037500 accuracy; NLL increases by 0.115146 and ECE by 0.013243.
- GAPS-SSDA minus MME-compatible: **-0.056438 Macro-F1**.

Thus the frozen C5 task has useful semi-supervised signal, because MME-compatible improves both discrimination and calibration relative to label-only A0T. The proposed GAPS-SSDA component does not capture that value under its bounded, pre-registered configuration.

## Selection and mechanism diagnostics

The six-item, two-fold labeled-only selection chose `tau=0.95`, `lambda_u=0.25` with mean validation Macro-F1 0.724545 and mean NLL 1.350980. All 12 selection endpoints completed without target-test access; the selected configuration was locked before final training.

The final EMA teacher accepted 210/240 unlabeled windows (87.5%) at the fixed threshold. Accepted pseudo-label precision was only 80.48% in the post-hoc truth audit despite 99.63% mean accepted confidence. The error was strongly class-asymmetric:

| Predicted class | Accepted | Post-hoc precision |
|---|---:|---:|
| 0 | 47 | 0.9574 |
| 1 | 11 | 0.7273 |
| 2 | 94 | 0.6170 |
| 3 | 58 | 1.0000 |

This is consistent with confirmation bias rather than insufficient pseudo-label coverage: class 2 attracted many high-confidence but incorrect pseudo-labels. On the sealed test, GAPS class-1 recall fell from 0.9412 (A0T) to 0.8088, while class-2 recall rose to 0.9824 but precision fell from 0.9446 to 0.8127. Mean online pseudo acceptance was 0.8994; the pseudo-label CE remained non-zero at step100 (0.4853 raw), so the unlabeled term was active rather than silently disabled.

## Scientific interpretation

- Generic unlabeled-target commissioning value: **supported by the MME-compatible comparator**.
- GAPS EMA pseudo-label + frozen source class-prototype advantage: **not supported**.
- No further `tau`, `lambda_u`, prototype, teacher, or update-budget search is scientifically allowed by this Gate.
- MME must remain labeled as **MME-compatible (existing linear head)**. The frozen biased linear classifier differs from the original temperature-scaled cosine head, so this is not an exact reproduction.
- Results are one fixed seed42 endpoint; no across-seed uncertainty or general SSDA superiority claim is made.

Because G2 is `SOURCE_DG_NOT_SUPPORTED` and the GAPS-specific G3 component is not supported, the cross-Gate story is `STORY_D`. G4 is disallowed and G5 is not launched automatically.
