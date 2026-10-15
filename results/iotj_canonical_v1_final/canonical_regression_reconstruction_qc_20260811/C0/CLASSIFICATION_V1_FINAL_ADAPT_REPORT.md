# Classification V1 Final Adaptation Report

- Decision: `V1_INTERLEAVED_RETAINED`.
- C0 isolates lifecycle timing: 25x100 interleaved target steps versus one final 100-step invocation.
- Optimizer, A4 losses, coefficients, source batch convention, calibration identities, seed and fixed endpoint were not searched.
- The C0-A device-residual input was baseline-unavailable and remained unavailable; prototype and semantic inputs retained parity.

| Target | Final Macro-F1 | Interleaved Macro-F1 | Delta | Pass |
|---|---:|---:|---:|:---:|
| C3 | 0.988768844 | 0.998506885 | -0.009738041 | False |
| C4 | 0.988204856 | 0.997794108 | -0.009589253 | False |
| C5 | 0.940641634 | 0.994126091 | -0.053484456 | False |
