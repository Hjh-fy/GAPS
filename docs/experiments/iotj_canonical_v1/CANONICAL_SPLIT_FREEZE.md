# Canonical Split Freeze

Physical roles reuse the frozen role-aware identities keyed by client, raw filename, and physical window start; they are independent of client traversal order and require no shared RNG state.

| Client | Role | Train | Calibration | Test |
|---|---|---:|---:|---:|
| C1 | source | 2360 | 320 | 680 |
| C2 | source | 2360 | 320 | 680 |
| C3 | target | 0 | 678 | 2677 |
| C4 | target | 0 | 320 | 1360 |
| C5 | target | 0 | 320 | 1360 |

C3 has five fewer included windows than the historical continuous-interpolation asset because canonical long-gap invalid windows are explicitly excluded. Calibration/test overlap is zero. Every included physical identity is unique and all client×class×concentration cells are covered. C5 Methane 225 ppm repeat 1 remains represented in processing and quality metadata.
