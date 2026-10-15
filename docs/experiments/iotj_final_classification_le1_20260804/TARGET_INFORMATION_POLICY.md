# Target Information Policy

Target test access outside fixed-endpoint final evaluation is an absolute hard failure. Target calibration access is method-specific and must be recorded by field rather than described as a global label prohibition.

| Method/stage | Split | x | class | phase | concentration | Purpose | Selection allowed |
|---|---|---:|---:|---:|---:|---|---:|
| E0 domain-shift diagnostic | target calibration | yes | no | no | no | sensor/feature distribution diagnostics | no |
| E1 FedAvg/FedProx/SCAFFOLD training | target calibration | no | no | no | no | unavailable | no |
| E2 CORAL | target calibration | yes | no | no | no | unconditional global covariance alignment | no |
| E2 MMD | target calibration | yes | no | no | no | unconditional global MMD² | no |
| E2 DANN | target calibration | yes | no | no | no | unconditional binary domain discrimination | no |
| E4 A0-A3 training | target calibration | no | no | no | no | unavailable; source client statistics only | no |
| E3 Full GAPS / E4 A4-A6 | target calibration | yes | yes | yes | no | registered global/class/phase conditional server DA; target CE remains zero | no |
| Any train/adapt/stop/select stage | target test | no | no | no | no | sealed | no |
| Fixed-endpoint final evaluation | target test | yes | yes | no | no | one-time classification metrics and predictions | no |

Every loader invocation must record method, stage, split, requested fields, purpose, allow/deny decision and reason. A target-test final-evaluation request is valid only after the exact method-target completion marker exists and cannot be reused for training, tuning, stopping, thresholding or checkpoint selection.
