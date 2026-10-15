# P6 factorial analysis

All four arms independently refit R84 on the corresponding calibration split. No test row enters alpha selection or fitting.

| Arm | Accuracy | Pipeline RMSE | S_CC RMSE | Oracle RMSE |
|---|---:|---:|---:|---:|
| A OLD checkpoint + OLD data | 98.46% | 16.0928 | 11.7965 | 12.0132 |
| B OLD checkpoint + NEW data | 98.31% | 27.0748 | 15.6551 | 20.9572 |
| C NEW checkpoint + OLD data | 97.35% | 30.8745 | 11.4326 | 12.0132 |
| D NEW checkpoint + NEW data | 98.97% | 27.2214 | 20.2864 | 20.9572 |

Oracle RMSE follows the data/calibration pipeline exactly (OLD 12.0132, NEW 20.9572) and is invariant to checkpoint. S_CC additionally changes with the checkpoint-specific correctness mask; on NEW data it is 15.6551 with OLD checkpoint and 20.2864 with NEW checkpoint.

## Fixed physical membership isolation

| Window representation | Membership | Oracle RMSE | Methane oracle RMSE |
|---|---|---:|---:|
| OLD | OLD | 12.0131 | 13.2915 |
| NEW | OLD | 22.4505 | 40.1372 |
| OLD | NEW | 12.3685 | 14.2117 |
| NEW | NEW | 20.9572 | 37.4413 |

Holding OLD physical calibration/test membership does not restore NEW preprocessing: RMSE changes from 12.0131 to 22.4505 (Methane 13.2915 to 40.1372). Conversely, under NEW membership, OLD numerical windows remain much better (12.3685 vs 20.9572). Preprocessing/numerical provenance is therefore primary; membership is secondary here.

Arms B/C are diagnostic-only because crossed tests can contain windows consumed by the checkpoint's original target calibration/adaptation.
