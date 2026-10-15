# S4 DG Multi-seed Confirmation Report

## Protocol

FedAvg and the exact frozen DG-P mechanism were compared under S4 C1-C4 source-only training for seeds 41/42/43. Seed42 is immutable Gate-A reuse; seeds41/43 are new fixed-round25 runs. C5 was unavailable to all training APIs and was evaluated only after all endpoints were locked.

## C5 result

| Seed | FedAvg Macro-F1 | GAPS-DG-P Macro-F1 | Paired gain |
|---:|---:|---:|---:|
| 41 | 0.499678 | 0.334948 | -0.164730 |
| 42 | 0.386605 | 0.461595 | +0.074990 |
| 43 | 0.715778 | 0.674327 | -0.041451 |

- Mean paired gain: -0.043730
- Paired-gain sample SD: 0.119876
- Decision: `SOURCE_DG_NOT_CONFIRMED`

## Scope and limitation

This is a registered three-seed C5 hardest-target source-DG confirmation. It does not establish universal cross-target superiority, and no additional seeds are authorized after observing the result.

## Next action

`STOP_SEED_EXPANSION_AND_ENTER_PHASE2` under the already frozen Phase-2 matrix.
