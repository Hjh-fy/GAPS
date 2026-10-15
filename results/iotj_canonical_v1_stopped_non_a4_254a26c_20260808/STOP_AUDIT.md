# Non-A4 Start Stop Audit

Status: `STOPPED_CONFIGURATION_MISMATCH_NOT_A_FORMAL_RESULT`

Freeze commit: `254a26c`

The LE1 C3 start correctly used canonical preprocessing and `local_epochs=1`, but command inspection showed that it inherited the complete `FCL-E3-GAPS` `proto_replay`/selective-aggregation configuration rather than the final A4 router. It was stopped after round 1 client aggregation, before any target test access. C4 and C5 were never started and no formal completion marker was produced.

All three Flower processes were terminated and confirmed absent. Local and ECS outputs were moved into explicit `stopped_non_a4_254a26c` audit locations. They are excluded from formal analysis.

The corrected formal command must inherit `FCL-E4-A4`: client/server profile `ce_stats`, ablation variant `A4`, target-information method `a4`, client semantic/replay unavailable, selective aggregation disabled, and the frozen 100-step A4 server adaptation. The canonical `(50,8)` input declaration and local epoch 1 remain unchanged.
