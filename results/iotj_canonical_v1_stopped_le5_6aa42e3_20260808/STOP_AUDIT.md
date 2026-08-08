# User-Requested LE5 Stop Audit

Status: `STOPPED_BY_USER_NOT_A_FORMAL_RESULT`

Freeze commit: `6aa42e3`

The C3 canonical run used 25 configured global rounds and 5 local epochs. The user requested that the experiment stop and that the formal rerun preserve the earlier frozen `local_epochs=1` training configuration so preprocessing remains the principal changed factor.

The server was terminated after round 5 client aggregation. All three Flower processes were confirmed stopped. C4 and C5 were never started, no formal completion marker was produced, and target test remained sealed. The local and ECS outputs were renamed into explicit `stopped_le5_6aa42e3` audit locations and are excluded from all formal tables.

The replacement protocol changes only `local_epochs` from 5 to 1 relative to this stopped attempt. It retains 25 global rounds, batch size 32, seed 42, Adam 5e-4, the frozen GAPS model/aggregation/server-adaptation settings, canonical data, and the explicit `(50,8)` DA input contract.
