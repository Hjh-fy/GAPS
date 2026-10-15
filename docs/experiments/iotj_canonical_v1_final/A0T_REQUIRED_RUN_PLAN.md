# A0T required run plan

Status: **SUBMISSION_BLOCKER_P0 / REQUIRED**.

Run C3, C4, and C5 independently on canonical-v1 with the same source roles, fresh seed42 initialization, backbone, 25 rounds, local_epochs=1, batch size 32, Adam lr=5e-4, calibration identities, and target-label budget as A4. The only target adaptation loss is supervised target CE for 100 fixed steps per round. MMD, CORAL, DANN, prototype, semantic, stage, consistency, and other A4-specific losses are unavailable/disabled. Checkpoint selection is fixed round25 and target test opens only after all three endpoints complete. No result may trigger A4 tuning.
