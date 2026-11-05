# Phase3.2 implementation plan

Goal: repair Phase3-only entry contracts without changing archived algorithms. User's A–G request is the approved scope; no new design approval or training authorization is inferred.

Architecture: separate spec/strict data and checkpoint validators, fixed-index loaders, and subclasses around archived Flower clients/strategies. Use full legal2520 source windows for serverCE; retain original10 CE batches plus1 feature-forward perstep. Target25 blocks×100 fixed batches16; reset Adam each block in all target-bearing paths. PureCE source clients can independently upload existing statistics; initial semantic prototypes empty, EMA.8, selective warmup3/minscale.3. No Test callbacks.

Files: contract.py(spec/identity/scaler-free50×8/checkpoint); runtime.py(verified imports, wrappers and real CLI); test_contract.py(negative validation); preflight.py(actual loops on synthetic fixtures + real no-grad loaders); build.py(registration, source-only input lock, commands); reports/verification/publication separate.

- [ ] Write failing behavior tests for missing contract, altered spec, shape/label/identity, unknown checkpoint, Test file and target batch consumption.
- [ ] Implement minimal validators/sampler; run tests.
- [ ] Build thin actual runtime; use original Client/Flower and native DA, no AST or numeric test doubles in final tests.
- [ ] Verify all5 target-bearing paths2500/40000 with synthetic Tiny64; no real data gradient. Check actual stats and feedback chain, TAC weight and negative mutations.
- [ ] Register executable source pretraining commands, source-only mount, random state and optimizer rules. Do not run them.
- [ ] Verify historical seals, new hashes, reimport/restore and fixed GitHub publication.

Review focus: checkpoint ancestry vs schema; exact identities rather than arbitrary self-declared SHA; source-only package excludes target; prototype stats must not replay target gradient loader; stale/wrong upload/received version must fail closed. Real pilot status remains BLOCKED_NO_SSTAR even if interface preflight passes.
