# Gate A Role-semantics Amendment

## Finding

`dataset/iotj_canonical_v1` provides `train/calibration/test` for C1/C2 and only `calibration/test` for C3/C4/C5. The legacy C1-C4-source dataset provides C3/C4 split arrays, but its per-window metadata omits physical window start/end fields. Those rows cannot be uniquely joined to canonical-v1 physical windows without inference from old numerical features.

## Frozen resolution

- C1/C2: copy every canonical-v1 split file byte-for-byte.
- C5: copy every canonical-v1 calibration/test file byte-for-byte. C5 never enters role-view RNG or Gate-A training.
- C3/C4: pool their canonical-v1 calibration+test rows, then partition independently per client and class×concentration stratum with seed42 client-local RNG into 70% train, 10% calibration, and 20% test. At least one calibration and one test row are retained per nontrivial stratum; the remainder is train.
- The partition manifest records every physical identity and source canonical file hash before any model run.
- Destination: `dataset/iotj_canonical_v1_s4_role_view`; fail if it already exists.

## Interpretation boundary

Gate A is a C5 hardest-target source-diversity sensitivity. S4 changes both the number of physical source domains and the amount/composition of labeled source data. It cannot support a pure single-factor claim that domain count alone caused the result.

## Leakage gate

C5 calibration/test hashes must equal canonical-v1, C5 files may not appear in any Flower client command, and the C5 test opens only after both S4 round25 endpoints are locked.

