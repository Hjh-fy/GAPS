# C5 SSDA Data Audit

- Canonical calibration pool: 320 identities.
- Labeled pool: 80 identities, exactly 2 in each of 40 pre-existing class×concentration strata.
- Unlabeled pool: the identity complement, 240 identities, exactly 6 per stratum.
- Labeled and unlabeled identity intersection: empty.
- The unlabeled training dataset stores and returns only `x` and `physical_identity`; it has no class, phase, or concentration field.
- The hidden-label array is not loaded before all final endpoints and the selected configuration are locked.
- Target-test manifest and arrays are not opened by the adaptation or selection stage.
- Stratum labels were used only to verify the already frozen nested calibration construction; they are not passed to any unlabeled loader, loss, sampler, selector, or checkpoint rule.

Calibration manifest SHA-256: `32fec2536e30399a525383c9b9fae73e1953b15b488523781c4aa76d818686bb`.
Labeled manifest SHA-256: `1024ba468e7afa54be08db075a490836ba11f42c8bb08be2fb140b8c73173d28`.
Unlabeled X tensor content SHA-256: `7559b020d6d765eda2cefadd96772cc617c3a2806527f855f4dc212b36ca7757`.
