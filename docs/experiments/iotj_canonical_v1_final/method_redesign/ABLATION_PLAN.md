# Minimal Ablation Plan

- G1 isolates commissioning mechanism while holding source checkpoint, C5 calibration identities, steps, optimizer, LR, batch size, seed, and test fixed.
- G2 changes only source-local loss from CE to CE plus registered semantic prototype alignment. Selective aggregation, replay, target DA, and target access remain disabled.
- G3 isolates use of unlabeled C5 calibration data: A0T uses 80 labels; MME and GAPS-SSDA use the same 80 labeled and 240 label-inaccessible windows with the same final update budget.
- No combinatorial ablation, coefficient search beyond the authorized G3 six-item grid, C3/C4 expansion, multi-seed expansion, or downstream R84/QC is part of this stage.

