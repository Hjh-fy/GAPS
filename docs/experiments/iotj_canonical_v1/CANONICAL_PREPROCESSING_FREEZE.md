# Canonical Preprocessing Freeze

The only formal preprocessing is `HZ5_MEAN_W10S`: stable real-time sort, duplicate mean merge, raw-observation mean conductance G0 over 20–50 s, 5 Hz/0.2 s mean physical bins, one-bin-only interpolation, no long-gap interpolation, 60–170 s crop, 10 s duration, 5 s stride, and 50×8 input.

No sampling-rate, window, baseline, aggregation, or gap-policy search is permitted after this freeze. `HZ2_MEAN_W10S` remains an engineering fallback record and receives no formal training.
