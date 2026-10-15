# Final candidate decision

## Recommended canonical candidate

**Candidate 1: `HZ5_MEAN_W10S` — 5 Hz, mean physical-time bin aggregation, 10-s physical windows, raw-observation mean G0, duplicate merge, and short-gap-only policy.**

It ranked first *before test access* by calibration validation (10.4821 ppm), while retaining 99.963% usable windows. Its frozen one-time test diagnostic was directionally consistent (weighted oracle RMSE 10.6692 ppm).

## Second frozen candidate

**Candidate 2: `HZ2_MEAN_W10S` — 2 Hz with the same aggregation/baseline/gap/window rules.** It has a modestly weaker calibration result (10.7805 ppm) but 100% usable-window coverage and lower temporal input cost (20 versus 50 points/window). It is retained as the engineering/robustness comparator.

## Answers

1. Raw-observation G0 is preferred on methodological grounds and the preceding G0 counterfactual supports it; this selection did not optimize target test.
2. Mean bin aggregation ranked above median in calibration-only screening.
3. 5 Hz is recommended; 2 Hz is the lower-cost backup.
4. 10-s windows ranked ahead of 20-s windows in calibration-only screening.
5. Lower rates maintained higher usable-window coverage; 5 Hz retained near-complete coverage.
6. C5 Methane 225 repeat 1 remains a quality anomaly and was retained.
7. Candidate 1 is the recommended canonical configuration above.
8. Candidate 2 is the 2-Hz configuration above.
9. The primary decision is calibration/source performance plus methodological correctness; engineering distinguishes the retained backup, not test RMSE.
10. **Yes, conditionally:** this freezes preprocessing sufficiently to authorize one pre-registered final 25-round GAPS→adaptation→R84 confirmation, provided its data regeneration and label-access audit are separately pre-run frozen.

The test diagnostic was confirmatory only; it did not alter the pre-test ranking.
