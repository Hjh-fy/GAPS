# P0 raw timestamp audit

`p0_raw_timestamp_stats.csv` records all 640 raw files.  It reports true timestamp deltas, duplicate timestamps, gap thresholds, and observed sample count per real 100-ms bin.  C5 methane 225 has 2 repeats; their maximum dt values are 0.15s, 0.05s and empty-bin ratios are 0.06717, 0.  Thus nominal 100 Hz must not be treated as proof of an exact row clock.
