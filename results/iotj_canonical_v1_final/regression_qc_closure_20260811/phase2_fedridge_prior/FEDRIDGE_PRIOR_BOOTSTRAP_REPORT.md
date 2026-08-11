# FedRidge prior grouped-bootstrap report

Primary C5 grouping is the highest retained raw experimental identity,
`filename`: 80 files and 1360 correlated
windows. Each of 5000 replicates resamples whole files with replacement and
evaluates paired M83/M84 predictions on identical resampled rows.

- M83 S_ALL RMSE: 27.094018 ppm.
- M84 S_ALL RMSE: 28.057496 ppm.
- Delta RMSE (M84-M83): +0.963478 ppm.
- 95% grouped-bootstrap CI: [-1.377391, +3.942463].
- M83 S_CC RMSE: 14.550551 ppm.
- M84 S_CC RMSE: 13.110113 ppm.

Decision: `C5_M84_PRIOR_NOT_SUPPORTED`. The prior improves the correctly routed subset but does
not improve C5 S_ALL point RMSE, and the paired C5 CI crosses zero. A pooled or
cross-target decision is blocked because C3/C4 final post-hoc endpoints do not
exist. This result does not authorize tuning the prior.
