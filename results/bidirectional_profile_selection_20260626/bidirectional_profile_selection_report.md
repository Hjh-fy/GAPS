# Bidirectional Target Profile Selection

This report makes the current selection policy explicit: the framework is shared, but the chosen profile is direction-specific.

## Summary

| direction | label | role | ALL_RMSE | ALL_NRMSE | CO_RMSE_by_target | CO_high_RMSE_by_target | nonCO_ALL_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C12_to_C345 | baseline final | baseline | 27.34 | 0.1578 | C3=33.70; C4=56.59; C5=46.12 | C3=41.70; C4=95.32; C5=60.00 | 19.00 |
| C12_to_C345 | H2.3 balanced mainline | balanced_mainline | 18.62 | 0.1326 | C3=16.15; C4=22.02; C5=26.85 | C3=20.02; C4=34.24; C5=34.82 | 17.83 |
| C12_to_C345 | H8 + formal C4 rescue | co_specialist_candidate | 18.30 | 0.1350 | C3=14.97; C4=17.16; C5=23.69 | C3=19.93; C4=26.79; C5=27.54 | 18.38 |
| C45_to_C123 | baseline final | baseline | 22.94 | 0.1473 | C1=37.68; C2=22.00; C3=32.31 | C1=51.97; C2=25.56; C3=38.16 | 19.34 |
| C45_to_C123 | target Ridge direct | balanced_mainline | 15.59 | 0.1123 | C1=23.77; C2=15.55; C3=14.68 | C1=38.30; C2=17.09; C3=16.37 | 14.50 |
| C45_to_C123 | H8-style source-aug CO else Ridge | diagnostic_co_specialist | 16.13 | 0.1192 | C1=24.70; C2=15.78; C3=10.73 | C1=39.70; C2=17.35; C3=11.04 | 15.44 |

## Decision

- C12 -> C345: keep H2.3 as balanced mainline and H8 + formal C4 rescue as deployable CO-specialist candidate.
- C45 -> C123: use target Ridge direct as the clean mainline; H8-style source-aug switching is diagnostic because it improves C3 CO/high-CO but worsens ALL/nonCO.
- Therefore the final system should expose a direction-specific profile selector, not a single hard-coded regression head.
