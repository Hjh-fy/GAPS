# DG-to-commissioning bridge report

| Identity | Budget | C5 Macro-F1 | Source retention delta | Seconds |
|---|---:|---:|---:|---:|
| I0 | 20 | 0.976544 | -0.337013 | 13.081 |
| I0 | 5 | 0.951568 | -0.339323 | 13.735 |
| I1 | 20 | 0.966918 | -0.309954 | 14.913 |
| I1 | 5 | 0.969067 | -0.285125 | 13.787 |
| I2 | 20 | 0.983821 | -0.279353 | 13.086 |
| I2 | 5 | 0.957275 | -0.372158 | 14.285 |

Decision: `DG_TO_COMMISSIONING_NOT_SUPPORTED`.

All six endpoints use Full A0T at fixed step100. I0+B20 is exact G1 reuse; the other five endpoints independently reload their registered original round25 source state. C5 test was not used for stopping, tuning, or selection.
