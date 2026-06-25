# C4 Route-Rescue Upper-Bound Sweep

Diagnostic only: gates are evaluated on target test to estimate possible headroom. Do not treat this as a formal selected rule.

| rank | classes | phase | max_final | min_risk | max_margin | rescue | hits | true high | false | ALL | C4 high | C4 nonCO | nonCO |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ethanol_ethylene | any | 20 | 4.0 | 1.0 | 250 | 15 | 15 | 0 | 18.05 | 14.81 | 8.86 | 18.38 |
| 2 | ethanol_ethylene | any | 30 | 4.0 | 1.0 | 250 | 17 | 17 | 0 | 18.05 | 14.81 | 8.86 | 18.38 |
| 3 | all_nonco | any | 20 | 4.0 | 1.0 | 250 | 15 | 15 | 0 | 18.05 | 14.81 | 8.86 | 18.38 |
| 4 | all_nonco | any | 30 | 4.0 | 1.0 | 250 | 17 | 17 | 0 | 18.05 | 14.81 | 8.86 | 18.38 |
| 5 | ethanol_ethylene | any | 50 | 4.0 | 1.0 | 250 | 18 | 17 | 1 | 18.30 | 14.81 | 11.32 | 18.72 |
| 6 | all_nonco | any | 50 | 4.0 | 1.0 | 250 | 18 | 17 | 1 | 18.30 | 14.81 | 11.32 | 18.72 |
| 7 | ethanol_ethylene | any | 20 | 2.0 | 1.0 | 250 | 17 | 15 | 2 | 18.56 | 14.81 | 8.86 | 18.38 |
| 8 | all_nonco | any | 20 | 2.0 | 1.0 | 250 | 17 | 15 | 2 | 18.56 | 14.81 | 8.86 | 18.38 |
| 9 | ethanol_ethylene | any | 30 | 2.0 | 1.0 | 250 | 20 | 17 | 3 | 18.78 | 14.81 | 11.11 | 18.68 |
| 10 | all_nonco | any | 30 | 2.0 | 1.0 | 250 | 20 | 17 | 3 | 18.78 | 14.81 | 11.11 | 18.68 |
| 11 | ethanol_ethylene | any | 50 | 2.0 | 1.0 | 250 | 21 | 17 | 4 | 19.03 | 14.81 | 13.15 | 19.02 |
| 12 | all_nonco | any | 50 | 2.0 | 1.0 | 250 | 21 | 17 | 4 | 19.03 | 14.81 | 13.15 | 19.02 |
| 13 | ethanol_ethylene | any | 20 | 4.0 | 1.0 | 225 | 15 | 15 | 0 | 18.07 | 16.57 | 8.86 | 18.38 |
| 14 | all_nonco | any | 20 | 4.0 | 1.0 | 225 | 15 | 15 | 0 | 18.07 | 16.57 | 8.86 | 18.38 |
| 15 | ethanol_ethylene | any | 20 | 2.0 | 1.0 | 225 | 17 | 15 | 2 | 18.48 | 16.57 | 8.86 | 18.38 |
| 16 | all_nonco | any | 20 | 2.0 | 1.0 | 225 | 17 | 15 | 2 | 18.48 | 16.57 | 8.86 | 18.38 |
| 17 | ethanol_ethylene | any | 30 | 4.0 | 1.0 | 225 | 17 | 17 | 0 | 18.08 | 16.94 | 8.86 | 18.38 |
| 18 | all_nonco | any | 30 | 4.0 | 1.0 | 225 | 17 | 17 | 0 | 18.08 | 16.94 | 8.86 | 18.38 |
| 19 | ethanol_ethylene | any | 50 | 4.0 | 1.0 | 225 | 18 | 17 | 1 | 18.28 | 16.94 | 10.85 | 18.65 |
| 20 | all_nonco | any | 50 | 4.0 | 1.0 | 225 | 18 | 17 | 1 | 18.28 | 16.94 | 10.85 | 18.65 |

## Reading

- The best rows estimate how much C4 high-CO could improve if a route-rescue gate were available.
- Rows with nonzero false hits are risky because they would overwrite non-CO or lower-concentration windows.
- A formal rule must be selected on calibration-validation and then evaluated on test.
