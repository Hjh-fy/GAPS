# Routing versus regression analysis

| Method | Target | S_ALL | S_CC | Oracle_ALL | routing gap | paired mapping gap |
|---|---|---|---|---|---|---|
| A0T | C3 | 10.0679 | 8.8101 | 9.0421 | 1.2578 | 0.0000 |
| A0T | C4 | 15.0181 | 9.8739 | 10.7417 | 5.1442 | 0.0000 |
| A0T | C5 | 22.1560 | 14.3028 | 14.4488 | 7.8531 | 0.0000 |
| A0T | POOLED_C3_C4_C5 | 15.1925 | 10.7020 | 11.0561 | 4.4905 | 0.0000 |
| A4 | C3 | 9.3327 | 8.8479 | 9.0421 | 0.4848 | 0.0000 |
| A4 | C4 | 13.8080 | 10.2452 | 10.7417 | 3.5627 | 0.0000 |
| A4 | C5 | 18.4765 | 14.3340 | 14.4488 | 4.1426 | 0.0000 |
| A4 | POOLED_C3_C4_C5 | 13.3144 | 10.8148 | 11.0561 | 2.4996 | 0.0000 |

The requested `routing_gap` is S_ALL minus S_CC. The requested `regression_gap` uses differently sized populations (S_CC minus Oracle_ALL), so mechanism attribution uses the paired S_CC minus Oracle_CC diagnostic. The paired gap is zero by construction here: once the route is correct, the deployed and Oracle feature/model paths coincide. The A4 gain is therefore attributable to avoiding or changing high-cost misroutes, not to a better correct-route Ridge mapping.
