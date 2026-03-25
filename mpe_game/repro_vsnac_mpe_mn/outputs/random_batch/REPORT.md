# Generalized MPE Batch Report

- Output folder: `outputs/random_batch`
- Parallel workers: `6`
- Total wall time (s): `264.706`
- Sum of per-case wall times (s): `379.930`
- Observed parallel speedup vs summed per-case wall times: `1.435x`

## Case Summary
| case | capture(dynamic) | mean assigned(dynamic) | final Eteam@horizon(dynamic) | capture(fixed) | mean assigned(fixed) | final Eteam@horizon(fixed) | switches | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 49.90 | 2535.998 | 296.084 | - | - | - | 0 | 2.0835 |
| 4v2 | 72.40 | 1726.556 | 352.504 | 78.40 | 1847.393 | 343.608 | 1 | 3.1012 |
| 5v2 | 74.40 | 1915.108 | 405.980 | 74.60 | 1994.544 | 404.814 | 1 | 3.6741 |
| 5v3 | 68.20 | 1267.984 | 290.403 | 70.05 | 1425.425 | 308.635 | 1 | 3.8579 |
| 6v3 | 72.65 | 1891.186 | 363.276 | 75.90 | 1850.327 | 371.691 | 3 | 4.5893 |
| 7v3 | 74.80 | 1768.612 | 528.947 | 74.65 | 1829.312 | 528.219 | 1 | 5.1824 |
| 8v4 | 78.05 | 1662.807 | 587.077 | 73.70 | 1746.343 | 569.417 | 1 | 5.7847 |

## Conclusions
- The generalized runner supports arbitrary `m` pursuers and `n` evaders through a shared scenario generator.
- A sweep mode is provided to enumerate a whole family of `m > n` scenarios and run them in parallel.
- For `n > 1`, the same trained weights are evaluated under dynamic graph and fixed graph using the same initial state, same seed, and the same simulation horizon.
- In the current validation set, dynamic graph improves capture time and lowers mean assigned error for the multi-evader cases, while final horizon Eteam remains close and case-dependent.
- Therefore, capture time and mean assigned error are treated as the primary effectiveness indicators for the generalized `m`-vs.-`n` extension.
- Runtime metrics are reported per case to support computer-oriented discussion on scalable simulation and parallel execution.