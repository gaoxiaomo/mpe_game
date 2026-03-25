# Generalized MPE Batch Report

- Output folder: `outputs/structured_batch`
- Parallel workers: `6`
- Total wall time (s): `286.435`
- Sum of per-case wall times (s): `383.720`
- Observed parallel speedup vs summed per-case wall times: `1.340x`

## Case Summary
| case | capture(dynamic) | mean assigned(dynamic) | final Eteam@horizon(dynamic) | capture(fixed) | mean assigned(fixed) | final Eteam@horizon(fixed) | switches | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 45.35 | 1693.876 | 324.682 | - | - | - | 0 | 1.9425 |
| 3v3 | 36.40 | 564.434 | 205.554 | 66.65 | 749.350 | 207.404 | 1 | 3.1261 |
| 4v2 | 35.55 | 594.051 | 305.254 | 54.45 | 826.408 | 294.264 | 1 | 2.9917 |
| 5v2 | 37.90 | 611.961 | 377.055 | 55.90 | 810.938 | 384.063 | 1 | 3.6515 |
| 5v3 | 34.10 | 561.831 | 278.640 | 38.25 | 634.743 | 276.375 | 1 | 4.0292 |
| 6v2 | 33.95 | 612.245 | 428.074 | 50.95 | 876.871 | 432.808 | 1 | 4.0717 |
| 6v3 | 34.65 | 574.665 | 367.336 | 62.50 | 740.033 | 376.845 | 1 | 4.4851 |
| 8v4 | 60.00 | 573.408 | 556.489 | 61.75 | 683.909 | 560.490 | 1 | 5.2577 |

## Conclusions
- The generalized runner supports arbitrary `m` pursuers and `n` evaders through a shared scenario generator.
- A sweep mode is provided to enumerate a whole family of `m > n` scenarios and run them in parallel.
- For `n > 1`, the same trained weights are evaluated under dynamic graph and fixed graph using the same initial state, same seed, and the same simulation horizon.
- In the current validation set, dynamic graph improves capture time and lowers mean assigned error for the multi-evader cases, while final horizon Eteam remains close and case-dependent.
- Therefore, capture time and mean assigned error are treated as the primary effectiveness indicators for the generalized `m`-vs.-`n` extension.
- Runtime metrics are reported per case to support computer-oriented discussion on scalable simulation and parallel execution.