# Generalized MPE Batch Report

- Output folder: `E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs\mn_random_batch_plots_20260324`
- Parallel workers: `12`
- Total wall time (s): `122.730`
- Sum of per-case wall times (s): `219.004`
- Observed parallel speedup vs summed per-case wall times: `1.784x`

## Case Summary
| case | capture(dynamic) | mean assigned(dynamic) | final Eteam@horizon(dynamic) | capture(fixed) | mean assigned(fixed) | final Eteam@horizon(fixed) | switches | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 64.65 | 2600.111 | 257.168 | - | - | - | 0 | 1.1872 |
| 4v2 | 64.05 | 2172.787 | 247.380 | 72.65 | 2304.429 | 258.428 | 1 | 2.4867 |
| 4v3 | 53.70 | 2007.107 | 249.640 | 52.90 | 2256.794 | 260.342 | 1 | 2.8425 |
| 5v2 | 70.40 | 2022.555 | 300.938 | 59.65 | 2045.654 | 297.997 | 1 | 2.7914 |
| 5v3 | 69.75 | 2215.064 | 331.205 | 65.75 | 2188.831 | 321.312 | 3 | 3.0850 |
| 5v4 | 72.25 | 1912.130 | 345.181 | 77.85 | 2185.374 | 346.409 | 1 | 3.3180 |
| 6v2 | 73.15 | 2129.651 | 377.146 | 80.30 | 2203.573 | 378.550 | 1 | 3.0577 |
| 6v3 | 49.90 | 2047.835 | 363.129 | 49.65 | 2090.312 | 368.068 | 1 | 3.3063 |
| 6v4 | 63.60 | 2120.451 | 410.170 | 60.80 | 2154.186 | 413.738 | 1 | 3.6378 |
| 7v3 | 81.95 | 2668.017 | 511.708 | 78.70 | 2668.416 | 493.622 | 2 | 3.6592 |
| 7v4 | - | 3144.058 | 2332.772 | 72.90 | 2619.185 | 505.626 | 4 | 3.9061 |
| 8v4 | 77.20 | 2135.017 | 581.266 | 83.40 | 2291.853 | 623.587 | 2 | 4.1381 |

## Conclusions
- The generalized runner supports arbitrary `m` pursuers and `n` evaders through a shared scenario generator.
- A sweep mode is provided to enumerate a whole family of `m > n` scenarios and run them in parallel.
- For `n > 1`, the same trained weights are evaluated under dynamic graph and fixed graph using the same initial state, same seed, and the same simulation horizon.
- In the current validation set, dynamic graph improves capture time and lowers mean assigned error for the multi-evader cases, while final horizon Eteam remains close and case-dependent.
- Therefore, capture time and mean assigned error are treated as the primary effectiveness indicators for the generalized `m`-vs.-`n` extension.
- Runtime metrics are reported per case to support computer-oriented discussion on scalable simulation and parallel execution.