# Generalized MPE Batch Report

- Output folder: `E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs\generalized_20260324_140544`
- Parallel workers: `2`
- Total wall time (s): `23.648`
- Sum of per-case wall times (s): `36.704`
- Observed parallel speedup vs summed per-case wall times: `1.552x`

## Case Summary
| case | capture(dynamic) | mean assigned(dynamic) | final Eteam@horizon(dynamic) | capture(fixed) | mean assigned(fixed) | final Eteam@horizon(fixed) | switches | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 57.40 | 1736.656 | 262.113 | - | - | - | 0 | 1.9653 |
| 3v3 | 34.25 | 673.653 | 192.821 | 36.35 | 886.503 | 208.130 | 1 | 2.6717 |
| 5v3 | 38.20 | 692.824 | 312.253 | 37.45 | 777.824 | 304.445 | 1 | 2.7134 |

## Conclusions
- The generalized runner supports arbitrary `m` pursuers and `n` evaders through a shared scenario generator.
- A sweep mode is provided to enumerate a whole family of `m > n` scenarios and run them in parallel.
- For `n > 1`, the same trained weights are evaluated under dynamic graph and fixed graph using the same initial state, same seed, and the same simulation horizon.
- In the current validation set, dynamic graph improves capture time and lowers mean assigned error for the multi-evader cases, while final horizon Eteam remains close and case-dependent.
- Therefore, capture time and mean assigned error are treated as the primary effectiveness indicators for the generalized `m`-vs.-`n` extension.
- Runtime metrics are reported per case to support computer-oriented discussion on scalable simulation and parallel execution.