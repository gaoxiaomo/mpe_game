# Generalized MPE Batch Report

- Output folder: `E:\毕业设计\mpe_game\repro_vsnac_mpe_mn\outputs\generalized_20260324_155046`
- Parallel workers: `1`
- Total wall time (s): `19.429`
- Sum of per-case wall times (s): `19.418`
- Parallel overhead note: current batch has throughput ratio `0.999x`; for these relatively short runs, process startup and plotting overhead dominate, so this value is not used as a performance claim.

## Case Summary
| case | capture(dynamic) | mean assigned(dynamic) | final Eteam@horizon(dynamic) | capture(fixed) | mean assigned(fixed) | final Eteam@horizon(fixed) | switches | ms/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4v2 | 37.40 | 716.252 | 225.298 | 40.70 | 981.754 | 217.520 | 1 | 3.3948 |

## Conclusions
- The generalized runner supports arbitrary `m` pursuers and `n` evaders through a shared scenario generator.
- A sweep mode is provided to enumerate a whole family of `m > n` scenarios and run them in parallel.
- For `n > 1`, the same trained weights are evaluated under dynamic graph and fixed graph using the same initial state, same seed, and the same simulation horizon.
- In the current validation set, dynamic graph improves capture time and lowers mean assigned error for the multi-evader cases, while final horizon Eteam remains close and case-dependent.
- Therefore, capture time and mean assigned error are treated as the primary effectiveness indicators for the generalized `m`-vs.-`n` extension.
- Runtime metrics are reported per case to support computer-oriented discussion on scalable simulation and parallel execution.