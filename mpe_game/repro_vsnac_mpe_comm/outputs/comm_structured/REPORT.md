# Communication-Augmented MPE Batch Report

- Output folder: `outputs/comm_structured`
- Parallel workers: `5`
- Total wall time (s): `214.703`
- Sum of per-case wall times (s): `327.086`

## Case Summary
| case | cap(full) | cap(none) | cap(drop) | err(full) | err(none) | err(drop) | d_min(full) | d_min(none) | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 55.05 | 47.15 | 53.85 | 1736.329 | 1693.165 | 1731.802 | 282.2 | 135.6 | 0 |
| 3v3 | 33.45 | 33.45 | 33.45 | 553.513 | 553.513 | 553.513 | 1380.6 | 1380.6 | 1 |
| 5v3 | 36.45 | 36.45 | 36.45 | 559.661 | 559.113 | 559.299 | 175.6 | 164.9 | 1 |
| 6v3 | 36.60 | 37.45 | 36.75 | 578.875 | 578.362 | 578.519 | 123.0 | 100.7 | 1 |
| 8v4 | 37.90 | 36.85 | 37.65 | 563.389 | 561.927 | 562.801 | 58.9 | 53.7 | 1 |

## Conclusions
- Training is performed ONCE with full communication (gamma > 0).
- The same trained weights are evaluated under three modes: full_comm, no_comm, and dropout.
- full_comm should provide the best team error; no_comm degrades to MN-equivalent behavior.
- dropout tests robustness of the learned policy to intermittent communication failures.
- d_min tracks minimum inter-pursuer distance to verify collision avoidance.