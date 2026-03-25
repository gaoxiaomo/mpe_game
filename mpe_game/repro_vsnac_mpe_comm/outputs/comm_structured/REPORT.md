# Communication-Augmented MPE Batch Report

- Output folder: `outputs/comm_structured`
- Parallel workers: `5`
- Total wall time (s): `210.938`
- Sum of per-case wall times (s): `317.683`

## Case Summary
| case | cap(full) | cap(none) | cap(drop) | err(full) | err(none) | err(drop) | d_min(full) | d_min(none) | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 55.65 | 45.25 | 55.50 | 1737.065 | 1694.103 | 1732.225 | 269.6 | 126.7 | 0 |
| 3v3 | 33.15 | 33.15 | 33.15 | 564.611 | 564.611 | 564.611 | 1393.9 | 1393.9 | 1 |
| 5v3 | 35.55 | 35.55 | 35.55 | 565.487 | 565.299 | 565.212 | 173.5 | 160.5 | 1 |
| 6v3 | 35.60 | 36.40 | 35.75 | 585.013 | 585.398 | 584.785 | 118.0 | 103.9 | 1 |
| 8v4 | 36.50 | 35.55 | 36.25 | 570.231 | 569.637 | 569.820 | 59.5 | 49.9 | 1 |

## Conclusions
- Training is performed ONCE with full communication (gamma > 0).
- The same trained weights are evaluated under three modes: full_comm, no_comm, and dropout.
- full_comm should provide the best team error; no_comm degrades to MN-equivalent behavior.
- dropout tests robustness of the learned policy to intermittent communication failures.
- d_min tracks minimum inter-pursuer distance to verify collision avoidance.