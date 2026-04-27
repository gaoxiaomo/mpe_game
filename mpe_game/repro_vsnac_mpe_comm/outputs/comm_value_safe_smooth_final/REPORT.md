# Communication-Augmented MPE Batch Report

- Output folder: `outputs\comm_value_safe_smooth_final`
- Parallel workers: `2`
- Total wall time (s): `124.158`
- Sum of per-case wall times (s): `70.314`

## Case Summary
| case | cap(full) | cap(none) | cap(drop) | err(full) | err(none) | err(drop) | d_min(full) | d_min(none) | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 52.10 | 47.15 | 51.15 | 1734.144 | 1693.242 | 1728.226 | 284.0 | 135.3 | 0 |
| 3v3 | 33.75 | 33.75 | 33.75 | 548.182 | 548.182 | 548.182 | 1361.3 | 1361.3 | 1 |
| 5v3 | 36.80 | 36.80 | 36.80 | 553.126 | 552.191 | 552.616 | 190.4 | 167.6 | 1 |
| 6v3 | 36.95 | 37.75 | 37.10 | 572.258 | 571.348 | 571.738 | 110.9 | 110.5 | 1 |
| 8v4 | 37.10 | 36.25 | 36.90 | 558.114 | 556.369 | 557.304 | 59.5 | 58.7 | 1 |

## Conclusions
- Training is performed ONCE with the structured communication-aware value function (gamma > 0).
- The analytic pairwise term uses a smooth approximation of the previous hard max distance factor.
- The same trained weights are evaluated under three modes: full_comm, no_comm, and dropout.
- full_comm is expected to improve inter-pursuer separation while preserving tracking performance; no_comm degrades to the baseline behavior.
- dropout tests robustness of the learned policy to intermittent communication failures.
- d_min tracks minimum inter-pursuer distance to verify collision avoidance.