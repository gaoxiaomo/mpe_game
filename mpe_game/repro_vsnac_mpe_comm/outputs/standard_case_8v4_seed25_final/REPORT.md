# Communication-Augmented MPE Batch Report

- Output folder: `outputs\standard_case_8v4_seed25_final`
- Parallel workers: `1`
- Total wall time (s): `53.093`
- Sum of per-case wall times (s): `18.543`

## Case Summary
| case | cap(full) | cap(none) | cap(drop) | err(full) | err(none) | err(drop) | d_min(full) | d_min(none) | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8v4 | 34.65 | 34.20 | 34.55 | 556.491 | 555.698 | 556.090 | 105.1 | 61.3 | 1 |

## Conclusions
- Training is performed ONCE with the structured communication-aware value function (gamma > 0).
- The analytic pairwise term uses a smooth approximation of the previous hard max distance factor.
- The same trained weights are evaluated under three modes: full_comm, no_comm, and dropout.
- full_comm is expected to improve inter-pursuer separation while preserving tracking performance; no_comm degrades to the baseline behavior.
- dropout tests robustness of the learned policy to intermittent communication failures.
- d_min tracks minimum inter-pursuer distance to verify collision avoidance.