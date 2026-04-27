# Communication-Augmented MPE Batch Report

- Output folder: `outputs\standard_case_6v3_final`
- Parallel workers: `1`
- Total wall time (s): `48.613`
- Sum of per-case wall times (s): `16.692`

## Case Summary
| case | cap(full) | cap(none) | cap(drop) | err(full) | err(none) | err(drop) | d_min(full) | d_min(none) | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6v3 | 36.95 | 37.75 | 37.10 | 572.258 | 571.348 | 571.738 | 110.9 | 110.5 | 1 |

## Conclusions
- Training is performed ONCE with the structured communication-aware value function (gamma > 0).
- The analytic pairwise term uses a smooth approximation of the previous hard max distance factor.
- The same trained weights are evaluated under three modes: full_comm, no_comm, and dropout.
- full_comm is expected to improve inter-pursuer separation while preserving tracking performance; no_comm degrades to the baseline behavior.
- dropout tests robustness of the learned policy to intermittent communication failures.
- d_min tracks minimum inter-pursuer distance to verify collision avoidance.