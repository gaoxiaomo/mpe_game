# Communication-Augmented MPE Batch Report

- Output folder: `outputs\standard_case_3v1_seed35_gamma01_final`
- Parallel workers: `1`
- Total wall time (s): `35.591`
- Sum of per-case wall times (s): `11.085`

## Case Summary
| case | cap(full) | cap(none) | cap(drop) | err(full) | err(none) | err(drop) | d_min(full) | d_min(none) | switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 42.40 | 43.95 | 42.05 | 1675.267 | 1672.600 | 1674.634 | 298.8 | 260.1 | 0 |

## Conclusions
- Training is performed ONCE with the structured communication-aware value function (gamma > 0).
- The analytic pairwise term uses a smooth approximation of the previous hard max distance factor.
- The same trained weights are evaluated under three modes: full_comm, no_comm, and dropout.
- full_comm is expected to improve inter-pursuer separation while preserving tracking performance; no_comm degrades to the baseline behavior.
- dropout tests robustness of the learned policy to intermittent communication failures.
- d_min tracks minimum inter-pursuer distance to verify collision avoidance.