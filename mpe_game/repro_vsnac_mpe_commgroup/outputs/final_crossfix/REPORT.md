# Generalized Team Communication Batch Summary

- Output folder: `outputs/final_crossfix`
- Parallel workers: `1`
- Total wall time (s): `450.385`

## Cases
| case | group sizes | capture (s) | final Eteam | final max assigned | switch count | min comm ratio | ms/step |
|---|---|---:|---:|---:|---:|---|---:|
| 4v2 | [2, 2] | 44.80 | 0.000 | 0.000 | 1 | [0.0, 0.0] | 4.0501 |
| 5v2 | [2, 3] | 50.30 | 0.000 | 0.000 | 1 | [0.0, 0.0] | 5.0369 |
| 6v2 | [3, 3] | 24.65 | 0.000 | 0.000 | 1 | [0.0, 1.0] | 5.5479 |
| 6v3 | [2, 2, 2] | 49.75 | 0.000 | 0.000 | 1 | [0.0, 0.0, 0.0] | 5.9261 |
| 8v4 | [2, 2, 2, 2] | 47.15 | 0.000 | 0.000 | 2 | [0.0, 0.0, 0.0, 0.0] | 7.5807 |

## Interpretation
- Each evader owns a fixed number of communication slots, so dynamic reassignment swaps slot occupants instead of changing group size.
- Every group uses a communication-aware team critic with masked decentralized execution, and only same-target pursuers share information.
- Random communication outages are resampled independently for each tracking group in every rollout and evaluation episode.
- Runtime is reduced by vectorized slot-cost computation plus optional process-level parallel case execution.
- Reported horizon-end errors use a zero tail after capture so the convergence plots visibly settle to 0 once capture is achieved.