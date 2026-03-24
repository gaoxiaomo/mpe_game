# Generalized Team Communication Batch Summary

- Output folder: `outputs/compare_summed`
- Parallel workers: `2`
- Total wall time (s): `81.950`

## Cases
| case | group sizes | capture (s) | final Eteam | final max assigned | switch count | min comm ratio | ms/step |
|---|---|---:|---:|---:|---:|---|---:|
| 4v2 | [2, 2] | - | 1643.994 | 534.772 | 1 | [1.0, 1.0] | 2.4196 |
| 5v2 | [2, 3] | - | 1482.207 | 432.343 | 1 | [1.0, 1.0] | 2.3697 |
| 6v3 | [2, 2, 2] | - | 1704.996 | 293.596 | 1 | [1.0, 1.0, 1.0] | 2.7223 |

## Interpretation
- Each evader owns a fixed number of communication slots, so dynamic reassignment swaps slot occupants instead of changing group size.
- Every group uses a communication-aware team critic with masked decentralized execution, and only same-target pursuers share information.
- Random communication outages are resampled independently for each tracking group in every rollout and evaluation episode.
- Runtime is reduced by vectorized slot-cost computation plus optional process-level parallel case execution.
- Reported horizon-end errors use a zero tail after capture so the convergence plots visibly settle to 0 once capture is achieved.