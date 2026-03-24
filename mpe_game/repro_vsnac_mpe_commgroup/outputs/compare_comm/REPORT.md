# Generalized Team Communication Batch Summary

- Output folder: `outputs/compare_comm`
- Parallel workers: `2`
- Total wall time (s): `114.884`

## Cases
| case | group sizes | capture (s) | final Eteam | final max assigned | switch count | min comm ratio | ms/step |
|---|---|---:|---:|---:|---:|---|---:|
| 4v2 | [2, 2] | 46.95 | 53.274 | 20.157 | 1 | [0.0, 0.0] | 3.4941 |
| 5v2 | [2, 3] | 54.60 | 61.581 | 26.852 | 1 | [0.0, 0.0] | 3.6171 |
| 6v3 | [2, 2, 2] | 49.35 | 51.429 | 15.049 | 1 | [0.0, 0.0, 0.0] | 4.0188 |

## Interpretation
- Each evader owns a fixed number of communication slots, so dynamic reassignment swaps slot occupants instead of changing group size.
- Every group uses a communication-aware team critic with masked decentralized execution, and only same-target pursuers share information.
- Random communication outages are resampled independently for each tracking group in every rollout and evaluation episode.
- Runtime is reduced by vectorized slot-cost computation plus optional process-level parallel case execution.
- Reported horizon-end errors use a zero tail after capture so the convergence plots visibly settle to 0 once capture is achieved.