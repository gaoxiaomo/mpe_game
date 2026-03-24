# Generalized Team Communication Batch Summary

- Output folder: `E:\毕业设计\mpe_game\repro_vsnac_mpe_commgroup\outputs\team_comm_generalized_20260324_164714`
- Parallel workers: `2`
- Total wall time (s): `321.604`

## Cases
| case | group sizes | capture (s) | final Eteam | final max assigned | switch count | min comm ratio | ms/step |
|---|---|---:|---:|---:|---:|---|---:|
| 4v2 | [2, 2] | 34.40 | 0.000 | 0.000 | 1 | [0.0, 0.0] | 37.8354 |
| 5v2 | [2, 3] | 35.50 | 0.000 | 0.000 | 1 | [0.0, 0.0] | 34.9800 |
| 6v3 | [2, 2, 2] | 35.25 | 0.000 | 0.000 | 1 | [0.0, 0.0, 0.0] | 19.5598 |

## Interpretation
- Each evader owns a fixed number of communication slots, so dynamic reassignment swaps slot occupants instead of changing group size.
- Every group uses a communication-aware team critic with masked decentralized execution, and only same-target pursuers share information.
- Random communication outages are resampled independently for each tracking group in every rollout and evaluation episode.
- Runtime is reduced by vectorized slot-cost computation plus optional process-level parallel case execution.
- Reported horizon-end errors use a zero tail after capture so the convergence plots visibly settle to 0 once capture is achieved.