# Generalized Team Communication Batch Summary

- Output folder: `outputs/final_gif`
- Parallel workers: `1`
- Total wall time (s): `1185.208`

## Cases
| case | group sizes | capture (s) | final Eteam | final max assigned | switch count | min comm ratio | ms/step |
|---|---|---:|---:|---:|---:|---|---:|
| 4v2 | [2, 2] | 47.65 | 24.509 | 8.574 | 1 | [0.0, 0.0] | 4.5339 |
| 5v2 | [2, 3] | 52.15 | 22.878 | 5.141 | 1 | [0.0, 0.0] | 5.5971 |
| 6v2 | [3, 3] | 50.40 | 36.227 | 8.107 | 1 | [0.0, 0.0] | 6.3624 |
| 6v3 | [2, 2, 2] | 38.55 | 26.084 | 5.618 | 1 | [0.0, 0.0, 0.0] | 6.6247 |
| 8v4 | [2, 2, 2, 2] | 45.90 | 56.437 | 14.164 | 1 | [0.0, 0.0, 0.0, 0.0] | 8.5305 |

## Interpretation
- Each evader owns a fixed number of communication slots, so dynamic reassignment swaps slot occupants instead of changing group size.
- Every group uses a communication-aware team critic with masked decentralized execution, and only same-target pursuers share information.
- Random communication outages are resampled independently for each tracking group in every rollout and evaluation episode.
- Runtime is reduced by vectorized slot-cost computation plus optional process-level parallel case execution.
- Reported horizon-end errors use a zero tail after capture so the convergence plots visibly settle to 0 once capture is achieved.