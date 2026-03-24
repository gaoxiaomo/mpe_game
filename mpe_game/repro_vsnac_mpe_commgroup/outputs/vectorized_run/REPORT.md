# Generalized Team Communication Batch Summary

- Output folder: `outputs/vectorized_run`
- Parallel workers: `2`
- Total wall time (s): `203.060`

## Cases
| case | group sizes | capture (s) | final Eteam | final max assigned | switch count | min comm ratio | ms/step |
|---|---|---:|---:|---:|---:|---|---:|
| 4v2 | [2, 2] | 46.95 | 53.274 | 20.157 | 1 | [0.0, 0.0] | 3.1045 |
| 5v2 | [2, 3] | 54.60 | 61.581 | 26.852 | 1 | [0.0, 0.0] | 3.1914 |
| 6v2 | [3, 3] | 46.50 | 55.263 | 11.038 | 1 | [0.0, 0.0] | 3.3960 |
| 6v3 | [2, 2, 2] | 45.70 | 52.737 | 10.666 | 1 | [0.0, 0.0, 0.0] | 4.1905 |
| 8v4 | [2, 2, 2, 2] | 71.00 | 282.641 | 128.913 | 1 | [0.0, 0.0, 0.0, 0.0] | 5.1993 |

## Interpretation
- Each evader owns a fixed number of communication slots, so dynamic reassignment swaps slot occupants instead of changing group size.
- Every group uses a communication-aware team critic with masked decentralized execution, and only same-target pursuers share information.
- Random communication outages are resampled independently for each tracking group in every rollout and evaluation episode.
- Runtime is reduced by vectorized slot-cost computation plus optional process-level parallel case execution.
- Reported horizon-end errors use a zero tail after capture so the convergence plots visibly settle to 0 once capture is achieved.