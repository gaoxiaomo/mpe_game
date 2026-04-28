# Assignment Compare: 12v6

- generated_at: 2026-04-27T16:39:02
- case: 12v6

## Offline comparison on random cost matrices
- trials: 50
- pairwise_swap mean gap: 13.03%, max gap: 44.32%
- critic_warm_auction mean gap: 3.74%, max gap: 22.90%
- pairwise_swap mean iterations: 3.12
- auction mean iterations: 488.42
- timings: pairwise=15.2 ms / hungarian=3.9 ms / auction=268.7 ms total

## Live MPE rollout (each solver drives the dynamic graph)
- pairwise_swap: mean_err = 1944.0 ± 0.0 m, avg min(d_min) = 61.3 m, avg switches = 1.0, captures = [None, None, None], wall = 12.62s
- hungarian: mean_err = 1614.8 ± 0.0 m, avg min(d_min) = 66.3 m, avg switches = 2.0, captures = [105.15, 105.15, 105.15], wall = 12.77s
- critic_warm_auction: mean_err = 1639.4 ± 0.0 m, avg min(d_min) = 66.4 m, avg switches = 1.0, captures = [None, None, None], wall = 55.77s