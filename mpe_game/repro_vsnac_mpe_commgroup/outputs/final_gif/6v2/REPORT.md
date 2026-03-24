# Generalized Team Communication Report: 6v2

## Scenario
- Pursuers: `6`
- Evaders: `2`
- Group sizes: `[3, 3]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 24.443370231033896, 'end_s': 39.57756372249451, 'isolated_slots': (0, 1, 2)}, {'group_idx': 1, 'start_s': 27.939775106853443, 'end_s': 42.56739679744929, 'isolated_slots': (1, 2)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 26.921028378541973, 'end_s': 42.16620817751283, 'isolated_slots': [0, 2]}, {'group_idx': 1, 'start_s': 31.376980437616854, 'end_s': 50.18656557173402, 'isolated_slots': [0, 1, 2]}]`

## Results
- Capture time: `50.4 s`
- Final total Eteam: `36.226578`
- Final max assigned error: `8.106785`
- Final group errors: `[17.537888, 18.68869]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `21`
- Best iteration: `13`
- Final delta per group: `[0.0025697, 0.00192079]`
- Final residual rms per group: `[133.90527372, 99.21009987]`
- Final weight norms: `[1.13726457, 0.9097356]`

## Runtime
- Total wall time (s): `99.444`

## Interpretation
- Training uses a curriculum: early iterations use full intra-group communication, later iterations fine-tune under random per-group outages.
- Only pursuers assigned to the same evader communicate; inter-group coordination is handled only by the dynamic slot-swap graph.
- Evaluation uses paper-style zero tail after capture, so curves continue to 0 once all assigned errors enter the capture radius.

## Files
- `summary.json`
- `fig_trajectory_xy.png`
- `fig_team_errors.png`
- `fig_assignment_timeline.png`
- `fig_comm_ratio.png`
- `fig_weight_convergence.png`
- `fig_assigned_errors.png`