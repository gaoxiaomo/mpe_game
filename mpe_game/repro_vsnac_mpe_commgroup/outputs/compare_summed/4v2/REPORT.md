# Generalized Team Communication Report: 4v2

## Scenario
- Pursuers: `4`
- Evaders: `2`
- Group sizes: `[2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 27.90975765523242, 'end_s': 41.48528043899851, 'isolated_slots': (0, 1)}, {'group_idx': 1, 'start_s': 30.511047465846936, 'end_s': 43.669354944230335, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 27.90975765523242, 'end_s': 41.48528043899851, 'isolated_slots': [0, 1]}, {'group_idx': 1, 'start_s': 30.511047465846936, 'end_s': 43.669354944230335, 'isolated_slots': [0, 1]}]`

## Results
- Capture time: `None s`
- Final total Eteam: `1643.994120`
- Final max assigned error: `534.772403`
- Final group errors: `[588.810877, 1055.183243]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `19`
- Best iteration: `19`
- Final delta per group: `[0.00078356, 0.00077976]`
- Final residual rms per group: `[112.13687289, 115.4645463]`
- Final weight norms: `[0.62369576, 0.50768735]`

## Runtime
- Total wall time (s): `30.632`

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