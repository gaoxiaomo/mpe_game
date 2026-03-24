# Generalized Team Communication Report: 4v2

## Scenario
- Pursuers: `4`
- Evaders: `2`
- Group sizes: `[2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 27.90975765523242, 'end_s': 41.48528043899851, 'isolated_slots': (0, 1)}, {'group_idx': 1, 'start_s': 30.511047465846936, 'end_s': 43.669354944230335, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 32.28771012120914, 'end_s': 50.31250694275982, 'isolated_slots': [0, 1]}, {'group_idx': 1, 'start_s': 28.6959777127064, 'end_s': 43.34942953728424, 'isolated_slots': [0]}]`

## Results
- Capture time: `47.65 s`
- Final total Eteam: `24.508973`
- Final max assigned error: `8.573894`
- Final group errors: `[10.103531, 14.405442]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `19`
- Best iteration: `12`
- Final delta per group: `[0.00147583, 0.00121453]`
- Final residual rms per group: `[54.96733639, 32.46271549]`
- Final weight norms: `[0.93134758, 0.77587294]`

## Runtime
- Total wall time (s): `64.089`

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