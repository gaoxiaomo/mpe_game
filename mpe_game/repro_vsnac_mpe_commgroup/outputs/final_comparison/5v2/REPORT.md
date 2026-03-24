# Generalized Team Communication Report: 5v2

## Scenario
- Pursuers: `5`
- Evaders: `2`
- Group sizes: `[2, 3]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 24.29864882262449, 'end_s': 40.30997672267749, 'isolated_slots': (0,)}, {'group_idx': 1, 'start_s': 24.87421311210548, 'end_s': 37.582633002181396, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 28.590707819226893, 'end_s': 46.84983261768117, 'isolated_slots': [0]}, {'group_idx': 1, 'start_s': 32.96513242989686, 'end_s': 49.99578019453162, 'isolated_slots': [1, 2]}]`

## Results
- Capture time: `52.15 s`
- Final total Eteam: `22.878258`
- Final max assigned error: `5.141056`
- Final group errors: `[9.62237, 13.255888]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `20`
- Best iteration: `12`
- Final delta per group: `[0.00146384, 0.00228711]`
- Final residual rms per group: `[47.15258037, 58.23644908]`
- Final weight norms: `[1.04314296, 0.98761626]`

## Runtime
- Total wall time (s): `88.476`

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