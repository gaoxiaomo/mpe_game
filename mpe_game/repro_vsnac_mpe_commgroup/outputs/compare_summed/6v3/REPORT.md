# Generalized Team Communication Report: 6v3

## Scenario
- Pursuers: `6`
- Evaders: `3`
- Group sizes: `[2, 2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 24.237374708740152, 'end_s': 42.87885345213063, 'isolated_slots': (1,)}, {'group_idx': 1, 'start_s': 27.99218887322616, 'end_s': 44.35702012170925, 'isolated_slots': (1,)}, {'group_idx': 2, 'start_s': 29.90450573108865, 'end_s': 42.53608375590326, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 24.237374708740152, 'end_s': 42.87885345213063, 'isolated_slots': [1]}, {'group_idx': 1, 'start_s': 27.99218887322616, 'end_s': 44.35702012170925, 'isolated_slots': [1]}, {'group_idx': 2, 'start_s': 29.90450573108865, 'end_s': 42.53608375590326, 'isolated_slots': [0, 1]}]`

## Results
- Capture time: `None s`
- Final total Eteam: `1704.996247`
- Final max assigned error: `293.596357`
- Final group errors: `[556.034555, 566.324938, 582.636754]`
- Final group communication ratios: `[1.0, 1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `21`
- Best iteration: `21`
- Final delta per group: `[0.00091931, 0.00074615, 0.00059541]`
- Final residual rms per group: `[145.77248683, 97.86781366, 116.88307025]`
- Final weight norms: `[0.66005834, 0.57477122, 0.45628111]`

## Runtime
- Total wall time (s): `42.549`

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