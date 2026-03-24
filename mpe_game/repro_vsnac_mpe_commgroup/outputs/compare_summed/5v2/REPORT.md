# Generalized Team Communication Report: 5v2

## Scenario
- Pursuers: `5`
- Evaders: `2`
- Group sizes: `[2, 3]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 24.29864882262449, 'end_s': 40.30997672267749, 'isolated_slots': (0,)}, {'group_idx': 1, 'start_s': 24.87421311210548, 'end_s': 37.582633002181396, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 24.29864882262449, 'end_s': 40.30997672267749, 'isolated_slots': [0]}, {'group_idx': 1, 'start_s': 24.87421311210548, 'end_s': 37.582633002181396, 'isolated_slots': [0, 1]}]`

## Results
- Capture time: `None s`
- Final total Eteam: `1482.207125`
- Final max assigned error: `432.343363`
- Final group errors: `[847.579857, 634.627268]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `20`
- Best iteration: `20`
- Final delta per group: `[0.00072842, 0.00068609]`
- Final residual rms per group: `[124.22716882, 171.48722623]`
- Final weight norms: `[0.655864, 0.5761929]`

## Runtime
- Total wall time (s): `33.412`

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