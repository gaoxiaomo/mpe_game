# Generalized Team Communication Report: 5v2

## Scenario
- Pursuers: `5`
- Evaders: `2`
- Group sizes: `[2, 3]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 20.24887401885374, 'end_s': 33.59164726889791, 'isolated_slots': (0,)}, {'group_idx': 1, 'start_s': 20.728510926754566, 'end_s': 31.318860835151163, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 23.825589849355744, 'end_s': 39.04152718140098, 'isolated_slots': [0]}, {'group_idx': 1, 'start_s': 27.470943691580718, 'end_s': 41.66315016210968, 'isolated_slots': [1, 2]}]`

## Results
- Capture time: `35.5 s`
- Final total Eteam: `0.000000`
- Final max assigned error: `0.000000`
- Final group errors: `[0.0, 0.0]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `10`
- Best iteration: `7`
- Final delta per group: `[0.00142259, 0.00273751]`
- Final residual rms per group: `[96.76378641, 148.29487514]`
- Final weight norms: `[0.97155531, 1.25770345]`

## Runtime
- Total wall time (s): `195.888`

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