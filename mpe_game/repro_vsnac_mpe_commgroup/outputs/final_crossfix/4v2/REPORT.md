# Generalized Team Communication Report: 4v2

## Scenario
- Pursuers: `4`
- Evaders: `2`
- Group sizes: `[2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 23.258131379360346, 'end_s': 34.57106703249875, 'isolated_slots': (0, 1)}, {'group_idx': 1, 'start_s': 25.42587288820578, 'end_s': 36.39112912019195, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 26.906425101007613, 'end_s': 41.927089118966514, 'isolated_slots': [0, 1]}, {'group_idx': 1, 'start_s': 23.913314760588666, 'end_s': 36.124524614403526, 'isolated_slots': [0]}]`

## Results
- Capture time: `44.8 s`
- Final total Eteam: `0.000000`
- Final max assigned error: `0.000000`
- Final group errors: `[0.0, 0.0]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `19`
- Best iteration: `12`
- Final delta per group: `[0.00114733, 0.00129136]`
- Final residual rms per group: `[11.6666986, 36.5628864]`
- Final weight norms: `[0.79825818, 0.64230695]`

## Runtime
- Total wall time (s): `49.654`

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