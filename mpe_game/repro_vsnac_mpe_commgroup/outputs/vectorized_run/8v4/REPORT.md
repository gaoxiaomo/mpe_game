# Generalized Team Communication Report: 8v4

## Scenario
- Pursuers: `8`
- Evaders: `4`
- Group sizes: `[2, 2, 2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 34.154999340728764, 'end_s': 50.0949183935589, 'isolated_slots': (0,)}, {'group_idx': 1, 'start_s': 25.088239328043127, 'end_s': 38.47312536117385, 'isolated_slots': (0,)}, {'group_idx': 2, 'start_s': 32.02218164093956, 'end_s': 49.7318947086867, 'isolated_slots': (0, 1)}, {'group_idx': 3, 'start_s': 23.985737171630955, 'end_s': 40.02491546022701, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 33.36088258760469, 'end_s': 52.52420131660783, 'isolated_slots': [0, 1]}, {'group_idx': 1, 'start_s': 29.751952906460055, 'end_s': 47.6989653933269, 'isolated_slots': [1]}, {'group_idx': 2, 'start_s': 26.322519231246112, 'end_s': 40.835666220486054, 'isolated_slots': [0]}, {'group_idx': 3, 'start_s': 25.061800204137594, 'end_s': 42.19562061835944, 'isolated_slots': [0]}]`

## Results
- Capture time: `71.0 s`
- Final total Eteam: `282.640849`
- Final max assigned error: `128.912550`
- Final group errors: `[14.692814, 23.704308, 236.576133, 7.667593]`
- Final group communication ratios: `[1.0, 1.0, 1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0, 0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `23`
- Best iteration: `23`
- Final delta per group: `[0.00145626, 0.00147044, 0.00139366, 0.00090987]`
- Final residual rms per group: `[91.3684648, 158.7225376, 123.85888075, 68.10243349]`
- Final weight norms: `[0.85090286, 0.79118864, 0.84802164, 0.8061883]`

## Runtime
- Total wall time (s): `98.578`

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