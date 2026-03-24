# Generalized Team Communication Report: 6v2

## Scenario
- Pursuers: `6`
- Evaders: `2`
- Group sizes: `[3, 3]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 32.95047187978154, 'end_s': 46.02737550687675, 'isolated_slots': (0, 1, 2)}, {'group_idx': 1, 'start_s': 33.36221811508572, 'end_s': 46.253982396516136, 'isolated_slots': (0, 1, 2)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 29.307741311086772, 'end_s': 43.166596657949114, 'isolated_slots': [0, 2]}, {'group_idx': 1, 'start_s': 28.346252756183194, 'end_s': 44.13307540565444, 'isolated_slots': [0, 2]}]`

## Results
- Capture time: `46.5 s`
- Final total Eteam: `55.262566`
- Final max assigned error: `11.037764`
- Final group errors: `[23.892897, 31.36967]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `21`
- Best iteration: `21`
- Final delta per group: `[0.0023876, 0.00186379]`
- Final residual rms per group: `[175.88052035, 71.1102254]`
- Final weight norms: `[0.9635836, 0.94849434]`

## Runtime
- Total wall time (s): `53.079`

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