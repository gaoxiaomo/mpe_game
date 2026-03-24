# Generalized Team Communication Report: 6v2

## Scenario
- Pursuers: `6`
- Evaders: `2`
- Group sizes: `[3, 3]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 20.369475192528245, 'end_s': 32.981303102078755, 'isolated_slots': (0, 1, 2)}, {'group_idx': 1, 'start_s': 23.28314592237787, 'end_s': 35.47283066454108, 'isolated_slots': (1, 2)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 22.434190315451644, 'end_s': 35.138506814594024, 'isolated_slots': [0, 2]}, {'group_idx': 1, 'start_s': 26.147483698014046, 'end_s': 41.82213797644501, 'isolated_slots': [0, 1, 2]}]`

## Results
- Capture time: `24.65 s`
- Final total Eteam: `0.000000`
- Final max assigned error: `0.000000`
- Final group errors: `[0.0, 0.0]`
- Final group communication ratios: `[1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `21`
- Best iteration: `13`
- Final delta per group: `[0.00242534, 0.00176291]`
- Final residual rms per group: `[131.38341327, 103.17567084]`
- Final weight norms: `[1.11629081, 0.98173972]`

## Runtime
- Total wall time (s): `84.494`

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