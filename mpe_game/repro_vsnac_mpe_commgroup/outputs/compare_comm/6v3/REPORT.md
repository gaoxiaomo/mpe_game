# Generalized Team Communication Report: 6v3

## Scenario
- Pursuers: `6`
- Evaders: `3`
- Group sizes: `[2, 2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 24.237374708740152, 'end_s': 42.87885345213063, 'isolated_slots': (1,)}, {'group_idx': 1, 'start_s': 27.99218887322616, 'end_s': 44.35702012170925, 'isolated_slots': (1,)}, {'group_idx': 2, 'start_s': 29.90450573108865, 'end_s': 42.53608375590326, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 29.307741311086772, 'end_s': 43.166596657949114, 'isolated_slots': [0]}, {'group_idx': 1, 'start_s': 29.080233974206855, 'end_s': 44.13854435692191, 'isolated_slots': [1]}, {'group_idx': 2, 'start_s': 31.53007309733382, 'end_s': 50.15765178552641, 'isolated_slots': [0, 1]}]`

## Results
- Capture time: `49.35 s`
- Final total Eteam: `51.428833`
- Final max assigned error: `15.048648`
- Final group errors: `[15.830843, 24.042761, 11.555229]`
- Final group communication ratios: `[1.0, 1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `21`
- Best iteration: `21`
- Final delta per group: `[0.00148421, 0.00142046, 0.0011569]`
- Final residual rms per group: `[94.56384795, 38.24342334, 43.8511384]`
- Final weight norms: `[0.80915216, 0.71130656, 0.72157401]`

## Runtime
- Total wall time (s): `62.813`

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