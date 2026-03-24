# Generalized Team Communication Report: 6v3

## Scenario
- Pursuers: `6`
- Evaders: `3`
- Group sizes: `[2, 2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 29.092996751336774, 'end_s': 44.11951348923907, 'isolated_slots': (0,)}, {'group_idx': 1, 'start_s': 28.49095531666444, 'end_s': 40.92081754599263, 'isolated_slots': (0,)}, {'group_idx': 2, 'start_s': 28.89128790995116, 'end_s': 41.526856424182654, 'isolated_slots': (0,)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 26.921028378541973, 'end_s': 42.16620817751283, 'isolated_slots': [0]}, {'group_idx': 1, 'start_s': 33.61437770117574, 'end_s': 49.85950776463127, 'isolated_slots': [0]}, {'group_idx': 2, 'start_s': 28.648597603539976, 'end_s': 41.217183484356525, 'isolated_slots': [0]}]`

## Results
- Capture time: `45.7 s`
- Final total Eteam: `52.736810`
- Final max assigned error: `10.666074`
- Final group errors: `[17.793919, 20.774593, 14.168298]`
- Final group communication ratios: `[1.0, 1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `21`
- Best iteration: `21`
- Final delta per group: `[0.00147781, 0.00111879, 0.00143296]`
- Final residual rms per group: `[124.83307104, 83.01619862, 89.5990767]`
- Final weight norms: `[0.78712363, 0.83297863, 0.65101856]`

## Runtime
- Total wall time (s): `65.498`

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