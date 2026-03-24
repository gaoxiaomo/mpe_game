# Generalized Team Communication Report: 6v3

## Scenario
- Pursuers: `6`
- Evaders: `3`
- Group sizes: `[2, 2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 20.19781225728346, 'end_s': 35.73237787677552, 'isolated_slots': (1,)}, {'group_idx': 1, 'start_s': 23.326824061021796, 'end_s': 36.9641834347577, 'isolated_slots': (1,)}, {'group_idx': 2, 'start_s': 24.920421442573875, 'end_s': 35.44673646325272, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 24.423117759238977, 'end_s': 35.97216388162426, 'isolated_slots': [0]}, {'group_idx': 1, 'start_s': 24.23352831183905, 'end_s': 36.78212029743493, 'isolated_slots': [1]}, {'group_idx': 2, 'start_s': 26.275060914444847, 'end_s': 41.79804315460534, 'isolated_slots': [0, 1]}]`

## Results
- Capture time: `35.25 s`
- Final total Eteam: `0.000000`
- Final max assigned error: `0.000000`
- Final group errors: `[0.0, 0.0, 0.0]`
- Final group communication ratios: `[1.0, 1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0, 0.0]`
- Switch count: `1`
- Switch times (s): `[0.0]`

## Training
- Iterations executed: `11`
- Best iteration: `8`
- Final delta per group: `[0.00175439, 0.00198764, 0.00191113]`
- Final residual rms per group: `[106.96304254, 123.15508017, 101.71049722]`
- Final weight norms: `[1.03329371, 0.98902665, 0.99312779]`

## Runtime
- Total wall time (s): `123.031`

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