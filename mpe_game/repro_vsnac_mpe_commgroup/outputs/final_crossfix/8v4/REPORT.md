# Generalized Team Communication Report: 8v4

## Scenario
- Pursuers: `8`
- Evaders: `4`
- Group sizes: `[2, 2, 2, 2]`
- Assignment mode: `shifted`
- Layout mode: `structured`
- Reference communication windows: `[{'group_idx': 0, 'start_s': 28.462499450607304, 'end_s': 41.74576532796575, 'isolated_slots': (0,)}, {'group_idx': 1, 'start_s': 20.90686610670261, 'end_s': 32.06093780097821, 'isolated_slots': (0,)}, {'group_idx': 2, 'start_s': 26.685151367449635, 'end_s': 41.44324559057226, 'isolated_slots': (0, 1)}, {'group_idx': 3, 'start_s': 19.988114309692463, 'end_s': 33.35409621685584, 'isolated_slots': (0, 1)}]`
- Evaluation communication windows: `[{'group_idx': 0, 'start_s': 27.800735489670576, 'end_s': 43.77016776383985, 'isolated_slots': [0, 1]}, {'group_idx': 1, 'start_s': 24.793294088716713, 'end_s': 39.749137827772415, 'isolated_slots': [1]}, {'group_idx': 2, 'start_s': 21.935432692705092, 'end_s': 34.029721850405046, 'isolated_slots': [0]}, {'group_idx': 3, 'start_s': 20.884833503447993, 'end_s': 35.1630171819662, 'isolated_slots': [0]}]`

## Results
- Capture time: `47.15 s`
- Final total Eteam: `0.000000`
- Final max assigned error: `0.000000`
- Final group errors: `[0.0, 0.0, 0.0, 0.0]`
- Final group communication ratios: `[1.0, 1.0, 1.0, 1.0]`
- Final group estimate errors: `[0.0, 0.0, 0.0, 0.0]`
- Switch count: `2`
- Switch times (s): `[0.0, 23.400000000000002]`

## Training
- Iterations executed: `23`
- Best iteration: `14`
- Final delta per group: `[0.0010437, 0.00126225, 0.00150224, 0.00120065]`
- Final residual rms per group: `[60.60348676, 71.40980363, 111.11757385, 103.56141419]`
- Final weight norms: `[0.87562363, 0.70412654, 0.95859161, 0.74393068]`

## Runtime
- Total wall time (s): `140.698`

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