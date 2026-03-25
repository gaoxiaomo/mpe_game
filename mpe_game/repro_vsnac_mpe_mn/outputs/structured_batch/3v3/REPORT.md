# Generalized MPE Case Report: 3v3

## Case
- Pursuers: 3
- Evaders: 3
- Seed: 17
- Assignment mode: `shifted`

## Runtime
- Total wall time (s): 34.575
- Total ms/step: 3.1261
- Train ms/step: 3.2275
- Eval(dynamic) ms/step: 3.2470
- Eval(fixed) ms/step: 2.8076

## Dynamic Evaluation
- Capture time (s): 36.4
- Final Eteam at common horizon: 205.554
- Mean assigned error: 564.434
- Final max assigned error: 102.849
- Switch count: 1
- Switch times (s): [0.0]

## Networks
- V-SNAC critics: 3
- AC estimated networks: 12
- Estimated reduction (%): 75.00

## Fixed-Graph Baseline
- Capture time (s): 66.65
- Final Eteam at common horizon: 207.404
- Mean assigned error: 749.350
- Dynamic graph better on final Eteam: True

## Files
- `summary.json`
- `fig_trajectory_xy.png`
- `fig_trajectory_3d.gif`
- `fig_trajectory_multiview.png`
- `fig_assigned_errors.png`
- `fig_assigned_residual_norm.png`
- `fig_control_inputs.png`
- `fig_control_input_deltas.png`
- `fig_weight_convergence.png`
- `fig_team_error_compare.png`
- `fig_old_new_errors.png`
- `fig_assignment_timeline.png`