# Generalized MPE Case Report: 8v4

## Case
- Pursuers: 8
- Evaders: 4
- Seed: 211
- Assignment mode: `random`

## Runtime
- Total wall time (s): 25.656
- Total ms/step: 4.1381
- Train ms/step: 3.7398
- Eval(dynamic) ms/step: 5.1087
- Eval(fixed) ms/step: 3.4934

## Dynamic Evaluation
- Capture time (s): 77.2
- Final Eteam at common horizon: 581.266
- Mean assigned error: 2135.017
- Final max assigned error: 85.905
- Switch count: 2
- Switch times (s): [0.0, 6.7]

## Networks
- V-SNAC critics: 8
- AC estimated networks: 24
- Estimated reduction (%): 66.67

## Fixed-Graph Baseline
- Capture time (s): 83.4
- Final Eteam at common horizon: 623.587
- Mean assigned error: 2291.853
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