# Generalized MPE Case Report: 4v2

## Case
- Pursuers: 4
- Evaders: 2
- Seed: 111
- Assignment mode: `random`

## Runtime
- Total wall time (s): 36.904
- Total ms/step: 3.1012
- Train ms/step: 3.1305
- Eval(dynamic) ms/step: 3.2629
- Eval(fixed) ms/step: 2.8735

## Dynamic Evaluation
- Capture time (s): 72.4
- Final Eteam at common horizon: 352.504
- Mean assigned error: 1726.556
- Final max assigned error: 109.959
- Switch count: 1
- Switch times (s): [0.0]

## Networks
- V-SNAC critics: 4
- AC estimated networks: 12
- Estimated reduction (%): 66.67

## Fixed-Graph Baseline
- Capture time (s): 78.4
- Final Eteam at common horizon: 343.608
- Mean assigned error: 1847.393
- Dynamic graph better on final Eteam: False

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