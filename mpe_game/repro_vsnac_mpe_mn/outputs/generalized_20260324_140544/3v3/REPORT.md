# Generalized MPE Case Report: 3v3

## Case
- Pursuers: 3
- Evaders: 3
- Seed: 17
- Assignment mode: `shifted`

## Runtime
- Total wall time (s): 14.962
- Total ms/step: 2.6717
- Train ms/step: 2.6862
- Eval(dynamic) ms/step: 2.7827
- Eval(fixed) ms/step: 2.5528

## Dynamic Evaluation
- Capture time (s): 34.25
- Final Eteam at common horizon: 192.821
- Mean assigned error: 673.653
- Final max assigned error: 82.138
- Switch count: 1
- Switch times (s): [0.0]

## Networks
- V-SNAC critics: 3
- AC estimated networks: 12
- Estimated reduction (%): 75.00

## Fixed-Graph Baseline
- Capture time (s): 36.35
- Final Eteam at common horizon: 208.130
- Mean assigned error: 886.503
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