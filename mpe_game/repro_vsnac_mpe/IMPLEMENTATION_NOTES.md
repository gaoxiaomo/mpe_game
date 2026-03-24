# Implementation Notes

## 1) Key files

- `mpe_repro/config.py`: all scenario constants and hyperparameters
- `mpe_repro/dynamics.py`: nonlinear aircraft model
- `mpe_repro/graph_switch.py`: target swap logic (Algorithm 1)
- `mpe_repro/features.py`: critic feature map
- `mpe_repro/controller.py`: policy + stage cost
- `mpe_repro/offpolicy_ls.py`: difference least-squares update
- `mpe_repro/simulator.py`: training/evaluation loop
- `mpe_repro/plotting.py`: figure generation (includes 3v3 GIF)
- `run_repro.py`: one-click experiment runner

## 2) 3v1 critic design

- This implementation uses 3 critics in 3v1 (`W1,s`, `W2,s`, `W3,s`), one per pursuer.
- This matches the paper text around Fig.4 where three converging weight curves are shown.

## 3) Why trajectories are clearer

- XY view is used for paper-like trajectory readability.
- Additional multiview trajectory plots are generated:
  - XY, XH, YH, and H-time in one figure.
- Simulation can stop at capture for trajectory plotting to avoid long post-capture drift.

## 4) Off-policy least-squares form

- Samples are collected from rollout under fixed policy:
  - `phi(x_t)`, `phi(x_{t+1})`, stage cost `r_t`
- Solved equation:
  - `(phi(x_{t+1}) - phi(x_t))^T W = -r_t * dt`
- Regularized least-squares with prior to previous iterate is used for numerical stability.
- Critic update step uses explicit learning rate (`alpha_i`):
  - `W_{i,s+1} = W_{i,s} + alpha_i * (W_i^LS - W_{i,s})`
  - in this reproduction: `alpha_i = 0.01`.

## 5) 3v3 dynamic graph focus

- `scenario_three_pursuer_three_evader()` uses paper-aligned initial states and parameters.
- Initial assignment is paper-style identity (`A_pe = I`).
- Swap threshold is paper-aligned (`50`).
- A paper-style reassignment trigger is configured to obtain `A'_pe = [2,1,0]`
  transition in the trajectory/error figures.
- Graph switching metric uses full-state weighted error; plotting metric uses position-only error.
- Outputs for this part:
  - `fig6_trajectory_3v3_dynamic.png`
  - `fig6b_trajectory_3v3_multiview.png`
  - `fig6_trajectory_3v3_dynamic.gif`
  - `fig7_old_new_errors_3v3.png`
  - `fig8_assignment_switch_3v3.png`
  - `fig9_errors_3v3_fixed.png`
  - `fig10_team_error_compare.png`

## 6) Main tuning knobs

- `LearningParams`:
  - `policy_iterations`
  - `rollout_steps`
  - `critic_learning_rate` (`alpha_i`)
  - `graph_update_interval`
  - `graph_update_start_step`
- `ControlParams`:
  - `u_bar_p`, `u_bar_e`
  - `u_bar_p_policy`, `u_bar_e_policy`
  - `k_pos_*`, `k_vel_*`
  - `q_diag`, `r1_diag`, `r2_diag`
- `ScenarioConfig`:
  - `swap_threshold`
  - `initial_assignment`
  - `displacement_matrix`
  - `evader_motion_mode`
  - `evader_script_amp`, `evader_script_omega`, `evader_script_decay`, `evader_script_mix`
