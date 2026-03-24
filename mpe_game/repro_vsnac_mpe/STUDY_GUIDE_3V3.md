# 3v3 Study Guide

This note is for quickly understanding and modifying the 3v3 reproduction.

## 1) Where to run

```powershell
cd E:\...\mpe_game\repro_vsnac_mpe
python .\run_repro.py
```

Latest full run from this update:
- `outputs/repro_20260308_231924`

## 2) Core pipeline (3v3)

1. Initialize pursuer/evader states.
2. Roll out fixed policy and collect off-policy samples.
3. Solve difference least squares for each critic.
4. Update critic weights with learning rate (`alpha_i=0.01`).
5. Run dynamic assignment swaps during pursuit.
6. Evaluate trajectory, switching, and cohesion metrics.

## 3) Files to read first

- `run_repro.py`
- `mpe_repro/simulator.py`
- `mpe_repro/graph_switch.py`
- `mpe_repro/offpolicy_ls.py`
- `mpe_repro/plotting.py`

## 4) Figure mapping

- `fig6_trajectory_3v3_dynamic.png` (XY)
- `fig6b_trajectory_3v3_multiview.png` (XY/XH/YH/H-time)
- `fig6_trajectory_3v3_dynamic.gif` (dynamic links)
- `fig7_old_new_errors_3v3.png`
- `fig8_assignment_switch_3v3.png`
- `fig9_errors_3v3_fixed.png`
- `fig10_team_error_compare.png`
- `fig11_weight_conv_3v3.png`

## 5) What is tuned now

- Paper-aligned initial states for 3v3 and identity initial assignment (`[0,1,2]`).
- Paper-aligned swap threshold (`50`) and `r_{j,i}=[100,50,50,0,0,0]`.
- Paper-style matrix transition trigger is enabled:
  - `[0,1,2] -> [2,1,0]` at configured step.
- Evader handling is pursuer-centric:
  - virtual `u_e` from HJI/tanh is used in critic update;
  - real evader motion can be scripted via scenario parameters.
- `graph_update_interval=1` for frequent swap checks.

## 6) Metrics to inspect

Open `metrics_summary.json`:
- `scenario_3v3_dynamic.switch_count`
- `scenario_3v3_dynamic.switch_times_s`
- `scenario_3v3_dynamic.initial_xy_distance_matrix`
- `scenario_3v3_dynamic.eval.final_team_error`
- `scenario_3v3_fixed_baseline.eval.final_team_error`

## 7) Fast knobs

- Later/earlier switching: `run_repro.py` -> `graph_update_start_step`
- Swap aggressiveness: `mpe_repro/config.py` -> `swap_threshold`, `max_switch_worsening`
- Evader trajectory shaping: `mpe_repro/config.py` -> `evader_script_*`
- Control aggressiveness: `ControlParams` gains and saturations
- Display scale: use XY and multiview figures together
