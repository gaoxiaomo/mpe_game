# MPE Reproduction (Refactored)

This folder is a clean reproduction workspace for:

- `1Approximate_Optimal_Strategy_for_Multiagent_System_PursuitEvasion_Game.pdf`

## What is implemented

- Nonlinear aircraft dynamics: Eq. (53), (54)
- Dynamic target graph switching: Algorithm 1
- Off-policy training with trajectory rollouts + difference least squares
- V-SNAC style critic update
- 3v1 and 3v3 experiments
- Saturation comparison (`u_p > u_e`, `u_p = u_e`, `u_p < u_e`)
- 3v3 dynamic-graph visualization:
  - switch timeline
  - animated XY trajectory with pursuer-target links
  - XY/XH/YH/H-time multiview panels
- Figures aligned to paper intent:
  - Fig.1,2,3,4,5,6,7,8,9,10,11 style outputs

## Important alignment choices

- 3v1 parameters follow paper values:
  - `alpha_i = 0.01`
  - input bounds: `u_bar_p = 25`, `u_bar_e = 15`
  - policy-scale saturation in actor: `u_bar_p(actor) = 35`, `u_bar_e(actor) = 15`
  - expected displacement:
    - `r_1,1 = [50, 10, 0, 0, 0, 0]`
    - `r_2,1 = [10, 50, 0, 0, 0, 0]`
    - `r_3,1 = [-10, 0, -50, 0, 0, 0]`
- 3v3 scenario uses paper initial states, `u_bar_p = u_bar_e = 20`, `r_{j,i} = [100, 50, 50, 0, 0, 0]`, and swap threshold `50`.
- 3v3 exchange trigger is aligned to paper matrix transition:
  - initial `A_pe = I`
  - after trigger step, assignment switches to `[2,1,0]` (paper `A'_pe` form).
- Pursuer-centric evader handling is implemented:
  - critic update uses virtual `u_e` from HJI/tanh policy;
  - real evader motion can be scripted (configurable) for trajectory alignment.
- 3v1 still uses three critics (`W1,s`, `W2,s`, `W3,s`), one per pursuer.

## Run (Windows PowerShell)

```powershell
cd E:\...\mpe_game\repro_vsnac_mpe
python .\run_repro.py
```

Quick run:

```powershell
python .\run_repro.py --quick
```

Custom output folder:

```powershell
python .\run_repro.py --output .\outputs\my_run
```

## Output files

Each run writes to `outputs/repro_<timestamp>/`:

- `fig1_trajectory_3v1.png`
- `fig1b_trajectory_3v1_multiview.png`
- `fig2_errors_3v1.png`
- `fig3_inputs_3v1.png`
- `fig3b_inputs_3v1_h_channel.png`
- `fig4_weight_conv_3v1.png`
- `fig5_saturation_compare.png`
- `fig6_trajectory_3v3_dynamic.png`
- `fig6b_trajectory_3v3_multiview.png`
- `fig6_trajectory_3v3_dynamic.gif`
- `fig7_old_new_errors_3v3.png`
- `fig8_assignment_switch_3v3.png`
- `fig9_errors_3v3_fixed.png`
- `fig10_team_error_compare.png`
- `fig11_weight_conv_3v3.png`
- `metrics_summary.json`
- `REPORT.md`
