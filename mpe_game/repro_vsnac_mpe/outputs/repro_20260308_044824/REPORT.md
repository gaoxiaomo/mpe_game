# V-SNAC MPE Reproduction Report

- Output folder: `outputs\repro_20260308_044824`
- Quick mode: `False`

## Scenario 1 (3 pursuers, 1 evader)
- Final max assigned error: 179.804
- Final team error: 207.789
- Capture time (s): 50.800000000000004

## Scenario 2 (3 pursuers, 3 evaders)
- Dynamic graph final team error: 34694.656
- Fixed graph final team error ([26] baseline run): 2482.198
- Fixed-reference final team error (same trajectory): 432462.601
- Dynamic graph switch count: 1
- Dynamic graph switch times (s): [6.0]
- Dynamic GIF generated: True

## Network Count
- V-SNAC critics: 3
- AC estimated networks: 12
- Estimated reduction: 75.00%

## Files
- `fig1_trajectory_3v1.png`
- `fig2_errors_3v1.png`
- `fig3_inputs_3v1.png`
- `fig4_weight_conv_3v1.png`
- `fig5_saturation_compare.png`
- `fig6_trajectory_3v3_dynamic.png`
- `fig6_trajectory_3v3_dynamic.gif`
- `fig7_old_new_errors_3v3.png`
- `fig8_assignment_switch_3v3.png`
- `fig9_errors_3v3_fixed.png`
- `fig10_team_error_compare.png`
- `fig11_weight_conv_3v3.png`
- `metrics_summary.json`