# V-SNAC MPE Reproduction Report

- Output folder: `outputs\repro_20260308_223406`
- Quick mode: `True`

## Scenario 1 (3 pursuers, 1 evader)
- Final max assigned error: 156.143
- Final team error: 344.756
- Capture time (s): 21.55

## Scenario 2 (3 pursuers, 3 evaders)
- Dynamic graph final team error: 480.967
- Fixed graph final team error ([26] baseline run): 482.148
- Fixed-reference final team error (same trajectory): 480.967
- Dynamic graph switch count: 0
- Dynamic graph switch times (s): []
- Dynamic GIF generated: True

## Network Count
- V-SNAC critics: 3
- AC estimated networks: 12
- Estimated reduction: 75.00%

## Files
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