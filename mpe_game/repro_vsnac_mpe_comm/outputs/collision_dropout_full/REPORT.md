# Collision-suite dropout robustness report

- generated_at: 2026-04-27T12:42:18
- d_safe_report (NMAC, 500 ft): 150.0 m
- d_safe_amp (smooth-amp scale): 150.0 m
- gamma: 1.5
- stochastic seeds per random mode: 5

## Per-scenario summary

| Tag | Scenario | mode | median min(d_min) [m] | median time below d_safe [s] |
|---|---|---|---:|---:|
| C1 | collision_C1_head_on_2v1 | no_comm | 2.9 | 0.75 |
| C1 | collision_C1_head_on_2v1 | full_comm | 215.3 | 0.00 |
| C1 | collision_C1_head_on_2v1 | iid_15 | 207.4 | 0.00 |
| C1 | collision_C1_head_on_2v1 | iid_30 | 79.1 | 1.95 |
| C1 | collision_C1_head_on_2v1 | iid_50 | 3.3 | 1.10 |
| C1 | collision_C1_head_on_2v1 | persistent_15 | 215.3 | 0.00 |
| C1 | collision_C1_head_on_2v1 | persistent_30 | 215.3 | 0.00 |
| C1 | collision_C1_head_on_2v1 | periodic_off25 | 221.4 | 0.00 |
| C1 | collision_C1_head_on_2v1 | periodic_off50 | 103.9 | 1.50 |

| C2 | collision_C2_trio_3v1 | no_comm | 7.6 | 1.35 |
| C2 | collision_C2_trio_3v1 | full_comm | 128.1 | 1.75 |
| C2 | collision_C2_trio_3v1 | iid_15 | 126.2 | 1.80 |
| C2 | collision_C2_trio_3v1 | iid_30 | 96.4 | 2.10 |
| C2 | collision_C2_trio_3v1 | iid_50 | 7.8 | 1.40 |
| C2 | collision_C2_trio_3v1 | persistent_15 | 128.1 | 1.75 |
| C2 | collision_C2_trio_3v1 | persistent_30 | 132.7 | 1.75 |
| C2 | collision_C2_trio_3v1 | periodic_off25 | 117.1 | 1.85 |
| C2 | collision_C2_trio_3v1 | periodic_off50 | 27.2 | 1.85 |

| C3 | collision_C3_stacked_4v1 | no_comm | 6.4 | 1.10 |
| C3 | collision_C3_stacked_4v1 | full_comm | 179.0 | 0.00 |
| C3 | collision_C3_stacked_4v1 | iid_15 | 174.7 | 0.00 |
| C3 | collision_C3_stacked_4v1 | iid_30 | 162.7 | 0.00 |
| C3 | collision_C3_stacked_4v1 | iid_50 | 64.1 | 1.90 |
| C3 | collision_C3_stacked_4v1 | persistent_15 | 197.3 | 0.00 |
| C3 | collision_C3_stacked_4v1 | persistent_30 | 144.0 | 0.95 |
| C3 | collision_C3_stacked_4v1 | periodic_off25 | 157.8 | 0.00 |
| C3 | collision_C3_stacked_4v1 | periodic_off50 | 48.2 | 2.55 |

| C4 | collision_C4_parallel_conflict_2v2 | no_comm | 14.2 | 0.65 |
| C4 | collision_C4_parallel_conflict_2v2 | full_comm | 278.5 | 0.00 |
| C4 | collision_C4_parallel_conflict_2v2 | iid_15 | 142.5 | 0.40 |
| C4 | collision_C4_parallel_conflict_2v2 | iid_30 | 54.9 | 2.05 |
| C4 | collision_C4_parallel_conflict_2v2 | iid_50 | 54.4 | 0.75 |
| C4 | collision_C4_parallel_conflict_2v2 | persistent_15 | 278.5 | 0.00 |
| C4 | collision_C4_parallel_conflict_2v2 | persistent_30 | 308.1 | 0.00 |
| C4 | collision_C4_parallel_conflict_2v2 | periodic_off25 | 73.1 | 1.20 |
| C4 | collision_C4_parallel_conflict_2v2 | periodic_off50 | 45.4 | 3.55 |

| C5 | collision_C5_vertical_funnel_3v1 | no_comm | 3.8 | 2.15 |
| C5 | collision_C5_vertical_funnel_3v1 | full_comm | 84.2 | 16.40 |
| C5 | collision_C5_vertical_funnel_3v1 | iid_15 | 75.7 | 18.55 |
| C5 | collision_C5_vertical_funnel_3v1 | iid_30 | 34.7 | 25.35 |
| C5 | collision_C5_vertical_funnel_3v1 | iid_50 | 7.0 | 4.80 |
| C5 | collision_C5_vertical_funnel_3v1 | persistent_15 | 84.2 | 16.40 |
| C5 | collision_C5_vertical_funnel_3v1 | persistent_30 | 84.2 | 10.80 |
| C5 | collision_C5_vertical_funnel_3v1 | periodic_off25 | 35.8 | 22.10 |
| C5 | collision_C5_vertical_funnel_3v1 | periodic_off50 | 1.3 | 8.50 |

| C6 | collision_C6_asymmetric_5v2 | no_comm | 4.2 | 1.15 |
| C6 | collision_C6_asymmetric_5v2 | full_comm | 138.6 | 0.65 |
| C6 | collision_C6_asymmetric_5v2 | iid_15 | 175.4 | 0.00 |
| C6 | collision_C6_asymmetric_5v2 | iid_30 | 145.6 | 0.60 |
| C6 | collision_C6_asymmetric_5v2 | iid_50 | 52.1 | 2.65 |
| C6 | collision_C6_asymmetric_5v2 | persistent_15 | 164.4 | 0.00 |
| C6 | collision_C6_asymmetric_5v2 | persistent_30 | 143.3 | 1.00 |
| C6 | collision_C6_asymmetric_5v2 | periodic_off25 | 150.4 | 0.00 |
| C6 | collision_C6_asymmetric_5v2 | periodic_off50 | 31.2 | 2.65 |
