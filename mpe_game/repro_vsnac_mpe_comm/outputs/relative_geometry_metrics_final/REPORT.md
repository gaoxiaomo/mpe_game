# Relative Geometry Metrics

## Metric
The relative-geometry metric is defined over same-target pursuer pairs.

e_rel(t)=1/|G_t| sum_(j,k in G_t, j<k) ||(p_j(t)-p_k(t))-(r_k(t)-r_j(t))||_2

where G_t contains pursuer pairs assigned to the same evader at time t.

## Standard Cases

| Case | full sustained capture/s | no sustained capture/s | full mean assigned err | no mean assigned err | full final rel-geom err/m | no final rel-geom err/m | full min d_min/m | no min d_min/m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 42.400 | 43.950 | 1675.267 | 1672.600 | 12.383 | 14.752 | 298.80 | 260.13 |
| 3v3 | 33.750 | 33.750 | 548.182 | 548.182 | — | — | 1361.29 | 1361.29 |
| 5v3 | 36.800 | 36.800 | 553.126 | 552.191 | 10.176 | 10.639 | 190.45 | 167.56 |
| 6v3 | 36.950 | 37.750 | 572.258 | 571.348 | 10.426 | 10.253 | 110.85 | 110.48 |
| 8v4 | 52.950 | 52.100 | 556.491 | 555.698 | 15.574 | 16.083 | 105.09 | 61.32 |

## Collision Case

| Method | sustained capture/s | final rel-geom err/m | min d_min/m |
|---|---:|---:|---:|
| no analytic | 61.900 | 15.575 | 51.934 |
| smooth analytic | 64.450 | 70.096 | 63.590 |