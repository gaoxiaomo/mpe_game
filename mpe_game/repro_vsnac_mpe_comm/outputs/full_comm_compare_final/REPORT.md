# Full-Communication Head-to-Head Batch Report

- Only the full-communication condition is compared.
- V-SNAC uses the communication-aware structured value function.
- Traditional AC uses the paper-inspired online actor-critic/Q-learning baseline.

| case | cap(vsnac) | cap(ac) | err(vsnac) | err(ac) | d_min(vsnac) | d_min(ac) | V-SNAC ms/step | AC ms/step | AC / V-SNAC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3v1 | 52.05 | - | 1734.301 | 29210.564 | 275.8 | 1852.3 | 0.294496 | 2.121388 | 7.203x |
| 3v3 | 33.75 | - | 548.182 | 28399.061 | 1361.3 | 1976.3 | 0.105888 | 1.602750 | 15.136x |
| 5v3 | 36.80 | - | 553.010 | 38821.784 | 185.1 | 194.2 | 0.509039 | 3.487520 | 6.851x |
| 6v3 | 36.90 | - | 572.029 | 35840.717 | 112.8 | 427.4 | 0.161624 | 0.897024 | 5.550x |
| 8v4 | 37.15 | - | 558.391 | 37779.476 | 57.0 | 215.2 | 0.713620 | 1.540262 | 2.158x |

## Conclusion
- This report no longer mixes in `no_comm` or `dropout` cases.
- The timing comparison is full-comm V-SNAC policy evaluation versus full-comm traditional AC online update.
- Each case folder contains aligned V-SNAC and traditional AC figure sets for direct visual comparison.