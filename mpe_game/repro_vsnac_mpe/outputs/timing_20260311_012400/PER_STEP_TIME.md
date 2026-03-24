# Per-step Time Benchmark (local machine)

| Scenario | This repo V-SNAC (ms/step) | Paper V-SNAC (ms/step) | Paper AC (ms/step) | Delta vs paper V-SNAC |
|---|---:|---:|---:|---:|
| 3v1 | 0.623 +/- 0.003 | 25.34 | 36.73 | -97.5% |
| 3v3 | 0.947 +/- 0.021 | 29.17 | 41.95 | -96.8% |

Notes:
- This benchmark uses `evaluate_policy(stop_on_capture=False)` runtime / total simulated steps.
- Hardware and Python runtime differ from paper environment, so absolute ms numbers are for relative local reference.
- AC is not re-implemented in this repo; AC values are cited from paper table for comparison.
