# Full-Communication Head-to-Head: 8v4

## Scope
- Only the full-communication setting is compared.
- Both methods produce rollout figures on the same scenario and same dynamic target-assignment logic.
- The traditional baseline is the paper-inspired online actor-critic/Q-learning implementation.

## Metrics
- V-SNAC capture time (s): 37.15
- Traditional AC capture time (s): None
- V-SNAC mean assigned error: 558.391
- Traditional AC mean assigned error: 37779.476
- V-SNAC min d_min: 56.951
- Traditional AC min d_min: 215.228

## Runtime
- V-SNAC full-comm policy ms/step: 0.713620
- Traditional AC full-comm online ms/step: 1.540262
- Traditional / V-SNAC ratio: 2.158x
- V-SNAC eval wall time (s): 11.518
- Traditional AC eval wall time (s): 22.151
- Traditional AC warmup wall time (s): 42.698

## Output Folders
- V-SNAC plots: `outputs\full_comm_compare_final\8v4\vsnac_full_comm`
- Traditional AC plots: `outputs\full_comm_compare_final\8v4\traditional_ac_full_comm`