# Full-Communication Head-to-Head: 3v3

## Scope
- Only the full-communication setting is compared.
- Both methods produce rollout figures on the same scenario and same dynamic target-assignment logic.
- The traditional baseline is the paper-inspired online actor-critic/Q-learning implementation.

## Metrics
- V-SNAC capture time (s): 33.75
- Traditional AC capture time (s): None
- V-SNAC mean assigned error: 548.182
- Traditional AC mean assigned error: 28399.061
- V-SNAC min d_min: 1361.290
- Traditional AC min d_min: 1976.250

## Runtime
- V-SNAC full-comm policy ms/step: 0.105888
- Traditional AC full-comm online ms/step: 1.602750
- Traditional / V-SNAC ratio: 15.136x
- V-SNAC eval wall time (s): 7.431
- Traditional AC eval wall time (s): 3.277
- Traditional AC warmup wall time (s): 8.614

## Output Folders
- V-SNAC plots: `outputs\full_comm_compare_final\3v3\vsnac_full_comm`
- Traditional AC plots: `outputs\full_comm_compare_final\3v3\traditional_ac_full_comm`