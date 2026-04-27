# Full-Communication Head-to-Head: 3v1

## Scope
- Only the full-communication setting is compared.
- Both methods produce rollout figures on the same scenario and same dynamic target-assignment logic.
- The traditional baseline is the paper-inspired online actor-critic/Q-learning implementation.

## Metrics
- V-SNAC capture time (s): 52.050000000000004
- Traditional AC capture time (s): None
- V-SNAC mean assigned error: 1734.301
- Traditional AC mean assigned error: 29210.564
- V-SNAC min d_min: 275.844
- Traditional AC min d_min: 1852.308

## Runtime
- V-SNAC full-comm policy ms/step: 0.294496
- Traditional AC full-comm online ms/step: 2.121388
- Traditional / V-SNAC ratio: 7.203x
- V-SNAC eval wall time (s): 5.546
- Traditional AC eval wall time (s): 10.193
- Traditional AC warmup wall time (s): 16.890

## Output Folders
- V-SNAC plots: `outputs\full_comm_compare_final\3v1\vsnac_full_comm`
- Traditional AC plots: `outputs\full_comm_compare_final\3v1\traditional_ac_full_comm`