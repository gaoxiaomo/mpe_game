# Full-Communication Head-to-Head: 5v3

## Scope
- Only the full-communication setting is compared.
- Both methods produce rollout figures on the same scenario and same dynamic target-assignment logic.
- The traditional baseline is the paper-inspired online actor-critic/Q-learning implementation.

## Metrics
- V-SNAC capture time (s): 36.800000000000004
- Traditional AC capture time (s): None
- V-SNAC mean assigned error: 553.010
- Traditional AC mean assigned error: 38821.784
- V-SNAC min d_min: 185.129
- Traditional AC min d_min: 194.234

## Runtime
- V-SNAC full-comm policy ms/step: 0.509039
- Traditional AC full-comm online ms/step: 3.487520
- Traditional / V-SNAC ratio: 6.851x
- V-SNAC eval wall time (s): 2.410
- Traditional AC eval wall time (s): 4.410
- Traditional AC warmup wall time (s): 6.697

## Output Folders
- V-SNAC plots: `outputs\full_comm_compare_final\5v3\vsnac_full_comm`
- Traditional AC plots: `outputs\full_comm_compare_final\5v3\traditional_ac_full_comm`