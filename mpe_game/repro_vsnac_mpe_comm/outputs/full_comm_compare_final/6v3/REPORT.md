# Full-Communication Head-to-Head: 6v3

## Scope
- Only the full-communication setting is compared.
- Both methods produce rollout figures on the same scenario and same dynamic target-assignment logic.
- The traditional baseline is the paper-inspired online actor-critic/Q-learning implementation.

## Metrics
- V-SNAC capture time (s): 36.9
- Traditional AC capture time (s): None
- V-SNAC mean assigned error: 572.029
- Traditional AC mean assigned error: 35840.717
- V-SNAC min d_min: 112.798
- Traditional AC min d_min: 427.366

## Runtime
- V-SNAC full-comm policy ms/step: 0.161624
- Traditional AC full-comm online ms/step: 0.897024
- Traditional / V-SNAC ratio: 5.550x
- V-SNAC eval wall time (s): 11.453
- Traditional AC eval wall time (s): 18.872
- Traditional AC warmup wall time (s): 34.000

## Output Folders
- V-SNAC plots: `outputs\full_comm_compare_final\6v3\vsnac_full_comm`
- Traditional AC plots: `outputs\full_comm_compare_final\6v3\traditional_ac_full_comm`