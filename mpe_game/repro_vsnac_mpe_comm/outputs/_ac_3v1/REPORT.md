# AC Train Compare: 3v1

- generated_at: 2026-04-27T12:40:13
- case: 3v1
- seed: 11
- AC hyperparams: critic_lr=0.05, actor_lr=0.0, episodes=40, steps=80

## Network inventory
- Traditional AC total networks: 2 * (N_p + N_e) = 8
- AC actor params per network: 18
- AC critic params per network: 78 (quadratic basis on z = [x, u, d])
- AC total scalar parameters: 384
- V-SNAC critics: 3 (one per pursuer)
- V-SNAC features per critic: 6
- V-SNAC total scalar parameters: 18

## Training wall time
- V-SNAC: 3.3 s
- Traditional AC: 8.1 s

## Frozen-policy evaluation (after AC training)
- mean assigned error: 2725.5 m
- min(d_min) over episode: 360.9 m
- capture time (s): None

## ms / step
- AC policy_only:  0.1181
- AC online step:  1.2658
- V-SNAC no_comm:  0.2518
- V-SNAC full_comm:0.4161