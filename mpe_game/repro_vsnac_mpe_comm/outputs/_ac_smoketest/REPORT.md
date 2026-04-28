# AC Train Compare: 6v3

- generated_at: 2026-04-27T12:39:28
- case: 6v3
- seed: 11
- AC hyperparams: critic_lr=0.05, actor_lr=0.0, episodes=60, steps=100

## Network inventory
- Traditional AC total networks: 2 * (N_p + N_e) = 18
- AC actor params per network: 18
- AC critic params per network: 78 (quadratic basis on z = [x, u, d])
- AC total scalar parameters: 864
- V-SNAC critics: 6 (one per pursuer)
- V-SNAC features per critic: 6
- V-SNAC total scalar parameters: 36

## Training wall time
- V-SNAC: 4.7 s
- Traditional AC: 18.5 s

## Frozen-policy evaluation (after AC training)
- mean assigned error: 1652.3 m
- min(d_min) over episode: 60.2 m
- capture time (s): None

## ms / step
- AC policy_only:  0.1221
- AC online step:  1.7034
- V-SNAC no_comm:  0.2203
- V-SNAC full_comm:0.4316