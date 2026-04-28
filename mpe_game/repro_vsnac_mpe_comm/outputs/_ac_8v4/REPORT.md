# AC Train Compare: 8v4

- generated_at: 2026-04-27T12:40:46
- case: 8v4
- seed: 11
- AC hyperparams: critic_lr=0.05, actor_lr=0.0, episodes=60, steps=100

## Network inventory
- Traditional AC total networks: 2 * (N_p + N_e) = 24
- AC actor params per network: 18
- AC critic params per network: 78 (quadratic basis on z = [x, u, d])
- AC total scalar parameters: 1152
- V-SNAC critics: 8 (one per pursuer)
- V-SNAC features per critic: 6
- V-SNAC total scalar parameters: 48

## Training wall time
- V-SNAC: 5.8 s
- Traditional AC: 21.3 s

## Frozen-policy evaluation (after AC training)
- mean assigned error: 1516.0 m
- min(d_min) over episode: 88.5 m
- capture time (s): None

## ms / step
- AC policy_only:  0.1240
- AC online step:  2.2369
- V-SNAC no_comm:  0.2294
- V-SNAC full_comm:0.4142