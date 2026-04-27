# Many-to-Many 6v3 Collision-Crossing Demo

## Scenario
- Three evaders move in parallel with fixed assignment.
- Each evader is pursued by exactly two pursuers.
- In every group, the two pursuers start on opposite sides of their desired left/right offsets, so reaching the target offsets requires an explicit lane swap near the evader.

## Metrics
- Unsafe-band threshold: 50.0 m
- Smooth analytic gamma: 0.300
- No analytic term: min d_min = 0.995 m, time below d_safe = 1.500 s, longest unsafe streak = 1.500 s, sustained capture time = 61.050000000000004, first-hit time = 33.25
- Smooth analytic term: min d_min = 82.404 m, time below d_safe = 0.000 s, longest unsafe streak = 0.000 s, sustained capture time = 64.15, first-hit time = 64.15

## Interpretation
- This is a many-to-many stress case, but the collision pressure is intentionally concentrated inside each pursuit group so that the effect of the analytic pairwise term is directly visible.
- Without the analytic term, each pair performs a near-collision lane swap while tracking its own offset.
- With the smooth analytic term, the group-level crossing is still completed, but the minimum inter-pursuer distance stays outside the chosen unsafe band.
- The safer behavior comes with slower convergence, so the demo should be presented as a safety-versus-aggressiveness tradeoff, not as a universal performance gain.
- Because the rollout continues after first entry into the capture radius, the report distinguishes between first-hit time and sustained capture time; the latter is used as the stricter comparison metric.