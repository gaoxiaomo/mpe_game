# Unified Safety Stress Report

## Reduced 2D Collision Illustration
- Purpose: mechanism illustration in a reduced-order crossing geometry.
- d_safe: 500.0 m
- No analytic term: min d_min = 0.039 m, time below d_safe = 15.600 s
- Smooth analytic term: min d_min = 1743.043 m, time below d_safe = 0.000 s

## Full 6D Unsafe-Band Stress Test
- Purpose: full-order evidence in the actual MPE simulator.
- d_safe: 30.0 m
- Gamma (smooth analytic): 1.500
- No analytic term: min d_min = 0.785 m, time below d_safe = 19.100 s, longest unsafe streak = 18.000 s, capture time = 33.1
- Smooth analytic term: min d_min = 33.638 m, time below d_safe = 0.000 s, longest unsafe streak = 0.000 s, capture time = 33.65

## Interpretation
- The reduced 2D model is used only to show an explicit collision-suppression mechanism.
- The full 6D scenario is the main evidence: the no-analytic controller remains inside the unsafe band for a long interval, whereas the smooth analytic term keeps the team outside the chosen safety threshold.