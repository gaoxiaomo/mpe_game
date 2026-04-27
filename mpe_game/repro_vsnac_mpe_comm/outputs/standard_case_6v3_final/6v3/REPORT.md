# Communication-Augmented MPE Case Report: 6v3

## Case
- Pursuers: 6
- Evaders: 3
- Seed: 37
- Assignment mode: `shifted`
- Gamma (training): 0.3
- d_safe: 100.0
- Pair factor: `smooth_max`

## Runtime
- Total wall time (s): 16.692
- Total ms/step: 0.9733
- Train ms/step: 1.0532

## Evaluation Modes
### full_comm
- Capture time (s): 36.95
- Final Eteam: 363.691
- Mean assigned error: 572.258
- Min d_min: 110.853
- Mean formation error: 114.091

### no_comm
- Capture time (s): 37.75
- Final Eteam: 363.474
- Mean assigned error: 571.348
- Min d_min: 110.479
- Mean formation error: 0.000

### dropout
- Capture time (s): 37.1
- Final Eteam: 363.508
- Mean assigned error: 571.738
- Min d_min: 94.460
- Mean formation error: 78.084

## Networks
- V-SNAC critics: 6
- AC estimated networks: 18
- Estimated reduction (%): 66.67

## Switch Info
- Switch count: 1
- Switch times (s): [0.0]