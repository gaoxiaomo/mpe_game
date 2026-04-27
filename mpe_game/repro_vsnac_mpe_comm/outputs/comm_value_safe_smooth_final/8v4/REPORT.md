# Communication-Augmented MPE Case Report: 8v4

## Case
- Pursuers: 8
- Evaders: 4
- Seed: 47
- Assignment mode: `shifted`
- Gamma (training): 0.3
- d_safe: 100.0
- Pair factor: `smooth_max`

## Runtime
- Total wall time (s): 17.004
- Total ms/step: 0.9343
- Train ms/step: 1.0338

## Evaluation Modes
### full_comm
- Capture time (s): 37.1
- Final Eteam: 502.909
- Mean assigned error: 558.114
- Min d_min: 59.481
- Mean formation error: 120.087

### no_comm
- Capture time (s): 36.25
- Final Eteam: 502.555
- Mean assigned error: 556.369
- Min d_min: 58.702
- Mean formation error: 0.000

### dropout
- Capture time (s): 36.9
- Final Eteam: 502.692
- Mean assigned error: 557.304
- Min d_min: 62.932
- Mean formation error: 81.910

## Networks
- V-SNAC critics: 8
- AC estimated networks: 24
- Estimated reduction (%): 66.67

## Switch Info
- Switch count: 1
- Switch times (s): [0.0]