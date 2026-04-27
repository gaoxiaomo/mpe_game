# Communication-Augmented MPE Case Report: 8v4

## Case
- Pursuers: 8
- Evaders: 4
- Seed: 25
- Assignment mode: `shifted`
- Gamma (training): 0.3
- d_safe: 100.0
- Pair factor: `smooth_max`

## Runtime
- Total wall time (s): 18.543
- Total ms/step: 1.0188
- Train ms/step: 1.1183

## Evaluation Modes
### full_comm
- Capture time (s): 34.65
- Final Eteam: 511.836
- Mean assigned error: 556.491
- Min d_min: 105.086
- Mean formation error: 99.465

### no_comm
- Capture time (s): 34.2
- Final Eteam: 512.248
- Mean assigned error: 555.698
- Min d_min: 61.316
- Mean formation error: 0.000

### dropout
- Capture time (s): 34.550000000000004
- Final Eteam: 512.007
- Mean assigned error: 556.090
- Min d_min: 95.014
- Mean formation error: 66.702

## Networks
- V-SNAC critics: 8
- AC estimated networks: 24
- Estimated reduction (%): 66.67

## Switch Info
- Switch count: 1
- Switch times (s): [0.0]