# 追逃博弈算法测试指南

## 概述

本文档描述了从简单到复杂的测试文件层次结构，帮助你逐步验证和调试算法。

**当前状态**：代码已恢复为论文原版公式（追捕者和逃避者控制都用负号）

---

## 测试文件层次结构

```
Level 1: 线性系统测试 (最简单)
├── test_simple_1v1.py          # 线性双积分器 + RL
├── test_game_rl_simple.py      # 饱和限制对比
├── test_paper_exact.py         # 论文原版公式验证
├── test_linear_improved.py     # 增加迭代次数测试 (200次)
└── test_linear_compare_signs.py # ⭐ 符号对比测试 (关键!)

Level 2: 动力学验证
└── test_proportional_control.py  # 非线性飞行器 + 比例控制

Level 3: 非线性系统 + RL (待调试)
├── simple_1v1_direct.py        # 简化非线性RL
├── simple_1v1_complete.py      # 完整非线性RL
└── test_simple_1v1_nonlinear.py # 非线性RL尝试

Level 4: 完整系统
├── simulation.py               # 核心仿真模块
├── main.py                     # 主入口
└── test_scenario1.py           # 场景1测试
```

---

## Level 1: 线性系统测试

### 1.1 `test_simple_1v1.py`

**目的**: 验证RL算法在最简单线性系统上的正确性

**系统**:
- 2D双积分器: `ẋ = Ax + Bu`
- 状态: `[x, y, vx, vy]`
- 控制: `[ax, ay]`

**关键代码段**:
```python
# 动力学
self.A = [[0,0,1,0], [0,0,0,1], [0,0,0,0], [0,0,0,0]]
self.B = [[0,0], [0,0], [1,0], [0,1]]

# 控制公式 (公式40)
u_p = -u_bar_p * np.tanh(rho_p)  # 追捕者
u_e = u_bar_e * np.tanh(rho_e)   # 逃避者 (测试版用正号)
```

**运行**:
```bash
python test_simple_1v1.py
```

**预期结果**:
- 当 `u_bar_p > u_bar_e` 时，追捕应该成功
- P矩阵应该正定并收敛

**待验证**: 使用论文原版负号是否也能工作

---

### 1.2 `test_game_rl_simple.py`

**目的**: 对比不同饱和限制配置下的博弈结果

**测试配置**:
| 场景 | u_bar_p | u_bar_e | 预期结果 |
|------|---------|---------|----------|
| 追捕者优势 | 25 | 15 | 追捕成功 |
| 能力相等 | 20 | 20 | 不确定 |
| 逃避者优势 | 15 | 25 | 追捕失败 |

**关键类**:
- `DoubleIntegrator2D`: 线性动力学
- `ValueFunction`: 二次型值函数 `V(x) = x^T P x`
- `SaturatedGameController`: 饱和控制器

**运行**:
```bash
python test_game_rl_simple.py
```

**观察指标**:
1. P矩阵特征值（正定性）
2. 距离变化趋势
3. 权重收敛性

---

### 1.3 `test_paper_exact.py`

**目的**: 严格按论文公式测试（两者都用负号）

**关键代码**:
```python
# 公式(40) - 论文原版
u_p = -u_bar_p * np.tanh(rho_p)  # 负号
u_e = -u_bar_e * np.tanh(rho_e)  # 负号

# 状态误差动力学
x_dot = A @ x + B @ u_p - B @ u_e
#                       ^
#              注意这里是减号！
```

**核心问题**:
论文中对抗性是通过状态误差动力学中的负号实现的：
```
ẋ̃ = ... + g_p u_p - g_e u_e
```

当两者都用负号时：
- `g_p u_p` 项贡献 `-∇V` 方向（减小V）
- `-g_e u_e = -g_e(-ū_e tanh) = +g_e ū_e tanh` 贡献 `+∇V` 方向（增大V）

**运行**:
```bash
python test_paper_exact.py
```

**待分析**:
- 为什么P矩阵不正定？
- Bellman方程的构建是否正确？

---

## Level 2: 动力学验证

### 2.1 `test_proportional_control.py`

**目的**: 验证非线性飞行器动力学模型正确性

**方法**: 使用简单比例控制（不需要RL）

**控制策略**:
```python
# 追捕者：向逃避者方向加速
u_p = -K_pos * pos_error - K_vel * vel_error

# 逃避者：远离追捕者
u_e = -K_pos * pos_error * 0.5
```

**运行**:
```bash
python test_proportional_control.py
```

**预期结果**:
- 距离应该减小（验证动力学正确）
- 控制应该在饱和限制内

**两个测试场景**:
1. 简单初始条件: `P=[600,0,200]`, `E=[0,0,200]`
2. 论文初始条件: `P=[600,-1900,300]`, `E=[500,1500,200]`

---

## Level 3: 非线性系统 + RL

### 3.1 `simple_1v1_direct.py`

**目的**: 简化的1v1非线性RL控制

**特点**:
- 使用原始归一化策略
- 增大Q矩阵补偿

**关键参数**:
```python
Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
```

**运行**:
```bash
python simple_1v1_direct.py
```

**当前问题**: 误差增加而非减小

---

### 3.2 `simple_1v1_complete.py`

**目的**: 完整的1v1非线性RL

**使用的缩放策略**:
```python
self.pos_scale = 1e-6   # 位置项缩放
self.vel_scale = 1e-4   # 速度项缩放
self.cross_scale = 1e-5 # 交叉项缩放
```

**运行**:
```bash
python simple_1v1_complete.py
```

---

### 3.3 `test_simple_1v1_nonlinear.py`

**目的**: 非线性飞行器动力学 + RL控制

**关键修改**: 不使用归一化

**运行**:
```bash
python test_simple_1v1_nonlinear.py
```

---

## Level 4: 完整系统

### 4.1 `simulation.py`

**核心模块**，包含:

1. **`offline_training()`**: 离线训练
   - 网格采样
   - 最小二乘求解权重
   - 对应论文公式(41)

2. **`run_simulation()`**: 在线仿真
   - 使用训练好的权重
   - RK4积分
   - 可选在线学习

3. **`run_scenario1()`**: 场景1仿真
4. **`run_scenario2()`**: 场景2仿真
5. **`run_saturation_comparison()`**: 饱和对比

---

### 4.2 `main.py`

**主入口**，运行完整仿真流程

**运行**:
```bash
python main.py
```

**输出**:
- 3D轨迹图
- 状态误差曲线
- 控制输入曲线
- 权重收敛曲线

---

### 4.3 核心模块说明

| 文件 | 功能 | 对应公式 |
|------|------|----------|
| `dynamics.py` | 飞行器动力学 | 公式(53), (54) |
| `agents.py` | 智能体类 | - |
| `graph.py` | 目标图算法 | Algorithm 1 |
| `networks.py` | 值函数网络 | 公式(37)-(39) |
| `controller.py` | 最优控制器 | 公式(40) |
| `config.py` | 参数配置 | Table I |
| `visualization.py` | 可视化 | - |

---

## 诊断工具

### `diagnose_original.py`

**功能**: 诊断原始代码问题

**包含的诊断**:
1. 控制方向检查
2. 动力学模型验证
3. 训练过程检查
4. 仿真行为分析

**运行**:
```bash
python diagnose_original.py
```

---

## 推荐调试顺序

```
Step 1: 运行 test_linear_compare_signs.py ⭐ 最重要!
        → 直接对比负号vs正号的效果
        → 观察P矩阵正定性和追捕结果

Step 2: 分析论文公式
        → 检查论文中代价函数定义
        → 检查状态误差动力学 ẋ̃ = ... + g_p u_p - g_e u_e
        → 理解负号为什么导致P不正定

Step 3: 运行 test_proportional_control.py
        → 确认动力学模型正确
        
Step 4: 运行 diagnose_original.py
        → 检查归一化问题
        → 检查梯度量级

Step 5: 决定修改策略
        → 如果确认正号是正确的：修改controller.py和simulation.py
        → 如果坚持论文原版：需要分析论文细节

Step 6: 运行完整系统 main.py
```

---

## 关键实验结果

### 符号对比测试 (test_linear_compare_signs.py)

运行命令: `python test_linear_compare_signs.py`

| 方法 | P正定 | 最终距离 | 状态 |
|------|-------|----------|------|
| **负号（论文）** | ❌ NO | 248.13 | UNSTABLE |
| **正号** | ✅ YES | 0.01 | SUCCESS |

**P矩阵特征值**:
- 负号: `[-3.51, -3.47, 3.43, 3.48]` → 不正定
- 正号: `[0.45, 0.45, 1.67, 1.68]` → 正定

---

## 关键问题清单

### 需要回答的问题

1. **论文公式两者都用负号时，为什么P矩阵不正定？**
   - 检查Bellman方程构建
   - 检查代价函数符号
   - **关键**: 状态误差动力学是 `g_p u_p - g_e u_e`，是否已正确考虑?

2. **论文中的代价函数定义**
   - 追捕者最小化、逃避者最大化
   - 值函数 V(x) 的符号约定
   - 检查论文公式(12)-(15)的推导

3. **归一化导致梯度过小的问题如何解决？**
   - 调整state_scale
   - 调整Q矩阵
   - 或不使用归一化

### 可能的修改点

| 文件 | 位置 | 修改内容 |
|------|------|----------|
| `networks.py` | 第24-26行 | state_scale值 |
| `config.py` | 第30行 | Q矩阵权重 |
| `simulation.py` | 第60-63行 | 采样范围 |
| `controller.py` | 第96行 | 逃避者控制符号 |

---

## 参考文献

- 论文PDF: `1Approximate_Optimal_Strategy_for_Multiagent_System_PursuitEvasion_Game.pdf`
- 算法文档: `ALGORITHM_DOCUMENTATION.md`
- 测试报告: `TEST_REPORT.md`
- MATLAB参考: `refer/` 目录

---

**文档版本**: 1.0  
**最后更新**: 2026-02-01

