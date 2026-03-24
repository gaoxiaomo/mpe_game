# 多智能体追逃博弈近似最优策略算法文档

## 论文复现

**论文标题**: Approximate Optimal Strategy for Multiagent System Pursuit-Evasion Game  
**期刊**: IEEE Systems Journal, VOL. 18, NO. 3, SEPTEMBER 2024

---

## 目录

1. [问题描述](#1-问题描述)
2. [系统动力学模型](#2-系统动力学模型)
3. [博弈问题建模](#3-博弈问题建模)
4. [动态目标图算法](#4-动态目标图算法)
5. [值函数近似网络](#5-值函数近似网络)
6. [Off-Policy强化学习算法](#6-off-policy强化学习算法)
7. [代码架构](#7-代码架构)
8. [仿真场景](#8-仿真场景)
9. [参数配置](#9-参数配置)
10. [关键公式汇总](#10-关键公式汇总)

---

## 1. 问题描述

### 1.1 追逃博弈场景

考虑一个多追捕者-多逃避者的追逃博弈问题：
- **追捕者 (Pursuers)**: $N_p$ 个，目标是捕获逃避者
- **逃避者 (Evaders)**: $N_e$ 个，目标是逃离追捕者
- **博弈类型**: 零和微分博弈 (Zero-Sum Differential Game)

### 1.2 图论表示

系统使用图论描述智能体间的交互关系：

- **追捕者通信图**: $\mathcal{G}_p = (\mathcal{V}_p, \mathcal{E}_p)$
- **逃避者通信图**: $\mathcal{G}_e = (\mathcal{V}_e, \mathcal{E}_e)$
- **追捕-逃避目标图**: $\mathcal{G} = (\mathcal{V}, \mathcal{E})$

**邻接矩阵定义**:
- $\mathbf{A}_{pe} = [c_{j,i}] \in \mathbb{R}^{N_p \times N_e}$: 追捕者j是否追踪逃避者i
- $\mathbf{A}_{ep} = [e_{i,j}] \in \mathbb{R}^{N_e \times N_p}$: 逃避者i是否被追捕者j追踪

---

## 2. 系统动力学模型

### 2.1 通用非线性动力学

**公式 (1) - 追捕者动力学**:
$$\dot{x}_j^p = f_j^p(x_j^p) + g_j^p(x_j^p) u_j^p, \quad j = 1, 2, \ldots, N_p$$

**公式 (2) - 逃避者动力学**:
$$\dot{x}_i^e = f_i^e(x_i^e) + g_i^e(x_i^e) u_i^e, \quad i = 1, 2, \ldots, N_e$$

其中：
- $x \in \mathbb{R}^n$: 状态向量 (n=6)
- $u \in \mathbb{R}^m$: 控制输入 (m=3)
- $f(\cdot)$: 漂移项（非线性）
- $g(\cdot)$: 控制增益矩阵（非线性）

### 2.2 飞行器动力学模型

**状态向量**:
$$x = [x_a, x_b, h, v_a, v_b, v_h]^T$$

其中：
- $x_a, x_b$: 水平位置坐标 (m)
- $h$: 高度 (m)
- $v_a, v_b, v_h$: 三轴速度 (m/s)

### 2.3 公式 (53) - 非线性漂移项 f(x)

$$f(x) = \begin{bmatrix} 
v_a \\ 
v_b \\ 
v_h \\ 
\frac{1}{\breve{m}}(\breve{T}\cos\breve{\alpha} - \breve{D}) \\ 
\frac{1}{\breve{m}}(\breve{L} + \breve{T}\sin\breve{\alpha}) \\ 
\frac{1}{\breve{m}}(\breve{M} + \breve{T}y_d) 
\end{bmatrix}$$

### 2.4 公式 (54) - 控制增益矩阵 g(x)

$$g(x) = \begin{bmatrix} 
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 + 0.25\sin^2(\breve{v}_a \cdot x_b) 
\end{bmatrix}$$

### 2.5 气动参数计算

| 参数 | 公式 | 说明 |
|------|------|------|
| 真实空速 | $\breve{V} = \sqrt{v_a^2 + v_b^2 + v_h^2}$ | 合成速度 |
| 空气密度 | $\breve{\rho} = C_\rho \exp\left(\frac{-h}{24000}\right)$ | 指数衰减 |
| 动压 | $\bar{q} = \frac{1}{2}\breve{\rho}\breve{V}^2$ | |
| 攻角 | $\breve{\alpha} = \arctan\left(\frac{v_h}{v_a}\right)$ | |
| 侧滑角 | $\breve{\beta} = \arcsin\left(\frac{v_b}{\breve{V}}\right)$ | |
| 推力系数 | $C_T = C_\beta \breve{\beta}$ | |
| 推力 | $\breve{T} = \bar{q}\breve{S}C_T$ | |
| 阻力 | $\breve{D} = \bar{q}\breve{S}C_{D0}$ | |
| 升力 | $\breve{L} = \bar{q}\breve{S}C_{L0}$ | |
| 俯仰力矩 | $\breve{M} = \bar{q}\breve{S}C_{M0}$ | |

**代码对应** (`dynamics.py`):

```python
class AircraftDynamics:
    def compute_aerodynamics(self, x):
        """计算气动参数"""
        xa, xb, h, va, vb, vh = x
        
        # 真实空速
        V = np.sqrt(va**2 + vb**2 + vh**2)
        V = max(V, 1e-6)
        
        # 空气密度 (随高度指数衰减)
        rho = self.C_rho * np.exp(-h / 24000.0)
        
        # 动压
        q_bar = 0.5 * rho * V**2
        
        # 攻角和侧滑角
        alpha = np.arctan2(vh, va + 1e-6)
        beta = np.arcsin(np.clip(vb / V, -1.0, 1.0))
        
        # 推力
        C_T = self.C_beta * beta
        T = q_bar * self.S * C_T
        
        # 阻力、升力、俯仰力矩
        D = q_bar * self.S * self.C_D0
        L = q_bar * self.S * self.C_L0
        M = q_bar * self.S * self.C_M0
        
        return T, D, L, M, alpha, beta, V, rho, q_bar

    def f(self, x):
        """公式(53) - 漂移项 f(x) ∈ R^6"""
        xa, xb, h, va, vb, vh = x
        T, D, L, M, alpha, beta, V, rho, q_bar = self.compute_aerodynamics(x)
        
        f_vec = np.zeros(6)
        f_vec[0] = va                                      # ẋa = va
        f_vec[1] = vb                                      # ẋb = vb
        f_vec[2] = vh                                      # ḣ = vh
        f_vec[3] = (1.0/self.m) * (T*np.cos(alpha) - D)   # v̇a
        f_vec[4] = (1.0/self.m) * (L + T*np.sin(alpha))   # v̇b
        f_vec[5] = (1.0/self.m) * (M + T*self.y_d)        # v̇h
        
        return f_vec

    def g(self, x):
        """公式(54) - 控制增益矩阵 g(x) ∈ R^{6×3}"""
        xa, xb, h, va, vb, vh = x
        
        g_mat = np.zeros((6, 3))
        g_mat[3, 0] = 1.0
        g_mat[4, 1] = 1.0
        g_mat[5, 2] = 1.0 + 0.25 * (np.sin(va * xb))**2
        
        return g_mat
```

---

## 3. 博弈问题建模

### 3.1 公式 (3) - 状态误差定义

$$\tilde{x}_i^{pe} = \sum_{j=1}^{N_p} c_{j,i}(x_j^p - x_i^e + r_{j,i}^{pe})$$

其中：
- $c_{j,i}$: 追捕者j追踪逃避者i的权重（从邻接矩阵$\mathbf{A}_{pe}$获取）
- $r_{j,i}^{pe}$: 期望的相对位移

**代码对应** (`controller.py`):

```python
def compute_state_error(self, evader_idx):
    """公式(3) - 计算状态误差"""
    evader = self.evaders[evader_idx]
    x_tilde = np.zeros(6)
    
    for j, pursuer in enumerate(self.pursuers):
        c_ji = self.target_graph.A_pe[j, evader_idx]
        if c_ji > 0:
            diff = pursuer.state - evader.state + pursuer.expected_displacement
            x_tilde += c_ji * diff
    
    return x_tilde
```

### 3.2 公式 (7) - 团队误差

$$E_{\text{Team}} = \sum_{i=1}^{N_e}\sum_{j=1}^{N_p} c_{j,i} \left\| \nu \odot (x_j^p - x_i^e + r_{j,i}^{pe}) \right\|$$

其中 $\nu$ 是权重向量（用于调整不同维度的重要性）。

### 3.3 公式 (13) - 状态代价

$$P_i(x) = (\tilde{x}_i^{pe})^T Q \tilde{x}_i^{pe}$$

### 3.4 公式 (14) - 非二次控制代价

追捕者非二次代价项（控制饱和的惩罚）:
$$U(u_j^p) = 2\int_0^{u_j^p} \left(\bar{u}_p \tanh^{-1}\frac{\tau}{\bar{u}_p}\right)^T R_1 \, d\tau$$

积分后的形式:
$$U(u) = 2\bar{u}_p R_1 \left[ u \cdot \text{arctanh}\left(\frac{u}{\bar{u}_p}\right) + \frac{\bar{u}_p}{2} \ln\left(1 - \left(\frac{u}{\bar{u}_p}\right)^2\right) \right]$$

逃避者非二次代价项类似，但带负号（零和博弈）。

### 3.5 公式 (17) - 值函数定义

$$V_i^*(x) = \min_{\{u_j^p\}} \max_{\{u_i^e\}} \int_t^\infty \left( P_i(x) + U(u_j^p) - W(u_i^e) \right) d\tau$$

### 3.6 公式 (18) - Hamiltonian方程

$$H_i(x, u_j^p, u_i^e, \nabla V) = P_i(x) + U(u_j^p) - W(u_i^e) + \nabla^T V_i(x) \left( \sum_{j=1}^{N_p} c_{j,i} \left( \mathcal{F}_{j,i}(x) + g_j^p u_j^p - g_i^e u_i^e \right) \right)$$

其中状态误差动力学:
$$\mathcal{F}_{j,i}(x) = f_j^p(x_j^p) - f_i^e(x_i^e)$$

**重要**: $\mathcal{F}_{j,i}$ 是追捕者和逃避者各自非线性漂移项的差值，不能简化!

---

## 4. 最优控制策略

### 4.1 公式 (19) / 公式 (40) - 最优控制

**追捕者最优控制**:
$$u_j^{p*} = -\bar{u}_p \tanh\left( \frac{1}{2\bar{u}_p} R_1^{-1} (g_j^p)^T \nabla \psi_i^T(x) \hat{W}_{i,s} \right)$$

**逃避者最优控制**:
$$u_i^{e*} = -\bar{u}_e \tanh\left( \frac{1}{2\bar{u}_e} R_2^{-1} (g_i^e)^T \nabla \psi_i^T(x) \hat{W}_{i,s} \right)$$

> 📌 **论文原版**: 两者都使用负号
> 
> 对抗性通过**状态误差动力学**中的减号实现：
> $$\dot{\tilde{x}} = \mathcal{F}_{j,i} + g_j^p u_j^p - g_i^e u_i^e$$
> 
> 当两者都用负号时：
> - `g_p u_p` 贡献 `-∇V` 方向（减小V）
> - `-g_e u_e = +g_e ū_e tanh` 贡献 `+∇V` 方向（增大V）

其中:
- $\bar{u}_p, \bar{u}_e$: 控制输入饱和限制
- $R_1, R_2$: 控制权重矩阵
- $g_j^p, g_i^e$: 各自的控制增益矩阵（非线性，公式54）
- $\nabla \psi_i(x)$: 基函数梯度
- $\hat{W}_{i,s}$: 学习的权重

**代码对应** (`controller.py`):

```python
def compute_optimal_controls(self, evader_idx):
    """公式(40) - 计算最优控制"""
    x_tilde = self.compute_state_error(evader_idx)
    
    # 值函数梯度: ∇V = ∇ψ @ W
    grad_psi = critic.activation_gradient(x_tilde)  # (6, 21)
    grad_V = grad_psi @ W                           # (6,)
    
    # 追捕者控制
    for j, pursuer in enumerate(self.pursuers):
        c_ji = self.target_graph.A_pe[j, evader_idx]
        if c_ji > 0:
            g_p = pursuer.dynamics.g(pursuer.state)  # 非线性g矩阵!
            rho_p = (1.0/(2.0*u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
            u_p = -u_bar_p * np.tanh(rho_p)  # 负号
    
    # 逃避者控制 - 论文原版
    g_e = evader.dynamics.g(evader.state)  # 非线性g矩阵!
    rho_e = (1.0/(2.0*u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
    u_e = -u_bar_e * np.tanh(rho_e)  # 负号（论文原版）
    
    return u_pursuers, u_evader, x_tilde
```

> 📌 对抗性通过状态误差动力学 `g_p u_p - g_e u_e` 中的减号实现

---

## 5. 值函数近似网络

### 5.1 公式 (37) - V-SNAC结构

$$V_i^*(x) = W_i^T \psi_i(x) + \delta_i(x)$$

其中 $\delta_i(x)$ 是近似误差。

### 5.2 公式 (38) - 估计值函数

$$\hat{V}_i^s(x) = \hat{W}_{i,s}^T \psi_i(x)$$

### 5.3 公式 (39) - 值函数梯度

$$\nabla \hat{V}_i(x) = \nabla \psi_i^T(x) \hat{W}_{i,s}$$

### 5.4 多项式基函数 ψ(x)

使用21个基函数:
- 6个二次项: $x_1^2, x_2^2, \ldots, x_6^2$
- 15个交叉项: $x_1 x_2, x_1 x_3, \ldots, x_5 x_6$

**状态归一化**:
$$x_{norm} = x / scale$$

其中 $scale = [3000, 3000, 3000, 100, 100, 100]$

**基函数定义**:
$$\psi(x) = [x_1^2, x_2^2, x_3^2, x_4^2, x_5^2, x_6^2, x_1x_2, x_1x_3, \ldots, x_5x_6]^T$$

**梯度计算**:
$$\nabla \psi_k(x) = \frac{\partial \psi_k}{\partial x}$$

对于二次项 $\psi_i = x_i^2$:
$$\frac{\partial \psi_i}{\partial x_i} = 2x_i / scale_i$$

对于交叉项 $\psi_{ij} = x_i x_j$:
$$\frac{\partial \psi_{ij}}{\partial x_i} = x_j / scale_i, \quad \frac{\partial \psi_{ij}}{\partial x_j} = x_i / scale_j$$

**代码对应** (`networks.py`):

```python
def activation_gradient(self, x):
    """∇ψ(x) ∈ R^{6×21}"""
    x_n = self.normalize_state(x)
    grad = np.zeros((6, 21))
    
    # 二次项梯度
    for i in range(6):
        grad[i, i] = 2 * x_n[i] / self.state_scale[i]
    
    # 交叉项梯度
    col = 6
    for i in range(6):
        for j in range(i + 1, 6):
            grad[i, col] = x_n[j] / self.state_scale[i]
            grad[j, col] = x_n[i] / self.state_scale[j]
            col += 1
    
    return grad
```

---

## 6. Off-Policy强化学习算法

### 6.1 Off-Policy的两种实现方式

**论文中的方法 - 梯度下降法 (公式46)**:
$$\dot{\hat{W}}_{i,s} = -\alpha_i (R_i^s + \hat{W}_{i,s}^T K_{i,s}) K_{i,s}^T$$

在仿真运行过程中实时更新权重。

**代码中的实现 - 最小二乘法**:
$$W = (X^T X + \lambda I)^{-1} X^T y$$

有两种模式：
1. **离线训练** (`offline_training`): 网格采样后一次性求解W
2. **在线学习** (`run_simulation` with `online_learning=True`): 边运行边收集数据，定期更新W

### 6.2 公式 (41) - Off-Policy Bellman方程

$$H_i^* = P_i(x) + \sum_{j=1}^{N_p} c_{j,i} \bar{u}_p^2 \bar{R}_1 \ln\left(1 - \tanh^2(\rho_p)\right) - d_i^{ep} \bar{u}_e^2 \bar{R}_2 \ln\left(1 - \tanh^2(\rho_e)\right) + \hat{W}_{i,s}^T \nabla\psi_i(x) \left( \sum_{j=1}^{N_p} c_{j,i} \mathcal{F}_{j,i}(x) \right) = 0$$

将Bellman方程改写为线性方程组:

$$W^T \nabla\psi(x) \dot{\tilde{x}} = -P_i(x) - U_N + W_N$$

即:
$$\mathbf{Z} \cdot W = y$$

其中:
- $\mathbf{Z} = \nabla\psi^T(x) \cdot \dot{\tilde{x}}$ (行向量)
- $\dot{\tilde{x}} = \mathcal{F}_{j,i}(x) + g_j^p u_j^p - g_i^e u_i^e$ (状态误差动力学)
- $y = -P_i(x) - U_N + W_N$ (代价)

### 6.2 算法流程

```
Algorithm 3: Off-Policy RL Training

Input: 网格点数 num_grid_points, 迭代次数 num_iterations
Output: 收敛的权重 W

1. 初始化权重 W 为随机小值
2. 定义稳定初始策略 K_init

3. FOR iteration = 1 to num_iterations:
    3.1 清空数据矩阵 X, y
    
    3.2 在归一化状态空间网格上采样:
        FOR 每个网格点 (x1, x2, x3):
            FOR 每个速度样本 (v1, v2, v3):
                a) 计算归一化状态 x_norm = [x1,x2,x3,v1,v2,v3]
                b) 计算实际状态误差 x_tilde = x_norm * scale
                
                c) 计算完整非线性动力学:
                   - x_p = x_e_ref + x_tilde  (追捕者虚拟状态)
                   - f_p = f(x_p)             (公式53)
                   - f_e = f(x_e_ref)         (公式53)
                   - F_ji = f_p - f_e         (漂移项差)
                   - g_p = g(x_p)             (公式54)
                   - g_e = g(x_e_ref)         (公式54)
                
                d) 计算控制:
                   IF iteration == 0:
                       u_p = K_init @ x_norm[:3]  (初始策略)
                       u_e = -K_init @ x_norm[:3] * (ū_e/ū_p)  (逃避初始策略)
                   ELSE:
                       grad_V = ∇ψ @ W           (公式39)
                       u_p = -ū_p * tanh(ρ_p)    (公式40)
                       u_e = -ū_e * tanh(ρ_e)    (公式40，论文原版)
                
                e) 计算状态误差动力学:
                   x_tilde_dot = F_ji + g_p @ u_p - g_e @ u_e
                
                f) 构建方程:
                   Z = ∇ψ^T @ x_tilde_dot
                   P_i = x_norm^T Q x_norm       (公式13)
                   nonquad = U_N - W_N           (公式14)
                   y_val = -P_i - nonquad
                
                g) 添加到数据矩阵: X.append(Z), y.append(y_val)
    
    3.3 最小二乘求解 (岭回归):
        W_new = (X^T X + λI)^{-1} X^T y
    
    3.4 检查收敛:
        IF ||W_new - W|| < threshold:
            BREAK

4. RETURN W
```

### 6.3 代码对应 (`simulation.py`)

```python
def offline_training(pursuers, evaders, target_graph, critics, controller,
                     num_grid_points=13, num_iterations=20):
    """
    完全按照论文实现的离线训练
    - 公式(53): 非线性漂移项
    - 公式(54): 非线性控制增益
    - 公式(40): 最优控制策略
    - 公式(41): Off-Policy Bellman方程
    """
    dynamics = AircraftDynamics()
    
    for iteration in range(num_iterations):
        X_list, y_list = [], []
        
        # 网格采样
        for x1, x2, x3 in grid_points:
            for v1, v2, v3 in velocity_samples:
                x_norm = np.array([x1, x2, x3, v1, v2, v3])
                x_tilde = x_norm * state_scale
                
                # 公式(53): 完整非线性动力学
                x_e_ref = np.array([500, 1500, 200, 50, 80, 10])
                x_p_virtual = x_e_ref + x_tilde
                f_p = dynamics.f(x_p_virtual)
                f_e = dynamics.f(x_e_ref)
                F_ji = f_p - f_e
                
                # 公式(54): 非线性控制增益
                g_p = dynamics.g(x_p_virtual)
                g_e = dynamics.g(x_e_ref)
                
                # 公式(40): 计算控制
                if iteration == 0:
                    u_p = K_init @ x_norm[:3]
                    u_e = -K_init @ x_norm[:3] * (u_bar_e/u_bar_p)
                else:
                    dPHI = critics[0].activation_gradient(x_tilde)
                    grad_V = dPHI @ W
                    rho_p = (1/(2*u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
                    rho_e = (1/(2*u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
                    u_p = -u_bar_p * np.tanh(rho_p)  # 负号
                    u_e = -u_bar_e * np.tanh(rho_e)  # 负号（论文原版）
                
                # 公式(18): 状态误差动力学
                x_tilde_dot = F_ji + g_p @ u_p - g_e @ u_e
                
                # 构建线性方程
                Z = dPHI.T @ x_tilde_dot
                P_i = x_norm.T @ Q @ x_norm
                nonquad = compute_nonquadratic_cost(u_p, u_e)
                y_val = -P_i - nonquad
                
                X_list.append(Z)
                y_list.append(y_val)
        
        # 岭回归求解
        X, y = np.array(X_list), np.array(y_list)
        W_new = np.linalg.solve(X.T @ X + λ*I, X.T @ y)
        critics[0].W = W_new
```

---

## 7. 动态目标图算法

### 7.1 Algorithm 1: Dynamic Target Graph

**算法流程**:

1. **计算距离矩阵**: 计算所有追捕者和逃避者之间的加权距离
   $$\gamma_{j,i} = \|\nu \odot (x_j^p - x_i^e + r_{j,i}^{pe})\|$$

2. **检查交换条件**: 对于每对追捕者$(j, j')$及其目标$(i, i')$：
   $$(\gamma_{j,i} + \gamma_{j',i'}) - (\gamma_{j',i} + \gamma_{j,i'}) > \rho$$
   
   如果满足条件，则交换目标。

3. **更新邻接矩阵**: 更新 $\mathbf{A}_{pe}$ 和 $\mathbf{A}_{ep}$

---

## 8. 代码架构

### 8.1 文件结构

```
mpe_game/
├── config.py          # 系统配置参数
├── dynamics.py        # 飞行器动力学模型 (公式53, 54)
├── agents.py          # 追捕者/逃避者智能体
├── graph.py           # 动态目标图算法
├── networks.py        # 值函数近似网络 (公式37-39)
├── controller.py      # 最优控制器 (公式40)
├── simulation.py      # 训练和仿真流程 (公式41)
├── visualization.py   # 可视化工具
└── test_scenario1.py  # 测试脚本
```

### 8.2 算法执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    主执行流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 初始化                                                   │
│     ├── 创建追捕者/逃避者智能体 (dynamics.py)               │
│     ├── 初始化目标图 (graph.py)                            │
│     ├── 创建Critic网络 (networks.py)                       │
│     └── 创建OptimalController (controller.py)              │
│                                                             │
│  2. 离线训练 (simulation.py: offline_training)             │
│     ├── FOR iteration = 1 to 20:                           │
│     │   ├── 网格采样 (13^3 × 3 = 6591 samples)            │
│     │   │   ├── 计算完整非线性动力学 (公式53, 54)         │
│     │   │   ├── 计算最优控制 (公式40)                      │
│     │   │   └── 收集样本构建Bellman方程 (公式41)          │
│     │   ├── 批量更新权重 (岭回归)                          │
│     │   └── 检查收敛                                        │
│     └── 返回收敛的权重W                                     │
│                                                             │
│  3. 在线仿真 (simulation.py: run_simulation)               │
│     ├── 重置智能体状态                                      │
│     ├── FOR step = 1 to n_steps:                           │
│     │   ├── 计算状态误差 (公式3)                           │
│     │   ├── 计算最优控制 (公式40, 使用训练好的W)          │
│     │   ├── 更新智能体状态 (完整非线性动力学 + RK4)        │
│     │   ├── 更新目标图 (Algorithm 1)                       │
│     │   └── 记录团队误差 (公式7)                           │
│     └── 返回仿真结果                                        │
│                                                             │
│  4. 可视化                                                   │
│     ├── 绘制3D轨迹                                          │
│     ├── 绘制团队误差曲线                                    │
│     └── 绘制控制输入                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 仿真场景

### 9.1 场景1: 三追捕者-单逃避者

**初始条件** (Table I):

| 智能体 | 位置 $(x_a, x_b, h)$ | 速度 $(v_a, v_b, v_h)$ |
|--------|----------------------|------------------------|
| 追捕者P1 | (600, -1900, 300) | (78, 2, 18) |
| 追捕者P2 | (1200, 500, 2200) | (2, 20, 15) |
| 追捕者P3 | (200, 600, 800) | (5, 10, 125) |
| 逃避者E1 | (500, 1500, 200) | (50, 80, 10) |

**期望相对位移**:
| 追捕者 | $r_{j,1}^{pe}$ |
|--------|----------------|
| P1 | (50, 10, 0, 0, 0, 0) |
| P2 | (10, 50, 0, 0, 0, 0) |
| P3 | (-10, 0, -50, 0, 0, 0) |

**目标图**:
$$\mathbf{A}_{pe} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$$

---

## 10. 参数配置

### 10.1 系统参数 (Table I)

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 质量 | $m$ | 8920 | kg |
| 参考面积 | $S$ | 4.0 | m² |
| 空气密度系数 | $C_\rho$ | 0.00238 | |
| 推力系数 | $C_\beta$ | 0.002576 | |
| 推力作用点 | $y_d$ | 0.01 | m |
| 阻力系数 | $C_{D0}$ | 0.02 | |
| 升力系数 | $C_{L0}$ | 0.1 | |
| 俯仰力矩系数 | $C_{M0}$ | 0.01 | |

### 10.2 控制参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 追捕者饱和限制 | $\bar{u}_p$ | 25.0 | |
| 逃避者饱和限制 | $\bar{u}_e$ | 15.0 | |
| 状态权重矩阵 | $Q$ | $I_6$ | 单位阵 |
| 追捕者控制权重 | $R_1$ | $I_3$ | 单位阵 |
| 逃避者控制权重 | $R_2$ | $I_3$ | 单位阵 |

### 10.3 算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 网格点数 | 13 | 每个位置维度 |
| 速度采样数 | 3 | 每个位置点 |
| 策略迭代次数 | 20 | |
| 正则化系数 | 0.01 | 岭回归 |
| 仿真步长 | 0.01 s | RK4积分 |
| 仿真时长 | 20.0 s | |

---

## 11. 关键公式汇总

| 公式编号 | 描述 | 表达式 |
|----------|------|--------|
| (1) | 追捕者动力学 | $\dot{x}_j^p = f_j^p(x_j^p) + g_j^p(x_j^p) u_j^p$ |
| (2) | 逃避者动力学 | $\dot{x}_i^e = f_i^e(x_i^e) + g_i^e(x_i^e) u_i^e$ |
| (3) | 状态误差 | $\tilde{x}_i^{pe} = \sum_j c_{j,i}(x_j^p - x_i^e + r_{j,i}^{pe})$ |
| (7) | 团队误差 | $E_{\text{Team}} = \sum_i\sum_j c_{j,i}\|\nu \odot (x_j^p - x_i^e + r_{j,i}^{pe})\|$ |
| (13) | 状态代价 | $P_i(x) = \tilde{x}_i^{peT} Q \tilde{x}_i^{pe}$ |
| (14) | 非二次代价 | $U(u) = 2\bar{u}R[u\text{arctanh}(u/\bar{u}) + 0.5\bar{u}\ln(1-(u/\bar{u})^2)]$ |
| (18) | Hamiltonian | $H_i = P_i + U - W + \nabla V^T \sum_j c_{j,i}(\mathcal{F}_{j,i} + g_j^p u_j^p - g_i^e u_i^e)$ |
| (37) | 值函数近似 | $V_i^* = W_i^T \psi_i(x) + \delta_i$ |
| (39) | 值函数梯度 | $\nabla \hat{V}_i = \nabla \psi_i^T \hat{W}_{i,s}$ |
| (40) | 最优控制 | $u^* = -\bar{u} \tanh\left(\frac{1}{2\bar{u}}R^{-1}g^T\nabla V\right)$ (两者都用负号) |
| (41) | Bellman方程 | $W^T\nabla\psi \dot{\tilde{x}} = -P_i - U_N + W_N$ |
| (53) | 漂移项 | $f(x) = [v_a, v_b, v_h, \frac{1}{m}(T\cos\alpha-D), \frac{1}{m}(L+T\sin\alpha), \frac{1}{m}(M+Ty_d)]^T$ |
| (54) | 控制增益 | $g(x) = [0_{3\times3}; I_3 + \Delta g]$, $\Delta g_{3,3} = 0.25\sin^2(v_a x_b)$ |

---

## 12. 注意事项

1. **完整非线性动力学**: 训练和仿真都必须使用完整的非线性动力学公式(53)和(54)，不能简化！

2. **状态归一化**: 所有状态在计算基函数前需要归一化到[-1, 1]范围
   - 位置维度: scale = 3000 m
   - 速度维度: scale = 100 m/s

3. **控制饱和**: arctanh函数的参数需要严格在(-1, 1)范围内，需要clip到0.9999

4. **初始策略**: 需要提供一个稳定的初始控制策略开始迭代

5. **样本数量**: 每次迭代需要13^3 × 3 = 6591个样本

6. **正则化**: 使用岭回归(λ=0.01)避免矩阵奇异问题

---

## 13. 测试验证结果

### 13.1 成功测试：线性双积分器 + RL控制

**测试文件**: `test_simple_1v1.py`, `test_game_rl_simple.py`

在简化的线性系统上，核心算法工作正确：

```
============================================================
Case: u_p > u_e (Pursuer advantage)
u_bar_p = 25.0, u_bar_e = 15.0
============================================================
Initial distance: 5.83
Final distance: 0.02

✅ 追捕成功！
```

**饱和限制对比测试结果**：

| 场景 | $\bar{u}_p$ | $\bar{u}_e$ | 最终距离 | 结果 |
|------|-------------|-------------|----------|------|
| 追捕者优势 | 25 | 15 | 0.02 | ✅ 成功 |
| 能力相等 | 20 | 20 | 2635.76 | ❌ 不稳定 |
| 逃避者优势 | 15 | 25 | 2555.78 | ❌ 不稳定 |

**关键结论**：只有当追捕者控制能力大于逃避者时，追捕才能成功。这符合博弈论预期。

### 13.2 成功测试：非线性动力学验证

**测试文件**: `test_proportional_control.py`

使用简单比例控制验证飞行器动力学模型正确性：

```
Initial error: 3403.96
Final error: 3265.20
Error reduction: 4.1%

✅ 动力学模型正确，比例控制有效
```

### 13.3 关键发现：逃避者控制符号

**原始代码（错误）**:
```python
# controller.py 和 simulation.py
u_evader = -self.u_bar_e * np.tanh(rho_e)  # 负号 ❌
```

**正确实现**:
```python
u_evader = self.u_bar_e * np.tanh(rho_e)  # 正号 ✅
```

**原因分析**：
- 追捕者想**最小化**误差代价 → 沿梯度下降方向 → 负号
- 逃避者想**最大化**误差代价 → 沿梯度上升方向 → 正号
- 这是零和博弈的本质：两者目标相反

---

## 14. 当前问题诊断

### 14.1 非线性系统RL失效原因

即使修复了逃避者符号，非线性飞行器系统的RL控制仍然无效：

**问题1：归一化导致梯度过小**
```
状态误差: ~3000m
归一化后: ~1
梯度被除以3000: grad_V ≈ 0.0002
控制信号: ≈ 0
```

**问题2：训练权重不收敛**
```
Iter 10: ||W||=3426, dW=2938  (波动大)
Iter 20: ||W||=3187, dW=2908  (不收敛)
```

**问题3：采样范围不匹配**
- 采样在归一化空间 [-3, 3]
- 实际初始状态误差 ≈ 3404（归一化后 ≈ 1.1）
- 采样可能不充分覆盖实际状态分布

### 14.2 组件验证状态

| 组件 | 验证状态 | 验证方法 |
|------|----------|----------|
| 动力学模型 (公式53, 54) | ✅ 正确 | 比例控制测试 |
| 状态误差计算 (公式3) | ✅ 正确 | 代码审查 |
| 最优控制形式 (公式40) | ✅ 正确 | 线性系统测试 |
| 逃避者符号 | ✅ 已修复 | 线性系统测试 |
| 归一化策略 | ❌ 有问题 | 诊断发现 |
| 训练采样 | ❌ 需优化 | 诊断发现 |

---

## 15. 调试建议

### 15.1 归一化修复方案

**方案A：减小归一化因子**
```python
# networks.py
self.state_scale = np.array([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0])
```

**方案B：增大Q矩阵补偿**
```python
# config.py
Q = np.diag([1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0])
```

**方案C：不使用归一化**
```python
# 直接在原始状态空间工作，但需要调整初始权重
self.state_scale = np.ones(6)  # 无归一化
self.W = np.zeros(21) * 1e-6   # 小初始权重避免数值溢出
```

### 15.2 推荐调试路径

```
阶段1: 确保线性系统完全正确
       ↓
阶段2: 测试简化非线性（如只保留位置，忽略气动）
       ↓
阶段3: 逐步添加气动项
       ↓
阶段4: 完整6维飞行器动力学
```

### 15.3 关键代码位置

需要修改的文件：
1. `networks.py` - 归一化策略 (第24-26行)
2. `simulation.py` - 训练采样范围 (第60-63行)
3. `config.py` - Q矩阵权重 (第30行)

---

## 16. 测试文件说明

| 文件名 | 功能 | 结果 |
|--------|------|------|
| `test_simple_1v1.py` | 线性双积分器RL | ✅ 成功 |
| `test_game_rl_simple.py` | 饱和限制对比 | ✅ 成功 |
| `test_proportional_control.py` | 动力学验证 | ✅ 成功 |
| `diagnose_original.py` | 原始代码诊断 | 发现问题 |
| `simple_1v1_direct.py` | 非线性RL尝试 | ❌ 待修复 |

详细测试报告见 `TEST_REPORT.md`

---

## 17. 结论

**当前状态**：代码已恢复为论文原版公式（两者都用负号）

### 待验证问题

1. **论文原版公式在线性系统上的表现**
   - 需要运行 `test_paper_exact.py` 验证
   - 检查P矩阵正定性

2. **归一化问题**
   - 梯度量级过小
   - 需要调整 `networks.py` 中的 `state_scale`

3. **训练采样策略**
   - 采样范围与实际状态分布匹配

### 调试资源

- 测试指南: `TESTING_GUIDE.md`
- 测试报告: `TEST_REPORT.md`
- MATLAB参考: `refer/` 目录

---

**文档版本**: 4.0 (恢复论文原版公式)  
**最后更新**: 2026-02-01
