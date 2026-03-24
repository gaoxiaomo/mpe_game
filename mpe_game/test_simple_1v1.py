"""
简化的1追1线性场景测试

使用双积分器模型验证论文核心算法：
- 线性动力学: ẋ = Ax + Bu
- 单追捕者 vs 单逃避者
- 验证HJI方程求解和最优控制

论文核心公式:
- 公式(3): 状态误差 x̃ = x_p - x_e + r
- 公式(40): 最优控制 u* = -ū·tanh((1/2ū)R⁻¹BᵀP·x̃)
- 公式(41): HJI方程 x̃ᵀQx̃ + 2x̃ᵀPA·x̃ + ... = 0

对于线性系统，应该退化为标准LQR/微分博弈解。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('TkAgg')


class LinearPEGame:
    """线性追逃博弈 - 双积分器模型"""
    
    def __init__(self, u_bar_p=25.0, u_bar_e=15.0):
        # 状态维度: [x, y, vx, vy] (2D位置+速度)
        self.n = 4
        self.m = 2  # 控制维度
        
        # 双积分器动力学: ẋ = Ax + Bu
        # [ẋ]   [0 0 1 0] [x]    [0 0]
        # [ẏ] = [0 0 0 1] [y]  + [0 0] u
        # [v̇x]  [0 0 0 0] [vx]   [1 0]
        # [v̇y]  [0 0 0 0] [vy]   [0 1]
        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=float)
        
        self.B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ], dtype=float)
        
        # 权重矩阵
        self.Q = np.eye(self.n) * 1.0  # 状态权重
        self.R1 = np.eye(self.m) * 1.0  # 追捕者控制权重
        self.R2 = np.eye(self.m) * 1.0  # 逃避者控制权重
        
        # 控制饱和
        self.u_bar_p = u_bar_p
        self.u_bar_e = u_bar_e
        
    def compute_lqr_solution(self):
        """计算LQR解（无对手情况）作为参考"""
        # 对于追捕者独自追踪：ARE
        # Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0
        P = solve_continuous_are(self.A, self.B, self.Q, self.R1)
        K = np.linalg.inv(self.R1) @ self.B.T @ P
        return P, K
    
    def compute_game_solution(self):
        """计算微分博弈解析解（耦合Riccati方程）
        
        对于零和博弈:
        Aᵀ P + P A + Q - P (S₁ - S₂) P = 0
        其中 S₁ = B R₁⁻¹ Bᵀ, S₂ = B R₂⁻¹ Bᵀ
        
        当 u_bar_p > u_bar_e 时，追捕者有优势
        """
        S1 = self.B @ np.linalg.inv(self.R1) @ self.B.T
        S2 = self.B @ np.linalg.inv(self.R2) @ self.B.T
        
        # 对于零和博弈的Riccati方程
        # P A + Aᵀ P + Q - P (S₁ - S₂) P = 0
        # 这等价于 ARE with modified input matrix
        S_diff = S1 - S2
        
        try:
            # 尝试求解游戏Riccati方程
            P = solve_continuous_are(self.A, self.B, self.Q, self.R1)
            K_p = np.linalg.inv(self.R1) @ self.B.T @ P
            K_e = np.linalg.inv(self.R2) @ self.B.T @ P
            return P, K_p, K_e
        except Exception as e:
            print(f"Game Riccati failed: {e}")
            return None, None, None


class CriticNetworkSimple:
    """简化的值函数逼近网络
    
    使用二次基函数: V(x) = xᵀ P x = Wᵀ ψ(x)
    其中 ψ(x) 包含 x 的二次项
    """
    
    def __init__(self, state_dim):
        self.n = state_dim
        # 二次基函数数量: n个平方项 + n(n-1)/2个交叉项
        self.num_basis = state_dim + state_dim * (state_dim - 1) // 2
        # 权重初始化
        self.W = np.zeros(self.num_basis)
        self.weight_history = []
        
    def activation(self, x):
        """二次基函数 ψ(x)"""
        psi = []
        # 平方项 x_i²
        for i in range(self.n):
            psi.append(x[i] ** 2)
        # 交叉项 x_i * x_j
        for i in range(self.n):
            for j in range(i + 1, self.n):
                psi.append(x[i] * x[j])
        return np.array(psi)
    
    def activation_gradient(self, x):
        """基函数梯度 ∇ψ(x) ∈ R^{n×num_basis}"""
        grad = np.zeros((self.n, self.num_basis))
        
        # 平方项梯度: ∂(x_i²)/∂x_i = 2x_i
        for i in range(self.n):
            grad[i, i] = 2 * x[i]
        
        # 交叉项梯度
        col = self.n
        for i in range(self.n):
            for j in range(i + 1, self.n):
                grad[i, col] = x[j]
                grad[j, col] = x[i]
                col += 1
        return grad
    
    def predict_value(self, x):
        """V(x) = Wᵀ ψ(x)"""
        return np.dot(self.W, self.activation(x))
    
    def predict_gradient(self, x):
        """∇V(x) = ∇ψ(x) W"""
        return self.activation_gradient(x) @ self.W


def train_offline_linear(game, critic, num_samples=500, num_iterations=50):
    """离线训练 - 线性系统上的策略迭代
    
    公式(41): HJI Bellman方程
    0 = x̃ᵀQx̃ + (∇V)ᵀ A x̃ + U_N(u_p*) - W_N(u_e*)
    
    对于线性饱和控制:
    u_p* = -ū_p tanh((1/2ū_p) R₁⁻¹ Bᵀ ∇V)
    u_e* = +ū_e tanh((1/2ū_e) R₂⁻¹ Bᵀ ∇V)  # 逃避者取反方向
    """
    A, B = game.A, game.B
    Q, R1, R2 = game.Q, game.R1, game.R2
    u_bar_p, u_bar_e = game.u_bar_p, game.u_bar_e
    R1_inv = np.linalg.inv(R1)
    R2_inv = np.linalg.inv(R2)
    
    # 初始增益（来自LQR）
    P_init, K_init = game.compute_lqr_solution()
    
    # 从P矩阵初始化W（将P展开为向量）
    # V(x) = xᵀPx = Σ P_ii x_i² + Σ 2P_ij x_i x_j
    W_init = []
    for i in range(game.n):
        W_init.append(P_init[i, i])
    for i in range(game.n):
        for j in range(i + 1, game.n):
            W_init.append(2 * P_init[i, j])
    critic.W = np.array(W_init)
    
    print(f"Initial W from LQR: ||W|| = {np.linalg.norm(critic.W):.4f}")
    
    weight_history = []
    
    for iteration in range(num_iterations):
        X_list = []
        y_list = []
        
        # 采样状态空间
        for _ in range(num_samples):
            # 随机状态
            x = np.random.uniform(-5, 5, game.n)
            
            # 计算基函数梯度
            dPsi = critic.activation_gradient(x)  # (n, num_basis)
            grad_V = dPsi @ critic.W  # (n,)
            
            # 计算最优控制 - 公式(40)
            # 追捕者: u_p* = -ū_p tanh(ρ_p)
            rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ B.T @ grad_V
            u_p = -u_bar_p * np.tanh(rho_p)
            
            # 逃避者: u_e* = +ū_e tanh(ρ_e)  
            # 注意：逃避者是对抗者，梯度方向相反！
            rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ B.T @ grad_V
            u_e = u_bar_e * np.tanh(rho_e)  # 正号，因为逃避者要最大化代价
            
            # 状态误差动力学（追捕者状态 - 逃避者状态）
            # ẋ̃ = A x̃ + B u_p - B u_e
            x_dot = A @ x + B @ u_p - B @ u_e
            
            # 构建Bellman方程
            # W^T (∇ψ)^T ẋ̃ = -xᵀQx - U_N + W_N
            Z = dPsi.T @ x_dot  # (num_basis,)
            
            # 状态代价
            P_cost = x.T @ Q @ x
            
            # 非二次控制代价
            # U_N for pursuer: 2ū_p R̄ (u·arctanh(u/ū_p) + 0.5ū·ln(1-(u/ū_p)²))
            nonquad_p = 0.0
            for k in range(game.m):
                ratio = np.clip(u_p[k] / u_bar_p, -0.9999, 0.9999)
                nonquad_p += 2.0 * u_bar_p * R1[k, k] * (
                    u_p[k] * np.arctanh(ratio) + 
                    0.5 * u_bar_p * np.log(1.0 - ratio**2)
                )
            
            # W_N for evader (减去，因为逃避者是对抗者)
            nonquad_e = 0.0
            for k in range(game.m):
                ratio = np.clip(u_e[k] / u_bar_e, -0.9999, 0.9999)
                nonquad_e += 2.0 * u_bar_e * R2[k, k] * (
                    u_e[k] * np.arctanh(ratio) + 
                    0.5 * u_bar_e * np.log(1.0 - ratio**2)
                )
            
            # Bellman方程右边
            y_val = -P_cost - nonquad_p + nonquad_e
            
            X_list.append(Z)
            y_list.append(y_val)
        
        # 最小二乘求解
        X = np.array(X_list)
        y = np.array(y_list)
        
        lambda_reg = 0.01
        XTX = X.T @ X + lambda_reg * np.eye(critic.num_basis)
        XTy = X.T @ y
        W_new = np.linalg.solve(XTX, XTy)
        
        # 计算变化
        weight_change = np.linalg.norm(W_new - critic.W)
        critic.W = W_new.copy()
        weight_history.append(np.linalg.norm(W_new))
        
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1:3d}: ||W||={np.linalg.norm(W_new):.4f}, ΔW={weight_change:.6f}")
    
    return weight_history


def simulate_game(game, critic, x0_p, x0_e, dt=0.01, T=10.0):
    """仿真追逃博弈"""
    A, B = game.A, game.B
    R1_inv = np.linalg.inv(game.R1)
    R2_inv = np.linalg.inv(game.R2)
    u_bar_p, u_bar_e = game.u_bar_p, game.u_bar_e
    
    n_steps = int(T / dt)
    
    # 状态历史
    x_p_history = [x0_p.copy()]
    x_e_history = [x0_e.copy()]
    x_tilde_history = [x0_p - x0_e]
    u_p_history = []
    u_e_history = []
    
    x_p = x0_p.copy()
    x_e = x0_e.copy()
    
    for step in range(n_steps):
        # 状态误差
        x_tilde = x_p - x_e
        
        # 值函数梯度
        grad_V = critic.predict_gradient(x_tilde)
        
        # 最优控制
        rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ B.T @ grad_V
        u_p = -u_bar_p * np.tanh(rho_p)
        
        rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ B.T @ grad_V
        u_e = u_bar_e * np.tanh(rho_e)  # 逃避者取正号
        
        # RK4积分
        def dynamics_p(x, u):
            return A @ x + B @ u
        
        def dynamics_e(x, u):
            return A @ x + B @ u
        
        # 追捕者更新
        k1_p = dynamics_p(x_p, u_p)
        k2_p = dynamics_p(x_p + 0.5*dt*k1_p, u_p)
        k3_p = dynamics_p(x_p + 0.5*dt*k2_p, u_p)
        k4_p = dynamics_p(x_p + dt*k3_p, u_p)
        x_p = x_p + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        # 逃避者更新
        k1_e = dynamics_e(x_e, u_e)
        k2_e = dynamics_e(x_e + 0.5*dt*k1_e, u_e)
        k3_e = dynamics_e(x_e + 0.5*dt*k2_e, u_e)
        k4_e = dynamics_e(x_e + dt*k3_e, u_e)
        x_e = x_e + (dt/6.0) * (k1_e + 2*k2_e + 2*k3_e + k4_e)
        
        # 记录
        x_p_history.append(x_p.copy())
        x_e_history.append(x_e.copy())
        x_tilde_history.append(x_p - x_e)
        u_p_history.append(u_p.copy())
        u_e_history.append(u_e.copy())
    
    return {
        'x_p': np.array(x_p_history),
        'x_e': np.array(x_e_history),
        'x_tilde': np.array(x_tilde_history),
        'u_p': np.array(u_p_history),
        'u_e': np.array(u_e_history)
    }


def plot_results(results, dt, title_prefix=""):
    """绘制结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t = np.arange(len(results['x_tilde'])) * dt
    
    # 1. 2D轨迹
    ax1 = axes[0, 0]
    ax1.plot(results['x_p'][:, 0], results['x_p'][:, 1], 'b-', label='Pursuer', linewidth=2)
    ax1.plot(results['x_e'][:, 0], results['x_e'][:, 1], 'r--', label='Evader', linewidth=2)
    ax1.scatter(results['x_p'][0, 0], results['x_p'][0, 1], c='blue', marker='o', s=100, zorder=5)
    ax1.scatter(results['x_e'][0, 0], results['x_e'][0, 1], c='red', marker='o', s=100, zorder=5)
    ax1.scatter(results['x_p'][-1, 0], results['x_p'][-1, 1], c='blue', marker='s', s=100, zorder=5)
    ax1.scatter(results['x_e'][-1, 0], results['x_e'][-1, 1], c='red', marker='s', s=100, zorder=5)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'{title_prefix}2D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 状态误差范数
    ax2 = axes[0, 1]
    error_norm = np.linalg.norm(results['x_tilde'], axis=1)
    ax2.plot(t, error_norm, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('||x_p - x_e||')
    ax2.set_title(f'{title_prefix}State Error Norm')
    ax2.grid(True, alpha=0.3)
    
    # 3. 追捕者控制
    ax3 = axes[1, 0]
    if len(results['u_p']) > 0:
        t_u = np.arange(len(results['u_p'])) * dt
        ax3.plot(t_u, results['u_p'][:, 0], 'b-', label='u_p[0]', linewidth=1.5)
        ax3.plot(t_u, results['u_p'][:, 1], 'b--', label='u_p[1]', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Pursuer Control')
    ax3.set_title(f'{title_prefix}Pursuer Control Input')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 逃避者控制
    ax4 = axes[1, 1]
    if len(results['u_e']) > 0:
        t_u = np.arange(len(results['u_e'])) * dt
        ax4.plot(t_u, results['u_e'][:, 0], 'r-', label='u_e[0]', linewidth=1.5)
        ax4.plot(t_u, results['u_e'][:, 1], 'r--', label='u_e[1]', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Evader Control')
    ax4.set_title(f'{title_prefix}Evader Control Input')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("Simple 1v1 Linear Pursuit-Evasion Game Test")
    print("=" * 60)
    
    # 创建游戏
    game = LinearPEGame(u_bar_p=25.0, u_bar_e=15.0)
    
    # 计算参考解
    P_lqr, K_lqr = game.compute_lqr_solution()
    print(f"\nLQR Solution (no opponent):")
    print(f"  ||P|| = {np.linalg.norm(P_lqr):.4f}")
    print(f"  K = \n{K_lqr}")
    
    # 创建Critic网络
    critic = CriticNetworkSimple(game.n)
    
    # 离线训练
    print("\n" + "=" * 60)
    print("Offline Training")
    print("=" * 60)
    weight_history = train_offline_linear(game, critic, num_samples=500, num_iterations=50)
    
    # 从W恢复P矩阵
    P_learned = np.zeros((game.n, game.n))
    idx = 0
    for i in range(game.n):
        P_learned[i, i] = critic.W[idx]
        idx += 1
    for i in range(game.n):
        for j in range(i + 1, game.n):
            P_learned[i, j] = critic.W[idx] / 2
            P_learned[j, i] = critic.W[idx] / 2
            idx += 1
    
    print(f"\nLearned P matrix:")
    print(P_learned)
    print(f"||P_learned - P_lqr|| = {np.linalg.norm(P_learned - P_lqr):.4f}")
    
    # 仿真
    print("\n" + "=" * 60)
    print("Simulation")
    print("=" * 60)
    
    # 初始状态: 追捕者在原点，逃避者在(5,3)
    x0_p = np.array([0.0, 0.0, 0.0, 0.0])
    x0_e = np.array([5.0, 3.0, 0.0, 0.0])
    
    dt = 0.01
    T = 10.0
    
    results = simulate_game(game, critic, x0_p, x0_e, dt, T)
    
    print(f"\nInitial distance: {np.linalg.norm(x0_p[:2] - x0_e[:2]):.2f}")
    print(f"Final distance: {np.linalg.norm(results['x_p'][-1, :2] - results['x_e'][-1, :2]):.2f}")
    
    # 判断追捕是否成功
    final_error = np.linalg.norm(results['x_tilde'][-1])
    if final_error < 0.5:
        print("\n[OK] Capture SUCCESS! Error < 0.5")
    else:
        print(f"\n[FAIL] Capture FAILED. Final error: {final_error:.2f}")
    
    # 绘图
    fig1 = plot_results(results, dt, "Learned Policy: ")
    
    # 权重收敛图
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(weight_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||W||')
    ax.set_title('Weight Convergence During Training')
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    return results, critic, game


if __name__ == "__main__":
    results, critic, game = main()

