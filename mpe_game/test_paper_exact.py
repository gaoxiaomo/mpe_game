"""
严格按照论文公式实现 - 验证两者都用负号

论文公式(40):
- 追捕者: u_p* = -ū_p * tanh(ρ_p)
- 逃避者: u_e* = -ū_e * tanh(ρ_e)

关键：状态误差动力学 (公式18)
ẋ̃ = F_{j,i} + g_p u_p - g_e u_e
           ↑
    注意这里是减号！

所以即使 u_e 用负号，整体效果是：
- g_p u_p 贡献：g_p (-ū_p tanh) = 沿着 -∇V 方向（减小V）
- -g_e u_e 贡献：-g_e (-ū_e tanh) = +g_e ū_e tanh = 沿着 +∇V 方向（增大V）

逃避者自然就在对抗！
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.linalg import solve_continuous_are


class LinearGame:
    """线性双积分器博弈"""
    
    def __init__(self, u_bar_p=25.0, u_bar_e=15.0):
        self.n = 4
        self.m = 2
        
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
        
        self.Q = np.eye(self.n)
        self.R1 = np.eye(self.m)
        self.R2 = np.eye(self.m)
        self.R1_inv = np.linalg.inv(self.R1)
        self.R2_inv = np.linalg.inv(self.R2)
        
        self.u_bar_p = u_bar_p
        self.u_bar_e = u_bar_e


class ValueFunction:
    """二次型值函数"""
    
    def __init__(self, n):
        self.n = n
        self.P = np.eye(n) * 0.1
        
    def gradient(self, x):
        return 2 * self.P @ x
    
    def to_vector(self):
        vec = []
        for i in range(self.n):
            for j in range(i, self.n):
                vec.append(self.P[i, j])
        return np.array(vec)
    
    def from_vector(self, vec):
        idx = 0
        for i in range(self.n):
            for j in range(i, self.n):
                self.P[i, j] = vec[idx]
                self.P[j, i] = vec[idx]
                idx += 1


def train_paper_exact(game, V, num_iterations=50, num_samples=500):
    """严格按照论文训练
    
    公式(40): 两者都用负号
    u_p* = -ū_p * tanh(ρ_p)
    u_e* = -ū_e * tanh(ρ_e)
    """
    
    print("=" * 60)
    print("Training with PAPER EXACT formulas (both negative)")
    print("=" * 60)
    
    # 初始化为LQR解
    try:
        P_lqr = solve_continuous_are(game.A, game.B, game.Q, game.R1)
        V.P = P_lqr.copy()
        print(f"Initialized with LQR: ||P|| = {np.linalg.norm(P_lqr):.4f}")
    except:
        pass
    
    num_params = game.n * (game.n + 1) // 2
    history = []
    
    for iteration in range(num_iterations):
        X_list, y_list = [], []
        
        for _ in range(num_samples):
            x = np.random.uniform(-5, 5, game.n)
            
            # 值函数梯度
            grad_V = V.gradient(x)
            
            # 公式(40) - 严格按照论文，两者都用负号
            rho_p = (1.0 / (2.0 * game.u_bar_p)) * game.R1_inv @ (game.B.T @ grad_V)
            rho_e = (1.0 / (2.0 * game.u_bar_e)) * game.R2_inv @ (game.B.T @ grad_V)
            
            u_p = -game.u_bar_p * np.tanh(rho_p)  # 负号
            u_e = -game.u_bar_e * np.tanh(rho_e)  # 负号（论文原版）
            
            # 饱和
            u_p = np.clip(u_p, -game.u_bar_p * 0.999, game.u_bar_p * 0.999)
            u_e = np.clip(u_e, -game.u_bar_e * 0.999, game.u_bar_e * 0.999)
            
            # 状态误差动力学：ẋ̃ = A x̃ + B u_p - B u_e
            # 注意：这里是 -B u_e，这是关键！
            x_dot = game.A @ x + game.B @ u_p - game.B @ u_e
            
            # 构建方程
            row = []
            for i in range(game.n):
                for j in range(i, game.n):
                    if i == j:
                        coef = 2 * x[i] * x_dot[i]
                    else:
                        coef = 2 * (x[i] * x_dot[j] + x[j] * x_dot[i])
                    row.append(coef)
            
            # 状态代价
            state_cost = x @ game.Q @ x
            
            # 非二次代价
            def nonquad(u, u_bar, R):
                cost = 0.0
                for k in range(len(u)):
                    ratio = np.clip(u[k] / u_bar, -0.999, 0.999)
                    cost += 2.0 * u_bar * R[k, k] * (
                        u[k] * np.arctanh(ratio) + 
                        0.5 * u_bar * np.log(max(1.0 - ratio**2, 1e-10))
                    )
                return cost
            
            cost_p = nonquad(u_p, game.u_bar_p, game.R1)
            cost_e = nonquad(u_e, game.u_bar_e, game.R2)
            
            # Bellman方程右边
            # 注意：逃避者代价是减去的（因为是零和博弈）
            rhs = -(state_cost + cost_p - cost_e)
            
            X_list.append(row)
            y_list.append(rhs)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 岭回归
        lambda_reg = 0.01
        try:
            p_new = np.linalg.solve(X.T @ X + lambda_reg * np.eye(num_params), X.T @ y)
        except:
            p_new = np.linalg.lstsq(X, y, rcond=None)[0]
        
        change = np.linalg.norm(p_new - V.to_vector())
        V.from_vector(p_new)
        history.append(np.linalg.norm(V.P))
        
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1:3d}: ||P||={np.linalg.norm(V.P):.4f}, dP={change:.6f}")
    
    print(f"\nFinal P:\n{V.P}")
    
    return history


def simulate_paper_exact(game, V, x0_p, x0_e, dt=0.01, T=10.0):
    """严格按照论文仿真"""
    
    n_steps = int(T / dt)
    
    x_p = x0_p.copy()
    x_e = x0_e.copy()
    
    x_p_hist = [x_p.copy()]
    x_e_hist = [x_e.copy()]
    u_p_hist = []
    u_e_hist = []
    
    for _ in range(n_steps):
        x_tilde = x_p - x_e
        grad_V = V.gradient(x_tilde)
        
        # 公式(40) - 两者都用负号
        rho_p = (1.0 / (2.0 * game.u_bar_p)) * game.R1_inv @ (game.B.T @ grad_V)
        rho_e = (1.0 / (2.0 * game.u_bar_e)) * game.R2_inv @ (game.B.T @ grad_V)
        
        u_p = -game.u_bar_p * np.tanh(rho_p)  # 负号
        u_e = -game.u_bar_e * np.tanh(rho_e)  # 负号
        
        # 更新（各自独立的动力学）
        def update(x, u):
            k1 = game.A @ x + game.B @ u
            k2 = game.A @ (x + 0.5*dt*k1) + game.B @ u
            k3 = game.A @ (x + 0.5*dt*k2) + game.B @ u
            k4 = game.A @ (x + dt*k3) + game.B @ u
            return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_p = update(x_p, u_p)
        x_e = update(x_e, u_e)
        
        x_p_hist.append(x_p.copy())
        x_e_hist.append(x_e.copy())
        u_p_hist.append(u_p.copy())
        u_e_hist.append(u_e.copy())
    
    return {
        'x_p': np.array(x_p_hist),
        'x_e': np.array(x_e_hist),
        'u_p': np.array(u_p_hist),
        'u_e': np.array(u_e_hist)
    }


def main():
    print("\n" + "=" * 70)
    print("TEST: Paper Exact Implementation (Both Negative Signs)")
    print("=" * 70)
    
    # 测试不同配置
    configs = [
        ("u_p > u_e", 25.0, 15.0),
        ("u_p = u_e", 20.0, 20.0),
        ("u_p < u_e", 15.0, 25.0),
    ]
    
    results = {}
    
    for name, u_bar_p, u_bar_e in configs:
        print(f"\n{'='*60}")
        print(f"Case: {name}")
        print(f"u_bar_p = {u_bar_p}, u_bar_e = {u_bar_e}")
        print(f"{'='*60}")
        
        game = LinearGame(u_bar_p, u_bar_e)
        V = ValueFunction(game.n)
        
        history = train_paper_exact(game, V, num_iterations=50, num_samples=500)
        
        # 检查P是否正定
        eigvals = np.linalg.eigvalsh(V.P)
        print(f"\nP eigenvalues: {eigvals}")
        if np.all(eigvals > 0):
            print("P is positive definite: YES")
        else:
            print("P is positive definite: NO (unstable)")
        
        # 仿真
        x0_p = np.array([5.0, 3.0, 0.0, 0.0])
        x0_e = np.array([0.0, 0.0, 0.0, 0.0])
        
        sim = simulate_paper_exact(game, V, x0_p, x0_e, dt=0.01, T=10.0)
        
        init_dist = np.linalg.norm(x0_p[:2] - x0_e[:2])
        final_dist = np.linalg.norm(sim['x_p'][-1, :2] - sim['x_e'][-1, :2])
        
        print(f"\nSimulation:")
        print(f"  Initial distance: {init_dist:.2f}")
        print(f"  Final distance: {final_dist:.2f}")
        
        if final_dist < init_dist * 0.5:
            print(f"  Result: SUCCESS (distance reduced by {(1-final_dist/init_dist)*100:.1f}%)")
        elif final_dist > init_dist * 2:
            print(f"  Result: UNSTABLE (distance increased)")
        else:
            print(f"  Result: PARTIAL")
        
        results[name] = {
            'sim': sim,
            'history': history,
            'V': V,
            'init_dist': init_dist,
            'final_dist': final_dist
        }
    
    # 绘图
    fig, axes = plt.subplots(len(configs), 3, figsize=(15, 4*len(configs)))
    
    for idx, (name, _, _) in enumerate(configs):
        sim = results[name]['sim']
        history = results[name]['history']
        
        # 轨迹
        ax = axes[idx, 0]
        ax.plot(sim['x_p'][:, 0], sim['x_p'][:, 1], 'b-', label='Pursuer')
        ax.plot(sim['x_e'][:, 0], sim['x_e'][:, 1], 'r--', label='Evader')
        ax.scatter([sim['x_p'][0, 0]], [sim['x_p'][0, 1]], c='blue', marker='o', s=100)
        ax.scatter([sim['x_e'][0, 0]], [sim['x_e'][0, 1]], c='red', marker='o', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{name}\nTrajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 距离
        ax = axes[idx, 1]
        t = np.arange(len(sim['x_p'])) * 0.01
        dist = np.linalg.norm(sim['x_p'][:, :2] - sim['x_e'][:, :2], axis=1)
        ax.plot(t, dist, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance')
        ax.set_title('Distance Over Time')
        ax.grid(True, alpha=0.3)
        
        # P收敛
        ax = axes[idx, 2]
        ax.plot(history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||P||')
        ax.set_title('P Matrix Convergence')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper_exact_test.png', dpi=150)
    plt.show()
    
    # 分析对比
    print("\n" + "=" * 70)
    print("ANALYSIS: Why Paper Formula Works")
    print("=" * 70)
    
    print("""
关键洞察：论文中两者都用负号是正确的！

原因在于状态误差动力学的结构：

    ẋ̃ = A x̃ + B u_p - B u_e
                      ↑
              这里是减号！

1. 追捕者控制项: B u_p = B (-ū_p tanh(ρ_p))
   - ρ_p ∝ B^T ∇V，与 ∇V 同向
   - 所以 tanh(ρ_p) 与 ∇V 同向
   - 因此 B u_p ∝ -B B^T ∇V，沿着 -∇V 方向
   - 使 ∇V^T B u_p < 0，即减小 V

2. 逃避者控制项: -B u_e = -B (-ū_e tanh(ρ_e)) = +B ū_e tanh(ρ_e)
   - ρ_e ∝ B^T ∇V，与 ∇V 同向
   - 所以 -B u_e ∝ +B B^T ∇V，沿着 +∇V 方向
   - 使 ∇V^T (-B u_e) > 0，即增大 V

结论：状态误差动力学中的负号 "-B u_e" 自动实现了对抗！
      不需要改变控制公式的符号！
""")
    
    return results


if __name__ == "__main__":
    results = main()

