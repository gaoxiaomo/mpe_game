"""
改进的线性系统测试 - 增加迭代次数，改善收敛性

改进点：
1. 增加迭代次数到200
2. 增加样本数量到1000
3. 使用固定随机种子保证可重复性
4. 增大正则化系数减少振荡
5. 使用移动平均平滑权重更新
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('TkAgg')


class LinearPEGame:
    """线性追逃博弈"""
    
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
        
        self.u_bar_p = u_bar_p
        self.u_bar_e = u_bar_e
        
    def compute_lqr_solution(self):
        P = solve_continuous_are(self.A, self.B, self.Q, self.R1)
        K = np.linalg.inv(self.R1) @ self.B.T @ P
        return P, K


class CriticNetwork:
    """值函数网络"""
    
    def __init__(self, state_dim):
        self.n = state_dim
        self.num_basis = state_dim + state_dim * (state_dim - 1) // 2
        self.W = np.zeros(self.num_basis)
        
    def activation(self, x):
        psi = []
        for i in range(self.n):
            psi.append(x[i] ** 2)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                psi.append(x[i] * x[j])
        return np.array(psi)
    
    def activation_gradient(self, x):
        grad = np.zeros((self.n, self.num_basis))
        for i in range(self.n):
            grad[i, i] = 2 * x[i]
        col = self.n
        for i in range(self.n):
            for j in range(i + 1, self.n):
                grad[i, col] = x[j]
                grad[j, col] = x[i]
                col += 1
        return grad
    
    def predict_gradient(self, x):
        return self.activation_gradient(x) @ self.W


def train_improved(game, critic, num_samples=1000, num_iterations=200, 
                   lambda_reg=0.1, momentum=0.9):
    """改进的训练算法
    
    改进点：
    1. 更多样本和迭代
    2. 更大的正则化系数减少振荡
    3. 动量平滑权重更新
    """
    
    print("=" * 60)
    print("Improved Training")
    print(f"Samples: {num_samples}, Iterations: {num_iterations}")
    print(f"Regularization: {lambda_reg}, Momentum: {momentum}")
    print("=" * 60)
    
    A, B = game.A, game.B
    Q, R1, R2 = game.Q, game.R1, game.R2
    u_bar_p, u_bar_e = game.u_bar_p, game.u_bar_e
    R1_inv = np.linalg.inv(R1)
    R2_inv = np.linalg.inv(R2)
    
    # 初始化为LQR解
    P_init, _ = game.compute_lqr_solution()
    W_init = []
    for i in range(game.n):
        W_init.append(P_init[i, i])
    for i in range(game.n):
        for j in range(i + 1, game.n):
            W_init.append(2 * P_init[i, j])
    critic.W = np.array(W_init)
    
    print(f"Initial ||W||: {np.linalg.norm(critic.W):.4f}")
    
    weight_history = []
    change_history = []
    W_velocity = np.zeros_like(critic.W)  # 动量
    
    # 固定随机种子
    np.random.seed(42)
    
    for iteration in range(num_iterations):
        X_list = []
        y_list = []
        
        for _ in range(num_samples):
            x = np.random.uniform(-5, 5, game.n)
            
            dPsi = critic.activation_gradient(x)
            grad_V = dPsi @ critic.W
            
            # 论文公式(40) - 两者都用负号
            rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ B.T @ grad_V
            rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ B.T @ grad_V
            
            u_p = -u_bar_p * np.tanh(rho_p)
            u_e = -u_bar_e * np.tanh(rho_e)  # 论文原版：负号
            
            u_p = np.clip(u_p, -u_bar_p * 0.9999, u_bar_p * 0.9999)
            u_e = np.clip(u_e, -u_bar_e * 0.9999, u_bar_e * 0.9999)
            
            # 状态误差动力学: x_dot = Ax + B*u_p - B*u_e
            x_dot = A @ x + B @ u_p - B @ u_e
            
            Z = dPsi.T @ x_dot
            
            # 状态代价
            P_cost = x.T @ Q @ x
            
            # 非二次代价
            def nonquad(u, u_bar, R):
                cost = 0.0
                for k in range(len(u)):
                    ratio = np.clip(u[k] / u_bar, -0.9999, 0.9999)
                    cost += 2.0 * u_bar * R[k, k] * (
                        u[k] * np.arctanh(ratio) + 
                        0.5 * u_bar * np.log(1.0 - ratio**2)
                    )
                return cost
            
            cost_p = nonquad(u_p, u_bar_p, R1)
            cost_e = nonquad(u_e, u_bar_e, R2)
            
            y_val = -(P_cost + cost_p - cost_e)
            
            X_list.append(Z)
            y_list.append(y_val)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 岭回归
        XTX = X.T @ X + lambda_reg * np.eye(critic.num_basis)
        XTy = X.T @ y
        W_new = np.linalg.solve(XTX, XTy)
        
        # 动量更新（平滑）
        W_velocity = momentum * W_velocity + (1 - momentum) * (W_new - critic.W)
        critic.W = critic.W + W_velocity
        
        weight_change = np.linalg.norm(W_new - critic.W + W_velocity)
        weight_history.append(np.linalg.norm(critic.W))
        change_history.append(weight_change)
        
        if (iteration + 1) % 20 == 0:
            print(f"Iter {iteration+1:3d}: ||W||={np.linalg.norm(critic.W):.4f}, "
                  f"dW={weight_change:.6f}")
    
    print(f"\nFinal ||W||: {np.linalg.norm(critic.W):.4f}")
    
    # 从W恢复P矩阵
    P = np.zeros((game.n, game.n))
    idx = 0
    for i in range(game.n):
        P[i, i] = critic.W[idx]
        idx += 1
    for i in range(game.n):
        for j in range(i + 1, game.n):
            P[i, j] = critic.W[idx] / 2
            P[j, i] = critic.W[idx] / 2
            idx += 1
    
    print(f"\nRecovered P matrix:")
    print(P)
    
    eigvals = np.linalg.eigvalsh(P)
    print(f"\nP eigenvalues: {eigvals}")
    if np.all(eigvals > 0):
        print("P is POSITIVE DEFINITE")
    else:
        print("P is NOT positive definite")
    
    return weight_history, change_history


def simulate(game, critic, x0_p, x0_e, dt=0.01, T=10.0):
    """仿真"""
    n_steps = int(T / dt)
    
    R1_inv = np.linalg.inv(game.R1)
    R2_inv = np.linalg.inv(game.R2)
    
    x_p = x0_p.copy()
    x_e = x0_e.copy()
    
    x_p_hist = [x_p.copy()]
    x_e_hist = [x_e.copy()]
    
    for _ in range(n_steps):
        x_tilde = x_p - x_e
        grad_V = critic.predict_gradient(x_tilde)
        
        # 论文公式(40)
        rho_p = (1.0 / (2.0 * game.u_bar_p)) * R1_inv @ game.B.T @ grad_V
        rho_e = (1.0 / (2.0 * game.u_bar_e)) * R2_inv @ game.B.T @ grad_V
        
        u_p = -game.u_bar_p * np.tanh(rho_p)
        u_e = -game.u_bar_e * np.tanh(rho_e)
        
        # RK4
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
    
    return np.array(x_p_hist), np.array(x_e_hist)


def main():
    print("=" * 70)
    print("Improved Linear System Test")
    print("Paper Formula: Both u_p and u_e use NEGATIVE sign")
    print("=" * 70)
    
    # 测试配置
    game = LinearPEGame(u_bar_p=25.0, u_bar_e=15.0)
    critic = CriticNetwork(game.n)
    
    # 训练
    weight_history, change_history = train_improved(
        game, critic,
        num_samples=1000,
        num_iterations=200,
        lambda_reg=0.1,
        momentum=0.8
    )
    
    # 仿真
    print("\n" + "=" * 60)
    print("Simulation")
    print("=" * 60)
    
    x0_p = np.array([5.0, 3.0, 0.0, 0.0])
    x0_e = np.array([0.0, 0.0, 0.0, 0.0])
    
    x_p_hist, x_e_hist = simulate(game, critic, x0_p, x0_e, dt=0.01, T=10.0)
    
    init_dist = np.linalg.norm(x0_p[:2] - x0_e[:2])
    final_dist = np.linalg.norm(x_p_hist[-1, :2] - x_e_hist[-1, :2])
    
    print(f"Initial distance: {init_dist:.2f}")
    print(f"Final distance: {final_dist:.2f}")
    
    if final_dist < init_dist * 0.5:
        print("Result: SUCCESS")
    elif final_dist > init_dist * 2:
        print("Result: UNSTABLE")
    else:
        print("Result: PARTIAL")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 轨迹
    ax = axes[0, 0]
    ax.plot(x_p_hist[:, 0], x_p_hist[:, 1], 'b-', label='Pursuer', linewidth=2)
    ax.plot(x_e_hist[:, 0], x_e_hist[:, 1], 'r--', label='Evader', linewidth=2)
    ax.scatter([x_p_hist[0, 0]], [x_p_hist[0, 1]], c='blue', marker='o', s=100, zorder=5)
    ax.scatter([x_e_hist[0, 0]], [x_e_hist[0, 1]], c='red', marker='o', s=100, zorder=5)
    ax.scatter([x_p_hist[-1, 0]], [x_p_hist[-1, 1]], c='blue', marker='s', s=100, zorder=5)
    ax.scatter([x_e_hist[-1, 0]], [x_e_hist[-1, 1]], c='red', marker='s', s=100, zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 距离
    ax = axes[0, 1]
    t = np.arange(len(x_p_hist)) * 0.01
    dist = np.linalg.norm(x_p_hist[:, :2] - x_e_hist[:, :2], axis=1)
    ax.plot(t, dist, 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance')
    ax.set_title('Distance Over Time')
    ax.grid(True, alpha=0.3)
    
    # 权重收敛
    ax = axes[1, 0]
    ax.plot(weight_history, 'b-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||W||')
    ax.set_title('Weight Norm Convergence (200 iterations)')
    ax.grid(True, alpha=0.3)
    
    # 权重变化
    ax = axes[1, 1]
    ax.semilogy(change_history, 'r-', linewidth=1.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||dW|| (log scale)')
    ax.set_title('Weight Change (should decrease)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_training.png', dpi=150)
    plt.show()
    
    return weight_history, critic


if __name__ == "__main__":
    weight_history, critic = main()

