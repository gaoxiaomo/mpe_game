"""
完整的简化1v1追逃博弈 - 修复所有问题

关键修复：
1. 移除归一化或调整归一化策略
2. 逃避者控制使用正号（对抗博弈）
3. 训练采样与实际状态范围匹配
4. 使用更稳定的训练方法
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

# 导入动力学模型
from dynamics import AircraftDynamics


class SimpleCritic:
    """简化的Critic网络 - 不使用归一化
    
    直接在原始状态空间工作，通过适当的权重初始化处理量级差异
    """
    
    def __init__(self, state_dim=6):
        self.n = state_dim
        self.num_basis = 21  # 6个平方项 + 15个交叉项
        
        # 权重初始化：考虑不同维度的量级
        # 位置量级 ~1000m, 速度量级 ~50m/s
        # V(x) ~ x^T P x, 期望值 ~10^6 (位置)^2 或 ~10^3 (速度)^2
        self.W = np.zeros(self.num_basis)
        
        # 缩放因子：使不同维度的贡献相当
        self.pos_scale = 1e-6  # 位置项缩放
        self.vel_scale = 1e-4  # 速度项缩放
        self.cross_scale = 1e-5  # 交叉项缩放
        
        self.weight_history = []
        
    def get_scales(self):
        """获取各基函数的缩放因子"""
        scales = []
        # 平方项
        for i in range(6):
            if i < 3:
                scales.append(self.pos_scale)
            else:
                scales.append(self.vel_scale)
        # 交叉项
        for i in range(6):
            for j in range(i + 1, 6):
                scales.append(self.cross_scale)
        return np.array(scales)
    
    def activation(self, x):
        """二次基函数"""
        psi = []
        scales = self.get_scales()
        idx = 0
        
        # 平方项
        for i in range(6):
            psi.append(x[i] ** 2 * scales[idx])
            idx += 1
        # 交叉项
        for i in range(6):
            for j in range(i + 1, 6):
                psi.append(x[i] * x[j] * scales[idx])
                idx += 1
        return np.array(psi)
    
    def activation_gradient(self, x):
        """基函数梯度"""
        grad = np.zeros((6, 21))
        scales = self.get_scales()
        
        # 平方项梯度
        for i in range(6):
            grad[i, i] = 2 * x[i] * scales[i]
        
        # 交叉项梯度
        col = 6
        for i in range(6):
            for j in range(i + 1, 6):
                grad[i, col] = x[j] * scales[col]
                grad[j, col] = x[i] * scales[col]
                col += 1
        return grad
    
    def predict_gradient(self, x):
        """值函数梯度"""
        return self.activation_gradient(x) @ self.W


def train_with_policy_iteration(dynamics, critic, u_bar_p, u_bar_e,
                                 Q, R1, R2, num_iterations=50, num_samples=1000):
    """策略迭代训练
    
    使用固定初始策略采样，然后最小二乘更新权重
    """
    
    print("\n" + "=" * 60)
    print("Policy Iteration Training")
    print("=" * 60)
    
    R1_inv = np.linalg.inv(R1)
    R2_inv = np.linalg.inv(R2)
    
    # 采样范围（基于论文初始条件）
    pos_range = 4000.0
    vel_range = 150.0
    
    # 参考逃避者状态
    x_e_ref = np.array([500, 1500, 200, 50, 80, 10])
    
    weight_history = []
    
    # 初始权重：正定对角阵
    critic.W = np.ones(21) * 1.0
    
    for iteration in range(num_iterations):
        X_list = []
        y_list = []
        
        for _ in range(num_samples):
            # 随机状态误差
            x_tilde = np.zeros(6)
            x_tilde[0] = np.random.uniform(-pos_range, pos_range)
            x_tilde[1] = np.random.uniform(-pos_range, pos_range)
            x_tilde[2] = np.random.uniform(-pos_range, pos_range)
            x_tilde[3] = np.random.uniform(-vel_range, vel_range)
            x_tilde[4] = np.random.uniform(-vel_range, vel_range)
            x_tilde[5] = np.random.uniform(-vel_range, vel_range)
            
            # 追捕者虚拟状态
            x_p = x_e_ref + x_tilde
            
            # 动力学
            f_p = dynamics.f(x_p)
            f_e = dynamics.f(x_e_ref)
            g_p = dynamics.g(x_p)
            g_e = dynamics.g(x_e_ref)
            F_ji = f_p - f_e
            
            # 基函数梯度
            dPsi = critic.activation_gradient(x_tilde)
            grad_V = dPsi @ critic.W
            
            # 最优控制
            rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
            u_p = -u_bar_p * np.tanh(rho_p)
            
            rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
            u_e = u_bar_e * np.tanh(rho_e)  # 正号
            
            # 饱和
            u_p = np.clip(u_p, -u_bar_p * 0.999, u_bar_p * 0.999)
            u_e = np.clip(u_e, -u_bar_e * 0.999, u_bar_e * 0.999)
            
            # 状态误差动力学
            x_tilde_dot = F_ji + g_p @ u_p - g_e @ u_e
            
            # Bellman方程
            Z = dPsi.T @ x_tilde_dot
            
            # 状态代价
            P_cost = x_tilde.T @ Q @ x_tilde
            
            # 控制代价
            def control_cost(u, u_bar, R):
                cost = 0.0
                for k in range(3):
                    ratio = np.clip(u[k] / u_bar, -0.999, 0.999)
                    cost += 2.0 * u_bar * R[k, k] * (
                        u[k] * np.arctanh(ratio) + 
                        0.5 * u_bar * np.log(max(1.0 - ratio**2, 1e-10))
                    )
                return cost
            
            cost_p = control_cost(u_p, u_bar_p, R1)
            cost_e = control_cost(u_e, u_bar_e, R2)
            
            # Bellman方程右边
            y_val = -(P_cost + cost_p - cost_e)
            
            X_list.append(Z)
            y_list.append(y_val)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 岭回归
        lambda_reg = 1.0
        XTX = X.T @ X + lambda_reg * np.eye(21)
        XTy = X.T @ y
        
        try:
            W_new = np.linalg.solve(XTX, XTy)
        except:
            W_new = np.linalg.lstsq(X, y, rcond=None)[0]
        
        weight_change = np.linalg.norm(W_new - critic.W)
        critic.W = W_new.copy()
        weight_history.append(np.linalg.norm(W_new))
        
        if (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1:3d}: ||W||={np.linalg.norm(W_new):.2f}, "
                  f"dW={weight_change:.4f}")
    
    print(f"\nFinal ||W|| = {np.linalg.norm(critic.W):.2f}")
    
    return weight_history


def simulate_pursuit_evasion(dynamics, critic, x0_p, x0_e, u_bar_p, u_bar_e,
                             R1, R2, dt=0.01, T=20.0):
    """仿真追逃博弈"""
    
    R1_inv = np.linalg.inv(R1)
    R2_inv = np.linalg.inv(R2)
    
    n_steps = int(T / dt)
    
    x_p = x0_p.copy()
    x_e = x0_e.copy()
    
    x_p_hist = [x_p.copy()]
    x_e_hist = [x_e.copy()]
    u_p_hist = []
    u_e_hist = []
    error_hist = [np.linalg.norm(x_p - x_e)]
    
    for step in range(n_steps):
        x_tilde = x_p - x_e
        
        # 计算控制
        g_p = dynamics.g(x_p)
        g_e = dynamics.g(x_e)
        grad_V = critic.predict_gradient(x_tilde)
        
        rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
        u_p = -u_bar_p * np.tanh(rho_p)
        
        rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
        u_e = u_bar_e * np.tanh(rho_e)
        
        # RK4更新
        def rk4_step(x, u):
            k1 = dynamics.dynamics(x, u)
            k2 = dynamics.dynamics(x + 0.5*dt*k1, u)
            k3 = dynamics.dynamics(x + 0.5*dt*k2, u)
            k4 = dynamics.dynamics(x + dt*k3, u)
            return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_p = rk4_step(x_p, u_p)
        x_e = rk4_step(x_e, u_e)
        
        x_p_hist.append(x_p.copy())
        x_e_hist.append(x_e.copy())
        u_p_hist.append(u_p.copy())
        u_e_hist.append(u_e.copy())
        error_hist.append(np.linalg.norm(x_p - x_e))
    
    return {
        'x_p': np.array(x_p_hist),
        'x_e': np.array(x_e_hist),
        'u_p': np.array(u_p_hist),
        'u_e': np.array(u_e_hist),
        'error': np.array(error_hist)
    }


def plot_results(results, dt, title_prefix=""):
    """绘制结果"""
    fig = plt.figure(figsize=(16, 10))
    
    # 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(results['x_p'][:, 0], results['x_p'][:, 1], results['x_p'][:, 2], 
             'b-', linewidth=1.5, label='Pursuer')
    ax1.plot(results['x_e'][:, 0], results['x_e'][:, 1], results['x_e'][:, 2], 
             'r--', linewidth=1.5, label='Evader')
    ax1.scatter([results['x_p'][0, 0]], [results['x_p'][0, 1]], [results['x_p'][0, 2]], 
                c='blue', marker='o', s=100, label='P Start')
    ax1.scatter([results['x_e'][0, 0]], [results['x_e'][0, 1]], [results['x_e'][0, 2]], 
                c='red', marker='o', s=100, label='E Start')
    ax1.scatter([results['x_p'][-1, 0]], [results['x_p'][-1, 1]], [results['x_p'][-1, 2]], 
                c='blue', marker='s', s=100, label='P End')
    ax1.scatter([results['x_e'][-1, 0]], [results['x_e'][-1, 1]], [results['x_e'][-1, 2]], 
                c='red', marker='s', s=100, label='E End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title_prefix}3D Trajectory')
    ax1.legend(loc='upper left', fontsize=8)
    
    # 2D轨迹 X-Y
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(results['x_p'][:, 0], results['x_p'][:, 1], 'b-', linewidth=1.5, label='Pursuer')
    ax2.plot(results['x_e'][:, 0], results['x_e'][:, 1], 'r--', linewidth=1.5, label='Evader')
    ax2.scatter([results['x_p'][0, 0]], [results['x_p'][0, 1]], c='blue', marker='o', s=100)
    ax2.scatter([results['x_e'][0, 0]], [results['x_e'][0, 1]], c='red', marker='o', s=100)
    ax2.scatter([results['x_p'][-1, 0]], [results['x_p'][-1, 1]], c='blue', marker='s', s=100)
    ax2.scatter([results['x_e'][-1, 0]], [results['x_e'][-1, 1]], c='red', marker='s', s=100)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'{title_prefix}2D Trajectory (X-Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 误差范数
    ax3 = fig.add_subplot(2, 3, 3)
    t = np.arange(len(results['error'])) * dt
    ax3.plot(t, results['error'], 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('||x_p - x_e||')
    ax3.set_title(f'{title_prefix}State Error Norm')
    ax3.grid(True, alpha=0.3)
    
    # 追捕者控制
    ax4 = fig.add_subplot(2, 3, 4)
    if len(results['u_p']) > 0:
        t_u = np.arange(len(results['u_p'])) * dt
        ax4.plot(t_u, results['u_p'][:, 0], 'b-', linewidth=1, label='u_p[0]')
        ax4.plot(t_u, results['u_p'][:, 1], 'b--', linewidth=1, label='u_p[1]')
        ax4.plot(t_u, results['u_p'][:, 2], 'b:', linewidth=1, label='u_p[2]')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control')
    ax4.set_title(f'{title_prefix}Pursuer Control')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 逃避者控制
    ax5 = fig.add_subplot(2, 3, 5)
    if len(results['u_e']) > 0:
        ax5.plot(t_u, results['u_e'][:, 0], 'r-', linewidth=1, label='u_e[0]')
        ax5.plot(t_u, results['u_e'][:, 1], 'r--', linewidth=1, label='u_e[1]')
        ax5.plot(t_u, results['u_e'][:, 2], 'r:', linewidth=1, label='u_e[2]')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Control')
    ax5.set_title(f'{title_prefix}Evader Control')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 位置分量
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, results['x_p'][:, 0], 'b-', label='P_x')
    ax6.plot(t, results['x_e'][:, 0], 'r--', label='E_x')
    ax6.plot(t, results['x_p'][:, 1], 'b:', alpha=0.7, label='P_y')
    ax6.plot(t, results['x_e'][:, 1], 'r:', alpha=0.7, label='E_y')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position (m)')
    ax6.set_title(f'{title_prefix}Position Components')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("Complete 1v1 Nonlinear Pursuit-Evasion Game")
    print("Using Aircraft Dynamics from Paper (Eq. 53-54)")
    print("=" * 70)
    
    # 创建动力学模型
    dynamics = AircraftDynamics()
    
    # 参数设置
    u_bar_p = 25.0
    u_bar_e = 15.0
    
    Q = np.eye(6) * 1.0  # 状态权重
    R1 = np.eye(3) * 1.0  # 追捕者控制权重
    R2 = np.eye(3) * 1.0  # 逃避者控制权重
    
    # 创建Critic
    critic = SimpleCritic()
    
    # 训练
    weight_history = train_with_policy_iteration(
        dynamics, critic, u_bar_p, u_bar_e, Q, R1, R2,
        num_iterations=50, num_samples=1000
    )
    
    # 仿真 - 使用论文的初始条件
    print("\n" + "=" * 60)
    print("Simulation with Paper Initial Conditions")
    print("=" * 60)
    
    x0_p = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
    x0_e = np.array([500, 1500, 200, 50, 80, 10], dtype=float)
    
    print(f"Initial pursuer: {x0_p[:3]}")
    print(f"Initial evader: {x0_e[:3]}")
    print(f"Initial error: {np.linalg.norm(x0_p - x0_e):.2f}")
    
    dt = 0.01
    T = 20.0
    
    results = simulate_pursuit_evasion(
        dynamics, critic, x0_p, x0_e, u_bar_p, u_bar_e, R1, R2, dt, T
    )
    
    print(f"\nFinal error: {results['error'][-1]:.2f}")
    print(f"Min error: {min(results['error']):.2f}")
    print(f"Error reduction: {(results['error'][0] - results['error'][-1])/results['error'][0]*100:.1f}%")
    
    # 绘图
    fig1 = plot_results(results, dt, "RL Control: ")
    
    # 权重收敛图
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(weight_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||W||')
    ax.set_title('Weight Convergence During Training')
    ax.grid(True, alpha=0.3)
    
    plt.show()
    
    return results, critic


if __name__ == "__main__":
    results, critic = main()

