"""
直接的1v1追逃博弈 - 使用原始归一化方案但增大Q补偿

关键：Q足够大才能使控制有效
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')

from dynamics import AircraftDynamics


class CriticDirect:
    """直接的Critic网络 - 使用归一化"""
    
    def __init__(self, state_dim=6):
        self.n = state_dim
        self.num_basis = 21
        
        # 状态归一化
        self.scale = np.array([3000.0, 3000.0, 3000.0, 100.0, 100.0, 100.0])
        
        # 权重
        self.W = np.zeros(self.num_basis)
        self.weight_history = []
        
    def activation(self, x):
        """归一化状态上的二次基函数"""
        x_n = x / self.scale
        psi = []
        for i in range(6):
            psi.append(x_n[i] ** 2)
        for i in range(6):
            for j in range(i + 1, 6):
                psi.append(x_n[i] * x_n[j])
        return np.array(psi)
    
    def activation_gradient(self, x):
        """基函数梯度（含链式法则）"""
        x_n = x / self.scale
        grad = np.zeros((6, 21))
        
        # 平方项
        for i in range(6):
            grad[i, i] = 2 * x_n[i] / self.scale[i]
        
        # 交叉项
        col = 6
        for i in range(6):
            for j in range(i + 1, 6):
                grad[i, col] = x_n[j] / self.scale[i]
                grad[j, col] = x_n[i] / self.scale[j]
                col += 1
        return grad
    
    def predict_gradient(self, x):
        return self.activation_gradient(x) @ self.W


def main():
    print("=" * 70)
    print("Direct 1v1 Pursuit-Evasion Game")
    print("=" * 70)
    
    dynamics = AircraftDynamics()
    
    # 参数
    u_bar_p = 25.0
    u_bar_e = 15.0
    
    # 关键：增大Q来补偿归一化
    # 归一化后位置量级~1，速度量级~1
    # 需要Q使得学到的W产生足够大的梯度
    Q = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
    R1 = np.eye(3) * 1.0
    R2 = np.eye(3) * 1.0
    R1_inv = np.linalg.inv(R1)
    R2_inv = np.linalg.inv(R2)
    
    critic = CriticDirect()
    
    # 采样范围（归一化后）
    pos_range = 1.5  # 对应 4500m
    vel_range = 1.5  # 对应 150m/s
    
    # 参考逃避者状态
    x_e_ref = np.array([500, 1500, 200, 50, 80, 10])
    
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    # 初始权重（正定）
    critic.W = np.ones(21) * 10.0
    
    num_iterations = 50
    num_samples = 1000
    weight_history = []
    
    for iteration in range(num_iterations):
        X_list = []
        y_list = []
        
        for _ in range(num_samples):
            # 归一化状态采样
            x_norm = np.zeros(6)
            x_norm[0] = np.random.uniform(-pos_range, pos_range)
            x_norm[1] = np.random.uniform(-pos_range, pos_range)
            x_norm[2] = np.random.uniform(-pos_range, pos_range)
            x_norm[3] = np.random.uniform(-vel_range, vel_range)
            x_norm[4] = np.random.uniform(-vel_range, vel_range)
            x_norm[5] = np.random.uniform(-vel_range, vel_range)
            
            # 实际状态误差
            x_tilde = x_norm * critic.scale
            
            # 虚拟追捕者状态
            x_p = x_e_ref + x_tilde
            
            # 动力学
            f_p = dynamics.f(x_p)
            f_e = dynamics.f(x_e_ref)
            g_p = dynamics.g(x_p)
            g_e = dynamics.g(x_e_ref)
            F_ji = f_p - f_e
            
            # 基函数梯度（在实际状态上计算）
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
            
            # 代价使用归一化状态（因为这样Q的量级才合理）
            P_cost = x_norm.T @ Q @ x_norm
            
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
            
            y_val = -(P_cost + cost_p - cost_e)
            
            X_list.append(Z)
            y_list.append(y_val)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 岭回归
        lambda_reg = 0.1
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
    
    print(f"\nFinal W[:6] = {critic.W[:6]}")
    
    # 仿真
    print("\n" + "=" * 60)
    print("Simulation")
    print("=" * 60)
    
    x_p = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
    x_e = np.array([500, 1500, 200, 50, 80, 10], dtype=float)
    
    dt = 0.01
    T = 20.0
    n_steps = int(T / dt)
    
    x_p_hist = [x_p.copy()]
    x_e_hist = [x_e.copy()]
    u_p_hist = []
    u_e_hist = []
    error_hist = [np.linalg.norm(x_p - x_e)]
    
    print(f"Initial error: {error_hist[0]:.2f}")
    
    for step in range(n_steps):
        x_tilde = x_p - x_e
        
        g_p = dynamics.g(x_p)
        g_e = dynamics.g(x_e)
        grad_V = critic.predict_gradient(x_tilde)
        
        rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
        u_p = -u_bar_p * np.tanh(rho_p)
        
        rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
        u_e = u_bar_e * np.tanh(rho_e)
        
        # RK4
        def rk4(x, u):
            k1 = dynamics.dynamics(x, u)
            k2 = dynamics.dynamics(x + 0.5*dt*k1, u)
            k3 = dynamics.dynamics(x + 0.5*dt*k2, u)
            k4 = dynamics.dynamics(x + dt*k3, u)
            return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_p = rk4(x_p, u_p)
        x_e = rk4(x_e, u_e)
        
        x_p_hist.append(x_p.copy())
        x_e_hist.append(x_e.copy())
        u_p_hist.append(u_p.copy())
        u_e_hist.append(u_e.copy())
        error_hist.append(np.linalg.norm(x_p - x_e))
        
        if (step + 1) % (n_steps // 5) == 0:
            print(f"  Step {step+1}: error = {error_hist[-1]:.2f}")
    
    x_p_hist = np.array(x_p_hist)
    x_e_hist = np.array(x_e_hist)
    u_p_hist = np.array(u_p_hist)
    u_e_hist = np.array(u_e_hist)
    error_hist = np.array(error_hist)
    
    print(f"\nFinal error: {error_hist[-1]:.2f}")
    print(f"Min error: {min(error_hist):.2f}")
    reduction = (error_hist[0] - error_hist[-1]) / error_hist[0] * 100
    print(f"Error reduction: {reduction:.1f}%")
    
    if reduction > 0:
        print("[OK] Pursuit effective!")
    else:
        print("[WARNING] Pursuit not effective")
    
    # 绘图
    fig = plt.figure(figsize=(16, 10))
    
    # 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x_p_hist[:, 0], x_p_hist[:, 1], x_p_hist[:, 2], 'b-', label='Pursuer')
    ax1.plot(x_e_hist[:, 0], x_e_hist[:, 1], x_e_hist[:, 2], 'r--', label='Evader')
    ax1.scatter([x_p_hist[0, 0]], [x_p_hist[0, 1]], [x_p_hist[0, 2]], c='blue', marker='o', s=100)
    ax1.scatter([x_e_hist[0, 0]], [x_e_hist[0, 1]], [x_e_hist[0, 2]], c='red', marker='o', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # 2D
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x_p_hist[:, 0], x_p_hist[:, 1], 'b-', label='Pursuer')
    ax2.plot(x_e_hist[:, 0], x_e_hist[:, 1], 'r--', label='Evader')
    ax2.scatter([x_p_hist[0, 0]], [x_p_hist[0, 1]], c='blue', marker='o', s=100)
    ax2.scatter([x_e_hist[0, 0]], [x_e_hist[0, 1]], c='red', marker='o', s=100)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D Trajectory (X-Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 误差
    ax3 = fig.add_subplot(2, 3, 3)
    t = np.arange(len(error_hist)) * dt
    ax3.plot(t, error_hist, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error')
    ax3.set_title('State Error Norm')
    ax3.grid(True, alpha=0.3)
    
    # 追捕者控制
    ax4 = fig.add_subplot(2, 3, 4)
    t_u = np.arange(len(u_p_hist)) * dt
    ax4.plot(t_u, u_p_hist[:, 0], 'b-', label='u0')
    ax4.plot(t_u, u_p_hist[:, 1], 'b--', label='u1')
    ax4.plot(t_u, u_p_hist[:, 2], 'b:', label='u2')
    ax4.axhline(y=u_bar_p, color='k', linestyle='--', alpha=0.3)
    ax4.axhline(y=-u_bar_p, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Control')
    ax4.set_title('Pursuer Control')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 逃避者控制
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t_u, u_e_hist[:, 0], 'r-', label='u0')
    ax5.plot(t_u, u_e_hist[:, 1], 'r--', label='u1')
    ax5.plot(t_u, u_e_hist[:, 2], 'r:', label='u2')
    ax5.axhline(y=u_bar_e, color='k', linestyle='--', alpha=0.3)
    ax5.axhline(y=-u_bar_e, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Control')
    ax5.set_title('Evader Control')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 权重收敛
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(weight_history, 'b-', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('||W||')
    ax6.set_title('Weight Convergence')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return x_p_hist, x_e_hist, error_hist, critic


if __name__ == "__main__":
    x_p_hist, x_e_hist, error_hist, critic = main()

