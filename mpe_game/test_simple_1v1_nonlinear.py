"""
简化的1追1场景 - 使用非线性飞行器动力学

修复原始代码的问题：
1. 归一化策略改进
2. 逃避者控制符号修正（正号用于对抗）
3. 训练采样范围与实际状态匹配
4. 使用更稳定的最小二乘方法
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from dynamics import AircraftDynamics


class CriticNetworkFixed:
    """修复的Critic网络"""
    
    def __init__(self, state_dim):
        self.n = state_dim
        # 只使用位置误差的二次项（简化）
        # 对于6维状态: 6个平方项 + 15个交叉项 = 21
        self.num_basis = 21
        self.W = np.zeros(self.num_basis)
        self.weight_history = []
        
        # 不使用归一化，直接在原始状态空间工作
        # 但使用适当的缩放使数值稳定
        self.value_scale = 1e-6  # 缩放值函数
        
    def activation(self, x):
        """二次基函数 psi(x)"""
        psi = []
        # 平方项
        for i in range(self.n):
            psi.append(x[i] ** 2)
        # 交叉项
        for i in range(self.n):
            for j in range(i + 1, self.n):
                psi.append(x[i] * x[j])
        return np.array(psi)
    
    def activation_gradient(self, x):
        """基函数梯度"""
        grad = np.zeros((self.n, self.num_basis))
        # 平方项梯度
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
    
    def predict_gradient(self, x):
        """值函数梯度"""
        return self.activation_gradient(x) @ self.W


def simple_1v1_game():
    """简化的1v1追逃博弈 - 非线性动力学"""
    
    print("=" * 60)
    print("Simple 1v1 Nonlinear Pursuit-Evasion Game")
    print("=" * 60)
    
    dynamics = AircraftDynamics()
    
    # 参数
    u_bar_p = 25.0
    u_bar_e = 15.0
    Q = np.eye(6) * 1.0
    R1 = np.eye(3) * 1.0
    R2 = np.eye(3) * 1.0
    R1_inv = np.linalg.inv(R1)
    R2_inv = np.linalg.inv(R2)
    
    # 创建Critic
    critic = CriticNetworkFixed(6)
    
    # ===== 阶段1: 简化训练 =====
    # 使用直接的LQR类型初始化
    # 对于追逃博弈，P矩阵应该是正定的
    print("\nPhase 1: Initialize with simple quadratic form")
    
    # 手动设置合理的初始权重
    # V(x) ≈ x^T P x, 其中P是对角阵
    # W = [P11, P22, P33, P44, P55, P66, 0, 0, ...]
    # 位置误差权重大，速度误差权重小
    critic.W = np.zeros(21)
    critic.W[0] = 1e-6  # x位置
    critic.W[1] = 1e-6  # y位置
    critic.W[2] = 1e-6  # z位置
    critic.W[3] = 1e-4  # vx
    critic.W[4] = 1e-4  # vy
    critic.W[5] = 1e-4  # vz
    
    # ===== 阶段2: 策略迭代训练 =====
    print("\nPhase 2: Policy Iteration Training")
    
    num_iterations = 30
    num_samples = 500
    
    # 采样范围（基于实际初始状态误差）
    pos_range = 3000.0  # 位置误差范围
    vel_range = 100.0   # 速度误差范围
    
    # 参考逃避者状态（训练时假设的）
    x_e_ref = np.array([500, 1500, 200, 50, 80, 10])
    
    weight_history = []
    
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
            
            # 计算动力学
            f_p = dynamics.f(x_p)
            f_e = dynamics.f(x_e_ref)
            g_p = dynamics.g(x_p)
            g_e = dynamics.g(x_e_ref)
            
            # 漂移项差
            F_ji = f_p - f_e
            
            # 基函数梯度
            dPsi = critic.activation_gradient(x_tilde)
            grad_V = dPsi @ critic.W
            
            # 最优控制 - 公式(40)
            # 追捕者: u_p = -u_bar_p * tanh(rho_p)
            rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
            u_p = -u_bar_p * np.tanh(rho_p)
            
            # 逃避者: u_e = +u_bar_e * tanh(rho_e) [关键修复: 正号！]
            # 因为逃避者想最大化代价，所以梯度方向相反
            rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
            u_e = u_bar_e * np.tanh(rho_e)  # 正号！
            
            # 饱和
            u_p = np.clip(u_p, -u_bar_p * 0.9999, u_bar_p * 0.9999)
            u_e = np.clip(u_e, -u_bar_e * 0.9999, u_bar_e * 0.9999)
            
            # 状态误差动力学
            # x_tilde_dot = f_p - f_e + g_p @ u_p - g_e @ u_e
            x_tilde_dot = F_ji + g_p @ u_p - g_e @ u_e
            
            # Bellman方程
            Z = dPsi.T @ x_tilde_dot
            
            # 状态代价
            P_cost = x_tilde.T @ Q @ x_tilde
            
            # 非二次控制代价
            nonquad_p = 0.0
            for k in range(3):
                ratio = u_p[k] / u_bar_p
                if abs(ratio) < 0.9999:
                    nonquad_p += 2.0 * u_bar_p * R1[k, k] * (
                        u_p[k] * np.arctanh(ratio) + 
                        0.5 * u_bar_p * np.log(1.0 - ratio**2)
                    )
            
            # 逃避者代价（减去）
            nonquad_e = 0.0
            for k in range(3):
                ratio = u_e[k] / u_bar_e
                if abs(ratio) < 0.9999:
                    nonquad_e += 2.0 * u_bar_e * R2[k, k] * (
                        u_e[k] * np.arctanh(ratio) + 
                        0.5 * u_bar_e * np.log(1.0 - ratio**2)
                    )
            
            y_val = -P_cost - nonquad_p + nonquad_e
            
            X_list.append(Z)
            y_list.append(y_val)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 岭回归
        lambda_reg = 1e-6  # 小正则化
        XTX = X.T @ X + lambda_reg * np.eye(21)
        XTy = X.T @ y
        
        try:
            W_new = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            W_new = np.linalg.lstsq(X, y, rcond=None)[0]
        
        weight_change = np.linalg.norm(W_new - critic.W)
        critic.W = W_new.copy()
        weight_history.append(np.linalg.norm(W_new))
        
        if (iteration + 1) % 5 == 0:
            print(f"Iter {iteration+1:2d}: ||W||={np.linalg.norm(W_new):.4e}, dW={weight_change:.4e}")
    
    print(f"\nFinal W (first 6): {critic.W[:6]}")
    
    # ===== 阶段3: 仿真 =====
    print("\n" + "=" * 60)
    print("Phase 3: Simulation")
    print("=" * 60)
    
    # 初始状态
    x_p = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
    x_e = np.array([500, 1500, 200, 50, 80, 10], dtype=float)
    
    dt = 0.01
    T = 10.0
    n_steps = int(T / dt)
    
    x_p_history = [x_p.copy()]
    x_e_history = [x_e.copy()]
    x_tilde_history = [x_p - x_e]
    u_p_history = []
    u_e_history = []
    
    print(f"\nInitial state error: {np.linalg.norm(x_p - x_e):.2f}")
    
    for step in range(n_steps):
        x_tilde = x_p - x_e
        
        # 计算控制
        g_p = dynamics.g(x_p)
        g_e = dynamics.g(x_e)
        grad_V = critic.predict_gradient(x_tilde)
        
        # 追捕者控制
        rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
        u_p = -u_bar_p * np.tanh(rho_p)
        
        # 逃避者控制（正号！）
        rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
        u_e = u_bar_e * np.tanh(rho_e)
        
        # RK4更新追捕者
        k1_p = dynamics.dynamics(x_p, u_p)
        k2_p = dynamics.dynamics(x_p + 0.5*dt*k1_p, u_p)
        k3_p = dynamics.dynamics(x_p + 0.5*dt*k2_p, u_p)
        k4_p = dynamics.dynamics(x_p + dt*k3_p, u_p)
        x_p = x_p + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        # RK4更新逃避者
        k1_e = dynamics.dynamics(x_e, u_e)
        k2_e = dynamics.dynamics(x_e + 0.5*dt*k1_e, u_e)
        k3_e = dynamics.dynamics(x_e + 0.5*dt*k2_e, u_e)
        k4_e = dynamics.dynamics(x_e + dt*k3_e, u_e)
        x_e = x_e + (dt/6.0) * (k1_e + 2*k2_e + 2*k3_e + k4_e)
        
        x_p_history.append(x_p.copy())
        x_e_history.append(x_e.copy())
        x_tilde_history.append(x_p - x_e)
        u_p_history.append(u_p.copy())
        u_e_history.append(u_e.copy())
        
        if (step + 1) % (n_steps // 5) == 0:
            error = np.linalg.norm(x_p - x_e)
            print(f"  Step {step+1:4d}: error = {error:.2f}")
    
    x_p_history = np.array(x_p_history)
    x_e_history = np.array(x_e_history)
    x_tilde_history = np.array(x_tilde_history)
    u_p_history = np.array(u_p_history)
    u_e_history = np.array(u_e_history)
    
    final_error = np.linalg.norm(x_tilde_history[-1])
    print(f"\nFinal state error: {final_error:.2f}")
    
    if final_error < np.linalg.norm(x_tilde_history[0]) * 0.5:
        print("[OK] Pursuit effective - error reduced!")
    else:
        print("[WARNING] Pursuit may not be effective")
    
    # ===== 绘图 =====
    fig = plt.figure(figsize=(16, 10))
    
    # 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x_p_history[:, 0], x_p_history[:, 1], x_p_history[:, 2], 'b-', label='Pursuer')
    ax1.plot(x_e_history[:, 0], x_e_history[:, 1], x_e_history[:, 2], 'r--', label='Evader')
    ax1.scatter([x_p_history[0, 0]], [x_p_history[0, 1]], [x_p_history[0, 2]], c='blue', marker='o', s=100)
    ax1.scatter([x_e_history[0, 0]], [x_e_history[0, 1]], [x_e_history[0, 2]], c='red', marker='o', s=100)
    ax1.scatter([x_p_history[-1, 0]], [x_p_history[-1, 1]], [x_p_history[-1, 2]], c='blue', marker='s', s=100)
    ax1.scatter([x_e_history[-1, 0]], [x_e_history[-1, 1]], [x_e_history[-1, 2]], c='red', marker='s', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # 2D轨迹 (X-Y)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x_p_history[:, 0], x_p_history[:, 1], 'b-', label='Pursuer')
    ax2.plot(x_e_history[:, 0], x_e_history[:, 1], 'r--', label='Evader')
    ax2.scatter([x_p_history[0, 0]], [x_p_history[0, 1]], c='blue', marker='o', s=100)
    ax2.scatter([x_e_history[0, 0]], [x_e_history[0, 1]], c='red', marker='o', s=100)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D Trajectory (X-Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 误差范数
    ax3 = fig.add_subplot(2, 3, 3)
    error_norm = np.linalg.norm(x_tilde_history, axis=1)
    t = np.arange(len(error_norm)) * dt
    ax3.plot(t, error_norm, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('||x_p - x_e||')
    ax3.set_title('State Error Norm')
    ax3.grid(True, alpha=0.3)
    
    # 追捕者控制
    ax4 = fig.add_subplot(2, 3, 4)
    t_u = np.arange(len(u_p_history)) * dt
    ax4.plot(t_u, u_p_history[:, 0], 'b-', label='u_p[0]')
    ax4.plot(t_u, u_p_history[:, 1], 'b--', label='u_p[1]')
    ax4.plot(t_u, u_p_history[:, 2], 'b:', label='u_p[2]')
    ax4.axhline(y=u_bar_p, color='k', linestyle='--', alpha=0.5)
    ax4.axhline(y=-u_bar_p, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pursuer Control')
    ax4.set_title('Pursuer Control (u_bar_p = 25)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 逃避者控制
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t_u, u_e_history[:, 0], 'r-', label='u_e[0]')
    ax5.plot(t_u, u_e_history[:, 1], 'r--', label='u_e[1]')
    ax5.plot(t_u, u_e_history[:, 2], 'r:', label='u_e[2]')
    ax5.axhline(y=u_bar_e, color='k', linestyle='--', alpha=0.5)
    ax5.axhline(y=-u_bar_e, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Evader Control')
    ax5.set_title('Evader Control (u_bar_e = 15)')
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
    
    return {
        'x_p': x_p_history,
        'x_e': x_e_history,
        'x_tilde': x_tilde_history,
        'u_p': u_p_history,
        'u_e': u_e_history,
        'weight_history': weight_history,
        'critic': critic
    }


if __name__ == "__main__":
    results = simple_1v1_game()

