"""
简化的追逃博弈RL控制器 - 使用双积分器动力学

这个版本专注于：
1. 正确实现HJI方程
2. 正确处理追捕者和逃避者的对抗关系
3. 验证收敛性

系统：
- 追捕者: x_p_dot = A x_p + B u_p
- 逃避者: x_e_dot = A x_e + B u_e
- 状态误差: x_tilde = x_p - x_e
- 误差动力学: x_tilde_dot = A x_tilde + B (u_p - u_e)

博弈结构（零和）：
- 追捕者最小化 J = integral(x^T Q x + u_p^T R1 u_p - u_e^T R2 u_e) dt
- 逃避者最大化同样的J

HJI方程：
0 = x^T Q x + u_p*^T R1 u_p* - u_e*^T R2 u_e* + (nabla V)^T (A x + B u_p* - B u_e*)

最优控制（饱和）：
- u_p* = -u_bar_p * tanh(rho_p), 其中 rho_p = (1/2u_bar_p) R1^{-1} B^T nabla V
- u_e* = +u_bar_e * tanh(rho_e), 其中 rho_e = (1/2u_bar_e) R2^{-1} B^T nabla V
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.linalg import solve_continuous_are


class DoubleIntegrator2D:
    """2D双积分器"""
    
    def __init__(self):
        # 状态: [x, y, vx, vy]
        # 控制: [ax, ay]
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
        
    def dynamics(self, x, u):
        return self.A @ x + self.B @ u


class ValueFunction:
    """二次型值函数 V(x) = x^T P x"""
    
    def __init__(self, n):
        self.n = n
        # P矩阵 (对称)
        self.P = np.eye(n) * 0.1
        
    def value(self, x):
        return x @ self.P @ x
    
    def gradient(self, x):
        # nabla V = 2 P x
        return 2 * self.P @ x
    
    def set_from_vector(self, vec):
        """从向量恢复P矩阵"""
        # vec包含上三角元素
        idx = 0
        for i in range(self.n):
            for j in range(i, self.n):
                self.P[i, j] = vec[idx]
                self.P[j, i] = vec[idx]
                idx += 1
    
    def to_vector(self):
        """将P矩阵转换为向量"""
        vec = []
        for i in range(self.n):
            for j in range(i, self.n):
                vec.append(self.P[i, j])
        return np.array(vec)


class SaturatedGameController:
    """饱和追逃博弈控制器"""
    
    def __init__(self, system, u_bar_p=25.0, u_bar_e=15.0):
        self.sys = system
        self.u_bar_p = u_bar_p
        self.u_bar_e = u_bar_e
        
        # 权重
        self.Q = np.eye(system.n) * 1.0
        self.R1 = np.eye(system.m) * 1.0
        self.R2 = np.eye(system.m) * 1.0
        self.R1_inv = np.linalg.inv(self.R1)
        self.R2_inv = np.linalg.inv(self.R2)
        
        # 值函数
        self.V = ValueFunction(system.n)
        
    def compute_controls(self, x):
        """计算最优控制
        
        追捕者: u_p = -u_bar_p * tanh(rho_p)
        逃避者: u_e = +u_bar_e * tanh(rho_e)
        """
        grad_V = self.V.gradient(x)  # 2 P x
        
        # rho = (1/2u_bar) R^{-1} B^T grad_V
        rho_p = (1.0 / (2.0 * self.u_bar_p)) * self.R1_inv @ (self.sys.B.T @ grad_V)
        rho_e = (1.0 / (2.0 * self.u_bar_e)) * self.R2_inv @ (self.sys.B.T @ grad_V)
        
        # 最优控制
        u_p = -self.u_bar_p * np.tanh(rho_p)
        u_e = self.u_bar_e * np.tanh(rho_e)  # 逃避者：正号！
        
        return u_p, u_e
    
    def compute_hamiltonian(self, x, u_p, u_e):
        """计算Hamiltonian
        
        H = x^T Q x + control_costs + grad_V^T (A x + B u_p - B u_e)
        
        在最优点H应该为0
        """
        grad_V = self.V.gradient(x)
        
        # 状态代价
        state_cost = x @ self.Q @ x
        
        # 控制代价（非二次形式）
        # 对于饱和控制: integral of 2u_bar R_ii (u arctanh(u/u_bar) + 0.5 u_bar ln(1-(u/u_bar)^2))
        def nonquad_cost(u, u_bar, R):
            cost = 0.0
            for k in range(len(u)):
                ratio = np.clip(u[k] / u_bar, -0.9999, 0.9999)
                cost += 2.0 * u_bar * R[k, k] * (
                    u[k] * np.arctanh(ratio) + 
                    0.5 * u_bar * np.log(1.0 - ratio**2)
                )
            return cost
        
        pursuer_cost = nonquad_cost(u_p, self.u_bar_p, self.R1)
        evader_cost = nonquad_cost(u_e, self.u_bar_e, self.R2)
        
        # 动力学项
        x_dot = self.sys.A @ x + self.sys.B @ (u_p - u_e)
        dynamics_term = grad_V @ x_dot
        
        # 总Hamiltonian
        H = state_cost + pursuer_cost - evader_cost + dynamics_term
        
        return H
    
    def train_policy_iteration(self, num_iterations=50, num_samples=500):
        """策略迭代训练"""
        
        print("\nPolicy Iteration Training")
        print("-" * 40)
        
        # 首先计算LQR解作为初始猜测（忽略饱和）
        try:
            P_lqr = solve_continuous_are(self.sys.A, self.sys.B, self.Q, self.R1)
            self.V.P = P_lqr.copy()
            print(f"Initialized with LQR: ||P|| = {np.linalg.norm(P_lqr):.4f}")
        except:
            print("LQR initialization failed, using identity")
        
        # P矩阵的独立元素数量
        n = self.sys.n
        num_params = n * (n + 1) // 2  # 上三角
        
        history = []
        
        for iteration in range(num_iterations):
            # 采样
            samples_X = []
            samples_y = []
            
            for _ in range(num_samples):
                # 随机状态
                x = np.random.uniform(-5, 5, n)
                
                # 当前策略下的控制
                u_p, u_e = self.compute_controls(x)
                
                # 构建Bellman方程
                # 0 = x^T Q x + cost(u_p) - cost(u_e) + grad_V^T @ x_dot
                # grad_V = 2 P x
                # 展开：0 = x^T Q x + costs + 2 x^T P (A x + B u_p - B u_e)
                # 这是关于P的线性方程
                
                # 为了简化，我们用最小化Hamiltonian的方法
                # H应该为0，所以 min ||H||^2
                
                x_dot = self.sys.A @ x + self.sys.B @ (u_p - u_e)
                
                # 目标：grad_V^T x_dot = -(state_cost + control_costs)
                # 2 x^T P x_dot = -RHS
                
                # 构建线性方程
                # 设 p = vec(P_upper), 则 2 x^T P x_dot = a^T p
                # 其中 a 是通过展开得到的系数
                
                # 简化方法：直接对P的每个元素构建系数
                row = []
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            coef = 2 * x[i] * x_dot[i]
                        else:
                            coef = 2 * (x[i] * x_dot[j] + x[j] * x_dot[i])
                        row.append(coef)
                
                samples_X.append(row)
                
                # RHS = -(state_cost + control_costs)
                state_cost = x @ self.Q @ x
                
                def nonquad_cost(u, u_bar, R):
                    cost = 0.0
                    for k in range(len(u)):
                        ratio = np.clip(u[k] / u_bar, -0.9999, 0.9999)
                        cost += 2.0 * u_bar * R[k, k] * (
                            u[k] * np.arctanh(ratio) + 
                            0.5 * u_bar * np.log(1.0 - ratio**2)
                        )
                    return cost
                
                pursuer_cost = nonquad_cost(u_p, self.u_bar_p, self.R1)
                evader_cost = nonquad_cost(u_e, self.u_bar_e, self.R2)
                
                rhs = -(state_cost + pursuer_cost - evader_cost)
                samples_y.append(rhs)
            
            X = np.array(samples_X)
            y = np.array(samples_y)
            
            # 岭回归
            lambda_reg = 0.01
            XTX = X.T @ X + lambda_reg * np.eye(num_params)
            XTy = X.T @ y
            
            try:
                p_new = np.linalg.solve(XTX, XTy)
            except:
                p_new = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # 更新P
            P_old = self.V.to_vector()
            change = np.linalg.norm(p_new - P_old)
            self.V.set_from_vector(p_new)
            
            history.append(np.linalg.norm(self.V.P))
            
            if (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1:3d}: ||P||={np.linalg.norm(self.V.P):.4f}, dP={change:.6f}")
        
        print(f"\nFinal P:\n{self.V.P}")
        
        return history


def simulate_game(controller, x0_p, x0_e, dt=0.01, T=10.0):
    """仿真博弈"""
    
    n_steps = int(T / dt)
    sys = controller.sys
    
    x_p = x0_p.copy()
    x_e = x0_e.copy()
    
    x_p_history = [x_p.copy()]
    x_e_history = [x_e.copy()]
    u_p_history = []
    u_e_history = []
    
    for _ in range(n_steps):
        x_tilde = x_p - x_e
        u_p, u_e = controller.compute_controls(x_tilde)
        
        # RK4更新
        def update(x, u):
            k1 = sys.dynamics(x, u)
            k2 = sys.dynamics(x + 0.5*dt*k1, u)
            k3 = sys.dynamics(x + 0.5*dt*k2, u)
            k4 = sys.dynamics(x + dt*k3, u)
            return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_p = update(x_p, u_p)
        x_e = update(x_e, u_e)
        
        x_p_history.append(x_p.copy())
        x_e_history.append(x_e.copy())
        u_p_history.append(u_p.copy())
        u_e_history.append(u_e.copy())
    
    return {
        'x_p': np.array(x_p_history),
        'x_e': np.array(x_e_history),
        'u_p': np.array(u_p_history),
        'u_e': np.array(u_e_history)
    }


def main():
    print("=" * 60)
    print("Saturated Zero-Sum Pursuit-Evasion Game")
    print("=" * 60)
    
    # 创建系统
    sys = DoubleIntegrator2D()
    
    # 测试不同的饱和限制比例
    cases = [
        ("u_p > u_e (Pursuer advantage)", 25.0, 15.0),
        ("u_p = u_e (Equal)", 20.0, 20.0),
        ("u_p < u_e (Evader advantage)", 15.0, 25.0),
    ]
    
    results = {}
    
    for name, u_bar_p, u_bar_e in cases:
        print(f"\n{'='*60}")
        print(f"Case: {name}")
        print(f"u_bar_p = {u_bar_p}, u_bar_e = {u_bar_e}")
        print(f"{'='*60}")
        
        controller = SaturatedGameController(sys, u_bar_p, u_bar_e)
        history = controller.train_policy_iteration(num_iterations=50, num_samples=500)
        
        # 仿真
        x0_p = np.array([5.0, 3.0, 0.0, 0.0])  # 追捕者初始位置
        x0_e = np.array([0.0, 0.0, 0.0, 0.0])  # 逃避者初始位置
        
        print(f"\nSimulation:")
        print(f"  Initial distance: {np.linalg.norm(x0_p[:2] - x0_e[:2]):.2f}")
        
        sim = simulate_game(controller, x0_p, x0_e, dt=0.01, T=10.0)
        
        final_dist = np.linalg.norm(sim['x_p'][-1, :2] - sim['x_e'][-1, :2])
        print(f"  Final distance: {final_dist:.2f}")
        
        results[name] = {
            'sim': sim,
            'history': history,
            'controller': controller
        }
    
    # 绘图
    fig, axes = plt.subplots(len(cases), 3, figsize=(15, 4*len(cases)))
    
    for idx, (name, _, _) in enumerate(cases):
        sim = results[name]['sim']
        history = results[name]['history']
        
        # 轨迹
        ax = axes[idx, 0]
        ax.plot(sim['x_p'][:, 0], sim['x_p'][:, 1], 'b-', label='Pursuer')
        ax.plot(sim['x_e'][:, 0], sim['x_e'][:, 1], 'r--', label='Evader')
        ax.scatter([sim['x_p'][0, 0]], [sim['x_p'][0, 1]], c='blue', marker='o', s=100)
        ax.scatter([sim['x_e'][0, 0]], [sim['x_e'][0, 1]], c='red', marker='o', s=100)
        ax.scatter([sim['x_p'][-1, 0]], [sim['x_p'][-1, 1]], c='blue', marker='s', s=100)
        ax.scatter([sim['x_e'][-1, 0]], [sim['x_e'][-1, 1]], c='red', marker='s', s=100)
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
        
        # 权重收敛
        ax = axes[idx, 2]
        ax.plot(history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||P||')
        ax.set_title('Weight Convergence')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    results = main()

