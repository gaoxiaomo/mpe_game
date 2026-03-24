"""
简化的1追1线性追逃博弈测试

参照MATLAB代码的实现方式，验证算法正确性

系统模型（线性）:
- 状态: x = [x1, x2, x3] (追捕者相对逃避者的状态误差)
- 追捕者控制: u_p (标量)
- 逃避者控制: u_e (标量)

动力学:
ẋ = A*x + B_p*u_p + B_e*u_e

追捕者目标: 最小化 J = ∫(x'Qx + r1*u_p² - r2*u_e²)dt
逃避者目标: 最大化 J

这是一个零和博弈（追逃博弈）
"""
import numpy as np
import matplotlib.pyplot as plt


class SimplePEGame:
    """简单的1追1追逃博弈"""

    def __init__(self, u_bar_p=3.0, u_bar_e=3.0):
        """
        Args:
            u_bar_p: 追捕者控制约束
            u_bar_e: 逃避者控制约束
        """
        # 线性系统矩阵 (类似MATLAB的例子)
        self.A = np.array([
            [2.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # 追捕者控制矩阵 (作用在x3上)
        self.B_p = np.array([0.0, 0.0, 1.0])

        # 逃避者控制矩阵 (作用在x2上)
        self.B_e = np.array([0.0, 1.0, 0.0])

        # 代价函数权重
        self.Q = np.eye(3)  # 状态权重
        self.R_p = 1.0  # 追捕者控制权重
        self.R_e = 1.0  # 逃避者控制权重

        # 控制约束
        self.u_bar_p = u_bar_p
        self.u_bar_e = u_bar_e

        # 基函数数量: 3个二次项 + 3个交叉项 = 6
        self.n_basis = 6

        # 值函数权重
        self.W = np.zeros(self.n_basis)

    def basis_functions(self, x):
        """计算基函数 ψ(x)

        ψ(x) = [x1², x2², x3², x1*x2, x1*x3, x2*x3]
        """
        x1, x2, x3 = x
        psi = np.array([
            x1 ** 2,
            x2 ** 2,
            x3 ** 2,
            x1 * x2,
            x1 * x3,
            x2 * x3
        ])
        return psi

    def basis_gradient(self, x):
        """计算基函数梯度 ∇ψ(x) ∈ R^{3×6}

        每行对应一个状态分量，每列对应一个基函数
        """
        x1, x2, x3 = x
        grad = np.array([
            [2 * x1, 0, 0, x2, x3, 0],  # ∂ψ/∂x1
            [0, 2 * x2, 0, x1, 0, x3],  # ∂ψ/∂x2
            [0, 0, 2 * x3, 0, x1, x2]  # ∂ψ/∂x3
        ])
        return grad

    def compute_optimal_controls(self, x):
        """计算最优控制

        对于零和博弈:
        u_p* = -ū_p * tanh((1/(2ū_p)) * (1/R_p) * B_p^T * ∇V)
        u_e* = +ū_e * tanh((1/(2ū_e)) * (1/R_e) * B_e^T * ∇V)  # 注意正号！

        其中 ∇V = ∇ψ^T @ W
        """
        grad_psi = self.basis_gradient(x)  # (3, 6)
        grad_V = grad_psi @ self.W  # (3,)

        # 追捕者控制 (最小化代价)
        rho_p = (1.0 / (2.0 * self.u_bar_p)) * (1.0 / self.R_p) * np.dot(self.B_p, grad_V)
        u_p = -self.u_bar_p * np.tanh(rho_p)

        # 逃避者控制 (最大化代价 = 追捕者的对手)
        rho_e = (1.0 / (2.0 * self.u_bar_e)) * (1.0 / self.R_e) * np.dot(self.B_e, grad_V)
        u_e = self.u_bar_e * np.tanh(rho_e)  # 注意：正号，因为逃避者想最大化

        return u_p, u_e

    def compute_nonquad_cost(self, u, u_bar, R):
        """计算非二次控制代价

        对于饱和控制: 2*ū*R*(u*atanh(u/ū) + 0.5*ū*log(1-(u/ū)²))
        """
        ratio = np.clip(u / u_bar, -0.9999, 0.9999)
        cost = 2.0 * u_bar * R * (
                u * np.arctanh(ratio) +
                0.5 * u_bar * np.log(1.0 - ratio ** 2)
        )
        return cost

    def offline_training(self, num_grid=13, num_iterations=20, verbose=True):
        """离线训练 - 类似MATLAB的实现

        使用网格采样和最小二乘法求解Bellman方程
        """
        if verbose:
            print("=" * 60)
            print("Offline Training - Simple Linear PE Game")
            print("=" * 60)

        # 网格参数
        lb, ub = -1.2, 1.2
        step = (ub - lb) / (num_grid - 1)

        # 初始稳定策略 (简单的比例控制)
        K_p = np.array([-4.0, -1.0, -2.0])  # 追捕者增益
        K_e = np.array([1.0, 0.5, 0.5])  # 逃避者增益

        weight_history = [np.linalg.norm(self.W)]

        for iteration in range(num_iterations):
            X_list = []
            y_list = []

            for i1 in range(num_grid):
                x1 = lb + i1 * step
                for i2 in range(num_grid):
                    x2 = lb + i2 * step
                    for i3 in range(num_grid):
                        x3 = lb + i3 * step

                        x = np.array([x1, x2, x3])

                        # 计算梯度 dPHI: (3, 6)
                        dPHI = self.basis_gradient(x)

                        # 系统漂移项 f(x) = A @ x
                        f = self.A @ x

                        # 计算控制
                        if iteration == 0:
                            # 初始稳定策略
                            u_p = np.clip(np.dot(K_p, x), -self.u_bar_p * 0.99, self.u_bar_p * 0.99)
                            u_e = np.clip(np.dot(K_e, x), -self.u_bar_e * 0.99, self.u_bar_e * 0.99)
                        else:
                            # 使用学到的策略
                            u_p, u_e = self.compute_optimal_controls(x)
                            # 确保不饱和
                            u_p = np.clip(u_p, -self.u_bar_p * 0.9999, self.u_bar_p * 0.9999)
                            u_e = np.clip(u_e, -self.u_bar_e * 0.9999, self.u_bar_e * 0.9999)

                        # 状态导数: ẋ = f + B_p*u_p + B_e*u_e
                        x_dot = f + self.B_p * u_p + self.B_e * u_e

                        # Z = dPHI.T @ ẋ  (6x3) @ (3,) = (6,)
                        # 但MATLAB的dPHI是 (n_basis, n_state)，所以
                        # Z = dPHI @ ẋ，其中dPHI.T是我们的 (6, 3).T = (3, 6)的转置
                        # 实际上 Z = dPHI.T @ x_dot 在MATLAB中就是 dPHI * x_dot
                        Z = dPHI.T @ x_dot  # (6,)

                        # 状态代价 P(x) = x'Qx
                        P_x = x.T @ self.Q @ x

                        # 非二次控制代价
                        # 追捕者: 正号 (代价)
                        nonquad_p = self.compute_nonquad_cost(u_p, self.u_bar_p, self.R_p)
                        # 逃避者: 负号 (因为在追逃博弈中逃避者代价被减去)
                        nonquad_e = self.compute_nonquad_cost(u_e, self.u_bar_e, self.R_e)

                        # HJI方程: W'Z = -P(x) - nonquad_p + nonquad_e
                        # 对于追逃博弈，逃避者的代价是负的（min-max问题）
                        y_val = -P_x - nonquad_p + nonquad_e

                        X_list.append(Z)
                        y_list.append(y_val)

            # 转换为数组
            X = np.array(X_list)  # (N, 6)
            y = np.array(y_list)  # (N,)

            # 最小二乘求解 (带正则化)
            lambda_reg = 0.001
            XTX = X.T @ X + lambda_reg * np.eye(self.n_basis)
            XTy = X.T @ y
            W_new = np.linalg.solve(XTX, XTy)

            # 更新权重
            weight_change = np.linalg.norm(W_new - self.W)
            self.W = W_new.copy()
            weight_history.append(np.linalg.norm(self.W))

            if verbose:
                print(f"Iter {iteration + 1:2d}: ||W||={np.linalg.norm(self.W):8.4f}, "
                      f"ΔW={weight_change:.6f}")

        if verbose:
            print(f"\nFinal W = {self.W}")

        return weight_history

    def simulate(self, x0, T=10.0, dt=0.01):
        """仿真

        Args:
            x0: 初始状态
            T: 仿真时间
            dt: 时间步长
        """
        n_steps = int(T / dt)
        x = np.array(x0, dtype=float)

        # 记录历史
        x_history = [x.copy()]
        u_p_history = []
        u_e_history = []
        error_history = [np.linalg.norm(x)]

        for step in range(n_steps):
            # 计算控制
            u_p, u_e = self.compute_optimal_controls(x)

            # 记录
            u_p_history.append(u_p)
            u_e_history.append(u_e)

            # RK4积分
            def dynamics(state):
                return self.A @ state + self.B_p * u_p + self.B_e * u_e

            k1 = dynamics(x)
            k2 = dynamics(x + 0.5 * dt * k1)
            k3 = dynamics(x + 0.5 * dt * k2)
            k4 = dynamics(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            x_history.append(x.copy())
            error_history.append(np.linalg.norm(x))

        return {
            'x': np.array(x_history),
            'u_p': np.array(u_p_history),
            'u_e': np.array(u_e_history),
            'error': np.array(error_history),
            't': np.arange(n_steps + 1) * dt
        }


def test_single_control():
    """测试单控制器情况 (无逃避者，对标MATLAB)"""
    print("\n" + "=" * 60)
    print("Test 1: Single Controller (No Evader) - Like MATLAB")
    print("=" * 60)

    class SingleControlSystem:
        """单控制器系统 - 对标MATLAB代码"""

        def __init__(self, A1=3.0, A2=20.0):
            self.A = np.array([
                [2.0, 1.0, 1.0],
                [1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            self.G = np.array([
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0]
            ])
            self.Q = np.eye(3)
            self.R = np.eye(2)
            self.A1 = A1  # u1的约束
            self.A2 = A2  # u2的约束
            self.n_basis = 21  # MATLAB用21个基函数
            self.W = np.zeros(self.n_basis)

        def basis_gradient(self, x):
            """MATLAB的dPHI矩阵 (21x3)"""
            x1, x2, x3 = x
            dPHI = np.array([
                [2 * x1, 0, 0],
                [0, 2 * x2, 0],
                [0, 0, 2 * x3],
                [x2, x1, 0],
                [x3, 0, x1],
                [0, x3, x2],
                [4 * x1 ** 3, 0, 0],
                [0, 4 * x2 ** 3, 0],
                [0, 0, 4 * x3 ** 3],
                [2 * x1 * x2 ** 2, 2 * x1 ** 2 * x2, 0],
                [2 * x1 * x3 ** 2, 0, 2 * x1 ** 2 * x3],
                [0, 2 * x2 * x3 ** 2, 2 * x2 ** 2 * x3],
                [2 * x1 * x2 * x3, x1 ** 2 * x3, x1 ** 2 * x2],
                [x2 ** 2 * x3, 2 * x1 * x2 * x3, x1 * x2 ** 2],
                [x2 * x3 ** 2, x1 * x3 ** 2, 2 * x1 * x2 * x3],
                [3 * x1 ** 2 * x2, x1 ** 3, 0],
                [3 * x1 ** 2 * x3, 0, x1 ** 3],
                [x2 ** 3, 3 * x1 * x2 ** 2, 0],
                [x3 ** 3, 0, 3 * x1 * x3 ** 2],
                [0, x3 ** 3, 3 * x2 * x3 ** 2],
                [0, 3 * x2 ** 2 * x3, x2 ** 3]
            ])
            return dPHI

        def compute_control(self, x):
            """计算最优控制 - 对标MATLAB"""
            dPHI = self.basis_gradient(x)  # (21, 3)
            # U = -0.5 * R^{-1} * G^T * dPHI^T * W
            grad_like = dPHI.T @ self.W  # (3,)
            U = -0.5 * np.linalg.inv(self.R) @ self.G.T @ grad_like  # (2,)

            u1 = self.A1 * np.tanh(U[0] / self.A1)
            u2 = self.A2 * np.tanh(U[1] / self.A2)
            return np.array([u1, u2])

        def train(self, num_grid=13, num_iterations=20):
            """训练 - 对标MATLAB"""
            print("Training single controller system...")
            lb, ub = -1.2, 1.2
            step = (ub - lb) / (num_grid - 1)

            # 初始稳定策略
            K = np.array([
                [-8.3056, -2.2827, -4.6607],
                [-8.5707, -2.7323, -2.2827]
            ])

            weight_history = []

            for iteration in range(num_iterations):
                X_list = []
                y_list = []

                for i1 in range(num_grid):
                    x1 = lb + i1 * step
                    for i2 in range(num_grid):
                        x2 = lb + i2 * step
                        for i3 in range(num_grid):
                            x3 = lb + i3 * step

                            x = np.array([x1, x2, x3])
                            dPHI = self.basis_gradient(x)
                            f = self.A @ x

                            if iteration == 0:
                                u = K @ x
                            else:
                                u = self.compute_control(x)

                            # 饱和
                            u1 = np.clip(u[0], -self.A1 * (1 - 1e-14), self.A1 * (1 - 1e-14))
                            u2 = np.clip(u[1], -self.A2 * (1 - 1e-14), self.A2 * (1 - 1e-14))
                            u = np.array([u1, u2])

                            # Z = dPHI @ (f + G @ u)
                            x_dot = f + self.G @ u
                            Z = dPHI @ x_dot  # (21,)

                            # y = -x'Qx - 非二次项
                            P_x = x.T @ self.Q @ x

                            # 非二次代价
                            ratio1 = u1 / self.A1
                            ratio2 = u2 / self.A2
                            nonquad = (
                                    2 * self.A1 * self.R[0, 0] * (
                                    u1 * np.arctanh(ratio1) + 0.5 * self.A1 * np.log(1 - ratio1 ** 2))
                                    + 2 * self.A2 * self.R[1, 1] * (
                                            u2 * np.arctanh(ratio2) + 0.5 * self.A2 * np.log(1 - ratio2 ** 2))
                            )

                            y_val = -P_x - nonquad

                            X_list.append(Z)
                            y_list.append(y_val)

                X = np.array(X_list)
                y = np.array(y_list)

                # 最小二乘
                W_new = np.linalg.lstsq(X, y, rcond=None)[0]
                weight_change = np.linalg.norm(W_new - self.W)
                self.W = W_new.copy()
                weight_history.append(np.linalg.norm(self.W))

                print(f"Iter {iteration + 1:2d}: ||W||={np.linalg.norm(self.W):10.4f}, "
                      f"ΔW={weight_change:.6f}")

            print(f"\nFinal W (first 6):\n{self.W[:6]}")
            return weight_history

        def simulate(self, x0, T=10.0, dt=0.01):
            """仿真"""
            n_steps = int(T / dt)
            x = np.array(x0, dtype=float)
            x_history = [x.copy()]
            u_history = []

            for step in range(n_steps):
                u = self.compute_control(x)
                u_history.append(u.copy())

                def dynamics(state):
                    return self.A @ state + self.G @ u

                k1 = dynamics(x)
                k2 = dynamics(x + 0.5 * dt * k1)
                k3 = dynamics(x + 0.5 * dt * k2)
                k4 = dynamics(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                x_history.append(x.copy())

            return {
                'x': np.array(x_history),
                'u': np.array(u_history),
                't': np.arange(n_steps + 1) * dt
            }

    # 运行测试
    system = SingleControlSystem()
    weight_history = system.train(num_grid=13, num_iterations=20)

    # 仿真
    x0 = [1.2, 1.2, 1.2]
    results = system.simulate(x0, T=10.0, dt=0.01)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 状态轨迹
    ax = axes[0, 0]
    ax.plot(results['t'], results['x'][:, 0], 'r-', label='x1')
    ax.plot(results['t'], results['x'][:, 1], 'b--', label='x2')
    ax.plot(results['t'], results['x'][:, 2], 'g-.', label='x3')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('States')
    ax.set_title('State Trajectory (Single Controller)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 控制输入
    ax = axes[0, 1]
    t_u = results['t'][:-1]
    ax.plot(t_u, results['u'][:, 0], 'r-', label='u1')
    ax.plot(t_u, results['u'][:, 1], 'b--', label='u2')
    ax.axhline(y=3, color='r', linestyle=':', alpha=0.5)
    ax.axhline(y=-3, color='r', linestyle=':', alpha=0.5)
    ax.axhline(y=20, color='b', linestyle=':', alpha=0.5)
    ax.axhline(y=-20, color='b', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.set_title('Control Inputs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 权重收敛
    ax = axes[1, 0]
    ax.plot(weight_history, 'b-o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||W||')
    ax.set_title('Weight Convergence')
    ax.grid(True, alpha=0.3)

    # 状态范数
    ax = axes[1, 1]
    state_norm = np.linalg.norm(results['x'], axis=1)
    ax.plot(results['t'], state_norm, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('||x||')
    ax.set_title('State Norm')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('single_control_test.png', dpi=150)
    print("\nFigure saved to single_control_test.png")

    return results


def test_pe_game():
    """测试追逃博弈"""
    print("\n" + "=" * 60)
    print("Test 2: Pursuit-Evasion Game")
    print("=" * 60)

    # 创建博弈
    game = SimplePEGame(u_bar_p=3.0, u_bar_e=2.0)  # 追捕者能力更强

    # 训练
    weight_history = game.offline_training(num_grid=13, num_iterations=30)

    # 仿真
    x0 = [1.0, 1.0, 1.0]
    results = game.simulate(x0, T=10.0, dt=0.01)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 状态轨迹
    ax = axes[0, 0]
    ax.plot(results['t'], results['x'][:, 0], 'r-', label='x1')
    ax.plot(results['t'], results['x'][:, 1], 'b--', label='x2')
    ax.plot(results['t'], results['x'][:, 2], 'g-.', label='x3')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('States')
    ax.set_title('State Error Trajectory (PE Game)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 控制输入
    ax = axes[0, 1]
    t_u = results['t'][:-1]
    ax.plot(t_u, results['u_p'], 'b-', label='Pursuer u_p')
    ax.plot(t_u, results['u_e'], 'r--', label='Evader u_e')
    ax.axhline(y=game.u_bar_p, color='b', linestyle=':', alpha=0.5, label=f'±{game.u_bar_p}')
    ax.axhline(y=-game.u_bar_p, color='b', linestyle=':', alpha=0.5)
    ax.axhline(y=game.u_bar_e, color='r', linestyle=':', alpha=0.5, label=f'±{game.u_bar_e}')
    ax.axhline(y=-game.u_bar_e, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.set_title('Control Inputs')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 误差范数
    ax = axes[1, 0]
    ax.plot(results['t'], results['error'], 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('||x||')
    ax.set_title('State Error Norm (Should decrease if pursuer wins)')
    ax.grid(True, alpha=0.3)

    # 权重收敛
    ax = axes[1, 1]
    ax.plot(weight_history, 'b-o')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||W||')
    ax.set_title('Weight Convergence')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pe_game_test.png', dpi=150)
    print("\nFigure saved to pe_game_test.png")

    # 打印最终状态
    print(f"\nInitial state: {x0}")
    print(f"Final state: {results['x'][-1]}")
    print(f"Initial error: {np.linalg.norm(x0):.4f}")
    print(f"Final error: {results['error'][-1]:.4f}")

    if results['error'][-1] < results['error'][0]:
        print("✓ Pursuer is winning (error decreasing)")
    else:
        print("✗ Evader is winning (error not decreasing)")

    return results


def test_pe_game_comparison():
    """对比不同控制约束下的追逃博弈"""
    print("\n" + "=" * 60)
    print("Test 3: PE Game - Control Bound Comparison")
    print("=" * 60)

    cases = [
        ('u_p > u_e', 4.0, 2.0),
        ('u_p = u_e', 3.0, 3.0),
        ('u_p < u_e', 2.0, 4.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x0 = [1.0, 1.0, 1.0]

    for idx, (name, u_bar_p, u_bar_e) in enumerate(cases):
        print(f"\n--- Case: {name} ---")
        game = SimplePEGame(u_bar_p=u_bar_p, u_bar_e=u_bar_e)
        game.offline_training(num_grid=11, num_iterations=20, verbose=False)
        results = game.simulate(x0, T=10.0, dt=0.01)

        ax = axes[idx]
        ax.plot(results['t'], results['error'], 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('||x||')
        ax.set_title(f'{name}\n$\\bar{{u}}_p$={u_bar_p}, $\\bar{{u}}_e$={u_bar_e}')
        ax.grid(True, alpha=0.3)

        outcome = "Pursuer wins" if results['error'][-1] < 0.5 else "Evader escapes"
        print(f"  Final error: {results['error'][-1]:.4f} - {outcome}")

    plt.tight_layout()
    plt.savefig('pe_comparison.png', dpi=150)
    print("\nFigure saved to pe_comparison.png")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')

    # 测试1: 单控制器（对标MATLAB）
    test_single_control()

    # 测试2: 追逃博弈
    test_pe_game()

    # 测试3: 不同约束对比
    test_pe_game_comparison()

    plt.show()

