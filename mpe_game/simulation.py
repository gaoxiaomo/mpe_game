"""
多智能体追逃博弈仿真模块

核心算法流程：
1. 离线训练：网格采样 + 最小二乘法求解权重W
2. 在线仿真：使用固定W计算最优控制

完全按照论文实现：
- 公式(53): 非线性飞行器漂移项 f(x)
- 公式(54): 控制增益矩阵 g(x)
- 公式(18): Hamiltonian方程
- 公式(40): 最优控制策略
- 公式(41): Off-Policy Bellman方程
"""
import numpy as np
from config import Config
from agents import Pursuer, Evader
from graph import TargetGraph
from networks import CriticNetwork
from controller import OptimalController
from dynamics import AircraftDynamics


def offline_training(pursuers, evaders, target_graph, critics, controller,
                     num_grid_points=13, num_iterations=20):
    """离线训练 - 网格采样 + 最小二乘
    
    完全按照论文实现：
    
    公式(41) - Off-Policy Bellman方程:
    H_i^* = P_i(x) + U_N - W_N + W^T ∇ψ(x) F_i(x) = 0
    
    其中：
    - P_i(x) = x̃^T Q x̃ (状态代价)
    - U_N: 追捕者非二次代价 (公式14)
    - W_N: 逃避者非二次代价 (公式14)
    - F_i(x) = Σ c_{j,i}(F_{j,i}(x) + g_j^p u_j^p - g_i^e u_i^e)
    - F_{j,i}(x) = f_j^p(x_j^p) - f_i^e(x_i^e) (公式53)
    
    算法流程：
    1. 在归一化状态空间网格上采样
    2. 构建线性方程组: X @ W = y
    3. 最小二乘求解: W = (X^T X + λI)^{-1} X^T y
    4. 迭代直到收敛
    
    Args:
        num_grid_points: 每个维度的网格点数
        num_iterations: 策略迭代次数
    """
    print(f"\n{'='*60}")
    print("Offline Training - Full Nonlinear Dynamics (Paper Eqs. 41, 53, 54)")
    print(f"{'='*60}")
    print(f"Grid: {num_grid_points}^3 x 3 = {num_grid_points**3 * 3} samples per iter")
    print(f"Iterations: {num_iterations}")
    
    # 创建动力学模型（用于计算完整的非线性f和g）
    dynamics = AircraftDynamics()
    
    # 网格参数（归一化状态空间）
    # 扩大范围以覆盖更大的初始状态误差
    lb = -3.0
    ub = 3.0
    step = (ub - lb) / (num_grid_points - 1)
    
    # 获取参数
    state_scale = critics[0].state_scale
    Q = controller.Q
    R1 = controller.R1
    R2 = controller.R2
    R1_inv = controller.R1_inv
    R2_inv = controller.R2_inv
    u_bar_p = controller.u_bar_p
    u_bar_e = controller.u_bar_e
    
    # 获取图结构参数
    # c_{j,i}: 追捕者j追踪逃避者i的权重
    c_ji = target_graph.A_pe[:, 0]  # 对于单逃避者情况
    N_p = len([c for c in c_ji if c > 0])  # 追踪该逃避者的追捕者数量
    d_ep = 1.0  # 逃避者的出度 (单逃避者情况)
    
    # 初始稳定策略 - 公式(40)的LQR近似
    # 作用在归一化位置维度上
    K_init = np.array([
        [-8.3, -2.3, -4.7],
        [-8.6, -2.7, -2.3],
        [-4.0, -1.0, -2.0]
    ])
    
    weight_history = []
    
    for iteration in range(num_iterations):
        X_list = []
        y_list = []
        
        # 在位置维度网格采样，每个位置随机采样几个速度
        num_vel_samples = 3
        
        for i1 in range(num_grid_points):
            x1 = lb + i1 * step
            for i2 in range(num_grid_points):
                x2 = lb + i2 * step
                for i3 in range(num_grid_points):
                    x3 = lb + i3 * step
                    
                    for _ in range(num_vel_samples):
                        # 速度归一化采样
                        v1, v2, v3 = np.random.uniform(-0.5, 0.5, 3)
                        
                        # 归一化状态误差
                        x_norm = np.array([x1, x2, x3, v1, v2, v3])
                        
                        # 实际状态误差 = 归一化状态 * 缩放因子
                        x_tilde = x_norm * state_scale
                        
                        # ===== 公式(53): 计算完整的非线性漂移项 =====
                        # F_{j,i}(x) = f_j^p(x_j^p) - f_i^e(x_i^e)
                        # 
                        # 假设逃避者在参考点（原点附近），追捕者状态 = 状态误差
                        # x_j^p ≈ x_tilde (相对于逃避者的误差)
                        # x_i^e ≈ [500, 1500, 200, 50, 80, 10] (逃避者初始状态)
                        
                        # 逃避者参考状态
                        x_e_ref = np.array([500, 1500, 200, 50, 80, 10])
                        
                        # 追捕者虚拟状态 = 逃避者参考 + 状态误差
                        x_p_virtual = x_e_ref + x_tilde
                        
                        # 计算各自的非线性动力学 - 公式(53)
                        f_p = dynamics.f(x_p_virtual)  # 追捕者漂移项
                        f_e = dynamics.f(x_e_ref)      # 逃避者漂移项
                        
                        # F_{j,i} = f_j^p - f_i^e
                        F_ji = f_p - f_e
                        
                        # ===== 公式(54): 控制增益矩阵 =====
                        g_p = dynamics.g(x_p_virtual)  # 追捕者g矩阵
                        g_e = dynamics.g(x_e_ref)      # 逃避者g矩阵
                        
                        # 计算基函数梯度 ∇ψ(x) ∈ R^{6×21}
                        dPHI = critics[0].activation_gradient(x_tilde)
                        
                        # ===== 公式(40): 计算最优控制 =====
                        if iteration == 0:
                            # 初始稳定策略 (作用在位置维度)
                            # 追捕者：追向目标（减小误差）
                            u_p = K_init @ x_norm[:3]
                            # 逃避者：初始策略
                            u_e = -K_init @ x_norm[:3] * (u_bar_e / u_bar_p)
                        else:
                            # 使用学到的策略 - 公式(40)
                            W = critics[0].W
                            grad_V = dPHI @ W  # ∇ψ^T(x) W̄_{i,s}
                            
                            # 追捕者控制: u_j^{p*} = -ū_p * tanh(ρ_p)
                            rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ (g_p.T @ grad_V)
                            u_p = -u_bar_p * np.tanh(rho_p)
                            
                            # 逃避者控制: u_i^{e*} = -ū_e * tanh(ρ_e)
                            # 论文原版：两者都用负号
                            rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ (g_e.T @ grad_V)
                            u_e = -u_bar_e * np.tanh(rho_e)  # 论文原版：负号
                        
                        # 控制饱和（确保在有效范围内）
                        u_p = np.clip(u_p, -u_bar_p * 0.9999, u_bar_p * 0.9999)
                        u_e = np.clip(u_e, -u_bar_e * 0.9999, u_bar_e * 0.9999)
                        
                        # ===== 公式(18): 构建状态误差动力学 =====
                        # ẋ̃ = Σ c_{j,i}(F_{j,i}(x) + g_j^p u_j^p - g_i^e u_i^e)
                        # 对于单逃避者情况，所有追捕者都追踪它
                        x_tilde_dot = F_ji + g_p @ u_p - g_e @ u_e
                        
                        # 构建方程: Z = ∇ψ^T @ ẋ̃
                        Z = dPHI.T @ x_tilde_dot  # (21,)
                        
                        # ===== 公式(13): 状态代价 =====
                        # 使用归一化状态计算代价，保持与梯度量级一致
                        # 注意：这里使用归一化状态是因为W是在归一化空间训练的
                        P_i = x_norm.T @ Q @ x_norm
                        
                        # ===== 公式(14): 非二次控制代价 =====
                        # 追捕者非二次项: Σ c_{j,i} * 2ū_p²R̄_1 * ln(1 - tanh²(ρ_p))
                        # 转换形式: 2ū_p R_1 * (u·arctanh(u/ū_p) + 0.5ū_p·ln(1-(u/ū_p)²))
                        
                        nonquad_p = 0.0
                        for k in range(3):
                            u_k = u_p[k]
                            ratio = u_k / u_bar_p
                            if abs(ratio) < 0.9999:
                                # 公式(14) 的积分形式
                                term = 2.0 * u_bar_p * R1[k, k] * (
                                    u_k * np.arctanh(ratio) + 
                                    0.5 * u_bar_p * np.log(1.0 - ratio**2)
                                )
                                nonquad_p += term
                        
                        # 逃避者非二次项 (减去，因为逃避者想最大化代价)
                        nonquad_e = 0.0
                        for k in range(3):
                            u_k = u_e[k]
                            ratio = u_k / u_bar_e
                            if abs(ratio) < 0.9999:
                                term = 2.0 * u_bar_e * R2[k, k] * (
                                    u_k * np.arctanh(ratio) + 
                                    0.5 * u_bar_e * np.log(1.0 - ratio**2)
                                )
                                nonquad_e += term
                        
                        # 总非二次代价: U_N - W_N
                        nonquad = nonquad_p - d_ep * nonquad_e
                        
                        # ===== 公式(41): Bellman方程右边 =====
                        # W^T ∇ψ ẋ̃ = -P_i - U_N + W_N
                        # 即 Z @ W = y
                        y_val = -P_i - nonquad
                        
                        X_list.append(Z)
                        y_list.append(y_val)
        
        # 转换为数组
        X = np.array(X_list)  # (N, 21)
        y = np.array(y_list)  # (N,)
        
        # 最小二乘求解（带正则化）- 岭回归
        lambda_reg = 0.01
        XTX = X.T @ X + lambda_reg * np.eye(21)
        XTy = X.T @ y
        W_new = np.linalg.solve(XTX, XTy)
        
        # 计算权重变化
        weight_change = np.linalg.norm(W_new - critics[0].W)
        critics[0].W = W_new.copy()
        weight_history.append(weight_change)
        
        # 记录权重历史（用于可视化）
        critics[0].weight_history.append(np.linalg.norm(W_new) ** 2)
        
        # 打印进度
        W_norm = np.linalg.norm(W_new)
        print(f"Iter {iteration+1:2d}/{num_iterations}: ||W||={W_norm:10.4f}, "
              f"ΔW={weight_change:.6f}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Final ||W|| = {np.linalg.norm(critics[0].W):.4f}")
    print(f"W = {critics[0].W}")
    print(f"{'='*60}")
    
    return weight_history


def run_simulation(pursuers, evaders, target_graph, controller, critics, dt, T,
                   online_learning=False, update_interval=100):
    """仿真阶段 - 使用训练好的权重W进行控制
    
    使用完整的非线性飞行器动力学进行状态更新
    
    Args:
        online_learning: 是否启用在线Off-Policy学习
        update_interval: 在线学习时的权重更新间隔（步数）
    """
    n_steps = int(T / dt)
    dynamics = AircraftDynamics()
    
    state_errors = {i: [] for i in range(len(evaders))}
    team_errors = []
    control_history = {'pursuers': [], 'evader': []}
    heights_history = []  # 记录高度用于可视化
    
    # 在线学习的数据缓冲区
    if online_learning:
        X_buffer = []
        y_buffer = []
        state_scale = critics[0].state_scale
        Q = controller.Q
        R1 = controller.R1
        R2 = controller.R2
        u_bar_p = controller.u_bar_p
        u_bar_e = controller.u_bar_e
    
    print(f"\nRunning simulation: {n_steps} steps, dt={dt}, T={T}")
    if online_learning:
        print(f"Online Off-Policy learning enabled, update every {update_interval} steps")
    
    for step in range(n_steps):
        for i, evader in enumerate(evaders):
            # 计算状态误差
            x_tilde = controller.compute_state_error(i)
            
            # 使用训练好的W计算最优控制
            u_pursuers, u_evader, _ = controller.compute_optimal_controls(i)
            
            # ===== 在线Off-Policy学习：收集样本 =====
            if online_learning:
                x_norm = x_tilde / state_scale
                dPHI = critics[i].activation_gradient(x_tilde)
                
                # 收集样本
                x_e_ref = evader.state.copy()
                for j, u_p in u_pursuers:
                    x_p = pursuers[j].state.copy()
                    f_p = dynamics.f(x_p)
                    f_e = dynamics.f(x_e_ref)
                    F_ji = f_p - f_e
                    g_p = dynamics.g(x_p)
                    g_e = dynamics.g(x_e_ref)
                    
                    x_tilde_dot = F_ji + g_p @ u_p - g_e @ u_evader
                    Z = dPHI.T @ x_tilde_dot
                    
                    P_i = x_norm.T @ Q @ x_norm
                    
                    # 非二次代价
                    nonquad = 0.0
                    for k in range(3):
                        r_p = np.clip(u_p[k] / u_bar_p, -0.9999, 0.9999)
                        r_e = np.clip(u_evader[k] / u_bar_e, -0.9999, 0.9999)
                        if abs(r_p) < 0.9999:
                            nonquad += 2.0 * u_bar_p * R1[k,k] * (
                                u_p[k] * np.arctanh(r_p) + 0.5 * u_bar_p * np.log(1.0 - r_p**2))
                        if abs(r_e) < 0.9999:
                            nonquad -= 2.0 * u_bar_e * R2[k,k] * (
                                u_evader[k] * np.arctanh(r_e) + 0.5 * u_bar_e * np.log(1.0 - r_e**2))
                    
                    y_val = -P_i - nonquad
                    X_buffer.append(Z)
                    y_buffer.append(y_val)
                
                # 定期更新权重 - 公式(46)的最小二乘版本
                if (step + 1) % update_interval == 0 and len(X_buffer) >= 50:
                    X = np.array(X_buffer)
                    y = np.array(y_buffer)
                    lambda_reg = 0.01
                    try:
                        XTX = X.T @ X + lambda_reg * np.eye(21)
                        XTy = X.T @ y
                        W_new = np.linalg.solve(XTX, XTy)
                        W_change = np.linalg.norm(W_new - critics[i].W)
                        critics[i].W = W_new.copy()
                        # 记录权重历史
                        critics[i].weight_history.append(np.linalg.norm(W_new) ** 2)
                        print(f"  Step {step+1}: Online update, ||W||={np.linalg.norm(W_new):.2f}, dW={W_change:.4f}")
                    except np.linalg.LinAlgError:
                        pass
                    X_buffer = []
                    y_buffer = []
            
            # 更新智能体状态（使用非线性动力学 + RK4积分）
            for j, u_p in u_pursuers:
                pursuers[j].update_state(u_p, dt)
            evader.update_state(u_evader, dt)
            
            state_errors[i].append(np.linalg.norm(x_tilde))
            
            # 记录控制
            if step % 100 == 0:
                control_history['pursuers'].append([u for _, u in u_pursuers])
                control_history['evader'].append(u_evader)
        
        # 记录高度（用于饱和对比图）
        h_pursuers = [p.state[2] for p in pursuers]
        h_evader = evaders[0].state[2]
        heights_history.append(h_pursuers + [h_evader])
        
        # 更新目标图（多逃避者场景）
        if len(evaders) > 1:
            target_graph.update_graph(pursuers, evaders)
        
        # 计算团队误差
        team_error = target_graph.compute_team_error(pursuers, evaders)
        team_errors.append(team_error)
        
        # 打印进度
        if (step + 1) % (n_steps // 10) == 0:
            print(f"Step {step+1}/{n_steps}: TeamError={team_error:.2f}")
    
    print(f"\nSimulation done! Final TeamError: {team_errors[-1]:.4f}")
    
    return {
        'state_errors': state_errors,
        'team_errors': team_errors,
        'control_history': control_history,
        'heights': np.array(heights_history)
    }


def run_scenario1():
    """场景1: 三追捕者-单逃避者"""
    print("=" * 60)
    print("Scenario 1: Three-Pursuer-One-Evader")
    print("=" * 60)

    # 初始化追捕者（使用论文Table I的参数）
    pursuers = [
        Pursuer(0, [600, -1900, 300, 78, 2, 18]),
        Pursuer(1, [1200, 500, 2200, 2, 20, 15]),
        Pursuer(2, [200, 600, 800, 5, 10, 125])
    ]
    
    # 期望相对位移
    pursuers[0].set_target(0, [50, 10, 0, 0, 0, 0])
    pursuers[1].set_target(0, [10, 50, 0, 0, 0, 0])
    pursuers[2].set_target(0, [-10, 0, -50, 0, 0, 0])

    # 初始化逃避者
    evaders = [Evader(0, [500, 1500, 200, 50, 80, 10])]

    # 目标图
    target_graph = TargetGraph(len(pursuers), len(evaders))
    target_graph.A_pe = np.array([[1], [1], [1]])
    target_graph.A_ep = np.array([[1, 1, 1]])

    # Critic网络和控制器
    critics = [CriticNetwork(Config.STATE_DIM) for _ in evaders]
    controller = OptimalController(pursuers, evaders, target_graph, critics)

    # ===== 阶段1: 离线训练 =====
    weight_history = offline_training(
        pursuers, evaders, target_graph, critics, controller,
        num_grid_points=13,
        num_iterations=30  # 增加迭代次数以确保收敛
    )
    
    # ===== 阶段2: 重置状态并仿真 =====
    # 重置到初始状态
    pursuers[0].state = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
    pursuers[1].state = np.array([1200, 500, 2200, 2, 20, 15], dtype=float)
    pursuers[2].state = np.array([200, 600, 800, 5, 10, 125], dtype=float)
    evaders[0].state = np.array([500, 1500, 200, 50, 80, 10], dtype=float)
    
    for p in pursuers:
        p.state_history = [p.state.copy()]
        p.control_history = []
    for e in evaders:
        e.state_history = [e.state.copy()]
        e.control_history = []

    # 运行仿真（启用在线Off-Policy学习）
    sim_results = run_simulation(
        pursuers, evaders, target_graph, controller, critics,
        Config.DT, Config.T_FINAL,
        online_learning=True,  # 启用在线Off-Policy学习
        update_interval=100    # 每100步更新一次权重
    )

    return {
        'pursuers': pursuers,
        'evaders': evaders,
        'state_errors': sim_results['state_errors'][0],
        'team_errors': sim_results['team_errors'],
        'heights': sim_results['heights'],
        'weight_history': weight_history,
        'critics': critics
    }


def run_scenario2():
    """场景2: 三追捕者-三逃避者"""
    print("=" * 60)
    print("Scenario 2: Three-Pursuer-Three-Evader")
    print("=" * 60)

    pursuers = [
        Pursuer(0, [263, -2380, 800, 123, 12, 18]),
        Pursuer(1, [-380, 3100, -200, 12, -40, 15]),
        Pursuer(2, [2500, -3280, 400, -15, 30, 15])
    ]

    for j, p in enumerate(pursuers):
        p.set_target(j, [100, 50, 50, 0, 0, 0])

    evaders = [
        Evader(0, [1037, 360, 100, 80, 40, 0]),
        Evader(1, [521, 1030, 34, 0, 80, 1.6]),
        Evader(2, [1816, -1887, -90, 40, 65, -0.1])
    ]

    target_graph = TargetGraph(len(pursuers), len(evaders))
    critics = [CriticNetwork(Config.STATE_DIM) for _ in evaders]
    controller = OptimalController(pursuers, evaders, target_graph, critics)

    # 离线训练
    weight_history = offline_training(
        pursuers, evaders, target_graph, critics, controller,
        num_grid_points=10,
        num_iterations=20
    )
    
    # 重置并仿真
    pursuers[0].state = np.array([263, -2380, 800, 123, 12, 18], dtype=float)
    pursuers[1].state = np.array([-380, 3100, -200, 12, -40, 15], dtype=float)
    pursuers[2].state = np.array([2500, -3280, 400, -15, 30, 15], dtype=float)
    evaders[0].state = np.array([1037, 360, 100, 80, 40, 0], dtype=float)
    evaders[1].state = np.array([521, 1030, 34, 0, 80, 1.6], dtype=float)
    evaders[2].state = np.array([1816, -1887, -90, 40, 65, -0.1], dtype=float)
    
    for p in pursuers:
        p.state_history = [p.state.copy()]
        p.control_history = []
    for e in evaders:
        e.state_history = [e.state.copy()]
        e.control_history = []

    sim_results = run_simulation(
        pursuers, evaders, target_graph, controller, critics,
        Config.DT, Config.T_FINAL,
        online_learning=True,  # 启用在线Off-Policy学习
        update_interval=100
    )

    return {
        'pursuers': pursuers,
        'evaders': evaders,
        'state_errors': sim_results['state_errors'],
        'team_errors': sim_results['team_errors'],
        'heights': sim_results['heights'],
        'critics': critics
    }


def run_saturation_comparison():
    """饱和约束对比实验"""
    print("=" * 60)
    print("Saturation Constraint Comparison")
    print("=" * 60)

    results = {}
    cases = [
        ('u_p > u_e', 25.0, 15.0),
        ('u_p = u_e', 20.0, 20.0),
        ('u_p < u_e', 15.0, 25.0)
    ]

    for name, u_bar_p, u_bar_e in cases:
        print(f"\n{'='*40}")
        print(f"Case: {name} (u_p={u_bar_p}, u_e={u_bar_e})")
        
        pursuers = [
            Pursuer(0, [600, -1900, 300, 78, 2, 18], u_bar=u_bar_p),
            Pursuer(1, [1200, 500, 2200, 2, 20, 15], u_bar=u_bar_p),
            Pursuer(2, [200, 600, 800, 5, 10, 125], u_bar=u_bar_p)
        ]
        pursuers[0].set_target(0, [50, 10, 0, 0, 0, 0])
        pursuers[1].set_target(0, [10, 50, 0, 0, 0, 0])
        pursuers[2].set_target(0, [-10, 0, -50, 0, 0, 0])

        evaders = [Evader(0, [500, 1500, 200, 50, 80, 10], u_bar=u_bar_e)]

        target_graph = TargetGraph(len(pursuers), len(evaders))
        target_graph.A_pe = np.array([[1], [1], [1]])
        target_graph.A_ep = np.array([[1, 1, 1]])

        critics = [CriticNetwork(Config.STATE_DIM)]
        controller = OptimalController(pursuers, evaders, target_graph, critics)
        controller.u_bar_p = u_bar_p
        controller.u_bar_e = u_bar_e

        # 训练
        offline_training(pursuers, evaders, target_graph, critics, controller,
                        num_grid_points=13, num_iterations=20)
        
        # 重置
        pursuers[0].state = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
        pursuers[1].state = np.array([1200, 500, 2200, 2, 20, 15], dtype=float)
        pursuers[2].state = np.array([200, 600, 800, 5, 10, 125], dtype=float)
        evaders[0].state = np.array([500, 1500, 200, 50, 80, 10], dtype=float)
        for p in pursuers:
            p.state_history = [p.state.copy()]
        for e in evaders:
            e.state_history = [e.state.copy()]

        # 仿真
        sim = run_simulation(pursuers, evaders, target_graph, controller, critics,
                           Config.DT, Config.T_FINAL)
        
        results[name] = {
            'team_errors': sim['team_errors'],
            'final_error': sim['team_errors'][-1],
            'heights': sim['heights'],
            'u_bar_p': u_bar_p,
            'u_bar_e': u_bar_e
        }
        print(f"Final Error: {sim['team_errors'][-1]:.2f}")

    return results
