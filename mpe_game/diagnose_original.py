"""
诊断原始代码问题

分析：
1. 状态误差计算是否正确
2. 控制方向是否正确（追捕者vs逃避者）
3. 值函数训练是否收敛
4. 非线性动力学的影响
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from config import Config
from agents import Pursuer, Evader
from graph import TargetGraph
from networks import CriticNetwork
from controller import OptimalController
from dynamics import AircraftDynamics


def diagnose_control_direction():
    """诊断控制方向问题"""
    print("=" * 60)
    print("Diagnose 1: Control Direction")
    print("=" * 60)
    
    # 设置简单场景
    pursuers = [Pursuer(0, [100, 0, 100, 50, 0, 0])]  # 在x=100
    pursuers[0].set_target(0, [0, 0, 0, 0, 0, 0])
    evaders = [Evader(0, [0, 0, 100, 50, 0, 0])]  # 在x=0
    
    target_graph = TargetGraph(1, 1)
    target_graph.A_pe = np.array([[1]])
    target_graph.A_ep = np.array([[1]])
    
    # 使用手动设置的权重
    critics = [CriticNetwork(Config.STATE_DIM)]
    # 设置一个简单的权重：只惩罚位置误差
    # W对应 [x1^2, x2^2, x3^2, v1^2, v2^2, v3^2, cross terms...]
    critics[0].W = np.zeros(21)
    critics[0].W[0] = 10.0  # 惩罚x误差
    critics[0].W[1] = 10.0  # 惩罚y误差
    critics[0].W[2] = 10.0  # 惩罚z误差
    
    controller = OptimalController(pursuers, evaders, target_graph, critics)
    
    # 计算状态误差
    x_tilde = controller.compute_state_error(0)
    print(f"\n状态误差 x_tilde = {x_tilde}")
    print(f"  位置误差: {x_tilde[:3]}")
    print(f"  速度误差: {x_tilde[3:]}")
    
    # 计算梯度
    grad_psi = critics[0].activation_gradient(x_tilde)
    W = critics[0].W
    grad_V = grad_psi @ W
    print(f"\n值函数梯度 grad_V = {grad_V}")
    
    # 计算控制
    u_pursuers, u_evader, _ = controller.compute_optimal_controls(0)
    
    print(f"\n追捕者控制: {u_pursuers[0][1]}")
    print(f"逃避者控制: {u_evader}")
    
    # 分析
    print("\n分析:")
    print(f"  x_tilde[0] = {x_tilde[0]:.2f} (追捕者在逃避者右边)")
    print(f"  追捕者要减小误差，应该向左移动（加速度为负）")
    print(f"  u_p[0] = {u_pursuers[0][1][0]:.4f}")
    if u_pursuers[0][1][0] < 0:
        print("  [OK] 追捕者控制方向正确（向左）")
    else:
        print("  [ERROR] 追捕者控制方向错误！应该为负")
    
    print(f"\n  逃避者要增大误差，应该向左移动（远离追捕者）")
    print(f"  u_e[0] = {u_evader[0]:.4f}")
    if u_evader[0] < 0:
        print("  [OK] 逃避者控制方向正确（向左逃离）")
    else:
        print("  [ERROR] 逃避者控制方向错误！应该为负")
    
    return x_tilde, grad_V, u_pursuers, u_evader


def diagnose_dynamics():
    """诊断动力学模型"""
    print("\n" + "=" * 60)
    print("Diagnose 2: Dynamics Model")
    print("=" * 60)
    
    dynamics = AircraftDynamics()
    
    # 测试状态
    x = np.array([500, 1000, 200, 80, 50, 10])
    
    f = dynamics.f(x)
    g = dynamics.g(x)
    
    print(f"\n状态 x = {x}")
    print(f"漂移项 f(x) = {f}")
    print(f"控制矩阵 g(x) =\n{g}")
    
    # 验证
    print("\n验证:")
    print(f"  f[0:3] = 速度 = {f[:3]} (应该等于 x[3:6] = {x[3:6]})")
    print(f"  f[3:6] = 加速度 (来自气动力)")
    
    # 控制对状态的影响
    u_test = np.array([1, 0, 0])  # 只有x方向控制
    x_dot = dynamics.dynamics(x, u_test)
    print(f"\n施加控制 u = {u_test}:")
    print(f"  x_dot = {x_dot}")
    print(f"  速度变化 = {x_dot[3:6]}")


def diagnose_training():
    """诊断训练过程"""
    print("\n" + "=" * 60)
    print("Diagnose 3: Training Process")
    print("=" * 60)
    
    # 简化场景
    pursuers = [Pursuer(0, [600, -1900, 300, 78, 2, 18])]
    pursuers[0].set_target(0, [0, 0, 0, 0, 0, 0])
    evaders = [Evader(0, [500, 1500, 200, 50, 80, 10])]
    
    target_graph = TargetGraph(1, 1)
    target_graph.A_pe = np.array([[1]])
    target_graph.A_ep = np.array([[1]])
    
    critics = [CriticNetwork(Config.STATE_DIM)]
    controller = OptimalController(pursuers, evaders, target_graph, critics)
    
    # 计算初始状态误差
    x_tilde = controller.compute_state_error(0)
    print(f"\n初始状态误差: {x_tilde}")
    print(f"  ||x_tilde|| = {np.linalg.norm(x_tilde):.2f}")
    
    # 训练前的控制
    u_before = controller.compute_optimal_controls(0)
    print(f"\n训练前控制:")
    print(f"  u_p = {u_before[0][0][1]}")
    print(f"  u_e = {u_before[1]}")
    
    # 执行训练
    from simulation import offline_training
    print("\n开始训练...")
    offline_training(pursuers, evaders, target_graph, critics, controller,
                    num_grid_points=10, num_iterations=20)
    
    # 训练后的控制
    # 重置状态
    pursuers[0].state = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
    evaders[0].state = np.array([500, 1500, 200, 50, 80, 10], dtype=float)
    
    u_after = controller.compute_optimal_controls(0)
    print(f"\n训练后控制:")
    print(f"  u_p = {u_after[0][0][1]}")
    print(f"  u_e = {u_after[1]}")
    
    # 分析
    print(f"\n权重分析:")
    W = critics[0].W
    print(f"  ||W|| = {np.linalg.norm(W):.4f}")
    print(f"  W前6项(平方项) = {W[:6]}")
    
    return critics, controller


def simulate_short(pursuers, evaders, target_graph, controller, critics, dt=0.01, T=2.0):
    """短时间仿真，观察运动趋势"""
    n_steps = int(T / dt)
    
    x_tilde_history = []
    u_p_history = []
    u_e_history = []
    
    for step in range(n_steps):
        x_tilde = controller.compute_state_error(0)
        x_tilde_history.append(x_tilde.copy())
        
        u_pursuers, u_evader, _ = controller.compute_optimal_controls(0)
        
        for j, u_p in u_pursuers:
            pursuers[j].update_state(u_p, dt)
            u_p_history.append(u_p.copy())
        evaders[0].update_state(u_evader, dt)
        u_e_history.append(u_evader.copy())
    
    return np.array(x_tilde_history), np.array(u_p_history), np.array(u_e_history)


def diagnose_simulation():
    """诊断仿真过程"""
    print("\n" + "=" * 60)
    print("Diagnose 4: Simulation Behavior")
    print("=" * 60)
    
    # 简化场景
    pursuers = [Pursuer(0, [600, 0, 200, 50, 0, 0])]  # 追捕者在右边
    pursuers[0].set_target(0, [0, 0, 0, 0, 0, 0])
    evaders = [Evader(0, [0, 0, 200, 50, 0, 0])]  # 逃避者在原点
    
    target_graph = TargetGraph(1, 1)
    target_graph.A_pe = np.array([[1]])
    target_graph.A_ep = np.array([[1]])
    
    critics = [CriticNetwork(Config.STATE_DIM)]
    controller = OptimalController(pursuers, evaders, target_graph, critics)
    
    # 训练
    from simulation import offline_training
    print("训练中...")
    offline_training(pursuers, evaders, target_graph, critics, controller,
                    num_grid_points=10, num_iterations=20)
    
    # 重置状态
    pursuers[0].state = np.array([600, 0, 200, 50, 0, 0], dtype=float)
    pursuers[0].state_history = [pursuers[0].state.copy()]
    evaders[0].state = np.array([0, 0, 200, 50, 0, 0], dtype=float)
    evaders[0].state_history = [evaders[0].state.copy()]
    
    # 短时间仿真
    print("\n仿真2秒...")
    x_tilde_hist, u_p_hist, u_e_hist = simulate_short(
        pursuers, evaders, target_graph, controller, critics, dt=0.01, T=2.0
    )
    
    # 分析
    print("\n状态误差变化:")
    print(f"  初始: {x_tilde_hist[0, :3]}")
    print(f"  最终: {x_tilde_hist[-1, :3]}")
    
    error_norm_init = np.linalg.norm(x_tilde_hist[0])
    error_norm_final = np.linalg.norm(x_tilde_hist[-1])
    print(f"\n误差范数:")
    print(f"  初始: {error_norm_init:.2f}")
    print(f"  最终: {error_norm_final:.2f}")
    
    if error_norm_final < error_norm_init:
        print("  [OK] 误差在减小，追捕有效！")
    else:
        print("  [ERROR] 误差在增大，追捕失效！")
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 轨迹
    ax1 = axes[0, 0]
    p_states = np.array(pursuers[0].state_history)
    e_states = np.array(evaders[0].state_history)
    ax1.plot(p_states[:, 0], p_states[:, 1], 'b-', label='Pursuer')
    ax1.plot(e_states[:, 0], e_states[:, 1], 'r--', label='Evader')
    ax1.scatter([p_states[0, 0]], [p_states[0, 1]], c='blue', marker='o', s=100)
    ax1.scatter([e_states[0, 0]], [e_states[0, 1]], c='red', marker='o', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('2D Trajectory (2s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 误差范数
    ax2 = axes[0, 1]
    error_norm = np.linalg.norm(x_tilde_hist, axis=1)
    ax2.plot(error_norm, 'g-')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('||x_tilde||')
    ax2.set_title('State Error Norm')
    ax2.grid(True, alpha=0.3)
    
    # 追捕者控制
    ax3 = axes[1, 0]
    ax3.plot(u_p_hist[:, 0], 'b-', label='u_p[0]')
    ax3.plot(u_p_hist[:, 1], 'b--', label='u_p[1]')
    ax3.plot(u_p_hist[:, 2], 'b:', label='u_p[2]')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Pursuer Control')
    ax3.set_title('Pursuer Control')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 逃避者控制
    ax4 = axes[1, 1]
    ax4.plot(u_e_hist[:, 0], 'r-', label='u_e[0]')
    ax4.plot(u_e_hist[:, 1], 'r--', label='u_e[1]')
    ax4.plot(u_e_hist[:, 2], 'r:', label='u_e[2]')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Evader Control')
    ax4.set_title('Evader Control')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return x_tilde_hist, u_p_hist, u_e_hist


def main():
    print("=" * 60)
    print("DIAGNOSTIC ANALYSIS OF ORIGINAL CODE")
    print("=" * 60)
    
    # 诊断1: 控制方向
    diagnose_control_direction()
    
    # 诊断2: 动力学
    diagnose_dynamics()
    
    # 诊断3: 训练过程
    diagnose_training()
    
    # 诊断4: 仿真行为
    diagnose_simulation()


if __name__ == "__main__":
    main()

