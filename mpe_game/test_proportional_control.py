"""
使用简单比例控制验证系统动力学

目的：
1. 验证动力学模型正确性
2. 验证状态误差减小的基本机制
3. 为RL控制器提供基准
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from dynamics import AircraftDynamics


def test_proportional_control():
    """使用简单比例控制测试追逃"""
    
    print("=" * 60)
    print("Test: Proportional Control")
    print("=" * 60)
    
    dynamics = AircraftDynamics()
    
    # 控制增益（作用在位置误差上产生加速度）
    K_p_pos = 0.01  # 追捕者位置增益
    K_p_vel = 0.5   # 追捕者速度增益
    K_e_pos = 0.005  # 逃避者位置增益
    K_e_vel = 0.3   # 逃避者速度增益
    
    u_bar_p = 25.0
    u_bar_e = 15.0
    
    # 初始状态
    x_p = np.array([600, 0, 200, 50, 0, 0], dtype=float)
    x_e = np.array([0, 0, 200, 50, 0, 0], dtype=float)
    
    dt = 0.01
    T = 20.0
    n_steps = int(T / dt)
    
    x_p_history = [x_p.copy()]
    x_e_history = [x_e.copy()]
    u_p_history = []
    u_e_history = []
    
    print(f"\nInitial positions: P={x_p[:3]}, E={x_e[:3]}")
    print(f"Initial distance: {np.linalg.norm(x_p[:3] - x_e[:3]):.2f}")
    
    for step in range(n_steps):
        # 状态误差
        pos_error = x_p[:3] - x_e[:3]  # 追捕者相对于逃避者的位置
        vel_error = x_p[3:] - x_e[3:]  # 速度误差
        
        # 追捕者控制：减小位置误差
        # 当 pos_error > 0（追捕者在右边），应该向左加速（负加速度）
        u_p = -K_p_pos * pos_error - K_p_vel * vel_error
        u_p = np.clip(u_p, -u_bar_p, u_bar_p)
        
        # 逃避者控制：增大位置误差（逃离）
        # 当追捕者在右边，逃避者应该向左加速（增大距离）
        u_e = -K_e_pos * pos_error - K_e_vel * vel_error
        u_e = np.clip(u_e, -u_bar_e, u_bar_e)
        
        # RK4更新
        k1_p = dynamics.dynamics(x_p, u_p)
        k2_p = dynamics.dynamics(x_p + 0.5*dt*k1_p, u_p)
        k3_p = dynamics.dynamics(x_p + 0.5*dt*k2_p, u_p)
        k4_p = dynamics.dynamics(x_p + dt*k3_p, u_p)
        x_p = x_p + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        k1_e = dynamics.dynamics(x_e, u_e)
        k2_e = dynamics.dynamics(x_e + 0.5*dt*k1_e, u_e)
        k3_e = dynamics.dynamics(x_e + 0.5*dt*k2_e, u_e)
        k4_e = dynamics.dynamics(x_e + dt*k3_e, u_e)
        x_e = x_e + (dt/6.0) * (k1_e + 2*k2_e + 2*k3_e + k4_e)
        
        x_p_history.append(x_p.copy())
        x_e_history.append(x_e.copy())
        u_p_history.append(u_p.copy())
        u_e_history.append(u_e.copy())
        
        if (step + 1) % (n_steps // 5) == 0:
            dist = np.linalg.norm(x_p[:3] - x_e[:3])
            print(f"  Step {step+1}: distance = {dist:.2f}")
    
    x_p_history = np.array(x_p_history)
    x_e_history = np.array(x_e_history)
    u_p_history = np.array(u_p_history)
    u_e_history = np.array(u_e_history)
    
    final_dist = np.linalg.norm(x_p_history[-1, :3] - x_e_history[-1, :3])
    init_dist = np.linalg.norm(x_p_history[0, :3] - x_e_history[0, :3])
    
    print(f"\nFinal distance: {final_dist:.2f}")
    if final_dist < init_dist:
        print("[OK] Distance reduced - pursuit effective!")
    else:
        print("[WARNING] Distance increased")
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 2D轨迹 X-Y
    ax = axes[0, 0]
    ax.plot(x_p_history[:, 0], x_p_history[:, 1], 'b-', label='Pursuer')
    ax.plot(x_e_history[:, 0], x_e_history[:, 1], 'r--', label='Evader')
    ax.scatter([x_p_history[0, 0]], [x_p_history[0, 1]], c='blue', marker='o', s=100, zorder=5)
    ax.scatter([x_e_history[0, 0]], [x_e_history[0, 1]], c='red', marker='o', s=100, zorder=5)
    ax.scatter([x_p_history[-1, 0]], [x_p_history[-1, 1]], c='blue', marker='s', s=100, zorder=5)
    ax.scatter([x_e_history[-1, 0]], [x_e_history[-1, 1]], c='red', marker='s', s=100, zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Trajectory (X-Y)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 距离
    ax = axes[0, 1]
    t = np.arange(len(x_p_history)) * dt
    dist = np.linalg.norm(x_p_history[:, :3] - x_e_history[:, :3], axis=1)
    ax.plot(t, dist, 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance')
    ax.set_title('Distance Between P and E')
    ax.grid(True, alpha=0.3)
    
    # X位置
    ax = axes[0, 2]
    ax.plot(t, x_p_history[:, 0], 'b-', label='Pursuer X')
    ax.plot(t, x_e_history[:, 0], 'r--', label='Evader X')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position')
    ax.set_title('X Position Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 追捕者控制
    ax = axes[1, 0]
    t_u = np.arange(len(u_p_history)) * dt
    ax.plot(t_u, u_p_history[:, 0], 'b-', label='u_p[0]')
    ax.plot(t_u, u_p_history[:, 1], 'b--', label='u_p[1]')
    ax.plot(t_u, u_p_history[:, 2], 'b:', label='u_p[2]')
    ax.axhline(y=u_bar_p, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=-u_bar_p, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.set_title('Pursuer Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 逃避者控制
    ax = axes[1, 1]
    ax.plot(t_u, u_e_history[:, 0], 'r-', label='u_e[0]')
    ax.plot(t_u, u_e_history[:, 1], 'r--', label='u_e[1]')
    ax.plot(t_u, u_e_history[:, 2], 'r:', label='u_e[2]')
    ax.axhline(y=u_bar_e, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=-u_bar_e, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control')
    ax.set_title('Evader Control')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 速度
    ax = axes[1, 2]
    ax.plot(t, x_p_history[:, 3], 'b-', label='Pursuer vx')
    ax.plot(t, x_e_history[:, 3], 'r--', label='Evader vx')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.set_title('X Velocity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return x_p_history, x_e_history


def test_with_paper_initial_conditions():
    """使用论文的初始条件测试"""
    
    print("\n" + "=" * 60)
    print("Test: Paper Initial Conditions")
    print("=" * 60)
    
    dynamics = AircraftDynamics()
    
    # 论文的初始条件
    x_p = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
    x_e = np.array([500, 1500, 200, 50, 80, 10], dtype=float)
    
    u_bar_p = 25.0
    u_bar_e = 15.0
    
    # 控制增益
    K_pos = 0.005
    K_vel = 0.3
    
    dt = 0.01
    T = 20.0
    n_steps = int(T / dt)
    
    x_p_history = [x_p.copy()]
    x_e_history = [x_e.copy()]
    
    print(f"\nInitial state error: {np.linalg.norm(x_p - x_e):.2f}")
    
    for step in range(n_steps):
        pos_error = x_p[:3] - x_e[:3]
        vel_error = x_p[3:] - x_e[3:]
        
        # 追捕者：追向逃避者
        u_p = -K_pos * pos_error - K_vel * vel_error
        u_p = np.clip(u_p, -u_bar_p, u_bar_p)
        
        # 逃避者：逃离追捕者
        u_e = -K_pos * pos_error * 0.5  # 弱一些的逃离策略
        u_e = np.clip(u_e, -u_bar_e, u_bar_e)
        
        # 更新
        k1_p = dynamics.dynamics(x_p, u_p)
        k2_p = dynamics.dynamics(x_p + 0.5*dt*k1_p, u_p)
        k3_p = dynamics.dynamics(x_p + 0.5*dt*k2_p, u_p)
        k4_p = dynamics.dynamics(x_p + dt*k3_p, u_p)
        x_p = x_p + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        k1_e = dynamics.dynamics(x_e, u_e)
        k2_e = dynamics.dynamics(x_e + 0.5*dt*k1_e, u_e)
        k3_e = dynamics.dynamics(x_e + 0.5*dt*k2_e, u_e)
        k4_e = dynamics.dynamics(x_e + dt*k3_e, u_e)
        x_e = x_e + (dt/6.0) * (k1_e + 2*k2_e + 2*k3_e + k4_e)
        
        x_p_history.append(x_p.copy())
        x_e_history.append(x_e.copy())
        
        if (step + 1) % (n_steps // 5) == 0:
            error = np.linalg.norm(x_p - x_e)
            print(f"  Step {step+1}: error = {error:.2f}")
    
    x_p_history = np.array(x_p_history)
    x_e_history = np.array(x_e_history)
    
    final_error = np.linalg.norm(x_p_history[-1] - x_e_history[-1])
    print(f"\nFinal state error: {final_error:.2f}")
    
    # 绘图
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(x_p_history[:, 0], x_p_history[:, 1], x_p_history[:, 2], 'b-', label='Pursuer')
    ax1.plot(x_e_history[:, 0], x_e_history[:, 1], x_e_history[:, 2], 'r--', label='Evader')
    ax1.scatter([x_p_history[0, 0]], [x_p_history[0, 1]], [x_p_history[0, 2]], c='blue', marker='o', s=100)
    ax1.scatter([x_e_history[0, 0]], [x_e_history[0, 1]], [x_e_history[0, 2]], c='red', marker='o', s=100)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory (Proportional Control)')
    ax1.legend()
    
    ax2 = fig.add_subplot(1, 2, 2)
    t = np.arange(len(x_p_history)) * dt
    error = np.linalg.norm(x_p_history - x_e_history, axis=1)
    ax2.plot(t, error, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('||x_p - x_e||')
    ax2.set_title('State Error Norm')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return x_p_history, x_e_history


if __name__ == "__main__":
    print("Test 1: Simple initial conditions")
    test_proportional_control()
    
    print("\n" + "#" * 60)
    print("Test 2: Paper initial conditions")
    test_with_paper_initial_conditions()

