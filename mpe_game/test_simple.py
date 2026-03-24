# -*- coding: utf-8 -*-
"""测试简单比例控制 - 验证系统基本工作"""

import numpy as np
np.random.seed(42)

from config import Config
from agents import Pursuer, Evader
from graph import TargetGraph

print("=" * 60)
print("Test: Simple Proportional Control")
print("=" * 60)

# 初始化
pursuers = [
    Pursuer(0, [600, -1900, 300, 78, 2, 18]),
    Pursuer(1, [1200, 500, 2200, 2, 20, 15]),
    Pursuer(2, [200, 600, 800, 5, 10, 125])
]

pursuers[0].set_target(0, [50, 10, 0, 0, 0, 0])
pursuers[1].set_target(0, [10, 50, 0, 0, 0, 0])
pursuers[2].set_target(0, [-10, 0, -50, 0, 0, 0])

evader = Evader(0, [500, 1500, 200, 50, 80, 10])

target_graph = TargetGraph(3, 1)
target_graph.A_pe = np.array([[1], [1], [1]])
target_graph.A_ep = np.array([[1, 1, 1]])

dt = 0.01
T = 20.0
n_steps = int(T / dt)

errors = []
K_p = 0.005  # 比例增益

print(f"\nRunning simulation with P-control (K={K_p})...")

for step in range(n_steps):
    # 计算状态误差
    x_tilde = np.zeros(6)
    for j, p in enumerate(pursuers):
        diff = p.state - evader.state + p.expected_displacement
        x_tilde += diff  # 所有追捕者权重为1
    
    errors.append(np.linalg.norm(x_tilde))
    
    # 简单比例控制：控制加速度来减小位置误差
    # 位置误差在x_tilde[:3]，我们要让追捕者的速度指向减小位置误差的方向
    pos_error = x_tilde[:3]  # x_p - x_e + r
    
    # 追捕者控制：减小位置误差
    # 目标是让 d(x_p)/dt 趋向于 -pos_error 方向
    # 由于控制输入影响加速度，我们用 u = -K * pos_error
    for j, p in enumerate(pursuers):
        u_p = -K_p * pos_error
        u_p = np.clip(u_p, -p.u_bar, p.u_bar)
        p.update_state(u_p, dt)
    
    # 逃避者控制：增大位置误差
    u_e = K_p * pos_error * 0.5  # 逃避者弱一些
    u_e = np.clip(u_e, -evader.u_bar, evader.u_bar)
    evader.update_state(u_e, dt)
    
    if (step + 1) % 200 == 0:
        print(f"  Step {step + 1}/{n_steps}, Error: {errors[-1]:.2f}")

print(f"\nInitial error: {errors[0]:.2f}")
print(f"Final error: {errors[-1]:.2f}")

if errors[-1] < errors[0]:
    print("[OK] Error decreased - basic control works!")
else:
    print("[WARNING] Error increased - check control direction")

# 绘图
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
t = np.arange(len(errors)) * dt
ax.plot(t, errors)
ax.set_xlabel('Time (s)')
ax.set_ylabel('State Error')
ax.set_title('Simple P-Control Test')
ax.grid(True)
plt.show()

