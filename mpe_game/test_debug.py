# -*- coding: utf-8 -*-
"""调试脚本 - 检查控制方向"""

import numpy as np
np.random.seed(42)

from config import Config
from agents import Pursuer, Evader
from graph import TargetGraph
from networks import CriticNetwork
from controller import OptimalController

print("=" * 60)
print("Debug: Checking Control Direction")
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

evaders = [Evader(0, [500, 1500, 200, 50, 80, 10])]

target_graph = TargetGraph(len(pursuers), len(evaders))
target_graph.A_pe = np.array([[1], [1], [1]])
target_graph.A_ep = np.array([[1, 1, 1]])

critics = [CriticNetwork(Config.STATE_DIM) for _ in evaders]
controller = OptimalController(pursuers, evaders, target_graph, critics)

# 计算初始状态误差
x_tilde = controller.compute_state_error(0)
print(f"\nState error: {x_tilde}")
print(f"State error norm: {np.linalg.norm(x_tilde):.2f}")

# 获取梯度
grad_V = critics[0].predict_gradient(x_tilde)
print(f"\nGradient of V: {grad_V}")

# 计算最优控制
u_pursuers, u_evader, _, _ = controller.compute_optimal_controls(0)

print("\nPursuer controls:")
for j, u_p in u_pursuers:
    print(f"  P{j}: {u_p}")

print(f"\nEvader control: {u_evader}")

# 检查控制方向：
# 如果追捕者控制正确，应该使状态误差减小
# 检查 u_p 与 x_tilde 的点积（应该为负，表示减小误差）

print("\n" + "="*60)
print("Checking if controls reduce state error:")
print("="*60)

# 简化分析：假设系统是位置控制
# 状态误差 x_tilde = x_p - x_e + r
# 追捕者想减小 |x_tilde|，所以控制应该使 x_p 向 x_e 移动

# 位置部分
x_pos_error = x_tilde[:3]
print(f"\nPosition error: {x_pos_error}")

# 理想的追捕者控制方向应该与位置误差相反（减小误差）
for j, u_p in u_pursuers:
    # u_p 作用在速度上 (g矩阵的后3行是控制输入)
    # 如果要减小位置误差，速度应该指向误差的反方向
    # 但控制输入影响的是加速度...
    dot_product = np.dot(u_p, x_pos_error[:3] if len(x_pos_error) >= 3 else u_p)
    print(f"  P{j} control: {u_p}")
    print(f"  P{j} dot with pos_error: {dot_product:.2f} (negative is good for pursuer)")

# 测试一下简单的比例控制
print("\n" + "="*60)
print("Compare with simple proportional control:")
print("="*60)

# 简单比例控制：u = -K * x_tilde[:3]
K = 0.1
u_simple = -K * x_tilde[:3]
u_simple = np.clip(u_simple, -25, 25)
print(f"Simple P-control: {u_simple}")
print(f"(This should reduce position error)")

# 结论
print("\n" + "="*60)
print("Issue: The learned gradient direction might be wrong!")
print("The gradient should point TOWARDS increasing V (state error)")
print("So pursuing agent should use -grad_V direction")
print("="*60)

