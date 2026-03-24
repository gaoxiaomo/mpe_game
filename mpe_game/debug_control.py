"""调试控制计算 - 包含训练"""
import numpy as np
from config import Config
from agents import Pursuer, Evader
from graph import TargetGraph
from networks import CriticNetwork
from controller import OptimalController
from simulation import offline_training

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

critics = [CriticNetwork(Config.STATE_DIM)]
controller = OptimalController(pursuers, evaders, target_graph, critics)

print("="*50)
print("Before Training")
print("="*50)

# 检查控制（训练前）
x_tilde = controller.compute_state_error(0)
print(f"State error: {x_tilde[:3]}")
u_pursuers, u_evader, _ = controller.compute_optimal_controls(0)
print(f"RL control (before): P0={u_pursuers[0][1]}")

# 离线训练
print("\n" + "="*50)
print("Training...")
print("="*50)
offline_training(pursuers, evaders, target_graph, critics, controller,
                num_grid_points=13, num_iterations=10)

print("\n" + "="*50)
print("After Training")
print("="*50)

# 重置状态
pursuers[0].state = np.array([600, -1900, 300, 78, 2, 18], dtype=float)
pursuers[1].state = np.array([1200, 500, 2200, 2, 20, 15], dtype=float)
pursuers[2].state = np.array([200, 600, 800, 5, 10, 125], dtype=float)
evaders[0].state = np.array([500, 1500, 200, 50, 80, 10], dtype=float)

# 检查控制（训练后）
x_tilde = controller.compute_state_error(0)
grad_psi = critics[0].activation_gradient(x_tilde)
W = critics[0].W
grad_V = grad_psi @ W

print(f"\nState error: {x_tilde[:3]}")
print(f"||W|| = {np.linalg.norm(W):.4f}")
print(f"||grad_V|| = {np.linalg.norm(grad_V):.4f}")
print(f"grad_V = {grad_V}")

u_pursuers, u_evader, _ = controller.compute_optimal_controls(0)
print(f"\nRL control (after training):")
for j, u_p in u_pursuers:
    print(f"  P{j}: {u_p}")
print(f"Evader: {u_evader}")

# 比较
K_p = 0.01
u_simple = -K_p * x_tilde[:3]
print(f"\nSimple proportional control: {u_simple}")

# 检查控制是否合理
# 状态误差 x_tilde[0]=550 意味着追捕者比逃避者x坐标大550
# 要减小误差，追捕者应该减速（如果是位置误差驱动速度变化）
# 控制u作用在速度上，所以u应该使速度变化能减小位置误差
print("\n" + "="*50)
print("Sanity Check:")
print("="*50)
print(f"x_tilde (pos): {x_tilde[:3]}")
print(f"To reduce error, we need:")
print(f"  If x_tilde[0] > 0: pursuer should move in -x direction (relative to evader)")
print(f"  The control affects acceleration, so sign depends on g matrix")
