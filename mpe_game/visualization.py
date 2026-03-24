import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter:
    """可视化工具类"""

    @staticmethod
    def plot_trajectory_3d(pursuers, evaders, title="MPE Game Trajectory"):
        """绘制3D轨迹 - 对应论文 Fig. 1 和 Fig. 6"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors_p = ['blue', 'cyan', 'navy']
        colors_e = ['red', 'orange', 'darkred']

        # 绘制追捕者轨迹
        for i, p in enumerate(pursuers):
            states = np.array(p.state_history)
            color = colors_p[i % len(colors_p)]
            ax.plot(states[:, 0], states[:, 1], states[:, 2],
                    color=color, linewidth=1.5, label=f'Pursuer {i + 1}')
            ax.scatter(states[0, 0], states[0, 1], states[0, 2],
                       c=color, marker='o', s=100, edgecolors='black')
            ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2],
                       c=color, marker='s', s=100, edgecolors='black')

        # 绘制逃避者轨迹
        for i, e in enumerate(evaders):
            states = np.array(e.state_history)
            color = colors_e[i % len(colors_e)]
            ax.plot(states[:, 0], states[:, 1], states[:, 2],
                    color=color, linewidth=1.5, linestyle='--', label=f'Evader {i + 1}')
            ax.scatter(states[0, 0], states[0, 1], states[0, 2],
                       c=color, marker='o', s=100, edgecolors='black')
            ax.scatter(states[-1, 0], states[-1, 1], states[-1, 2],
                       c=color, marker='s', s=100, edgecolors='black')

        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Height (m)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper left')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_state_errors(state_errors, dt, title="State Errors"):
        """绘制状态误差 - 对应论文 Fig. 2 和 Fig. 7"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(state_errors, dict):
            for i, errors in state_errors.items():
                t = np.arange(len(errors)) * dt
                ax.plot(t, errors, linewidth=1.5, label=f'Evader {i + 1}')
        else:
            t = np.arange(len(state_errors)) * dt
            ax.plot(t, state_errors, linewidth=1.5, label='State Error')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('State Error Norm', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_control_inputs(pursuers, evaders, dt, title="Control Inputs"):
        """绘制控制输入 - 对应论文 Fig. 3"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 追捕者控制输入
        ax1 = axes[0]
        for i, p in enumerate(pursuers):
            if len(p.control_history) > 0:
                controls = np.array(p.control_history)
                t = np.arange(len(controls)) * dt
                for j in range(controls.shape[1]):
                    ax1.plot(t, controls[:, j], linewidth=1,
                             label=f'P{i + 1}-u{j + 1}', alpha=0.8)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Pursuer Control', fontsize=12)
        ax1.set_title('Pursuer Control Inputs', fontsize=12)
        ax1.legend(loc='upper right', ncol=3, fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 逃避者控制输入
        ax2 = axes[1]
        for i, e in enumerate(evaders):
            if len(e.control_history) > 0:
                controls = np.array(e.control_history)
                t = np.arange(len(controls)) * dt
                for j in range(controls.shape[1]):
                    ax2.plot(t, controls[:, j], linewidth=1,
                             label=f'E{i + 1}-u{j + 1}', alpha=0.8)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Evader Control', fontsize=12)
        ax2.set_title('Evader Control Inputs', fontsize=12)
        ax2.legend(loc='upper right', ncol=3, fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_weight_convergence(critics, dt, title="Weight Convergence"):
        """绘制权重收敛曲线 - 对应论文 Fig. 4 和 Fig. 11"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, critic in enumerate(critics):
            weight_norms = critic.weight_history
            t = np.arange(len(weight_norms)) * dt
            ax.plot(t, weight_norms, linewidth=1.5, label=f'Critic {i + 1}')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(r'$||\hat{W}_{i,s}||^2$', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_team_cohesion(team_errors, dt, title="Team State Error"):
        """绘制团队状态误差 - 对应论文 Fig. 10"""
        fig, ax = plt.subplots(figsize=(10, 6))

        t = np.arange(len(team_errors)) * dt
        ax.plot(t, team_errors, 'b-', linewidth=1.5)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Team State Error', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_saturation_comparison(results, dt):
        """绘制饱和约束对比 - 对应论文 Fig. 5"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        case_names = list(results.keys())
        titles = [
            r'(a) $\bar{u}_p > \bar{u}_e$',
            r'(b) $\bar{u}_p = \bar{u}_e$',
            r'(c) $\bar{u}_p < \bar{u}_e$'
        ]

        for idx, (case_name, ax, title) in enumerate(zip(case_names, axes, titles)):
            heights = results[case_name]['heights']
            t = np.arange(len(heights)) * dt

            # 绘制追捕者高度
            for i in range(3):
                ax.plot(t, heights[:, i], linewidth=1.5, label=f'P{i + 1}')

            # 绘制逃避者高度
            ax.plot(t, heights[:, 3], 'r--', linewidth=1.5, label='E')

            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Height (m)', fontsize=12)
            ax.set_title(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig