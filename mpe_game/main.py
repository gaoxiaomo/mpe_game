import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 导入所有模块
from config import Config
from simulation import run_scenario1, run_scenario2, run_saturation_comparison
from visualization import Plotter


def main():
    print("\n" + "=" * 70)
    print("Multi-Agent Pursuit-Evasion Game")
    print("Approximate Optimal Strategy using Reinforcement Learning")
    print("Paper: IEEE Systems Journal, Vol. 18, No. 3, September 2024")
    print("=" * 70 + "\n")

    np.random.seed(42)

    #场景1
    print("\n" + "-" * 50)
    print("Running Scenario 1: Multiple-Pursuer-One-Evader")
    print("-" * 50)

    results1 = run_scenario1()

    # 可视化场景1
    fig1 = Plotter.plot_trajectory_3d(
        results1['pursuers'],
        results1['evaders'],
        "Scenario 1: Three-Pursuer-One-Evader Trajectory (Fig. 1)"
    )

    fig2 = Plotter.plot_state_errors(
        results1['state_errors'],
        Config.DT,
        "Scenario 1: State Error (Fig. 2)"
    )

    fig3 = Plotter.plot_control_inputs(
        results1['pursuers'],
        results1['evaders'],
        Config.DT,
        "Scenario 1: Control Inputs (Fig. 3)"
    )

    fig4 = Plotter.plot_weight_convergence(
        results1['critics'],
        Config.DT,
        "Scenario 1: Weight Convergence (Fig. 4)"
    )

    #场景2
    print("\n" + "-" * 50)
    print("Running Scenario 2: Multiple-Pursuer-Multiple-Evader")
    print("-" * 50)

    results2 = run_scenario2()

    # 可视化场景2
    fig5 = Plotter.plot_trajectory_3d(
        results2['pursuers'],
        results2['evaders'],
        "Scenario 2: Three-Pursuer-Three-Evader Trajectory (Fig. 6)"
    )

    fig6 = Plotter.plot_state_errors(
        results2['state_errors'],
        Config.DT,
        "Scenario 2: State Errors (Fig. 7)"
    )

    fig7 = Plotter.plot_team_cohesion(
        results2['team_errors'],
        Config.DT,
        "Scenario 2: Team State Error (Fig. 10)"
    )

    fig8 = Plotter.plot_weight_convergence(
        results2['critics'],
        Config.DT,
        "Scenario 2: Weight Convergence (Fig. 11)"
    )

    #饱和约束对比
    print("\n" + "-" * 50)
    print("Running Saturation Constraint Comparison")
    print("-" * 50)

    sat_results = run_saturation_comparison()
    fig9 = Plotter.plot_saturation_comparison(sat_results, Config.DT)

    # 显示所有图形
    plt.show()

    print("\n" + "=" * 70)
    print("All Simulations Completed Successfully!")
    print("=" * 70)

    return results1, results2, sat_results


if __name__ == "__main__":
    results1, results2, sat_results = main()