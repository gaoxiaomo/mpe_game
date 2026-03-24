"""测试场景1：三追捕者-单逃避者"""
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_scenario1

if __name__ == "__main__":
    # 运行场景1
    results = run_scenario1()
    
    pursuers = results['pursuers']
    evaders = results['evaders']
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    colors_p = ['blue', 'green', 'orange']
    for j, p in enumerate(pursuers):
        traj = np.array(p.state_history)
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=colors_p[j], label=f'Pursuer {j+1}')
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                   color=colors_p[j], marker='o', s=100)
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                   color=colors_p[j], marker='x', s=100)
    
    e_traj = np.array(evaders[0].state_history)
    ax1.plot(e_traj[:, 0], e_traj[:, 1], e_traj[:, 2], 
            color='red', label='Evader', linewidth=2)
    ax1.scatter(e_traj[0, 0], e_traj[0, 1], e_traj[0, 2], 
               color='red', marker='o', s=100)
    ax1.scatter(e_traj[-1, 0], e_traj[-1, 1], e_traj[-1, 2], 
               color='red', marker='x', s=100)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectories')
    ax1.legend()
    
    # 2. 团队误差曲线
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(results['team_errors'], 'b-', linewidth=1)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Team Error')
    ax2.set_title('Team State Error')
    ax2.grid(True)
    
    # 3. 权重变化曲线
    ax3 = fig.add_subplot(2, 2, 3)
    if 'weight_history' in results:
        ax3.plot(results['weight_history'], 'g-o', linewidth=1, markersize=4)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Weight Change')
        ax3.set_title('Training Convergence')
        ax3.grid(True)
        ax3.set_yscale('log')
    
    # 4. 状态误差曲线
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(results['state_errors'], 'r-', linewidth=1)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('||x_tilde||')
    ax4.set_title('State Error Norm')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('scenario1_results.png', dpi=150)
    plt.show()
    
    # 打印最终状态
    print("\n" + "="*50)
    print("Final States:")
    print("="*50)
    for j, p in enumerate(pursuers):
        print(f"Pursuer {j+1}: pos=({p.state[0]:.1f}, {p.state[1]:.1f}, {p.state[2]:.1f})")
    print(f"Evader:    pos=({evaders[0].state[0]:.1f}, {evaders[0].state[1]:.1f}, {evaders[0].state[2]:.1f})")
    
    # 打印最终权重
    print("\n" + "="*50)
    print("Final Critic Weights:")
    print("="*50)
    print(f"W = {results['critics'][0].W}")
    print(f"||W|| = {np.linalg.norm(results['critics'][0].W):.6f}")
