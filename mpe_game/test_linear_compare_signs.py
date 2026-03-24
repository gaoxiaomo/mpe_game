"""
对比测试：逃避者用负号 vs 正号

验证两种实现方式的差异
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('TkAgg')


class LinearPEGame:
    def __init__(self, u_bar_p=25.0, u_bar_e=15.0):
        self.n = 4
        self.m = 2
        self.A = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], dtype=float)
        self.B = np.array([[0,0],[0,0],[1,0],[0,1]], dtype=float)
        self.Q = np.eye(self.n)
        self.R1 = np.eye(self.m)
        self.R2 = np.eye(self.m)
        self.u_bar_p = u_bar_p
        self.u_bar_e = u_bar_e
        
    def compute_lqr_solution(self):
        P = solve_continuous_are(self.A, self.B, self.Q, self.R1)
        return P


class CriticNetwork:
    def __init__(self, n):
        self.n = n
        self.num_basis = n + n*(n-1)//2
        self.W = np.zeros(self.num_basis)
        
    def activation_gradient(self, x):
        grad = np.zeros((self.n, self.num_basis))
        for i in range(self.n):
            grad[i, i] = 2 * x[i]
        col = self.n
        for i in range(self.n):
            for j in range(i + 1, self.n):
                grad[i, col] = x[j]
                grad[j, col] = x[i]
                col += 1
        return grad
    
    def predict_gradient(self, x):
        return self.activation_gradient(x) @ self.W


def train_with_sign(game, evader_positive_sign, num_iterations=200, num_samples=1000):
    """训练，可选择逃避者符号"""
    
    critic = CriticNetwork(game.n)
    
    A, B = game.A, game.B
    Q, R1, R2 = game.Q, game.R1, game.R2
    u_bar_p, u_bar_e = game.u_bar_p, game.u_bar_e
    R1_inv, R2_inv = np.linalg.inv(R1), np.linalg.inv(R2)
    
    # LQR初始化
    P_init = game.compute_lqr_solution()
    W_init = []
    for i in range(game.n):
        W_init.append(P_init[i, i])
    for i in range(game.n):
        for j in range(i + 1, game.n):
            W_init.append(2 * P_init[i, j])
    critic.W = np.array(W_init)
    
    weight_history = []
    np.random.seed(42)
    
    for iteration in range(num_iterations):
        X_list, y_list = [], []
        
        for _ in range(num_samples):
            x = np.random.uniform(-5, 5, game.n)
            dPsi = critic.activation_gradient(x)
            grad_V = dPsi @ critic.W
            
            rho_p = (1.0 / (2.0 * u_bar_p)) * R1_inv @ B.T @ grad_V
            rho_e = (1.0 / (2.0 * u_bar_e)) * R2_inv @ B.T @ grad_V
            
            u_p = -u_bar_p * np.tanh(rho_p)
            
            if evader_positive_sign:
                u_e = u_bar_e * np.tanh(rho_e)   # 正号
            else:
                u_e = -u_bar_e * np.tanh(rho_e)  # 负号（论文）
            
            u_p = np.clip(u_p, -u_bar_p*0.9999, u_bar_p*0.9999)
            u_e = np.clip(u_e, -u_bar_e*0.9999, u_bar_e*0.9999)
            
            x_dot = A @ x + B @ u_p - B @ u_e
            Z = dPsi.T @ x_dot
            
            P_cost = x.T @ Q @ x
            
            def nonquad(u, u_bar, R):
                cost = 0.0
                for k in range(len(u)):
                    ratio = np.clip(u[k]/u_bar, -0.9999, 0.9999)
                    cost += 2.0*u_bar*R[k,k]*(u[k]*np.arctanh(ratio)+0.5*u_bar*np.log(1.0-ratio**2))
                return cost
            
            cost_p = nonquad(u_p, u_bar_p, R1)
            cost_e = nonquad(u_e, u_bar_e, R2)
            y_val = -(P_cost + cost_p - cost_e)
            
            X_list.append(Z)
            y_list.append(y_val)
        
        X, y = np.array(X_list), np.array(y_list)
        W_new = np.linalg.solve(X.T @ X + 0.1*np.eye(critic.num_basis), X.T @ y)
        
        # 动量平滑
        critic.W = 0.2*critic.W + 0.8*W_new
        weight_history.append(np.linalg.norm(critic.W))
    
    # 恢复P矩阵
    P = np.zeros((game.n, game.n))
    idx = 0
    for i in range(game.n):
        P[i, i] = critic.W[idx]
        idx += 1
    for i in range(game.n):
        for j in range(i + 1, game.n):
            P[i, j] = critic.W[idx] / 2
            P[j, i] = critic.W[idx] / 2
            idx += 1
    
    return critic, weight_history, P


def simulate(game, critic, evader_positive_sign, x0_p, x0_e, T=10.0, dt=0.01):
    """仿真"""
    R1_inv = np.linalg.inv(game.R1)
    R2_inv = np.linalg.inv(game.R2)
    
    x_p, x_e = x0_p.copy(), x0_e.copy()
    x_p_hist, x_e_hist = [x_p.copy()], [x_e.copy()]
    
    for _ in range(int(T/dt)):
        x_tilde = x_p - x_e
        grad_V = critic.predict_gradient(x_tilde)
        
        rho_p = (1.0/(2.0*game.u_bar_p)) * R1_inv @ game.B.T @ grad_V
        rho_e = (1.0/(2.0*game.u_bar_e)) * R2_inv @ game.B.T @ grad_V
        
        u_p = -game.u_bar_p * np.tanh(rho_p)
        if evader_positive_sign:
            u_e = game.u_bar_e * np.tanh(rho_e)
        else:
            u_e = -game.u_bar_e * np.tanh(rho_e)
        
        def update(x, u):
            k1 = game.A @ x + game.B @ u
            k2 = game.A @ (x + 0.5*dt*k1) + game.B @ u
            k3 = game.A @ (x + 0.5*dt*k2) + game.B @ u
            k4 = game.A @ (x + dt*k3) + game.B @ u
            return x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_p = update(x_p, u_p)
        x_e = update(x_e, u_e)
        x_p_hist.append(x_p.copy())
        x_e_hist.append(x_e.copy())
    
    return np.array(x_p_hist), np.array(x_e_hist)


def main():
    print("=" * 70)
    print("COMPARISON: Evader Negative Sign (Paper) vs Positive Sign")
    print("=" * 70)
    
    game = LinearPEGame(u_bar_p=25.0, u_bar_e=15.0)
    x0_p = np.array([5.0, 3.0, 0.0, 0.0])
    x0_e = np.array([0.0, 0.0, 0.0, 0.0])
    
    results = {}
    
    for name, positive_sign in [("Negative (Paper)", False), ("Positive", True)]:
        print(f"\n{'='*60}")
        print(f"Case: Evader {name}")
        print(f"{'='*60}")
        
        critic, weight_history, P = train_with_sign(game, positive_sign, 
                                                     num_iterations=200, num_samples=1000)
        
        eigvals = np.linalg.eigvalsh(P)
        print(f"P eigenvalues: {eigvals}")
        print(f"P positive definite: {np.all(eigvals > 0)}")
        
        x_p_hist, x_e_hist = simulate(game, critic, positive_sign, x0_p, x0_e)
        
        init_dist = np.linalg.norm(x0_p[:2] - x0_e[:2])
        final_dist = np.linalg.norm(x_p_hist[-1, :2] - x_e_hist[-1, :2])
        
        print(f"Initial distance: {init_dist:.2f}")
        print(f"Final distance: {final_dist:.2f}")
        
        if final_dist < 0.5:
            status = "SUCCESS"
        elif final_dist > init_dist * 2:
            status = "UNSTABLE"
        else:
            status = "PARTIAL"
        print(f"Result: {status}")
        
        results[name] = {
            'weight_history': weight_history,
            'x_p_hist': x_p_hist,
            'x_e_hist': x_e_hist,
            'P': P,
            'eigvals': eigvals,
            'final_dist': final_dist,
            'status': status
        }
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, (name, positive_sign) in enumerate([("Negative (Paper)", False), ("Positive", True)]):
        res = results[name]
        
        # 轨迹
        ax = axes[idx, 0]
        ax.plot(res['x_p_hist'][:, 0], res['x_p_hist'][:, 1], 'b-', label='Pursuer')
        ax.plot(res['x_e_hist'][:, 0], res['x_e_hist'][:, 1], 'r--', label='Evader')
        ax.scatter([res['x_p_hist'][0,0]], [res['x_p_hist'][0,1]], c='blue', marker='o', s=100)
        ax.scatter([res['x_e_hist'][0,0]], [res['x_e_hist'][0,1]], c='red', marker='o', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{name}\nTrajectory - {res["status"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 距离
        ax = axes[idx, 1]
        t = np.arange(len(res['x_p_hist'])) * 0.01
        dist = np.linalg.norm(res['x_p_hist'][:,:2] - res['x_e_hist'][:,:2], axis=1)
        ax.plot(t, dist, 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance')
        ax.set_title(f'Distance (Final: {res["final_dist"]:.2f})')
        ax.grid(True, alpha=0.3)
        
        # 权重收敛
        ax = axes[idx, 2]
        ax.plot(res['weight_history'], 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||W||')
        ax.set_title(f'Weight Convergence\neigvals: [{res["eigvals"][0]:.2f}, {res["eigvals"][-1]:.2f}]')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sign_comparison.png', dpi=150)
    plt.show()
    
    # 总结
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'P Positive Def':<15} {'Final Dist':<12} {'Status'}")
    print("-" * 70)
    for name in results:
        res = results[name]
        pd = "YES" if np.all(res['eigvals'] > 0) else "NO"
        print(f"{name:<25} {pd:<15} {res['final_dist']:<12.2f} {res['status']}")
    
    return results


if __name__ == "__main__":
    results = main()

