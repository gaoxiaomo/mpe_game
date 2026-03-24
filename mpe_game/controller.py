"""
最优控制器模块

实现论文核心公式:
- 公式(40): 最优控制 u = -ū*tanh((1/2ū)*R^{-1}*g^T*∇ψ^T*W)
- 公式(41): HJI方程
"""
import numpy as np


class OptimalController:
    """最优控制求解器"""

    def __init__(self, pursuers, evaders, target_graph, critic_networks):
        from config import Config

        self.pursuers = pursuers
        self.evaders = evaders
        self.target_graph = target_graph
        self.critics = critic_networks

        # 权重矩阵
        self.Q = Config.Q
        self.R1 = Config.R1
        self.R2 = Config.R2
        self.R1_inv = np.linalg.inv(self.R1)
        self.R2_inv = np.linalg.inv(self.R2)

        # 饱和限制
        self.u_bar_p = Config.U_BAR_P
        self.u_bar_e = Config.U_BAR_E

        # R̄ 对角元素
        self.R1_bar = np.diag(self.R1)
        self.R2_bar = np.diag(self.R2)

    def compute_state_error(self, evader_idx):
        """计算状态误差 - 公式(3)
        
        x̃_i^{pe} = Σ_{j=1}^{N_p} c_{j,i}(x_j^p - x_i^e + r_{j,i}^{pe})
        """
        evader = self.evaders[evader_idx]
        x_tilde = np.zeros(6)

        for j, pursuer in enumerate(self.pursuers):
            c_ji = self.target_graph.A_pe[j, evader_idx]
            if c_ji > 0:
                diff = pursuer.state - evader.state + pursuer.expected_displacement
                x_tilde += c_ji * diff

        return x_tilde

    def compute_optimal_controls(self, evader_idx):
        """计算最优控制 - 公式(40)
        
        u_j^{p*} = -ū_p * tanh((1/(2ū_p)) * R_1^{-1} * g_j^{pT} * ∇V)
        u_i^{e*} = +ū_e * tanh((1/(2ū_e)) * R_2^{-1} * g_i^{eT} * ∇V)
        
        其中 ∇V = ∇ψ^T(x) @ W
        
        注意：逃避者取正号（对抗博弈）
        """
        evader = self.evaders[evader_idx]
        critic = self.critics[evader_idx]
        W = critic.W

        # 计算状态误差
        x_tilde = self.compute_state_error(evader_idx)

        # 计算基函数梯度 ∇ψ(x) ∈ R^{6×21}
        grad_psi = critic.activation_gradient(x_tilde)
        
        # 值函数梯度 ∇V = ∇ψ @ W ∈ R^6
        grad_V = grad_psi @ W

        # 追捕者控制
        u_pursuers = []
        for j, pursuer in enumerate(self.pursuers):
            c_ji = self.target_graph.A_pe[j, evader_idx]
            if c_ji > 0:
                # g_j^p(x) 控制增益矩阵 (6, 3)
                g_p = pursuer.dynamics.g(pursuer.state)
                
                # ρ_p = (1/(2ū_p)) * R_1^{-1} * g^T * ∇V
                rho_p = (1.0 / (2.0 * self.u_bar_p)) * self.R1_inv @ (g_p.T @ grad_V)
                
                # u_p = -ū_p * tanh(ρ_p)
                u_p = -self.u_bar_p * np.tanh(rho_p)
                u_pursuers.append((j, u_p))

        # 逃避者控制 - 公式(40)下半部分
        # 论文原版：u_i^{e*} = -ū_e * tanh(ρ_e)
        # 注意：状态误差动力学中是 g_p u_p - g_e u_e，负号在那里体现对抗
        g_e = evader.dynamics.g(evader.state)
        rho_e = (1.0 / (2.0 * self.u_bar_e)) * self.R2_inv @ (g_e.T @ grad_V)
        u_evader = -self.u_bar_e * np.tanh(rho_e)  # 论文原版：负号

        return u_pursuers, u_evader, x_tilde

    def compute_hamiltonian(self, evader_idx, x_tilde=None):
        """计算哈密顿函数 - 公式(41)
        
        H_i^* = P_i(x) + U_N - W_N + ∇V^T @ Σ c_{j,i} F_{j,i}
        
        当达到最优时 H_i^* = 0
        """
        if x_tilde is None:
            x_tilde = self.compute_state_error(evader_idx)
            
        critic = self.critics[evader_idx]
        W = critic.W
        grad_psi = critic.activation_gradient(x_tilde)
        grad_V = grad_psi @ W
        
        evader = self.evaders[evader_idx]
        d_ep = self.target_graph.get_evader_indegree()[evader_idx]

        # P_i(x) = x̃^T Q x̃
        P_i = x_tilde.T @ self.Q @ x_tilde

        # 非二次项 U_N
        U_N = 0.0
        for j, pursuer in enumerate(self.pursuers):
            c_ji = self.target_graph.A_pe[j, evader_idx]
            if c_ji > 0:
                g_p = pursuer.dynamics.g(pursuer.state)
                rho_p = (1.0 / (2.0 * self.u_bar_p)) * self.R1_inv @ (g_p.T @ grad_V)
                tanh_sq = np.tanh(rho_p) ** 2
                log_term = np.log(np.maximum(1.0 - tanh_sq, 1e-10))
                U_N += c_ji * (self.u_bar_p ** 2) * np.dot(self.R1_bar, log_term)

        # 非二次项 W_N
        g_e = evader.dynamics.g(evader.state)
        rho_e = (1.0 / (2.0 * self.u_bar_e)) * self.R2_inv @ (g_e.T @ grad_V)
        tanh_sq_e = np.tanh(rho_e) ** 2
        log_term_e = np.log(np.maximum(1.0 - tanh_sq_e, 1e-10))
        W_N = -d_ep * (self.u_bar_e ** 2) * np.dot(self.R2_bar, log_term_e)

        # 漂移项: ∇V^T @ Σ c_{j,i} F_{j,i}
        F_sum = np.zeros(6)
        for j, pursuer in enumerate(self.pursuers):
            c_ji = self.target_graph.A_pe[j, evader_idx]
            if c_ji > 0:
                f_p = pursuer.dynamics.f(pursuer.state)
                f_e = evader.dynamics.f(evader.state)
                F_sum += c_ji * (f_p - f_e)
        drift_term = np.dot(grad_V, F_sum)

        H = P_i + U_N + W_N + drift_term
        return H
