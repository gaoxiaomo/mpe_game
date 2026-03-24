import numpy as np


class TargetGraph:
    """动态目标图

    实现 Algorithm 1: Dynamic Target Graph Algorithm

    图论基础 (Section II-A):
    - G_p = (V_p, E_p): 追捕者通信拓扑
    - G_e = (V_e, E_e): 逃避者通信拓扑
    - G = (V, E): 追捕者-逃避者目标拓扑

    邻接矩阵:
    - A_pe = [c_{m,n}]: 追捕者m是否追踪逃避者n
    - A_ep = [e_{n,m}]: 逃避者n是否被追捕者m追踪
    """

    def __init__(self, n_pursuers, n_evaders, swap_threshold=None):
        from config import Config

        self.n_pursuers = n_pursuers  # N_p
        self.n_evaders = n_evaders  # N_e

        if swap_threshold is None:
            swap_threshold = Config.SWAP_THRESHOLD
        self.swap_threshold = swap_threshold  # ρ

        # 追捕者-逃避者邻接矩阵 A_pe = [c_{j,i}] ∈ R^{N_p × N_e}
        # c_{j,i} = 1 如果追捕者j追踪逃避者i
        self.A_pe = np.zeros((n_pursuers, n_evaders))

        # 逃避者-追捕者邻接矩阵 A_ep = [e_{i,j}] ∈ R^{N_e × N_p}
        # e_{i,j} = 1 如果逃避者i被追捕者j追踪
        self.A_ep = np.zeros((n_evaders, n_pursuers))

        # 初始化目标分配
        self._initialize_assignment()

    def _initialize_assignment(self):
        """初始化目标分配 - 每个追捕者追踪一个逃避者"""
        for j in range(self.n_pursuers):
            i = j % self.n_evaders
            self.A_pe[j, i] = 1.0
            self.A_ep[i, j] = 1.0

    def compute_distance(self, pursuer, evader, nu=None):
        """计算加权距离

        对应公式 (7) 中的 γ_{j,i}:
        γ_{j,i} = ||ν ⊙ (x_j^p - x_i^e + r_{j,i}^{pe})||

        其中 ⊙ 表示Hadamard积 (逐元素乘法)
        """
        if nu is None:
            nu = np.ones(6)

        diff = pursuer.state - evader.state + pursuer.expected_displacement
        return np.linalg.norm(nu * diff)

    def compute_all_distances(self, pursuers, evaders, nu=None):
        """计算所有追捕者和逃避者之间的加权距离矩阵"""
        distances = np.zeros((self.n_pursuers, self.n_evaders))

        for j, pursuer in enumerate(pursuers):
            for i, evader in enumerate(evaders):
                distances[j, i] = self.compute_distance(pursuer, evader, nu)

        return distances

    def update_graph(self, pursuers, evaders, nu=None):
        """更新目标图 - Algorithm 1

        算法流程:
        1. 计算所有追捕者和逃避者之间的距离 γ_{j,i}
        2. 对于每对追捕者(j, j')和它们的目标(i, i'):
           如果 (γ_{j,i} + γ_{j',i'}) - (γ_{j',i} + γ_{j,i'}) > ρ
           则交换边权重
        3. 更新邻接矩阵 A_pe 和 A_ep

        Returns:
            A_pe, A_ep: 更新后的邻接矩阵
        """
        distances = self.compute_all_distances(pursuers, evaders, nu)

        # 迭代直到没有交换发生
        swapped = True
        while swapped:
            swapped = False

            for j in range(self.n_pursuers):
                for j_prime in range(j + 1, self.n_pursuers):
                    # 找到当前目标
                    targets_j = np.where(self.A_pe[j, :] > 0)[0]
                    targets_j_prime = np.where(self.A_pe[j_prime, :] > 0)[0]

                    if len(targets_j) == 0 or len(targets_j_prime) == 0:
                        continue

                    i = targets_j[0]
                    i_prime = targets_j_prime[0]

                    if i == i_prime:
                        continue

                    # 检查交换条件: (γ_{j,i} + γ_{j',i'}) - (γ_{j',i} + γ_{j,i'}) > ρ
                    current_cost = distances[j, i] + distances[j_prime, i_prime]
                    swap_cost = distances[j_prime, i] + distances[j, i_prime]

                    if current_cost - swap_cost > self.swap_threshold:
                        # 执行交换
                        # 更新 A_pe
                        self.A_pe[j, i] = 0
                        self.A_pe[j, i_prime] = 1
                        self.A_pe[j_prime, i_prime] = 0
                        self.A_pe[j_prime, i] = 1

                        # 更新 A_ep
                        self.A_ep[i, j] = 0
                        self.A_ep[i, j_prime] = 1
                        self.A_ep[i_prime, j_prime] = 0
                        self.A_ep[i_prime, j] = 1

                        # 更新追捕者的目标
                        pursuers[j].set_target(i_prime, pursuers[j].expected_displacement)
                        pursuers[j_prime].set_target(i, pursuers[j_prime].expected_displacement)

                        swapped = True

        return self.A_pe, self.A_ep

    def get_pursuer_indegree_matrix(self):
        """获取追捕者入度矩阵 D_pe = diag{d_m^{pe}}

        d_m^{pe} = Σ_{n=1}^{N_e} c_{m,n}
        """
        d_pe = np.sum(self.A_pe, axis=1)
        return np.diag(d_pe)

    def get_evader_indegree(self):
        """获取逃避者入度向量 d_i^{ep}

        d_n^{ep} = Σ_{m=1}^{N_p} e_{n,m}
        """
        return np.sum(self.A_ep, axis=1)

    def compute_team_error(self, pursuers, evaders, Q=None):
        """计算团队状态误差

        对应公式 (7):
        E_{Team} = Σ_{i=1}^{N_e} Σ_{j=1}^{N_p} (c_{j,i} ||ν ⊙ (x_j^p - x_i^e + r_{j,i}^{pe})||)

        Args:
            pursuers: 追捕者列表
            evaders: 逃避者列表
            Q: 权重矩阵 (用于计算 ν = sqrt(diag(Q)))

        Returns:
            team_error: 团队状态误差
        """
        from config import Config
        if Q is None:
            Q = Config.Q

        nu = np.sqrt(np.diag(Q))
        distances = self.compute_all_distances(pursuers, evaders, nu)

        # E_{Team} = Σ_{i,j} c_{j,i} * γ_{j,i}
        team_error = np.sum(self.A_pe * distances)

        return team_error