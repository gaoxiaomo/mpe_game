import numpy as np


class AircraftDynamics:
    """非线性飞行器动力学模型

    对应公式 (53) 和 (54):

    f(x) = [v̆_{a,i}, v̆_{b,i}, v_{h,i},
            1/m̆(T̆cos(ᾰ) - D̆),
            1/m̆(L̆ + T̆sin(ᾰ)),
            1/m̆(M̆ + T̆y_d)]^T

    g(x) = [0_{3×3}; I_3 + nonlinear_term]

    其中 nonlinear_term 在 g[5,2] 位置有 0.25*sin(v̆_{a,i}*x_{b,i})^2
    """

    def __init__(self, params=None):
        from config import Config
        if params is None:
            params = Config.AIRCRAFT_PARAMS

        self.m = params['m']  # 质量
        self.S = params['S']  # 参考面积
        self.C_rho = params['C_rho']  # 空气密度系数
        self.C_beta = params['C_beta']  # 推力系数
        self.y_d = params['y_d']  # 推力作用点距离
        self.C_D0 = params['C_D0']  # 阻力系数
        self.C_L0 = params['C_L0']  # 升力系数
        self.C_M0 = params['C_M0']  # 俯仰力矩系数

    def compute_aerodynamics(self, x):
        """计算气动参数

        x = [xa, xb, h, va, vb, vh]

        计算公式:
        - V̆ = sqrt(v̆_a^2 + v̆_b^2 + v_h^2)  真实空速
        - ρ̆ = C_ρ * exp(-h/24000)           空气密度
        - q̄ = 0.5 * ρ̆ * V̆^2                动压
        - ᾰ = arctan(v_h / v̆_a)            攻角
        - β̆ = arcsin(v̆_b / V̆)              侧滑角
        - C_T = C_β * β̆                     推力系数
        - T̆ = q̄ * S̆ * C_T                  推力
        """
        xa, xb, h, va, vb, vh = x

        # 真实空速 (避免除零)
        V = np.sqrt(va ** 2 + vb ** 2 + vh ** 2)
        V = max(V, 1e-6)

        # 空气密度 (随高度指数衰减)
        rho = self.C_rho * np.exp(-h / 24000.0)

        # 动压
        q_bar = 0.5 * rho * V ** 2

        # 攻角
        alpha = np.arctan2(vh, va + 1e-6)

        # 侧滑角
        beta = np.arcsin(np.clip(vb / V, -1.0, 1.0))

        # 推力系数和推力
        C_T = self.C_beta * beta
        T = q_bar * self.S * C_T

        # 阻力、升力、俯仰力矩
        D = q_bar * self.S * self.C_D0
        L = q_bar * self.S * self.C_L0
        M = q_bar * self.S * self.C_M0

        return T, D, L, M, alpha, beta, V, rho, q_bar

    def f(self, x):
        """漂移项 f(x) ∈ R^6

        对应公式 (53):
        f(x) = [v̆_{a,i}
                v̆_{b,i}
                v_{h,i}
                1/m̆ * (T̆*cos(ᾰ) - D̆)
                1/m̆ * (L̆ + T̆*sin(ᾰ))
                1/m̆ * (M̆ + T̆*y_d)]
        """
        xa, xb, h, va, vb, vh = x
        T, D, L, M, alpha, beta, V, rho, q_bar = self.compute_aerodynamics(x)

        f_vec = np.zeros(6)
        f_vec[0] = va  # ẋa = va
        f_vec[1] = vb  # ẋb = vb
        f_vec[2] = vh  # ḣ = vh
        f_vec[3] = (1.0 / self.m) * (T * np.cos(alpha) - D)  # v̇a
        f_vec[4] = (1.0 / self.m) * (L + T * np.sin(alpha))  # v̇b
        f_vec[5] = (1.0 / self.m) * (M + T * self.y_d)  # v̇h

        return f_vec

    def g(self, x):
        """控制增益矩阵 g(x) ∈ R^{6×3}

        对应公式 (54):
        g(x) = [0_{3×3}
                [1, 0, 0]
                [0, 1, 0]
                [0, 0, 1 + 0.25*sin(v̆_{a,i}*x_{b,i})^2]]
        """
        xa, xb, h, va, vb, vh = x

        g_mat = np.zeros((6, 3))
        g_mat[3, 0] = 1.0
        g_mat[4, 1] = 1.0
        g_mat[5, 2] = 1.0 + 0.25 * (np.sin(va * xb)) ** 2

        return g_mat

    def dynamics(self, x, u):
        """完整动力学方程

        ẋ = f(x) + g(x)u

        对应公式 (1) 和 (2)
        """
        return self.f(x) + self.g(x) @ u
