import numpy as np


class CriticNetwork:
    """值函数逼近神经网络 (V-SNAC结构)
    
    使用多项式基函数逼近值函数
    
    公式 (37): V_i^*(x) = W_i^T ψ_i(x) + δ_i(x)
    公式 (38): V̂_i^s(x) = Ŵ_{i,s}^T ψ_i(x)
    公式 (39): ∇V̂_i(x) = ∇ψ_i^T(x) Ŵ_{i,s}
    """

    def __init__(self, state_dim, learning_rate=None):
        from config import Config

        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE

        self.state_dim = state_dim  # n = 6
        self.alpha = learning_rate

        # 状态归一化因子
        # 位置量级~3000m, 速度量级~100m/s
        self.state_scale = np.array([3000.0, 3000.0, 3000.0, 100.0, 100.0, 100.0])

        # 多项式基函数数量: 6个二次项 + 15个交叉项 = 21
        self.hidden_neurons = 21  # h = 21

        # 权重初始化 - 较小的随机值
        self.W = np.random.randn(self.hidden_neurons) * 0.1

        # 记录权重历史
        self.weight_history = [np.linalg.norm(self.W) ** 2]

    def normalize_state(self, x):
        """归一化状态到合理范围"""
        return x / self.state_scale

    def activation(self, x):
        """多项式激活函数 ψ(x) ∈ R^h
        
        在归一化状态上计算多项式基
        ψ(x_n) = [x1², x2², ..., x6², x1x2, x1x3, ..., x5x6]
        
        Returns:
            psi: (21,) 基函数值
        """
        x_n = self.normalize_state(x)
        psi = []

        # 二次项 x_i² (6个)
        for i in range(6):
            psi.append(x_n[i] ** 2)

        # 交叉项 x_i * x_j, i < j (15个)
        for i in range(6):
            for j in range(i + 1, 6):
                psi.append(x_n[i] * x_n[j])

        return np.array(psi)

    def activation_gradient(self, x):
        """激活函数梯度 ∇ψ(x) ∈ R^{n×h}
        
        包含链式法则：∂ψ/∂x = (∂ψ/∂x_n) * (∂x_n/∂x) = (∂ψ/∂x_n) / scale
        
        Returns:
            grad: (6, 21) 梯度矩阵
        """
        x_n = self.normalize_state(x)
        grad = np.zeros((6, 21))

        # 二次项梯度: ∂(x_n[i]²)/∂x[i] = 2*x_n[i] / scale[i]
        for i in range(6):
            grad[i, i] = 2 * x_n[i] / self.state_scale[i]

        # 交叉项梯度
        col = 6
        for i in range(6):
            for j in range(i + 1, 6):
                # ∂(x_n[i]*x_n[j])/∂x[i] = x_n[j] / scale[i]
                # ∂(x_n[i]*x_n[j])/∂x[j] = x_n[i] / scale[j]
                grad[i, col] = x_n[j] / self.state_scale[i]
                grad[j, col] = x_n[i] / self.state_scale[j]
                col += 1

        return grad

    def predict_value(self, x):
        """预测值函数 V̂(x) = W^T @ ψ(x)"""
        psi = self.activation(x)
        return np.dot(self.W, psi)

    def predict_gradient(self, x):
        """预测值函数梯度 ∇V̂(x) = ∇ψ^T(x) @ W = ∇ψ(x) @ W"""
        grad_psi = self.activation_gradient(x)  # (6, 21)
        grad_V = grad_psi @ self.W  # (6,)
        return grad_V
