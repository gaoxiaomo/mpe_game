import numpy as np


class Config:
    """系统配置参数 - 对应论文Table I和Section IV"""

    # 系统维度
    STATE_DIM = 6  # 状态维度 [xa, xb, h, va, vb, vh]
    CONTROL_DIM = 3  # 控制维度 [ua, ub, uh]

    # 飞行器参数 (Table I)
    AIRCRAFT_PARAMS = {
        'm': 8920.0,  # 质量 (kg)
        'S': 4.0,  # 参考面积 (m^2)
        'C_rho': 0.00238,  # 海平面空气密度
        'C_beta': 0.002576,  # 推力系数
        'y_d': 0.5,  # 推力作用点距离
        'C_D0': 0.02,  # 零升阻力系数
        'C_L0': 0.1,  # 升力系数
        'C_M0': 0.01,  # 俯仰力矩系数
    }

    # 控制约束 (Section IV-B)
    U_BAR_P = 25.0  # 追捕者输入饱和限制 ū_p
    U_BAR_E = 15.0  # 逃避者输入饱和限制 ū_e

    # 代价函数权重矩阵 (Section IV-B: Q=I, R1=R2=I)
    # 注意：由于使用归一化状态，需要放大Q以获得足够大的梯度
    # 这等价于在原始状态空间使用 Q_original = Q * scale²
    Q = 100.0 * np.eye(STATE_DIM)  # 状态权重（放大以补偿归一化）
    R1 = np.eye(CONTROL_DIM)  # 追捕者控制权重
    R2 = np.eye(CONTROL_DIM)  # 逃避者控制权重

    # 神经网络参数 (Section IV)
    HIDDEN_NEURONS = 21  # 多项式基函数数量 (6个二次项 + 15个交叉项)
    LEARNING_RATE = 0.01  # 学习率 α_i

    # 算法参数
    SWAP_THRESHOLD = 1.0  # 目标交换阈值 ρ (Remark 2)
    CONVERGENCE_THRESHOLD = 1e-4  # 收敛阈值 ε
    MAX_ITERATIONS = 10000  # 最大迭代次数

    # 仿真参数
    DT = 0.01  # 时间步长
    T_FINAL = 20.0  # 仿真总时间