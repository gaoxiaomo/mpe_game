"""
智能体模块

实现追捕者和逃避者的动力学更新
使用论文中的非线性飞行器模型（公式53-54）
"""
import numpy as np
from dynamics import AircraftDynamics


class BaseAgent:
    """智能体基类"""

    def __init__(self, agent_id, initial_state, u_bar):
        self.id = agent_id
        self.state = np.array(initial_state, dtype=float)
        self.u_bar = u_bar
        self.dynamics = AircraftDynamics()
        self.control_history = []
        self.state_history = [self.state.copy()]

    def update_state(self, u, dt):
        """使用RK4积分更新状态
        
        ẋ = f(x) + g(x)u  (公式1, 2)
        """
        u_sat = np.clip(u, -self.u_bar, self.u_bar)
        
        # RK4积分
        k1 = self.dynamics.dynamics(self.state, u_sat)
        k2 = self.dynamics.dynamics(self.state + 0.5 * dt * k1, u_sat)
        k3 = self.dynamics.dynamics(self.state + 0.5 * dt * k2, u_sat)
        k4 = self.dynamics.dynamics(self.state + dt * k3, u_sat)
        
        self.state = self.state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.control_history.append(u_sat.copy())
        self.state_history.append(self.state.copy())
        
        return self.state


class Pursuer(BaseAgent):
    """追捕者智能体"""

    def __init__(self, agent_id, initial_state, u_bar=None):
        from config import Config
        if u_bar is None:
            u_bar = Config.U_BAR_P
        super().__init__(agent_id, initial_state, u_bar)
        self.target_evader_id = None
        self.expected_displacement = np.zeros(6)

    def set_target(self, evader_id, displacement=None):
        """设置目标逃避者和期望相对位移"""
        self.target_evader_id = evader_id
        if displacement is not None:
            self.expected_displacement = np.array(displacement, dtype=float)


class Evader(BaseAgent):
    """逃避者智能体"""

    def __init__(self, agent_id, initial_state, u_bar=None):
        from config import Config
        if u_bar is None:
            u_bar = Config.U_BAR_E
        super().__init__(agent_id, initial_state, u_bar)
        self.targeting_pursuers = []
