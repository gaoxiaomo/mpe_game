from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class AircraftParams:
    m: float = 8920.0
    S: float = 4.0
    c_rho: float = 0.00238
    c_beta: float = 0.002576
    y_d: float = 0.5
    c_d0: float = 0.02
    c_l0: float = 0.10
    c_m0: float = 0.01


@dataclass(frozen=True)
class LearningParams:
    dt: float = 0.05
    ridge_lambda: float = 1e-3
    replay_capacity: int = 30000
    min_samples_per_evader: int = 50
    policy_iterations: int = 40
    rollout_steps: int = 200
    exploration_std_start: float = 1.0
    exploration_std_end: float = 0.1
    critic_learning_rate: float = 0.05
    critic_lr_decay: float = 0.90
    convergence_tol: float = 1e-3
    random_perturb_scale: float = 0.03
    graph_update_interval: int = 10
    graph_update_start_step: int = 0


@dataclass(frozen=True)
class ControlParams:
    u_bar_p: float = 25.0
    u_bar_e: float = 15.0
    # Paper text uses a separate saturation scale in policy expression and hard
    # input bounds in simulation. If None, fallback to input bounds.
    u_bar_p_policy: float | None = None
    u_bar_e_policy: float | None = None
    policy_gain: float = 20.0
    k_pos_p: float = 0.050
    k_vel_p: float = 0.350
    k_pos_e: float = 0.010
    k_vel_e: float = 0.080
    # Keep XY as dominant terms, with mild H-channel penalty for better 3D behavior.
    q_diag: Tuple[float, ...] = (8.0, 8.0, 2.5, 3.0, 3.0, 1.2)
    r1_diag: Tuple[float, ...] = (1.0, 1.0, 1.0)
    r2_diag: Tuple[float, ...] = (1.0, 1.0, 1.0)

    @property
    def q(self) -> np.ndarray:
        return np.diag(self.q_diag)

    @property
    def r1(self) -> np.ndarray:
        return np.diag(self.r1_diag)

    @property
    def r2(self) -> np.ndarray:
        return np.diag(self.r2_diag)

    @property
    def u_bar_p_actor(self) -> float:
        return self.u_bar_p if self.u_bar_p_policy is None else float(self.u_bar_p_policy)

    @property
    def u_bar_e_actor(self) -> float:
        return self.u_bar_e if self.u_bar_e_policy is None else float(self.u_bar_e_policy)


@dataclass(frozen=True)
class FeatureParams:
    n_features: int = 6
    state_scale: Tuple[float, ...] = (800.0, 800.0, 800.0, 80.0, 80.0, 80.0)
    feature_gain: float = 4.0
    random_seed: int = 42


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    pursuer_init: np.ndarray
    evader_init: np.ndarray
    displacement_matrix: np.ndarray
    swap_threshold: float
    max_switch_worsening: float
    initial_assignment: np.ndarray
    t_final: float
    capture_radius: float
    evader_motion_mode: str = "policy"
    evader_script_amp: Tuple[float, float, float] = (8.0, 6.0, 3.0)
    evader_script_omega: float = 0.55
    evader_script_decay: float = 0.07
    evader_script_mix: float = 0.15
    # Reactive mode: evader command magnitude rises with pursuer effort.
    evader_reactive_base: float = 0.45
    evader_reactive_response: float = 0.85
    evader_reactive_vel_gain: float = 0.25
    evader_reactive_decay: float = 0.02
    evader_reactive_floor: float = 0.20
    swap_lookahead_time: float = 0.0

    @property
    def n_pursuers(self) -> int:
        return self.pursuer_init.shape[0]

    @property
    def n_evaders(self) -> int:
        return self.evader_init.shape[0]


def _build_displacements(n_p: int, n_e: int, per_pursuer: List[List[float]]) -> np.ndarray:
    mat = np.zeros((n_p, n_e, 6), dtype=float)
    for j in range(n_p):
        mat[j, :, :] = np.asarray(per_pursuer[j], dtype=float)
    return mat


def scenario_three_pursuer_one_evader() -> ScenarioConfig:
    pursuers = np.array(
        [
            [600.0, -1900.0, 300.0, 78.0, 2.0, 18.0],
            [1200.0, 500.0, 2200.0, 2.0, 20.0, 15.0],
            [200.0, 600.0, 800.0, 5.0, 10.0, 125.0],
        ],
        dtype=float,
    )
    evaders = np.array([[500.0, 1500.0, 200.0, 50.0, 80.0, 10.0]], dtype=float)
    displacements = _build_displacements(
        n_p=3,
        n_e=1,
        per_pursuer=[
            [50.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            [-10.0, 0.0, -50.0, 0.0, 0.0, 0.0],
        ],
    )
    assignment = np.array([0, 0, 0], dtype=int)
    return ScenarioConfig(
        name="scenario_3v1",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=1e9,  # no target switching in 3v1
        max_switch_worsening=0.0,
        initial_assignment=assignment,
        t_final=100.0,
        capture_radius=160.0,
        evader_motion_mode="scripted",
        evader_script_amp=(10.0, 8.0, 4.0),
        evader_script_omega=0.58,
        evader_script_decay=0.075,
        evader_script_mix=1.0,
        swap_lookahead_time=0.0,
    )


def scenario_three_pursuer_three_evader() -> ScenarioConfig:
    pursuers = np.array(
        [
            [263.0, -2380.0, 800.0, 123.0, 12.0, 18.0],
            [-380.0, 3100.0, -200.0, 12.0, -40.0, 15.0],
            [2500.0, -3280.0, 400.0, -15.0, 30.0, 15.0],
        ],
        dtype=float,
    )
    evaders = np.array(
        [
            [1037.0, 360.0, 100.0, 80.0, 40.0, 0.0],
            [521.0, 1030.0, 34.0, 0.0, 80.0, 1.6],
            [1816.0, -1887.0, -90.0, 40.0, 65.0, -0.1],
        ],
        dtype=float,
    )
    displacements = _build_displacements(
        n_p=3,
        n_e=3,
        per_pursuer=[
            [100.0, 50.0, 50.0, 0.0, 0.0, 0.0],
            [100.0, 50.0, 50.0, 0.0, 0.0, 0.0],
            [100.0, 50.0, 50.0, 0.0, 0.0, 0.0],
        ],
    )
    assignment = np.array([0, 1, 2], dtype=int)
    return ScenarioConfig(
        name="scenario_3v3",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=5.0,
        max_switch_worsening=0.0,
        initial_assignment=assignment,
        t_final=130.0,
        capture_radius=220.0,
        evader_motion_mode="scripted",
        evader_script_amp=(8.0, 8.0, 3.0),
        evader_script_omega=0.14,
        evader_script_decay=0.01,
        evader_script_mix=0.2,
        swap_lookahead_time=0.0,
    )
