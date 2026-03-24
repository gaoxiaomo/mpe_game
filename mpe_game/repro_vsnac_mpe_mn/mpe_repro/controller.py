from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .dynamics import AircraftDynamics
from .features import SigmoidFeatureMap


def safe_atanh(x: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(x, -0.999999, 0.999999))


def nonquadratic_integral_cost(u: np.ndarray, u_bar: float, r_diag: np.ndarray) -> float:
    ratio = np.clip(u / u_bar, -0.999999, 0.999999)
    term = u * safe_atanh(ratio) + 0.5 * u_bar * np.log(1.0 - ratio * ratio)
    return float(np.sum(2.0 * u_bar * r_diag * term))


class VSNACController:
    """Policy and stage-cost computation from equations (14), (40), (44)."""

    def __init__(
        self,
        dynamics: AircraftDynamics,
        features: SigmoidFeatureMap,
        q: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        u_bar_p: float,
        u_bar_e: float,
        u_bar_p_actor: float,
        u_bar_e_actor: float,
        policy_gain: float,
        k_pos_p: float,
        k_vel_p: float,
        k_pos_e: float,
        k_vel_e: float,
        state_cost_scale: np.ndarray,
    ) -> None:
        self.dynamics = dynamics
        self.features = features
        self.q = q
        self.r1 = r1
        self.r2 = r2
        self.r1_inv = np.linalg.inv(r1)
        self.r2_inv = np.linalg.inv(r2)
        self.r1_diag = np.diag(r1)
        self.r2_diag = np.diag(r2)
        self.u_bar_p = u_bar_p
        self.u_bar_e = u_bar_e
        self.u_bar_p_actor = u_bar_p_actor
        self.u_bar_e_actor = u_bar_e_actor
        self.policy_gain = policy_gain
        self.k_pos_p = k_pos_p
        self.k_vel_p = k_vel_p
        self.k_pos_e = k_pos_e
        self.k_vel_e = k_vel_e
        self.state_cost_scale = state_cost_scale

    def individual_error(
        self,
        pursuer_idx: int,
        evader_idx: int,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        displacements: np.ndarray,
    ) -> np.ndarray:
        return (
            pursuer_states[pursuer_idx]
            - evader_states[evader_idx]
            + displacements[pursuer_idx, evader_idx]
        )

    def value_gradient(self, x_tilde: np.ndarray, w: np.ndarray) -> np.ndarray:
        jac = self.features.jacobian(x_tilde)  # [h, n]
        return jac.T @ w  # [n]

    def policy(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        displacements: np.ndarray,
        weights: List[np.ndarray],
        rng: np.random.Generator,
        exploration_std: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_p = pursuer_states.shape[0]
        n_e = evader_states.shape[0]

        x_err = np.zeros((n_p, 6), dtype=float)
        grad_p = np.zeros((n_p, 6), dtype=float)
        for j in range(n_p):
            i = int(assignment[j])
            x_err[j] = self.individual_error(j, i, pursuer_states, evader_states, displacements)
            grad_p[j] = self.value_gradient(x_err[j], weights[j])

        u_p = np.zeros((n_p, 3), dtype=float)
        u_p_tanh = np.zeros((n_p, 3), dtype=float)
        for j in range(n_p):
            g_p = self.dynamics.g(pursuer_states[j])
            # Paper Eq. (40): u_p* = -u_bar_p * tanh((1/(2u_bar_p)) * R1^{-1} * g_p^T * dV/dx)
            rho_p = (self.r1_inv @ (g_p.T @ grad_p[j])) / (2.0 * self.u_bar_p_actor)
            u_rl = -self.u_bar_p_actor * np.tanh(rho_p)
            u = u_rl.copy()
            if exploration_std > 0.0:
                u += rng.normal(0.0, exploration_std, size=3)
            u_p[j] = np.clip(u, -self.u_bar_p, self.u_bar_p)
            u_p_tanh[j] = u_rl

        u_e = np.zeros((n_e, 3), dtype=float)
        u_e_tanh = np.zeros((n_e, 3), dtype=float)
        for i in range(n_e):
            pursuers_targeting = np.where(assignment == i)[0]
            if pursuers_targeting.size > 0:
                x_mean = np.mean(x_err[pursuers_targeting], axis=0)
                grad_mean = np.mean(grad_p[pursuers_targeting], axis=0)
            else:
                grad_mean = np.zeros(6, dtype=float)
            g_e = self.dynamics.g(evader_states[i])
            # Paper Eq. (40): u_e* = -u_bar_e * tanh((1/(2u_bar_e)) * R2^{-1} * g_e^T * dV/dx)
            rho_e = (self.r2_inv @ (g_e.T @ grad_mean)) / (2.0 * self.u_bar_e_actor)
            u_rl = -self.u_bar_e_actor * np.tanh(rho_e)
            u = u_rl.copy()
            if exploration_std > 0.0:
                u += rng.normal(0.0, exploration_std, size=3)
            u_e[i] = np.clip(u, -self.u_bar_e, self.u_bar_e)
            u_e_tanh[i] = u_rl

        return u_p, u_e, x_err, u_p_tanh, u_e_tanh

    def stage_costs(
        self,
        x_err: np.ndarray,
        u_p: np.ndarray,
        u_e: np.ndarray,
        assignment: np.ndarray,
    ) -> np.ndarray:
        n_p = u_p.shape[0]
        costs = np.zeros(n_p, dtype=float)
        for j in range(n_p):
            i = int(assignment[j])
            x_norm = x_err[j] / self.state_cost_scale
            state_cost = float(x_norm.T @ self.q @ x_norm)
            pursuer_cost = nonquadratic_integral_cost(u_p[j], self.u_bar_p, self.r1_diag)
            evader_cost = nonquadratic_integral_cost(u_e[i], self.u_bar_e, self.r2_diag)
            costs[j] = state_cost + pursuer_cost - evader_cost
        return costs
