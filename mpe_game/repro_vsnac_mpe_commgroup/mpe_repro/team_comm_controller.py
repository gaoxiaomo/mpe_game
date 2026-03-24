from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import nonquadratic_integral_cost
from .dynamics import AircraftDynamics
from .team_comm_features import TeamCommunicationFeatureMap


@dataclass
class TeamControlStep:
    true_team_state: np.ndarray
    team_estimates: np.ndarray
    team_masks: np.ndarray
    block_gradients_true: np.ndarray
    block_gradients_est: np.ndarray
    pursuer_u: np.ndarray
    evader_u: np.ndarray
    pursuer_u_tanh: np.ndarray
    evader_u_tanh: np.ndarray
    stage_cost: float
    block_errors: np.ndarray
    team_error: float


class TeamVSNACController:
    """Centralized team critic with decentralized execution under communication limits."""

    def __init__(
        self,
        dynamics: AircraftDynamics,
        features: TeamCommunicationFeatureMap,
        q: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        u_bar_p: float,
        u_bar_e: float,
        u_bar_p_actor: float,
        u_bar_e_actor: float,
        state_cost_scale: np.ndarray,
        nu_eval: np.ndarray,
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
        self.u_bar_p = float(u_bar_p)
        self.u_bar_e = float(u_bar_e)
        self.u_bar_p_actor = float(u_bar_p_actor)
        self.u_bar_e_actor = float(u_bar_e_actor)
        self.state_cost_scale = np.asarray(state_cost_scale, dtype=float)
        self.nu_eval = np.asarray(nu_eval, dtype=float)
        self._r1_inv_diag = np.diag(self.r1_inv)
        self._q_diag = np.diag(self.q)

    def individual_error(
        self,
        pursuer_idx: int,
        pursuer_states: np.ndarray,
        evader_state: np.ndarray,
        displacements: np.ndarray,
    ) -> np.ndarray:
        return pursuer_states[pursuer_idx] - evader_state + displacements[pursuer_idx, 0]

    def team_state(
        self,
        pursuer_states: np.ndarray,
        evader_state: np.ndarray,
        displacements: np.ndarray,
    ) -> np.ndarray:
        return (pursuer_states - evader_state + displacements[:, 0, :]).ravel()

    def block_errors(self, team_state: np.ndarray) -> np.ndarray:
        blocks = np.asarray(team_state, dtype=float).reshape(-1, 6)
        return np.linalg.norm(blocks * self.nu_eval, axis=1)

    def team_error(self, team_state: np.ndarray) -> float:
        return float(np.sum(self.block_errors(team_state)))

    def stage_cost(self, team_state: np.ndarray, u_p: np.ndarray, u_e: np.ndarray) -> float:
        blocks = np.asarray(team_state, dtype=float).reshape(-1, 6)
        x_norm = blocks / self.state_cost_scale
        state_cost = float(np.sum(x_norm * x_norm * self._q_diag))
        ratio_p = np.clip(u_p / self.u_bar_p, -0.999999, 0.999999)
        term_p = u_p * np.arctanh(ratio_p) + 0.5 * self.u_bar_p * np.log(1.0 - ratio_p * ratio_p)
        pursuer_cost = float(np.sum(2.0 * self.u_bar_p * self.r1_diag * term_p))
        evader_cost = nonquadratic_integral_cost(u_e, self.u_bar_e, self.r2_diag)
        return state_cost + pursuer_cost - evader_cost

    def policy(
        self,
        pursuer_states: np.ndarray,
        evader_state: np.ndarray,
        displacements: np.ndarray,
        team_estimates: np.ndarray,
        team_masks: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        exploration_std: float = 0.0,
    ) -> TeamControlStep:
        n_p = pursuer_states.shape[0]
        true_team_state = self.team_state(pursuer_states, evader_state, displacements)

        full_grad_true = self.features.gradient(true_team_state, weights)
        block_gradients_true = full_grad_true.reshape(n_p, 6)

        block_gradients_est = np.zeros((n_p, 6), dtype=float)
        for j in range(n_p):
            grad_j = self.features.gradient(team_estimates[j], weights, visibility_mask=team_masks[j])
            block_gradients_est[j] = grad_j[j * 6 : (j + 1) * 6]

        g33 = self.dynamics.g33_batch(pursuer_states)
        gt_grad = np.column_stack([
            block_gradients_est[:, 3],
            block_gradients_est[:, 4],
            g33 * block_gradients_est[:, 5],
        ])
        rho_all = self._r1_inv_diag * gt_grad / (2.0 * self.u_bar_p_actor)
        pursuer_u_tanh = -self.u_bar_p_actor * np.tanh(rho_all)
        pursuer_u = pursuer_u_tanh.copy()
        if exploration_std > 0.0:
            pursuer_u = pursuer_u + rng.normal(0.0, exploration_std, size=(n_p, 3))
        pursuer_u = np.clip(pursuer_u, -self.u_bar_p, self.u_bar_p)

        evader_gradient = np.sum(block_gradients_true, axis=0)
        g_e = self.dynamics.g(evader_state)
        rho_e = (self.r2_inv @ (g_e.T @ evader_gradient)) / (2.0 * self.u_bar_e_actor)
        u_e_tanh = -self.u_bar_e_actor * np.tanh(rho_e)
        evader_u = np.clip(u_e_tanh, -self.u_bar_e, self.u_bar_e)

        stage_cost = self.stage_cost(true_team_state, pursuer_u, evader_u)
        block_errors = self.block_errors(true_team_state)
        team_error = float(np.sum(block_errors))

        return TeamControlStep(
            true_team_state=true_team_state,
            team_estimates=team_estimates.copy(),
            team_masks=team_masks.copy(),
            block_gradients_true=block_gradients_true,
            block_gradients_est=block_gradients_est,
            pursuer_u=pursuer_u,
            evader_u=evader_u,
            pursuer_u_tanh=pursuer_u_tanh,
            evader_u_tanh=u_e_tanh,
            stage_cost=stage_cost,
            block_errors=block_errors,
            team_error=team_error,
        )
