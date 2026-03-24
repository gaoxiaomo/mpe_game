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
        blocks = [
            self.individual_error(j, pursuer_states, evader_state, displacements)
            for j in range(pursuer_states.shape[0])
        ]
        return np.concatenate(blocks, axis=0)

    def block_errors(self, team_state: np.ndarray) -> np.ndarray:
        blocks = np.asarray(team_state, dtype=float).reshape(-1, 6)
        return np.array([np.linalg.norm(self.nu_eval * block) for block in blocks], dtype=float)

    def team_error(self, team_state: np.ndarray) -> float:
        return float(np.sum(self.block_errors(team_state)))

    def stage_cost(self, team_state: np.ndarray, u_p: np.ndarray, u_e: np.ndarray) -> float:
        blocks = np.asarray(team_state, dtype=float).reshape(-1, 6)
        state_cost = 0.0
        for block in blocks:
            x_norm = block / self.state_cost_scale
            state_cost += float(x_norm.T @ self.q @ x_norm)
        pursuer_cost = float(
            sum(nonquadratic_integral_cost(u_p[j], self.u_bar_p, self.r1_diag) for j in range(u_p.shape[0]))
        )
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
        block_gradients_true = np.zeros((n_p, 6), dtype=float)
        block_gradients_est = np.zeros((n_p, 6), dtype=float)
        pursuer_u = np.zeros((n_p, 3), dtype=float)
        pursuer_u_tanh = np.zeros((n_p, 3), dtype=float)

        for j in range(n_p):
            block_gradients_true[j] = self.features.block_gradient(true_team_state, weights, j)
            block_gradients_est[j] = self.features.block_gradient(
                team_estimates[j],
                weights,
                j,
                visibility_mask=team_masks[j],
            )
            g_p = self.dynamics.g(pursuer_states[j])
            rho = (self.r1_inv @ (g_p.T @ block_gradients_est[j])) / (2.0 * self.u_bar_p_actor)
            u_rl = -self.u_bar_p_actor * np.tanh(rho)
            u = u_rl.copy()
            if exploration_std > 0.0:
                u += rng.normal(0.0, exploration_std, size=3)
            pursuer_u[j] = np.clip(u, -self.u_bar_p, self.u_bar_p)
            pursuer_u_tanh[j] = u_rl

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
