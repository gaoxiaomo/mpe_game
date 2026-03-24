from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import nonquadratic_integral_cost
from .dynamics import AircraftDynamics
from .summed_features import SummedStateFeatureMap
from .team_comm_controller import TeamControlStep


class SummedVSNACController:
    """V-SNAC controller using summed error state (original paper).

    All pursuers tracking the same evader get the same gradient direction.
    Individual controls differ only because g(x_j^p) differs per pursuer.
    """

    def __init__(
        self,
        dynamics: AircraftDynamics,
        features: SummedStateFeatureMap,
        n_pursuers: int,
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
        self.n_pursuers = n_pursuers
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

    def team_state(
        self,
        pursuer_states: np.ndarray,
        evader_state: np.ndarray,
        displacements: np.ndarray,
    ) -> np.ndarray:
        """Compute the SUMMED error state (6D).

        Returns tilde_x = sum_j (x_j^p - x^e + d_j), a 6-dim vector.
        """
        errors = pursuer_states - evader_state + displacements[:, 0, :]
        return errors.sum(axis=0)

    def individual_errors(
        self,
        pursuer_states: np.ndarray,
        evader_state: np.ndarray,
        displacements: np.ndarray,
    ) -> np.ndarray:
        """Individual error vectors (n_pursuers, 6) -- needed for block_errors."""
        return pursuer_states - evader_state + displacements[:, 0, :]

    def block_errors(self, pursuer_states: np.ndarray, evader_state: np.ndarray, displacements: np.ndarray) -> np.ndarray:
        """Per-pursuer error norms (same metric as stacked version)."""
        errors = self.individual_errors(pursuer_states, evader_state, displacements)
        return np.linalg.norm(errors * self.nu_eval, axis=1)

    def team_error(self, pursuer_states: np.ndarray, evader_state: np.ndarray, displacements: np.ndarray) -> float:
        return float(np.sum(self.block_errors(pursuer_states, evader_state, displacements)))

    def stage_cost(
        self,
        pursuer_states: np.ndarray,
        evader_state: np.ndarray,
        displacements: np.ndarray,
        u_p: np.ndarray,
        u_e: np.ndarray,
    ) -> float:
        """Stage cost using individual error blocks (same as stacked version)."""
        errors = self.individual_errors(pursuer_states, evader_state, displacements)
        x_norm = errors / self.state_cost_scale
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
        weights: np.ndarray,
        rng: np.random.Generator,
        exploration_std: float = 0.0,
    ) -> TeamControlStep:
        """Compute control using summed-state value function.

        Key difference: ONE gradient is computed on the summed state, shared
        by all pursuers. Each pursuer's control differs only through g(x_j^p).
        """
        n_p = pursuer_states.shape[0]
        summed_state = self.team_state(pursuer_states, evader_state, displacements)

        # Single gradient on the 6D summed state
        grad_6d = self.features.gradient(summed_state, weights)

        # All pursuers get the same gradient direction, but each uses its own g(x_j^p)
        g33 = self.dynamics.g33_batch(pursuer_states)
        # grad_6d[3:] are the velocity-space components
        gt_grad = np.column_stack([
            np.full(n_p, grad_6d[3]),
            np.full(n_p, grad_6d[4]),
            g33 * grad_6d[5],
        ])
        rho_all = self._r1_inv_diag * gt_grad / (2.0 * self.u_bar_p_actor)
        pursuer_u_tanh = -self.u_bar_p_actor * np.tanh(rho_all)
        pursuer_u = pursuer_u_tanh.copy()
        if exploration_std > 0.0:
            pursuer_u = pursuer_u + rng.normal(0.0, exploration_std, size=(n_p, 3))
        pursuer_u = np.clip(pursuer_u, -self.u_bar_p, self.u_bar_p)

        # Evader policy: uses the same gradient (summed state gradient)
        g_e = self.dynamics.g(evader_state)
        rho_e = (self.r2_inv @ (g_e.T @ grad_6d)) / (2.0 * self.u_bar_e_actor)
        u_e_tanh = -self.u_bar_e_actor * np.tanh(rho_e)
        evader_u = np.clip(u_e_tanh, -self.u_bar_e, self.u_bar_e)

        block_errs = self.block_errors(pursuer_states, evader_state, displacements)
        team_err = float(np.sum(block_errs))
        sc = self.stage_cost(pursuer_states, evader_state, displacements, pursuer_u, evader_u)

        # For compatibility, create block_gradients arrays
        # In summed mode, every pursuer sees the same gradient
        block_grad = np.tile(grad_6d, (n_p, 1))

        # Build dummy team_estimates and team_masks (not used in summed mode)
        dummy_estimates = np.tile(summed_state, (n_p, 1))
        dummy_masks = np.ones((n_p, n_p), dtype=float)

        return TeamControlStep(
            true_team_state=summed_state,
            team_estimates=dummy_estimates,
            team_masks=dummy_masks,
            block_gradients_true=block_grad,
            block_gradients_est=block_grad,
            pursuer_u=pursuer_u,
            evader_u=evader_u,
            pursuer_u_tanh=pursuer_u_tanh,
            evader_u_tanh=u_e_tanh,
            stage_cost=sc,
            block_errors=block_errs,
            team_error=team_err,
        )
