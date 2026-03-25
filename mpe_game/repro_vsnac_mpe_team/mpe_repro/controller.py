from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .dynamics import AircraftDynamics
from .features import CouplingFeatureMap, SigmoidFeatureMap


def safe_atanh(x: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(x, -0.999999, 0.999999))


def nonquadratic_integral_cost(u: np.ndarray, u_bar: float, r_diag: np.ndarray) -> float:
    ratio = np.clip(u / u_bar, -0.999999, 0.999999)
    term = u * safe_atanh(ratio) + 0.5 * u_bar * np.log(1.0 - ratio * ratio)
    return float(np.sum(2.0 * u_bar * r_diag * term))


def separation_penalty(pursuer_states: np.ndarray, d_ref: float) -> np.ndarray:
    """Gaussian separation penalty S_j for each pursuer."""
    n_p = pursuer_states.shape[0]
    S = np.zeros(n_p, dtype=float)
    d_ref_sq_2 = 2.0 * d_ref * d_ref
    for j in range(n_p):
        for k in range(n_p):
            if k == j:
                continue
            d_sq = float(np.sum((pursuer_states[j, :3] - pursuer_states[k, :3]) ** 2))
            S[j] += np.exp(-d_sq / d_ref_sq_2)
    return S


class TeamVSNACController:
    """V-SNAC controller with two-stage team value function.

    Stage 1 (individual): V_j = W_j^T phi(x̃_j)
    Stage 2 (coupling):   V_j^team = W_j^T phi(x̃_j) + W_c^T psi_j

    The control law is always u_j = -ū tanh(1/(2ū) R⁻¹ g^T ∇V_j),
    where ∇V_j includes the coupling gradient when W_c is provided.
    """

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
        jac = self.features.jacobian(x_tilde)  # (6, 6)
        return jac.T @ w  # (6,)

    def policy(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        displacements: np.ndarray,
        weights: List[np.ndarray],
        rng: np.random.Generator,
        exploration_std: float = 0.0,
        # --- coupling parameters (Stage 2 / eval) ---
        coupling: Optional[CouplingFeatureMap] = None,
        w_c: Optional[np.ndarray] = None,
        prev_controls: Optional[np.ndarray] = None,
        teammate_cache: Optional[np.ndarray] = None,
        control_cache: Optional[np.ndarray] = None,
        comm_mode: str = "full",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_p = pursuer_states.shape[0]
        n_e = evader_states.shape[0]

        use_coupling = (
            coupling is not None
            and w_c is not None
            and comm_mode != "local"
        )
        state_only = comm_mode == "state_only"

        x_err = np.zeros((n_p, 6), dtype=float)
        grad_p = np.zeros((n_p, 6), dtype=float)
        for j in range(n_p):
            i = int(assignment[j])
            x_err[j] = self.individual_error(j, i, pursuer_states, evader_states, displacements)
            # Individual V-SNAC gradient.
            grad_p[j] = self.value_gradient(x_err[j], weights[j])

            # Add coupling gradient from team value function.
            if use_coupling:
                mask = np.ones(n_p, dtype=bool)
                mask[j] = False
                # Use cached teammate data during dropout; live data otherwise.
                src_states = teammate_cache if teammate_cache is not None else pursuer_states
                src_controls = control_cache if control_cache is not None else prev_controls
                tm_states = src_states[mask]
                tm_controls = (
                    np.zeros((n_p - 1, 3), dtype=float)
                    if src_controls is None
                    else src_controls[mask]
                )
                jac_psi = coupling.jac_psi(
                    pursuer_states[j], tm_states, tm_controls, state_only=state_only
                )  # (9, 6)
                grad_p[j] += jac_psi.T @ w_c  # (6,)

        # Pursuer controls.
        u_p = np.zeros((n_p, 3), dtype=float)
        u_p_tanh = np.zeros((n_p, 3), dtype=float)
        for j in range(n_p):
            g_p = self.dynamics.g(pursuer_states[j])
            rho_p = (self.r1_inv @ (g_p.T @ grad_p[j])) / (2.0 * self.u_bar_p_actor)
            u_rl = -self.u_bar_p_actor * np.tanh(rho_p)
            u = u_rl.copy()
            if exploration_std > 0.0:
                u += rng.normal(0.0, exploration_std, size=3)
            u_p[j] = np.clip(u, -self.u_bar_p, self.u_bar_p)
            u_p_tanh[j] = u_rl

        # Evader controls — uses sum of team gradients.
        u_e = np.zeros((n_e, 3), dtype=float)
        u_e_tanh = np.zeros((n_e, 3), dtype=float)
        for i in range(n_e):
            pursuers_targeting = np.where(assignment == i)[0]
            if pursuers_targeting.size > 0:
                grad_mean = np.mean(grad_p[pursuers_targeting], axis=0)
            else:
                grad_mean = np.zeros(6, dtype=float)
            g_e = self.dynamics.g(evader_states[i])
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
