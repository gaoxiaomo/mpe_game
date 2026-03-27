from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .dynamics import AircraftDynamics
from .features import SigmoidFeatureMap


def safe_atanh(x: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(x, -0.999999, 0.999999))


def nonquadratic_integral_cost(u: np.ndarray, u_bar: float, r_diag: np.ndarray) -> float:
    ratio = np.clip(u / u_bar, -0.999999, 0.999999)
    term = u * safe_atanh(ratio) + 0.5 * u_bar * np.log(1.0 - ratio * ratio)
    return float(np.sum(2.0 * u_bar * r_diag * term))


def formation_gradient(
    j: int,
    pursuer_states: np.ndarray,
    A_p: np.ndarray,
    delta_matrix: np.ndarray,
    gamma: float,
    ref_dist: float = 500.0,
    d_safe: float = 0.0,
) -> np.ndarray:
    """Compute formation potential gradient in velocity space for pursuer j.

    PD-like coordination that drives inter-pursuer relative states toward
    the desired formation displacement Δ_{jk} = r_j - r_k:

        grad_v = gamma/d_j * sum_k a_{jk} * (dv + kp * pos_err / ref_dist)

    where pos_err = (p_j - p_k) - Δ_{jk} and kp scales position deviation
    into a velocity correction.  This naturally provides:
    - Repulsion when too close (pos_err points outward)
    - Damping of relative velocity (dv term)
    - Formation convergence when near the evader
    """
    n_p = pursuer_states.shape[0]
    grad = np.zeros(6, dtype=float)
    if gamma <= 0.0 or A_p is None:
        return grad

    d_j = float(A_p[j].sum())
    if d_j <= 0.0:
        return grad

    p_j = pursuer_states[j, :3]
    v_j = pursuer_states[j, 3:]
    kp = 1.0

    for k in range(n_p):
        if k == j or A_p[j, k] <= 0.0:
            continue
        p_k = pursuer_states[k, :3]
        v_k = pursuer_states[k, 3:]
        dv = v_j - v_k
        delta_p = delta_matrix[j, k, :3] if delta_matrix is not None else np.zeros(3)
        pos_err = (p_j - p_k) - delta_p

        # Per-pair adaptive gamma amplification near d_safe
        if d_safe > 0.0:
            dp = p_j - p_k
            dist_sq = float(np.dot(dp, dp))
            amp = max(1.0, d_safe * d_safe / max(dist_sq, 1e-8))
        else:
            amp = 1.0

        grad[3:] += amp * A_p[j, k] * (dv + kp * pos_err / ref_dist)

    grad[3:] *= gamma / d_j
    return grad


class CommVSNACController:
    """Communication-augmented V-SNAC controller.

    Training uses standard tracking error (same as original paper).
    During evaluation, an additive formation gradient from the
    communication graph is injected into the value gradient:

        u_j = -u_bar * tanh(1/(2u_bar) R^{-1} g^T [nabla_psi^T W_j + gamma * grad_Phi])

    This keeps training unchanged while adding communication-based
    formation coordination at runtime.
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
        self.state_cost_scale = state_cost_scale

    def individual_error(
        self,
        pursuer_idx: int,
        evader_idx: int,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        displacements: np.ndarray,
    ) -> np.ndarray:
        """Standard tracking error (no communication coupling)."""
        return (
            pursuer_states[pursuer_idx]
            - evader_states[evader_idx]
            + displacements[pursuer_idx, evader_idx]
        )

    def value_gradient(self, x_tilde: np.ndarray, w: np.ndarray) -> np.ndarray:
        jac = self.features.jacobian(x_tilde)
        return jac.T @ w

    def policy(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        displacements: np.ndarray,
        weights: List[np.ndarray],
        rng: np.random.Generator,
        exploration_std: float = 0.0,
        gamma: float = 0.0,
        A_p: np.ndarray | None = None,
        delta_matrix: np.ndarray | None = None,
        formation_ref_dist: float = 500.0,
        d_safe: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_p = pursuer_states.shape[0]
        n_e = evader_states.shape[0]

        x_err = np.zeros((n_p, 6), dtype=float)
        grad_p = np.zeros((n_p, 6), dtype=float)
        for j in range(n_p):
            i = int(assignment[j])
            # Standard tracking error (same as original paper)
            x_err[j] = self.individual_error(j, i, pursuer_states, evader_states, displacements)
            # Tracking value gradient
            grad_p[j] = self.value_gradient(x_err[j], weights[j])

        u_p = np.zeros((n_p, 3), dtype=float)
        u_p_tanh = np.zeros((n_p, 3), dtype=float)
        for j in range(n_p):
            g_p = self.dynamics.g(pursuer_states[j])
            # Add formation gradient from communication (zero during training)
            total_grad = grad_p[j].copy()
            if gamma > 0.0 and A_p is not None:
                fg = formation_gradient(j, pursuer_states, A_p, delta_matrix,
                                        gamma, formation_ref_dist, d_safe=d_safe)
                total_grad += fg
            rho_p = (self.r1_inv @ (g_p.T @ total_grad)) / (2.0 * self.u_bar_p_actor)
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
