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


def _separation_gradient(
    pursuer_j_state: np.ndarray,
    teammate_states: np.ndarray,
    alpha: float,
    d_ref: float,
    evader_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Lateral separation gradient injected into velocity components (3,4,5).

    To avoid slowing down the pursuit, the repulsive force is projected
    PERPENDICULAR to the pursuit direction.  This steers the pursuer
    sideways rather than decelerating it.

    The Gaussian kernel focuses the effect on close proximity only.
    """
    grad = np.zeros(6, dtype=float)
    if teammate_states.shape[0] == 0 or alpha == 0.0:
        return grad

    # Pursuit direction (toward evader).
    if evader_pos is not None:
        pursuit_vec = evader_pos[:3] - pursuer_j_state[:3]
        pn = float(np.linalg.norm(pursuit_vec))
        if pn > 1e-6:
            pursuit_dir = pursuit_vec / pn
        else:
            pursuit_dir = np.array([1.0, 0.0, 0.0])
    else:
        pursuit_dir = np.array([1.0, 0.0, 0.0])

    d_ref_sq = d_ref * d_ref
    for k in range(teammate_states.shape[0]):
        delta_pos = pursuer_j_state[:3] - teammate_states[k, :3]
        d_sq = float(np.dot(delta_pos, delta_pos))
        d = np.sqrt(d_sq + 1e-8)
        raw_dir = delta_pos / d  # away from teammate

        # Project out the pursuit component → lateral only.
        proj = float(np.dot(raw_dir, pursuit_dir))
        lateral = raw_dir - proj * pursuit_dir
        lat_norm = float(np.linalg.norm(lateral))
        if lat_norm > 1e-8:
            lateral = lateral / lat_norm
        else:
            # Teammate exactly on pursuit line → pick arbitrary perpendicular.
            perp = np.array([0.0, 1.0, 0.0]) if abs(pursuit_dir[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            lateral = perp - float(np.dot(perp, pursuit_dir)) * pursuit_dir
            ln = float(np.linalg.norm(lateral))
            lateral = lateral / max(ln, 1e-8)

        kernel = np.exp(-d_sq / (2.0 * d_ref_sq))
        # Negative sign: steers u in the lateral direction AWAY from teammate.
        # u = -ū tanh(R⁻¹ g^T ∇V), so adding NEGATIVE to ∇V[3:6]
        # makes u more positive in that direction → accelerate laterally.
        grad[3:6] -= alpha * lateral * kernel

    grad[3:6] /= max(teammate_states.shape[0], 1)
    return grad


class TeamVSNACController:
    """V-SNAC controller with optional analytical separation gradient.

    The value function is the standard 6-feature V-SNAC (proven convergence).
    Communication enables an additional **separation gradient** that pushes
    pursuers apart during control computation.  This separation is an
    analytical potential-field correction, not a learned coupling.
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
        sep_alpha: float = 0.0,
        sep_d_ref: float = 400.0,
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
        self.sep_alpha = sep_alpha
        self.sep_d_ref = sep_d_ref

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
        use_separation: bool = False,
        teammate_cache: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_p = pursuer_states.shape[0]
        n_e = evader_states.shape[0]

        x_err = np.zeros((n_p, 6), dtype=float)
        grad_p = np.zeros((n_p, 6), dtype=float)
        for j in range(n_p):
            i = int(assignment[j])
            x_err[j] = self.individual_error(j, i, pursuer_states, evader_states, displacements)
            # Standard V-SNAC gradient.
            grad_p[j] = self.value_gradient(x_err[j], weights[j])

            # Add analytical separation gradient if communication available.
            if use_separation and self.sep_alpha > 0.0:
                src = teammate_cache if teammate_cache is not None else pursuer_states
                mask = np.ones(n_p, dtype=bool)
                mask[j] = False
                tm = src[mask]
                i = int(assignment[j])
                sep_grad = _separation_gradient(
                    pursuer_states[j], tm, self.sep_alpha, self.sep_d_ref,
                    evader_pos=evader_states[i],
                )
                grad_p[j] = grad_p[j] + sep_grad

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
