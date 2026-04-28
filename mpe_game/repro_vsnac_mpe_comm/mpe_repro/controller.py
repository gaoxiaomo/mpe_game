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


def nonquadratic_integral_cost_batch(u: np.ndarray, u_bar: float, r_diag: np.ndarray) -> np.ndarray:
    ratio = np.clip(u / u_bar, -0.999999, 0.999999)
    term = u * safe_atanh(ratio) + 0.5 * u_bar * np.log(1.0 - ratio * ratio)
    return np.sum(2.0 * u_bar * r_diag[None, :] * term, axis=1)


def smooth_pair_amplification(dist_sq: np.ndarray, d_safe: float) -> np.ndarray:
    """Smooth approximation of max(1, d_safe^2 / dist_sq)."""
    if d_safe <= 0.0:
        return np.ones_like(dist_sq, dtype=float)
    ratio = (d_safe * d_safe) / np.maximum(dist_sq, 1e-8)
    delta = ratio - 1.0
    return 1.0 + 0.5 * (delta + np.sqrt(delta * delta + 1e-4))


def coordination_potential_and_gradient(
    pursuer_states: np.ndarray,
    A_p: np.ndarray | None,
    delta_matrix: np.ndarray | None,
    gamma: float,
    ref_dist: float = 500.0,
    d_safe: float = 0.0,
    degree_norm: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pursuer coordination potential and its state gradient.

    At the exact level we decompose the value function as

        V_j^*(x) = Vbar_j^*(x) + Phi_j(x),

    and only approximate the residual value Vbar_j^* with the critic.

    The coordination potential is

        Phi_j = (gamma / norm(d_j)) * sum_k a_jk * amp_jk *
                [0.5 ||v_j - v_k||^2 +
                 ((p_j - p_k) - Delta_jk)^T v_j / ref_dist].

    The ``degree_norm`` argument controls how the per-agent group size
    ``d_j`` enters the prefactor:

    - ``"linear"`` (default): ``Phi_j ∝ 1/d_j``. Coordination strength is
      *per-pair*, group-size-invariant. This is the original Xu-style
      averaging that was found to under-coordinate larger groups
      (degree-dilution).
    - ``"sqrt"``: ``Phi_j ∝ 1/sqrt(d_j)``. Sublinear scaling (Olfati-Saber
      consensus convention) that partially compensates for dilution while
      avoiding tanh saturation in very large groups.
    - ``"none"``: no degree normalisation. ``Phi_j ∝ 1`` (sums all pair
      contributions). Strongest coordination, may saturate tanh in groups
      with d_j > ~5 but works very well for d_j <= 4.

    Because g(x)^T extracts only the velocity-space component, the
    returned gradient is zero on the position channels and nonzero on the
    velocity channels.
    """
    n_p = pursuer_states.shape[0]
    potentials = np.zeros(n_p, dtype=float)
    gradients = np.zeros((n_p, 6), dtype=float)
    if gamma <= 0.0 or A_p is None:
        return potentials, gradients

    degree = np.sum(A_p, axis=1)
    valid = degree > 0.0
    if not np.any(valid):
        return potentials, gradients

    if degree_norm == "linear":
        norm_factor = degree[valid]
    elif degree_norm == "sqrt":
        norm_factor = np.sqrt(degree[valid])
    elif degree_norm == "none":
        norm_factor = np.ones_like(degree[valid])
    else:
        raise ValueError(f"unknown degree_norm: {degree_norm!r}")

    pos = pursuer_states[:, :3]
    vel = pursuer_states[:, 3:]
    sep = pos[:, None, :] - pos[None, :, :]
    delta_p = np.zeros_like(sep) if delta_matrix is None else delta_matrix[:, :, :3]
    pos_err = sep - delta_p
    dv = vel[:, None, :] - vel[None, :, :]

    dist_sq = np.sum(sep * sep, axis=2)
    amp = smooth_pair_amplification(dist_sq, d_safe)

    pair_weight = A_p * amp
    np.fill_diagonal(pair_weight, 0.0)

    pair_grad = dv + pos_err / ref_dist
    grad_v = np.sum(pair_weight[:, :, None] * pair_grad, axis=1)
    gradients[valid, 3:] = gamma * grad_v[valid] / norm_factor[:, None]

    pair_energy = 0.5 * np.sum(dv * dv, axis=2) + np.sum(pos_err * vel[:, None, :], axis=2) / ref_dist
    potentials[valid] = gamma * np.sum(pair_weight * pair_energy, axis=1)[valid] / norm_factor
    return potentials, gradients


def formation_gradient(
    j: int,
    pursuer_states: np.ndarray,
    A_p: np.ndarray | None,
    delta_matrix: np.ndarray | None,
    gamma: float,
    ref_dist: float = 500.0,
    d_safe: float = 0.0,
) -> np.ndarray:
    """Single-agent wrapper kept for backward compatibility."""
    _, gradients = coordination_potential_and_gradient(
        pursuer_states=pursuer_states,
        A_p=A_p,
        delta_matrix=delta_matrix,
        gamma=gamma,
        ref_dist=ref_dist,
        d_safe=d_safe,
    )
    return gradients[j]


class CommVSNACController:
    """Communication-aware V-SNAC controller with a structured value decomposition.

    The exact theory first decomposes the value as

        V_j^*(x) = Vbar_j^*(x) + Phi_j(x),

    and the implementation only approximates Vbar_j^* with the critic:

        Vhat_j(x) = W_j^T psi(x_tilde_j) + Phi_j(x).

    The control law is derived from the gradient of this total approximate
    value, rather than from an external post-hoc gradient correction.
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
        self.q_diag = np.diag(q)
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
        """Standard tracking error."""
        return (
            pursuer_states[pursuer_idx]
            - evader_states[evader_idx]
            + displacements[pursuer_idx, evader_idx]
        )

    def value_gradient(self, x_tilde: np.ndarray, w: np.ndarray) -> np.ndarray:
        jac = self.features.jacobian(x_tilde)
        return jac.T @ w

    def value_gradients(self, x_tilde: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.features.value_gradient_batch(x_tilde, w)

    def coordination_terms(
        self,
        pursuer_states: np.ndarray,
        A_p: np.ndarray | None,
        delta_matrix: np.ndarray | None,
        gamma: float,
        formation_ref_dist: float = 500.0,
        d_safe: float = 0.0,
        degree_norm: str = "linear",
    ) -> tuple[np.ndarray, np.ndarray]:
        return coordination_potential_and_gradient(
            pursuer_states=pursuer_states,
            A_p=A_p,
            delta_matrix=delta_matrix,
            gamma=gamma,
            ref_dist=formation_ref_dist,
            d_safe=d_safe,
            degree_norm=degree_norm,
        )

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
        coordination_gradients: np.ndarray | None = None,
        degree_norm: str = "linear",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_p = pursuer_states.shape[0]
        n_e = evader_states.shape[0]

        assigned = assignment.astype(int, copy=False)
        x_err = pursuer_states - evader_states[assigned] + displacements[np.arange(n_p), assigned]
        weight_matrix = np.vstack(weights)
        grad_p = self.value_gradients(x_err, weight_matrix)

        total_grad = grad_p.copy()
        if coordination_gradients is None and gamma > 0.0 and A_p is not None:
            _, coordination_gradients = self.coordination_terms(
                pursuer_states=pursuer_states,
                A_p=A_p,
                delta_matrix=delta_matrix,
                gamma=gamma,
                formation_ref_dist=formation_ref_dist,
                d_safe=d_safe,
                degree_norm=degree_norm,
            )
        if coordination_gradients is not None:
            total_grad += coordination_gradients

        proj_p = self.dynamics.g_transpose_dot_batch(pursuer_states, total_grad)
        rho_p = (proj_p @ self.r1_inv.T) / (2.0 * self.u_bar_p_actor)
        u_p_tanh = -self.u_bar_p_actor * np.tanh(rho_p)
        u_p = u_p_tanh.copy()
        if exploration_std > 0.0:
            u_p += rng.normal(0.0, exploration_std, size=u_p.shape)
        u_p = np.clip(u_p, -self.u_bar_p, self.u_bar_p)

        grad_mean = np.zeros((n_e, 6), dtype=float)
        counts = np.bincount(assigned, minlength=n_e).astype(float)
        np.add.at(grad_mean, assigned, grad_p)
        valid = counts > 0.0
        grad_mean[valid] /= counts[valid, None]

        proj_e = self.dynamics.g_transpose_dot_batch(evader_states, grad_mean)
        rho_e = (proj_e @ self.r2_inv.T) / (2.0 * self.u_bar_e_actor)
        u_e_tanh = -self.u_bar_e_actor * np.tanh(rho_e)
        u_e = u_e_tanh.copy()
        if exploration_std > 0.0:
            u_e += rng.normal(0.0, exploration_std, size=u_e.shape)
        u_e = np.clip(u_e, -self.u_bar_e, self.u_bar_e)

        return u_p, u_e, x_err, u_p_tanh, u_e_tanh

    def stage_costs(
        self,
        x_err: np.ndarray,
        u_p: np.ndarray,
        u_e: np.ndarray,
        assignment: np.ndarray,
    ) -> np.ndarray:
        x_norm = x_err / self.state_cost_scale[None, :]
        state_cost = np.sum(x_norm * self.q_diag[None, :] * x_norm, axis=1)
        pursuer_cost = nonquadratic_integral_cost_batch(u_p, self.u_bar_p, self.r1_diag)
        evader_cost = nonquadratic_integral_cost_batch(u_e, self.u_bar_e, self.r2_diag)
        return state_cost + pursuer_cost - evader_cost[assignment.astype(int, copy=False)]
