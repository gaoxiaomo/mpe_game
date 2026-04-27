from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _upper_triangular_count(dim: int) -> int:
    return dim * (dim + 1) // 2


@dataclass(frozen=True)
class NetworkParameterSummary:
    pursuer_actors: int
    evader_actors: int
    pursuer_critics: int
    evader_critics: int
    actor_parameters_per_network: int
    critic_parameters_per_network: int
    total_networks: int
    total_scalar_parameters: int
    augmented_input_dim: int
    critic_basis_dim: int


class TraditionalACEquivalent:
    """Paper-inspired actor-critic/Q-learning baseline.

    The local IEEE brief uses:
    - one actor and one action-dependent critic for each pursuer and evader
    - a quadratic critic basis on the augmented vector z = [x, u, d]
    - a normalized critic update and a gradient actor update

    This implementation keeps that structure, while adapting the paper's
    2D/single-input example to this repository's 6D local-error state and
    3-axis controls.
    """

    def __init__(
        self,
        state_scale: np.ndarray,
        n_p: int,
        n_e: int,
        u_bar_p: float,
        u_bar_e: float,
        q_diag: np.ndarray,
        r1_diag: np.ndarray,
        r2_diag: np.ndarray,
        seed: int = 0,
        critic_lr: float = 1.1,
        actor_lr: float = 0.1,
        critic_regularization: float = 1e-3,
        critic_weight_clip: float = 50.0,
        actor_weight_clip: float = 10.0,
        actor_target_clip: float = 4.0,
    ) -> None:
        self.n_p = int(n_p)
        self.n_e = int(n_e)
        self.state_dim = 6
        self.action_dim = 3
        self.augmented_dim = self.state_dim + 2 * self.action_dim
        self.critic_basis_dim = _upper_triangular_count(self.augmented_dim)

        self.state_scale = np.maximum(np.asarray(state_scale, dtype=float), 1.0)
        if self.state_scale.shape != (self.state_dim,):
            raise ValueError("state_scale must have shape (6,)")

        self.u_bar_p = float(u_bar_p)
        self.u_bar_e = float(u_bar_e)
        self.critic_lr = float(critic_lr)
        self.actor_lr = float(actor_lr)
        self.critic_regularization = float(critic_regularization)
        self.critic_weight_clip = float(critic_weight_clip)
        self.actor_weight_clip = float(actor_weight_clip)
        self.actor_target_clip = float(actor_target_clip)
        self._seed = int(seed)

        self.q_diag = np.asarray(q_diag, dtype=float)
        self.r1_diag = np.asarray(r1_diag, dtype=float)
        self.r2_diag = np.asarray(r2_diag, dtype=float)
        if self.q_diag.shape != (self.state_dim,):
            raise ValueError("q_diag must have shape (6,)")
        if self.r1_diag.shape != (self.action_dim,) or self.r2_diag.shape != (self.action_dim,):
            raise ValueError("r1_diag and r2_diag must have shape (3,)")

        rng = np.random.default_rng(seed)
        self.pursuer_actor = rng.normal(0.0, 0.08, size=(self.n_p, self.action_dim, self.state_dim))
        self.evader_actor = rng.normal(0.0, 0.08, size=(self.n_e, self.action_dim, self.state_dim))
        self.pursuer_critic = rng.normal(0.0, 0.05, size=(self.n_p, self.critic_basis_dim))
        self.evader_critic = rng.normal(0.0, 0.05, size=(self.n_e, self.critic_basis_dim))

        tri_i, tri_j = np.triu_indices(self.augmented_dim)
        self._tri_i = tri_i
        self._tri_j = tri_j
        self._off_diag_mask = (tri_i != tri_j).astype(float)

        self._x_slice = slice(0, self.state_dim)
        self._u_slice = slice(self.state_dim, self.state_dim + self.action_dim)
        self._d_slice = slice(self.state_dim + self.action_dim, self.augmented_dim)

    def copy(self) -> "TraditionalACEquivalent":
        other = TraditionalACEquivalent(
            state_scale=self.state_scale.copy(),
            n_p=self.n_p,
            n_e=self.n_e,
            u_bar_p=self.u_bar_p,
            u_bar_e=self.u_bar_e,
            q_diag=self.q_diag.copy(),
            r1_diag=self.r1_diag.copy(),
            r2_diag=self.r2_diag.copy(),
            seed=self._seed,
            critic_lr=self.critic_lr,
            actor_lr=self.actor_lr,
            critic_regularization=self.critic_regularization,
            critic_weight_clip=self.critic_weight_clip,
            actor_weight_clip=self.actor_weight_clip,
            actor_target_clip=self.actor_target_clip,
        )
        other.pursuer_actor = self.pursuer_actor.copy()
        other.evader_actor = self.evader_actor.copy()
        other.pursuer_critic = self.pursuer_critic.copy()
        other.evader_critic = self.evader_critic.copy()
        return other

    def parameter_summary(self) -> NetworkParameterSummary:
        actor_params = self.action_dim * self.state_dim
        critic_params = self.critic_basis_dim
        total_banks = self.n_p + self.n_e
        return NetworkParameterSummary(
            pursuer_actors=self.n_p,
            evader_actors=self.n_e,
            pursuer_critics=self.n_p,
            evader_critics=self.n_e,
            actor_parameters_per_network=actor_params,
            critic_parameters_per_network=critic_params,
            total_networks=2 * total_banks,
            total_scalar_parameters=total_banks * (actor_params + critic_params),
            augmented_input_dim=self.augmented_dim,
            critic_basis_dim=self.critic_basis_dim,
        )

    def policy_only(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        displacements: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_p, x_e = self._local_errors(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            assignment=assignment,
            displacements=displacements,
        )
        _, u_p = self._actor_forward(self.pursuer_actor, x_p, self.u_bar_p)
        _, u_e = self._actor_forward(self.evader_actor, x_e, self.u_bar_e)
        return u_p, u_e

    def online_qlearning_step(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        next_pursuer_states: np.ndarray,
        next_evader_states: np.ndarray,
        next_assignment: np.ndarray,
        displacements: np.ndarray,
        dt: float,
    ) -> None:
        x_p, x_e = self._local_errors(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            assignment=assignment,
            displacements=displacements,
        )
        next_x_p, next_x_e = self._local_errors(
            pursuer_states=next_pursuer_states,
            evader_states=next_evader_states,
            assignment=next_assignment,
            displacements=displacements,
        )

        pre_p, u_p = self._actor_forward(self.pursuer_actor, x_p, self.u_bar_p)
        pre_e, u_e = self._actor_forward(self.evader_actor, x_e, self.u_bar_e)
        _, next_u_p = self._actor_forward(self.pursuer_actor, next_x_p, self.u_bar_p)
        _, next_u_e = self._actor_forward(self.evader_actor, next_x_e, self.u_bar_e)

        d_p = self._pursuer_disturbance(u_p, u_e, assignment)
        d_e = self._evader_disturbance(u_p, u_e, assignment)
        next_d_p = self._pursuer_disturbance(next_u_p, next_u_e, next_assignment)
        next_d_e = self._evader_disturbance(next_u_p, next_u_e, next_assignment)

        z_p = self._critic_input(x_p, u_p, d_p, self.u_bar_p, self.u_bar_e)
        z_e = self._critic_input(x_e, u_e, d_e, self.u_bar_e, self.u_bar_p)
        z_p_next = self._critic_input(next_x_p, next_u_p, next_d_p, self.u_bar_p, self.u_bar_e)
        z_e_next = self._critic_input(next_x_e, next_u_e, next_d_e, self.u_bar_e, self.u_bar_p)

        phi_p = self._quadratic_features(z_p)
        phi_e = self._quadratic_features(z_e)
        phi_p_next = self._quadratic_features(z_p_next)
        phi_e_next = self._quadratic_features(z_e_next)

        r_p = self._pursuer_stage_cost(x_p, u_p, d_p)
        r_e = self._evader_stage_cost(x_e, u_e, d_e)

        self._critic_update(self.pursuer_critic, phi_p, phi_p_next, r_p, dt)
        self._critic_update(self.evader_critic, phi_e, phi_e_next, r_e, dt)

        target_p = self._actor_targets(self.pursuer_critic, x_p)
        target_e = self._actor_targets(self.evader_critic, x_e)
        self._actor_update(self.pursuer_actor, x_p, pre_p, target_p)
        self._actor_update(self.evader_actor, x_e, pre_e, target_e)

    def _actor_forward(
        self,
        actor_bank: np.ndarray,
        state_errors: np.ndarray,
        u_bar: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        preactivation = np.einsum("nij,nj->ni", actor_bank, state_errors)
        actions = float(u_bar) * np.tanh(preactivation)
        return preactivation, actions

    def _critic_input(
        self,
        state_errors: np.ndarray,
        own_actions: np.ndarray,
        disturbance_actions: np.ndarray,
        own_u_bar: float,
        disturbance_u_bar: float,
    ) -> np.ndarray:
        own_scaled = own_actions / max(float(own_u_bar), 1e-6)
        disturbance_scaled = disturbance_actions / max(float(disturbance_u_bar), 1e-6)
        return np.concatenate([state_errors, own_scaled, disturbance_scaled], axis=1)

    def _local_errors(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        displacements: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assigned = assignment.astype(int, copy=False)
        p_err = pursuer_states - evader_states[assigned] + displacements[np.arange(self.n_p), assigned]
        p_err = p_err / self.state_scale[None, :]

        adjusted_pursuers = pursuer_states - displacements[np.arange(self.n_p), assigned]
        pursuer_centers = np.zeros((self.n_e, self.state_dim), dtype=float)
        counts = np.bincount(assigned, minlength=self.n_e).astype(float)
        np.add.at(pursuer_centers, assigned, adjusted_pursuers)
        valid = counts > 0.0
        if np.any(valid):
            pursuer_centers[valid] /= counts[valid, None]
        default_center = np.mean(adjusted_pursuers, axis=0)
        pursuer_centers[~valid] = default_center

        if self.n_e > 1:
            evader_sum = np.sum(evader_states, axis=0, keepdims=True)
            team_term = evader_states - (evader_sum - evader_states) / max(self.n_e - 1, 1)
        else:
            team_term = np.zeros_like(evader_states)
        opponent_term = evader_states - pursuer_centers
        e_err = (team_term - opponent_term) / self.state_scale[None, :]
        return p_err, e_err

    def _pursuer_disturbance(
        self,
        pursuer_actions: np.ndarray,
        evader_actions: np.ndarray,
        assignment: np.ndarray,
    ) -> np.ndarray:
        disturbances = np.zeros_like(pursuer_actions)
        assigned = assignment.astype(int, copy=False)
        for i in range(self.n_p):
            group = np.where(assigned == assigned[i])[0]
            peers = group[group != i]
            terms = [evader_actions[assigned[i]]]
            if peers.size > 0:
                terms.append(np.mean(pursuer_actions[peers], axis=0))
            disturbances[i] = np.mean(np.vstack(terms), axis=0)
        return disturbances

    def _evader_disturbance(
        self,
        pursuer_actions: np.ndarray,
        evader_actions: np.ndarray,
        assignment: np.ndarray,
    ) -> np.ndarray:
        disturbances = np.zeros((self.n_e, self.action_dim), dtype=float)
        assigned = assignment.astype(int, copy=False)
        evader_mean = np.mean(evader_actions, axis=0) if self.n_e > 0 else np.zeros(self.action_dim, dtype=float)
        for j in range(self.n_e):
            attackers = np.where(assigned == j)[0]
            terms = []
            if attackers.size > 0:
                terms.append(np.mean(pursuer_actions[attackers], axis=0))
            if self.n_e > 1:
                others = np.arange(self.n_e) != j
                terms.append(np.mean(evader_actions[others], axis=0))
            elif terms:
                terms.append(evader_actions[j])
            else:
                terms.append(evader_mean)
            disturbances[j] = np.mean(np.vstack(terms), axis=0)
        return disturbances

    def _quadratic_features(self, z: np.ndarray) -> np.ndarray:
        feats = z[:, self._tri_i] * z[:, self._tri_j]
        if feats.shape[1] != self.critic_basis_dim:
            raise RuntimeError("unexpected critic feature dimension")
        feats[:, self._off_diag_mask.astype(bool)] *= 2.0
        return feats

    def _critic_update(
        self,
        critic_weights: np.ndarray,
        phi_t: np.ndarray,
        phi_tp1: np.ndarray,
        stage_cost: np.ndarray,
        dt: float,
    ) -> None:
        xi = phi_tp1 - phi_t
        ec = np.sum(critic_weights * xi, axis=1) + stage_cost * float(dt)
        norm = 1.0 + np.sum(xi * xi, axis=1)
        critic_weights -= self.critic_lr * ((ec / norm)[:, None] * xi)
        np.clip(critic_weights, -self.critic_weight_clip, self.critic_weight_clip, out=critic_weights)

    def _actor_targets(self, critic_weights: np.ndarray, state_errors: np.ndarray) -> np.ndarray:
        hess = self._weights_to_hessian(critic_weights)
        q_uu = hess[:, self._u_slice, self._u_slice]
        q_ux = hess[:, self._u_slice, self._x_slice]

        eye = np.eye(self.action_dim, dtype=float)[None, :, :]
        q_uu_reg = q_uu + self.critic_regularization * eye
        feedback = np.linalg.solve(q_uu_reg, q_ux)
        targets = -np.einsum("nij,nj->ni", feedback, state_errors)
        return np.clip(targets, -self.actor_target_clip, self.actor_target_clip)

    def _actor_update(
        self,
        actor_bank: np.ndarray,
        state_errors: np.ndarray,
        current_preactivation: np.ndarray,
        target_preactivation: np.ndarray,
    ) -> None:
        actor_error = current_preactivation - target_preactivation
        actor_bank -= self.actor_lr * actor_error[:, :, None] * state_errors[:, None, :]
        np.clip(actor_bank, -self.actor_weight_clip, self.actor_weight_clip, out=actor_bank)

    def _weights_to_hessian(self, critic_weights: np.ndarray) -> np.ndarray:
        count = critic_weights.shape[0]
        hess = np.zeros((count, self.augmented_dim, self.augmented_dim), dtype=float)
        hess[:, self._tri_i, self._tri_j] = critic_weights
        hess[:, self._tri_j, self._tri_i] = critic_weights
        return hess

    def _pursuer_stage_cost(
        self,
        state_errors: np.ndarray,
        own_actions: np.ndarray,
        disturbance_actions: np.ndarray,
    ) -> np.ndarray:
        own_scaled = own_actions / max(self.u_bar_p, 1e-6)
        disturbance_scaled = disturbance_actions / max(self.u_bar_e, 1e-6)
        state_cost = np.sum(self.q_diag[None, :] * state_errors * state_errors, axis=1)
        control_cost = np.sum(self.r1_diag[None, :] * own_scaled * own_scaled, axis=1)
        disturbance_cost = np.sum(self.r2_diag[None, :] * disturbance_scaled * disturbance_scaled, axis=1)
        return state_cost + control_cost - disturbance_cost

    def _evader_stage_cost(
        self,
        state_errors: np.ndarray,
        own_actions: np.ndarray,
        disturbance_actions: np.ndarray,
    ) -> np.ndarray:
        own_scaled = own_actions / max(self.u_bar_e, 1e-6)
        disturbance_scaled = disturbance_actions / max(self.u_bar_p, 1e-6)
        state_cost = np.sum(self.q_diag[None, :] * state_errors * state_errors, axis=1)
        control_cost = np.sum(self.r2_diag[None, :] * own_scaled * own_scaled, axis=1)
        disturbance_cost = np.sum(self.r1_diag[None, :] * disturbance_scaled * disturbance_scaled, axis=1)
        return state_cost + control_cost - disturbance_cost
