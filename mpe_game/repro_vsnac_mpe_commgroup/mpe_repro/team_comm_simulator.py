from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import AircraftParams, ControlParams, LearningParams
from .dynamics import AircraftDynamics
from .team_comm_config import CommunicationParams, TeamCommScenario, TeamFeatureParams
from .team_comm_controller import TeamControlStep, TeamVSNACController
from .team_comm_features import TeamCommunicationFeatureMap
from .team_comm_ls import TeamReplayLeastSquares


@dataclass
class TeamCommTrainResult:
    weights: np.ndarray
    weight_history: np.ndarray
    delta_history: np.ndarray
    residual_history: np.ndarray
    sample_history: np.ndarray
    exploration_history: np.ndarray
    validation_history: np.ndarray
    best_iteration: int


@dataclass
class TeamCommEvalResult:
    pursuer_traj: np.ndarray
    evader_traj: np.ndarray
    pursuer_u: np.ndarray
    evader_u: np.ndarray
    pursuer_u_tanh: np.ndarray
    evader_u_tanh: np.ndarray
    team_errors: np.ndarray
    block_errors: np.ndarray
    estimate_errors: np.ndarray
    communication_ratio: np.ndarray
    communication_matrix: np.ndarray
    capture_time: Optional[float]
    step_logs: Optional[list[dict[str, Any]]] = None


class CommunicationSchedule:
    def __init__(self, params: CommunicationParams, n_agents: int) -> None:
        self.params = params
        self.n_agents = n_agents

    def matrix(self, time_s: float) -> np.ndarray:
        mat = np.ones((self.n_agents, self.n_agents), dtype=bool)
        np.fill_diagonal(mat, True)
        for window in self.params.windows:
            if window.start_s <= time_s <= window.end_s:
                for agent in window.isolated_agents:
                    mat[agent, :] = False
                    mat[:, agent] = False
                    mat[agent, agent] = True
        return mat

    def availability_ratio(self, time_s: float) -> float:
        mat = self.matrix(time_s)
        total = self.n_agents * (self.n_agents - 1)
        active = int(np.sum(mat) - self.n_agents)
        return float(active / max(total, 1))


class TeamCommunicationSimulator:
    def __init__(
        self,
        scenario: TeamCommScenario,
        aircraft_params: AircraftParams,
        control_params: ControlParams,
        learning_params: LearningParams,
        feature_params: TeamFeatureParams,
    ) -> None:
        self.scenario = scenario
        self.learning = learning_params
        self.control_params = control_params
        self.dynamics = AircraftDynamics(aircraft_params)
        self.features = TeamCommunicationFeatureMap(feature_params, n_pursuers=scenario.n_pursuers)
        self.controller = TeamVSNACController(
            dynamics=self.dynamics,
            features=self.features,
            q=control_params.q,
            r1=control_params.r1,
            r2=control_params.r2,
            u_bar_p=control_params.u_bar_p,
            u_bar_e=control_params.u_bar_e,
            u_bar_p_actor=control_params.u_bar_p_actor,
            u_bar_e_actor=control_params.u_bar_e_actor,
            state_cost_scale=np.asarray(feature_params.state_scale, dtype=float),
            nu_eval=scenario.nu_eval,
        )
        self.schedule = CommunicationSchedule(scenario.communication, scenario.n_pursuers)

    def _comm_matrix_for_mode(self, time_s: float, visibility_mode: str) -> np.ndarray:
        if visibility_mode == "full":
            mat = np.ones((self.scenario.n_pursuers, self.scenario.n_pursuers), dtype=bool)
        elif visibility_mode == "dropout":
            mat = self.schedule.matrix(time_s)
        elif visibility_mode == "self_only":
            mat = np.eye(self.scenario.n_pursuers, dtype=bool)
        else:
            raise ValueError(f"Unsupported visibility_mode: {visibility_mode}")
        np.fill_diagonal(mat, True)
        return mat

    def _availability_ratio_for_mode(self, time_s: float, visibility_mode: str) -> float:
        if visibility_mode == "full":
            return 1.0
        if visibility_mode == "dropout":
            return self.schedule.availability_ratio(time_s)
        if visibility_mode == "self_only":
            return 0.0
        raise ValueError(f"Unsupported visibility_mode: {visibility_mode}")

    def _scripted_evader_input(self, time_s: float) -> np.ndarray:
        amp = np.asarray(self.scenario.evader_script_amp, dtype=float)
        omega = float(self.scenario.evader_script_omega)
        decay = float(self.scenario.evader_script_decay)
        u = np.array(
            [
                amp[0] * np.sin(omega * time_s),
                amp[1] * np.sin(1.1 * omega * time_s + 0.8),
                amp[2] * np.sin(0.9 * omega * time_s + 1.6),
            ],
            dtype=float,
        )
        u = u * np.exp(-decay * time_s)
        return np.clip(u, -self.controller.u_bar_e, self.controller.u_bar_e)

    def _applied_evader_input(self, virtual_u: np.ndarray, time_s: float) -> np.ndarray:
        mix = float(np.clip(self.scenario.evader_script_mix, 0.0, 1.0))
        scripted = self._scripted_evader_input(time_s)
        applied = (1.0 - mix) * virtual_u + mix * scripted
        return np.clip(applied, -self.controller.u_bar_e, self.controller.u_bar_e)

    def _reset_states(
        self,
        rng: np.random.Generator,
        perturb_scale: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.scenario.pursuer_init.copy()
        e = self.scenario.evader_init.copy()
        if perturb_scale > 0.0:
            p += rng.normal(0.0, perturb_scale, size=p.shape) * np.maximum(np.abs(p), 1.0)
            e += rng.normal(0.0, perturb_scale, size=e.shape) * np.maximum(np.abs(e), 1.0)
        return p, e

    def _init_weights(self, rng: np.random.Generator) -> np.ndarray:
        local_w = rng.uniform(0.22, 0.34, size=self.features.local_feature_count)
        cross_w = rng.uniform(0.10, 0.20, size=self.features.n_features - self.features.local_feature_count)
        return np.concatenate([local_w, cross_w], axis=0)

    def _initial_estimates(self, true_team_state: np.ndarray) -> np.ndarray:
        return np.repeat(true_team_state[None, :], self.scenario.n_pursuers, axis=0)

    def _build_team_estimates_and_masks(
        self,
        true_team_state: np.ndarray,
        comm_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        updated = np.zeros((self.scenario.n_pursuers, true_team_state.shape[0]), dtype=float)
        masks = np.zeros((self.scenario.n_pursuers, self.scenario.n_pursuers), dtype=float)
        true_blocks = true_team_state.reshape(self.scenario.n_pursuers, 6)
        for receiver in range(self.scenario.n_pursuers):
            for sender in range(self.scenario.n_pursuers):
                if comm_matrix[sender, receiver]:
                    updated[receiver, sender * 6 : (sender + 1) * 6] = true_blocks[sender]
                    masks[receiver, sender] = 1.0
        return updated, masks

    def _single_step(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        team_estimates: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        exploration_std: float,
        time_s: float,
        visibility_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, TeamControlStep, np.ndarray]:
        comm_matrix = self._comm_matrix_for_mode(time_s, visibility_mode)

        true_team_state = self.controller.team_state(
            pursuer_states=pursuer_states,
            evader_state=evader_states[0],
            displacements=self.scenario.displacement_matrix,
        )
        updated_estimates, team_masks = self._build_team_estimates_and_masks(true_team_state, comm_matrix)
        control_step = self.controller.policy(
            pursuer_states=pursuer_states,
            evader_state=evader_states[0],
            displacements=self.scenario.displacement_matrix,
            team_estimates=updated_estimates,
            team_masks=team_masks,
            weights=weights,
            rng=rng,
            exploration_std=exploration_std,
        )
        evader_u_virtual = control_step.evader_u.copy()
        control_step.evader_u = self._applied_evader_input(evader_u_virtual, time_s)
        control_step.stage_cost = self.controller.stage_cost(
            control_step.true_team_state,
            control_step.pursuer_u,
            control_step.evader_u,
        )

        next_p = pursuer_states.copy()
        next_e = evader_states.copy()
        for j in range(self.scenario.n_pursuers):
            next_p[j] = self.dynamics.rk4_step(pursuer_states[j], control_step.pursuer_u[j], self.learning.dt)
        next_e[0] = self.dynamics.rk4_step(evader_states[0], control_step.evader_u, self.learning.dt)

        return next_p, next_e, updated_estimates, control_step, comm_matrix

    def _validation_metric(self, weights: np.ndarray, horizon_s: float = 60.0, visibility_mode: str = "full") -> float:
        p = self.scenario.pursuer_init.copy()
        e = self.scenario.evader_init.copy()
        team_state = self.controller.team_state(p, e[0], self.scenario.displacement_matrix)
        team_estimates = self._initial_estimates(team_state)

        horizon_steps = min(int(horizon_s / self.learning.dt), int(self.scenario.t_final / self.learning.dt))
        accum = 0.0
        for step_idx in range(horizon_steps):
            time_s = float(step_idx * self.learning.dt)
            p, e, team_estimates, step, _ = self._single_step(
                pursuer_states=p,
                evader_states=e,
                team_estimates=team_estimates,
                weights=weights,
                rng=np.random.default_rng(0),
                exploration_std=0.0,
                time_s=time_s,
                visibility_mode=visibility_mode,
            )
            accum += float(step.team_error)
            if float(np.max(step.block_errors)) <= self.scenario.capture_radius:
                capture_time = float(step_idx + 1) * self.learning.dt
                mean_error = accum / float(step_idx + 1)
                return capture_time + 1e-3 * mean_error

        final_state = self.controller.team_state(p, e[0], self.scenario.displacement_matrix)
        final_error = self.controller.team_error(final_state)
        mean_error = accum / float(max(horizon_steps, 1))
        return horizon_s + 5e-3 * mean_error + 1e-2 * final_error

    def train_policy(self, seed: int = 0, visibility_mode: str = "full") -> TeamCommTrainResult:
        rng = np.random.default_rng(seed)
        weights = self._init_weights(rng)
        rollouts_per_iter = max(
            3,
            int(np.ceil(self.learning.min_samples_per_evader / max(self.learning.rollout_steps, 1))),
        )
        weight_hist = [weights.copy()]
        delta_hist = []
        residual_hist = []
        sample_hist = []
        exploration_hist = []
        validation_hist = []
        current_metric = self._validation_metric(weights, visibility_mode=visibility_mode)
        validation_hist.append(np.array([current_metric], dtype=float))

        total_iters = self.learning.policy_iterations
        for s in range(total_iters):
            frac = 1.0 if total_iters == 1 else s / (total_iters - 1)
            explore = self.learning.exploration_std_start + frac * (
                self.learning.exploration_std_end - self.learning.exploration_std_start
            )
            exploration_hist.append(explore)

            replay = TeamReplayLeastSquares(
                n_features=self.features.n_features,
                capacity=rollouts_per_iter * self.learning.rollout_steps,
                local_feature_count=self.features.local_feature_count,
                local_bounds=(0.06, 0.45),
                cross_bounds=(-0.30, 0.30),
                recent_weight_floor=0.75,
            )

            for _ in range(rollouts_per_iter):
                p, e = self._reset_states(rng, self.learning.random_perturb_scale)
                team_state = self.controller.team_state(p, e[0], self.scenario.displacement_matrix)
                team_estimates = self._initial_estimates(team_state)

                for k in range(self.learning.rollout_steps):
                    time_s = float(k * self.learning.dt)
                    phi_t = self.features.phi(team_state)
                    p, e, team_estimates, step, _ = self._single_step(
                        pursuer_states=p,
                        evader_states=e,
                        team_estimates=team_estimates,
                        weights=weights,
                        rng=rng,
                        exploration_std=explore,
                        time_s=time_s,
                        visibility_mode=visibility_mode,
                    )
                    team_state_next = self.controller.team_state(p, e[0], self.scenario.displacement_matrix)
                    phi_tp1 = self.features.phi(team_state_next)
                    replay.add_sample(phi_t=phi_t, phi_tp1=phi_tp1, stage_cost=step.stage_cost, dt=self.learning.dt)
                    team_state = team_state_next

            solved_weights, stats = replay.solve(
                current_weights=weights,
                ridge_lambda=self.learning.ridge_lambda,
                min_samples=self.learning.min_samples_per_evader,
            )

            alpha = float(np.clip(self.learning.critic_learning_rate, 0.0, 1.0)) / (1.0 + 0.15 * s)
            prev_norm_sq = float(np.sum(weights * weights))
            accepted = False
            updated = weights.copy()
            updated_metric = current_metric
            step_alpha = alpha

            for _ in range(8):
                candidate = weights + step_alpha * (solved_weights - weights)
                candidate[: self.features.local_feature_count] = np.clip(
                    candidate[: self.features.local_feature_count], 0.06, 0.45
                )
                candidate[self.features.local_feature_count :] = np.clip(
                    candidate[self.features.local_feature_count :], -0.30, 0.30
                )
                candidate_norm_sq = float(np.sum(candidate * candidate))
                candidate_metric = self._validation_metric(candidate, visibility_mode=visibility_mode)

                # Keep the critic update inside a stable basin: do not accept
                # steps that increase the overall weight energy, and only allow
                # mild validation degradation while exploration shrinks.
                if candidate_norm_sq <= prev_norm_sq + 1.0e-6 and candidate_metric <= current_metric + 0.15:
                    updated = candidate
                    updated_metric = candidate_metric
                    accepted = True
                    break
                step_alpha *= 0.5

            if not accepted:
                updated = weights.copy()
                updated_metric = current_metric

            delta = float(np.linalg.norm(updated - weights))
            weights = updated
            current_metric = updated_metric
            weight_hist.append(weights.copy())
            delta_hist.append(np.array([delta], dtype=float))
            residual_hist.append(np.array([stats.residual_rms], dtype=float))
            sample_hist.append(np.array([stats.sample_count], dtype=float))
            validation_hist.append(np.array([current_metric], dtype=float))

            if delta < self.learning.convergence_tol and stats.sample_count >= self.learning.min_samples_per_evader:
                break

        return TeamCommTrainResult(
            weights=weights,
            weight_history=np.asarray(weight_hist, dtype=float),
            delta_history=np.asarray(delta_hist, dtype=float),
            residual_history=np.asarray(residual_hist, dtype=float),
            sample_history=np.asarray(sample_hist, dtype=float),
            exploration_history=np.asarray(exploration_hist, dtype=float),
            validation_history=np.asarray(validation_hist, dtype=float),
            best_iteration=int(len(weight_hist) - 1),
        )

    def evaluate_policy(
        self,
        weights: np.ndarray,
        seed: int = 1,
        use_dropout: bool = False,
        visibility_mode: str | None = None,
        stop_on_capture: bool = True,
        record_logs: bool = False,
    ) -> TeamCommEvalResult:
        rng = np.random.default_rng(seed)
        p, e = self._reset_states(rng, perturb_scale=0.0)
        team_state = self.controller.team_state(p, e[0], self.scenario.displacement_matrix)
        team_estimates = self._initial_estimates(team_state)

        steps = int(self.scenario.t_final / self.learning.dt)
        p_traj = [p.copy()]
        e_traj = [e.copy()]
        p_u = []
        e_u = []
        p_u_tanh = []
        e_u_tanh = []
        team_errors = []
        block_errors = []
        estimate_errors = []
        communication_ratio = []
        communication_matrix = []
        step_logs: Optional[list[dict[str, Any]]] = [] if record_logs else None
        capture_time: Optional[float] = None

        mode = visibility_mode if visibility_mode is not None else ("dropout" if use_dropout else "full")
        for step_idx in range(steps):
            time_s = float(step_idx * self.learning.dt)
            p_before = p.copy()
            e_before = e.copy()
            true_team_state_before = self.controller.team_state(p_before, e_before[0], self.scenario.displacement_matrix)
            p, e, team_estimates, step, comm_matrix = self._single_step(
                pursuer_states=p,
                evader_states=e,
                team_estimates=team_estimates,
                weights=weights,
                rng=rng,
                exploration_std=0.0,
                time_s=time_s,
                visibility_mode=mode,
            )
            estimate_gap = np.array(
                [
                    np.linalg.norm(team_estimates[j] - true_team_state_before)
                    for j in range(self.scenario.n_pursuers)
                ],
                dtype=float,
            )

            p_traj.append(p.copy())
            e_traj.append(e.copy())
            p_u.append(step.pursuer_u.copy())
            e_u.append(step.evader_u.copy())
            p_u_tanh.append(step.pursuer_u_tanh.copy())
            e_u_tanh.append(step.evader_u_tanh.copy())
            team_errors.append(float(step.team_error))
            block_errors.append(step.block_errors.copy())
            estimate_errors.append(estimate_gap)
            communication_ratio.append(self._availability_ratio_for_mode(time_s, mode))
            communication_matrix.append(comm_matrix.astype(int))

            if step_logs is not None:
                step_logs.append(
                    {
                        "step": int(step_idx),
                        "time_s": time_s,
                        "dropout_enabled": bool(mode == "dropout"),
                        "visibility_mode": mode,
                        "communication_ratio": float(communication_ratio[-1]),
                        "communication_matrix": comm_matrix.astype(int).tolist(),
                        "pursuer_states_before": np.round(p_before, 6).tolist(),
                        "evader_state_before": np.round(e_before[0], 6).tolist(),
                        "true_team_state": np.round(true_team_state_before, 6).tolist(),
                        "team_estimates": np.round(team_estimates, 6).tolist(),
                        "team_masks": np.round(step.team_masks, 6).tolist(),
                        "estimate_errors": np.round(estimate_gap, 6).tolist(),
                        "block_gradients_true": np.round(step.block_gradients_true, 6).tolist(),
                        "block_gradients_est": np.round(step.block_gradients_est, 6).tolist(),
                        "block_errors": np.round(step.block_errors, 6).tolist(),
                        "team_error": float(step.team_error),
                        "stage_cost": float(step.stage_cost),
                        "u_p": np.round(step.pursuer_u, 6).tolist(),
                        "u_e_applied": np.round(step.evader_u, 6).tolist(),
                        "u_e_virtual": np.round(step.evader_u_tanh, 6).tolist(),
                        "weights": np.round(weights, 6).tolist(),
                    }
                )

            if capture_time is None and float(np.max(step.block_errors)) <= self.scenario.capture_radius:
                capture_time = time_s
                if stop_on_capture:
                    break

        return TeamCommEvalResult(
            pursuer_traj=np.asarray(p_traj, dtype=float),
            evader_traj=np.asarray(e_traj, dtype=float),
            pursuer_u=np.asarray(p_u, dtype=float),
            evader_u=np.asarray(e_u, dtype=float),
            pursuer_u_tanh=np.asarray(p_u_tanh, dtype=float),
            evader_u_tanh=np.asarray(e_u_tanh, dtype=float),
            team_errors=np.asarray(team_errors, dtype=float),
            block_errors=np.asarray(block_errors, dtype=float),
            estimate_errors=np.asarray(estimate_errors, dtype=float),
            communication_ratio=np.asarray(communication_ratio, dtype=float),
            communication_matrix=np.asarray(communication_matrix, dtype=int),
            capture_time=capture_time,
            step_logs=step_logs,
        )
