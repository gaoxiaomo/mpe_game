from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import AircraftParams, ControlParams, LearningParams
from .dynamics import AircraftDynamics
from .team_comm_config import TeamFeatureParams
from .team_comm_controller import TeamVSNACController
from .team_comm_features import TeamCommunicationFeatureMap
from .team_comm_ls import TeamReplayLeastSquares
from .team_comm_multi_config import (
    GroupCommunicationParams,
    GroupCommunicationWindow,
    TeamCommMultiScenario,
    sample_group_communication_windows,
)
from .team_comm_multi_graph import DynamicGroupSlotGraph


@dataclass
class TeamCommMultiTrainResult:
    weights: list[np.ndarray]
    weight_history: list[list[np.ndarray]]
    weight_norm_history: np.ndarray
    delta_history: np.ndarray
    residual_history: np.ndarray
    sample_history: np.ndarray
    exploration_history: np.ndarray
    validation_history: np.ndarray
    best_iteration: int


@dataclass
class TeamCommMultiEvalResult:
    pursuer_traj: np.ndarray
    evader_traj: np.ndarray
    pursuer_u: np.ndarray
    evader_u: np.ndarray
    pursuer_u_tanh: np.ndarray
    evader_u_tanh: np.ndarray
    team_errors: np.ndarray
    group_errors: np.ndarray
    assigned_errors: np.ndarray
    assignment_history: np.ndarray
    communication_ratio: np.ndarray
    estimate_errors: np.ndarray
    capture_time: Optional[float]
    communication_windows: list[dict[str, Any]]
    step_logs: Optional[list[dict[str, Any]]] = None


@dataclass
class MultiTeamStepRecord:
    phi_t: list[np.ndarray]
    phi_tp1: list[np.ndarray]
    stage_costs: np.ndarray
    group_errors: np.ndarray
    total_team_error: float
    assigned_errors: np.ndarray
    assignment: np.ndarray
    pursuer_u: np.ndarray
    evader_u: np.ndarray
    pursuer_u_tanh: np.ndarray
    evader_u_tanh: np.ndarray
    communication_ratio: np.ndarray
    estimate_errors: np.ndarray
    group_members: list[np.ndarray]
    communication_matrices: list[np.ndarray]


class GroupCommunicationSchedule:
    def __init__(self, params: GroupCommunicationParams, group_sizes: np.ndarray) -> None:
        self.params = params
        self.group_sizes = np.asarray(group_sizes, dtype=int)
        self.windows_by_group: dict[int, list[Any]] = {}
        for window in self.params.windows:
            self.windows_by_group.setdefault(int(window.group_idx), []).append(window)

    def matrix(self, group_idx: int, time_s: float) -> np.ndarray:
        group_size = int(self.group_sizes[int(group_idx)])
        mat = np.ones((group_size, group_size), dtype=bool)
        np.fill_diagonal(mat, True)
        for window in self.windows_by_group.get(int(group_idx), []):
            if window.start_s <= time_s <= window.end_s:
                for slot_idx in window.isolated_slots:
                    s = int(slot_idx)
                    if s < 0 or s >= group_size:
                        continue
                    mat[s, :] = False
                    mat[:, s] = False
                    mat[s, s] = True
        return mat

    def availability_ratio(self, group_idx: int, time_s: float) -> float:
        mat = self.matrix(group_idx, time_s)
        group_size = mat.shape[0]
        if group_size <= 1:
            return 1.0
        total = group_size * (group_size - 1)
        active = int(np.sum(mat) - group_size)
        return float(active / max(total, 1))


class MultiTeamCommunicationSimulator:
    def __init__(
        self,
        scenario: TeamCommMultiScenario,
        aircraft_params: AircraftParams,
        control_params: ControlParams,
        learning_params: LearningParams,
        feature_params: TeamFeatureParams,
    ) -> None:
        self.scenario = scenario
        self.learning = learning_params
        self.control_params = control_params
        self.dynamics = AircraftDynamics(aircraft_params)
        self.nu_eval = np.asarray(self.scenario.nu_eval, dtype=float)
        self.graph_metric = self.nu_eval.copy()

        self.group_sizes = self.scenario.group_sizes
        self.group_features: list[TeamCommunicationFeatureMap] = []
        self.group_controllers: list[TeamVSNACController] = []
        self.group_slot_displacements: list[np.ndarray] = []
        for evader_idx, slot_ids in enumerate(self.scenario.group_slot_ids):
            group_size = int(slot_ids.shape[0])
            features = TeamCommunicationFeatureMap(feature_params, n_pursuers=group_size)
            controller = TeamVSNACController(
                dynamics=self.dynamics,
                features=features,
                q=control_params.q,
                r1=control_params.r1,
                r2=control_params.r2,
                u_bar_p=control_params.u_bar_p,
                u_bar_e=control_params.u_bar_e,
                u_bar_p_actor=control_params.u_bar_p_actor,
                u_bar_e_actor=control_params.u_bar_e_actor,
                state_cost_scale=np.asarray(feature_params.state_scale, dtype=float),
                nu_eval=self.nu_eval,
            )
            self.group_features.append(features)
            self.group_controllers.append(controller)
            disp = self.scenario.slot_displacements[np.asarray(slot_ids, dtype=int)][:, None, :]
            self.group_slot_displacements.append(disp)

        self.schedule = GroupCommunicationSchedule(self.scenario.communication, self.group_sizes)

    def _make_schedule(
        self,
        rng: np.random.Generator,
        *,
        randomize_episode: bool,
    ) -> GroupCommunicationSchedule:
        params = self.scenario.communication
        windows = params.windows
        if randomize_episode and params.randomize_each_episode and params.windows_per_group > 0:
            windows = sample_group_communication_windows(
                t_final=float(self.scenario.t_final),
                group_slot_ids=self.scenario.group_slot_ids,
                rng=rng,
                windows_per_group=params.windows_per_group,
                dropout_fraction=params.dropout_fraction,
                full_group_prob=params.full_group_prob,
            )
        schedule_params = GroupCommunicationParams(
            windows=tuple(windows),
            windows_per_group=params.windows_per_group,
            dropout_fraction=params.dropout_fraction,
            full_group_prob=params.full_group_prob,
            randomize_each_episode=params.randomize_each_episode,
        )
        return GroupCommunicationSchedule(schedule_params, self.group_sizes)

    @staticmethod
    def _windows_payload(windows: tuple[GroupCommunicationWindow, ...]) -> list[dict[str, Any]]:
        return [
            {
                "group_idx": int(window.group_idx),
                "start_s": float(window.start_s),
                "end_s": float(window.end_s),
                "isolated_slots": [int(v) for v in window.isolated_slots],
            }
            for window in windows
        ]

    def _new_graph(self) -> DynamicGroupSlotGraph:
        return DynamicGroupSlotGraph(
            initial_assignment=self.scenario.initial_assignment,
            slot_evaders=self.scenario.slot_evaders,
            group_slot_ids=self.scenario.group_slot_ids,
        )

    def _comm_matrix_for_group(self, group_idx: int, time_s: float, visibility_mode: str) -> np.ndarray:
        group_size = int(self.group_sizes[int(group_idx)])
        if visibility_mode == "full":
            mat = np.ones((group_size, group_size), dtype=bool)
        elif visibility_mode == "dropout":
            mat = self.schedule.matrix(group_idx, time_s)
        else:
            raise ValueError(f"Unsupported visibility_mode: {visibility_mode}")
        np.fill_diagonal(mat, True)
        return mat

    def _availability_ratio_for_group(self, group_idx: int, time_s: float, visibility_mode: str) -> float:
        if visibility_mode == "full":
            return 1.0
        if visibility_mode == "dropout":
            return self.schedule.availability_ratio(group_idx, time_s)
        raise ValueError(f"Unsupported visibility_mode: {visibility_mode}")

    def _build_team_estimates_and_masks(
        self,
        true_team_state: np.ndarray,
        comm_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        group_size = int(comm_matrix.shape[0])
        updated = np.zeros((group_size, true_team_state.shape[0]), dtype=float)
        masks = np.zeros((group_size, group_size), dtype=float)
        true_blocks = true_team_state.reshape(group_size, 6)
        for receiver in range(group_size):
            for sender in range(group_size):
                if comm_matrix[sender, receiver]:
                    updated[receiver, sender * 6 : (sender + 1) * 6] = true_blocks[sender]
                    masks[receiver, sender] = 1.0
        return updated, masks

    def _scripted_evader_input(self, step_idx: int, evader_idx: int) -> np.ndarray:
        t = float(step_idx) * self.learning.dt
        amp = np.asarray(self.scenario.evader_script_amp, dtype=float)
        omega = float(self.scenario.evader_script_omega)
        decay = float(self.scenario.evader_script_decay)
        phase = 0.9 * float(evader_idx)
        u = np.array(
            [
                amp[0] * np.sin(omega * t + phase),
                amp[1] * np.sin(1.1 * omega * t + phase + 0.8),
                amp[2] * np.sin(0.9 * omega * t + phase + 1.6),
            ],
            dtype=float,
        )
        u = u * np.exp(-decay * t)
        return np.clip(u, -self.control_params.u_bar_e, self.control_params.u_bar_e)

    def _applied_evader_inputs(self, step_idx: int, virtual_u: np.ndarray) -> np.ndarray:
        mode = self.scenario.evader_motion_mode
        if mode == "policy":
            return virtual_u.copy()

        if mode == "scripted":
            mix = float(np.clip(self.scenario.evader_script_mix, 0.0, 1.0))
            applied = virtual_u.copy()
            for evader_idx in range(applied.shape[0]):
                scripted = self._scripted_evader_input(step_idx, evader_idx)
                applied[evader_idx] = np.clip(
                    (1.0 - mix) * virtual_u[evader_idx] + mix * scripted,
                    -self.control_params.u_bar_e,
                    self.control_params.u_bar_e,
                )
            return applied

        raise ValueError(f"Unsupported evader_motion_mode: {mode}")

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

    def _graph_states_for_update(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tau = float(max(self.scenario.swap_lookahead_time, 0.0))
        if tau <= 0.0:
            return pursuer_states, evader_states
        p_pred = pursuer_states.copy()
        e_pred = evader_states.copy()
        p_pred[:, :3] = p_pred[:, :3] + tau * p_pred[:, 3:6]
        e_pred[:, :3] = e_pred[:, :3] + tau * e_pred[:, 3:6]
        return p_pred, e_pred

    def _init_weights(self, rng: np.random.Generator) -> list[np.ndarray]:
        weights: list[np.ndarray] = []
        for features in self.group_features:
            local_w = rng.uniform(0.15, 0.30, size=features.local_feature_count)
            cross_w = rng.uniform(0.02, 0.12, size=features.n_features - features.local_feature_count)
            weights.append(np.concatenate([local_w, cross_w], axis=0))
        return weights

    def _clip_weights(self, group_idx: int, weights: np.ndarray) -> np.ndarray:
        out = weights.copy()
        local_count = self.group_features[group_idx].local_feature_count
        out[:local_count] = np.clip(out[:local_count], 0.001, 0.80)
        out[local_count:] = np.clip(out[local_count:], -0.60, 0.60)
        return out

    def _weight_norm_sq(self, weights: list[np.ndarray]) -> float:
        return float(sum(np.sum(w * w) for w in weights))

    def _weight_norms(self, weights: list[np.ndarray]) -> np.ndarray:
        return np.asarray([float(np.linalg.norm(w)) for w in weights], dtype=float)

    def _step(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        graph: DynamicGroupSlotGraph,
        weights: list[np.ndarray],
        rng: np.random.Generator,
        exploration_std: float,
        dynamic_graph: bool,
        visibility_mode: str,
        step_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, MultiTeamStepRecord]:
        start_step = max(int(self.learning.graph_update_start_step), 0)
        interval = max(int(self.learning.graph_update_interval), 1)
        if dynamic_graph and (step_idx >= start_step) and ((step_idx - start_step) % interval == 0):
            p_for_graph, e_for_graph = self._graph_states_for_update(pursuer_states, evader_states)
            graph.update(
                pursuer_states=p_for_graph,
                evader_states=e_for_graph,
                slot_displacements=self.scenario.slot_displacements,
                switch_threshold=self.scenario.swap_threshold,
                max_switch_worsening=self.scenario.max_switch_worsening,
                nu=self.graph_metric,
            )

        slot_costs = graph.cost_matrix_to_slots(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            slot_displacements=self.scenario.slot_displacements,
            nu=self.nu_eval,
        )
        assignment = graph.assignment
        assigned_errors = graph.assigned_costs(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            slot_displacements=self.scenario.slot_displacements,
            nu=self.nu_eval,
            slot_costs=slot_costs,
        )

        pursuer_u = np.zeros((self.scenario.n_pursuers, 3), dtype=float)
        pursuer_u_tanh = np.zeros((self.scenario.n_pursuers, 3), dtype=float)
        evader_u_virtual = np.zeros((self.scenario.n_evaders, 3), dtype=float)
        evader_u_tanh = np.zeros((self.scenario.n_evaders, 3), dtype=float)
        stage_costs = np.zeros(self.scenario.n_evaders, dtype=float)
        group_errors = np.zeros(self.scenario.n_evaders, dtype=float)
        communication_ratio = np.ones(self.scenario.n_evaders, dtype=float)
        estimate_errors = np.zeros(self.scenario.n_evaders, dtype=float)

        phi_t: list[np.ndarray] = []
        phi_tp1: list[np.ndarray] = []
        group_members: list[np.ndarray] = []
        communication_matrices: list[np.ndarray] = []
        group_true_states: list[np.ndarray] = []

        for evader_idx in range(self.scenario.n_evaders):
            members = graph.group_members(evader_idx)
            group_members.append(members.copy())
            controller = self.group_controllers[evader_idx]
            features = self.group_features[evader_idx]
            displacements = self.group_slot_displacements[evader_idx]
            p_group = pursuer_states[members]
            e_state = evader_states[evader_idx]

            true_team_state = controller.team_state(
                pursuer_states=p_group,
                evader_state=e_state,
                displacements=displacements,
            )
            group_true_states.append(true_team_state.copy())
            phi_t.append(features.phi(true_team_state))

            comm_matrix = self._comm_matrix_for_group(evader_idx, float(step_idx * self.learning.dt), visibility_mode)
            communication_matrices.append(comm_matrix.astype(int))
            team_estimates, team_masks = self._build_team_estimates_and_masks(true_team_state, comm_matrix)
            step = controller.policy(
                pursuer_states=p_group,
                evader_state=e_state,
                displacements=displacements,
                team_estimates=team_estimates,
                team_masks=team_masks,
                weights=weights[evader_idx],
                rng=rng,
                exploration_std=exploration_std,
            )

            pursuer_u[members] = step.pursuer_u
            pursuer_u_tanh[members] = step.pursuer_u_tanh
            evader_u_virtual[evader_idx] = step.evader_u
            evader_u_tanh[evader_idx] = step.evader_u_tanh
            group_errors[evader_idx] = float(step.team_error)
            communication_ratio[evader_idx] = self._availability_ratio_for_group(
                evader_idx,
                float(step_idx * self.learning.dt),
                visibility_mode,
            )
            estimate_errors[evader_idx] = float(
                np.mean([np.linalg.norm(team_estimates[j] - true_team_state) for j in range(team_estimates.shape[0])])
            )

        evader_u_applied = self._applied_evader_inputs(step_idx, evader_u_virtual)

        next_p = pursuer_states.copy()
        next_e = evader_states.copy()
        for pursuer_idx in range(self.scenario.n_pursuers):
            next_p[pursuer_idx] = self.dynamics.rk4_step(pursuer_states[pursuer_idx], pursuer_u[pursuer_idx], self.learning.dt)
        for evader_idx in range(self.scenario.n_evaders):
            next_e[evader_idx] = self.dynamics.rk4_step(evader_states[evader_idx], evader_u_applied[evader_idx], self.learning.dt)

        for evader_idx in range(self.scenario.n_evaders):
            controller = self.group_controllers[evader_idx]
            features = self.group_features[evader_idx]
            displacements = self.group_slot_displacements[evader_idx]
            members = group_members[evader_idx]
            next_true_state = controller.team_state(
                pursuer_states=next_p[members],
                evader_state=next_e[evader_idx],
                displacements=displacements,
            )
            phi_tp1.append(features.phi(next_true_state))
            stage_costs[evader_idx] = controller.stage_cost(
                team_state=group_true_states[evader_idx],
                u_p=pursuer_u[members],
                u_e=evader_u_applied[evader_idx],
            )

        total_team_error = float(np.sum(assigned_errors))
        record = MultiTeamStepRecord(
            phi_t=phi_t,
            phi_tp1=phi_tp1,
            stage_costs=stage_costs,
            group_errors=group_errors,
            total_team_error=total_team_error,
            assigned_errors=assigned_errors,
            assignment=assignment,
            pursuer_u=pursuer_u,
            evader_u=evader_u_applied,
            pursuer_u_tanh=pursuer_u_tanh,
            evader_u_tanh=evader_u_tanh,
            communication_ratio=communication_ratio,
            estimate_errors=estimate_errors,
            group_members=group_members,
            communication_matrices=communication_matrices,
        )
        return next_p, next_e, record

    def _validation_metric(
        self,
        weights: list[np.ndarray],
        *,
        visibility_mode: str,
        dynamic_graph: bool,
        horizon_s: float | None = None,
    ) -> float:
        p = self.scenario.pursuer_init.copy()
        e = self.scenario.evader_init.copy()
        graph = self._new_graph()
        rng = np.random.default_rng(0)
        self.schedule = self._make_schedule(rng, randomize_episode=(visibility_mode == "dropout"))

        if horizon_s is None:
            horizon_s = min(25.0, max(12.0, 0.18 * float(self.scenario.t_final)))
        horizon_steps = min(int(horizon_s / self.learning.dt), int(self.scenario.t_final / self.learning.dt))
        accum = 0.0
        for step_idx in range(horizon_steps):
            p, e, step = self._step(
                pursuer_states=p,
                evader_states=e,
                graph=graph,
                weights=weights,
                rng=rng,
                exploration_std=0.0,
                dynamic_graph=dynamic_graph,
                visibility_mode=visibility_mode,
                step_idx=step_idx,
            )
            accum += float(step.total_team_error)
            if float(np.max(step.assigned_errors)) <= self.scenario.capture_radius:
                capture_time = float(step_idx + 1) * self.learning.dt
                mean_error = accum / float(step_idx + 1)
                return capture_time + 1.0e-3 * mean_error

        mean_error = accum / float(max(horizon_steps, 1))
        return horizon_s + 5.0e-3 * mean_error + 1.0e-2 * float(step.total_team_error)

    def train_policy(
        self,
        seed: int = 0,
        *,
        dynamic_graph: bool = True,
        visibility_mode: str = "dropout",
        initial_weights: Optional[list[np.ndarray]] = None,
    ) -> TeamCommMultiTrainResult:
        rng = np.random.default_rng(seed)
        if initial_weights is None:
            weights = self._init_weights(rng)
        else:
            weights = [np.asarray(w, dtype=float).copy() for w in initial_weights]
        rollouts_per_iter = max(
            3,
            int(np.ceil(self.learning.min_samples_per_evader / max(self.learning.rollout_steps, 1))),
        )

        buffer_capacity = max(rollouts_per_iter * self.learning.rollout_steps * 8, 30000)
        replays: list[TeamReplayLeastSquares] = []
        for group_idx, features in enumerate(self.group_features):
            # Stronger cross-feature ridge for larger groups to stabilise
            # the coupled team Bellman LS.
            cross_scale = 4.0 if features.cross_feature_count <= 3 else 6.0
            replays.append(
                TeamReplayLeastSquares(
                    n_features=features.n_features,
                    capacity=buffer_capacity,
                    local_feature_count=features.local_feature_count,
                    local_bounds=(0.001, 0.80),
                    cross_bounds=(-0.60, 0.60),
                    recent_weight_floor=0.35,
                    cross_ridge_scale=cross_scale,
                    svd_threshold=1e-4,
                )
            )

        weight_history: list[list[np.ndarray]] = [[w.copy() for w in weights]]
        weight_norm_history = [self._weight_norms(weights)]
        delta_history: list[np.ndarray] = []
        residual_history: list[np.ndarray] = []
        sample_history: list[np.ndarray] = []
        exploration_history: list[float] = []
        validation_history: list[np.ndarray] = []

        total_iters = self.learning.policy_iterations
        curriculum_iters = max(1, int(np.ceil(0.55 * total_iters)))

        def rollout_mode_for_iteration(iteration: int) -> str:
            if visibility_mode != "curriculum":
                return visibility_mode
            return "full" if iteration < curriculum_iters else "dropout"

        current_validation_mode = rollout_mode_for_iteration(0)
        current_metric = self._validation_metric(
            weights,
            visibility_mode=current_validation_mode,
            dynamic_graph=dynamic_graph,
        )
        validation_history.append(np.array([current_metric], dtype=float))
        best_weights = [w.copy() for w in weights]
        best_iteration = 0
        best_metric = float("inf")
        selection_mode = "dropout" if visibility_mode == "curriculum" else current_validation_mode
        if current_validation_mode == selection_mode:
            best_metric = float(current_metric)
        for iteration in range(total_iters):
            iter_visibility_mode = rollout_mode_for_iteration(iteration)
            if iter_visibility_mode != current_validation_mode:
                current_validation_mode = iter_visibility_mode
                current_metric = self._validation_metric(
                    weights,
                    visibility_mode=current_validation_mode,
                    dynamic_graph=dynamic_graph,
                )
            frac = 1.0 if total_iters == 1 else iteration / (total_iters - 1)
            explore = self.learning.exploration_std_start + frac * (
                self.learning.exploration_std_end - self.learning.exploration_std_start
            )
            exploration_history.append(explore)

            for _ in range(rollouts_per_iter):
                self.schedule = self._make_schedule(
                    rng,
                    randomize_episode=(iter_visibility_mode == "dropout"),
                )
                p, e = self._reset_states(rng, self.learning.random_perturb_scale)
                graph = self._new_graph()
                for step_idx in range(self.learning.rollout_steps):
                    p, e, step = self._step(
                        pursuer_states=p,
                        evader_states=e,
                        graph=graph,
                        weights=weights,
                        rng=rng,
                        exploration_std=explore,
                        dynamic_graph=dynamic_graph,
                        visibility_mode=iter_visibility_mode,
                        step_idx=step_idx,
                    )
                    for group_idx, replay in enumerate(replays):
                        replay.add_sample(
                            phi_t=step.phi_t[group_idx],
                            phi_tp1=step.phi_tp1[group_idx],
                            stage_cost=float(step.stage_costs[group_idx]),
                            dt=self.learning.dt,
                        )

            solved_weights: list[np.ndarray] = []
            residual_row = np.zeros(self.scenario.n_evaders, dtype=float)
            sample_row = np.zeros(self.scenario.n_evaders, dtype=float)
            for group_idx, replay in enumerate(replays):
                solved, stats = replay.solve(
                    current_weights=weights[group_idx],
                    ridge_lambda=self.learning.ridge_lambda,
                    min_samples=self.learning.min_samples_per_evader,
                )
                solved_weights.append(self._clip_weights(group_idx, solved))
                residual_row[group_idx] = float(stats.residual_rms)
                sample_row[group_idx] = float(stats.sample_count)

            # Simple dampened update — always apply, no backtracking.
            frac_done = iteration / max(total_iters - 1, 1)
            alpha = float(np.clip(self.learning.critic_learning_rate, 0.0, 1.0)) * ((1.0 - frac_done) ** 2.0 + 0.025)

            new_weights: list[np.ndarray] = []
            for group_idx in range(self.scenario.n_evaders):
                cand = weights[group_idx] + alpha * (solved_weights[group_idx] - weights[group_idx])
                new_weights.append(self._clip_weights(group_idx, cand))

            deltas = np.asarray(
                [
                    float(np.linalg.norm(new_weights[group_idx] - weights[group_idx]))
                    for group_idx in range(self.scenario.n_evaders)
                ],
                dtype=float,
            )
            weights = new_weights
            weight_history.append([w.copy() for w in weights])
            weight_norm_history.append(self._weight_norms(weights))
            delta_history.append(deltas)
            residual_history.append(residual_row)
            sample_history.append(sample_row)

            # Periodic validation for best-weight selection
            _val_interval = max(3, total_iters // 8)
            _do_val = (iteration % _val_interval == 0) or (iteration == total_iters - 1)
            if _do_val:
                current_metric = self._validation_metric(
                    weights,
                    visibility_mode=current_validation_mode,
                    dynamic_graph=dynamic_graph,
                )
            validation_history.append(np.array([current_metric], dtype=float))

            if current_validation_mode == selection_mode and current_metric < best_metric - 1.0e-9:
                best_metric = float(current_metric)
                best_weights = [w.copy() for w in weights]
                best_iteration = int(len(weight_history) - 1)

            min_iters_before_stop = curriculum_iters if visibility_mode == "curriculum" else 8
            if iteration + 1 >= min_iters_before_stop and float(np.nanmax(deltas)) < self.learning.convergence_tol and np.all(
                sample_row >= self.learning.min_samples_per_evader
            ):
                break

        if best_metric == float("inf"):
            best_weights = [w.copy() for w in weights]
            best_iteration = int(len(weight_history) - 1)

        return TeamCommMultiTrainResult(
            weights=[w.copy() for w in best_weights],
            weight_history=weight_history,
            weight_norm_history=np.asarray(weight_norm_history, dtype=float),
            delta_history=np.asarray(delta_history, dtype=float),
            residual_history=np.asarray(residual_history, dtype=float),
            sample_history=np.asarray(sample_history, dtype=float),
            exploration_history=np.asarray(exploration_history, dtype=float),
            validation_history=np.asarray(validation_history, dtype=float),
            best_iteration=best_iteration,
        )

    def evaluate_policy(
        self,
        weights: list[np.ndarray],
        seed: int = 1,
        *,
        dynamic_graph: bool = True,
        visibility_mode: str = "dropout",
        stop_on_capture: bool = True,
        zero_tail_after_capture: bool = False,
        record_logs: bool = False,
    ) -> TeamCommMultiEvalResult:
        rng = np.random.default_rng(seed)
        self.schedule = self._make_schedule(rng, randomize_episode=(visibility_mode == "dropout"))
        p, e = self._reset_states(rng, perturb_scale=0.0)
        graph = self._new_graph()

        steps = int(self.scenario.t_final / self.learning.dt)
        p_traj = [p.copy()]
        e_traj = [e.copy()]
        p_u = []
        e_u = []
        p_u_tanh = []
        e_u_tanh = []
        team_errors = []
        group_errors = []
        assigned_errors = []
        assignment_history = []
        communication_ratio = []
        estimate_errors = []
        step_logs: Optional[list[dict[str, Any]]] = [] if record_logs else None

        capture_time: Optional[float] = None
        prev_assignment = self.scenario.initial_assignment.copy()
        terminal_mode = False

        for step_idx in range(steps):
            if terminal_mode:
                p_traj.append(p.copy())
                e_traj.append(e.copy())
                p_u.append(np.zeros((self.scenario.n_pursuers, 3), dtype=float))
                e_u.append(np.zeros((self.scenario.n_evaders, 3), dtype=float))
                p_u_tanh.append(np.zeros((self.scenario.n_pursuers, 3), dtype=float))
                e_u_tanh.append(np.zeros((self.scenario.n_evaders, 3), dtype=float))
                team_errors.append(0.0)
                group_errors.append(np.zeros(self.scenario.n_evaders, dtype=float))
                assigned_errors.append(np.zeros(self.scenario.n_pursuers, dtype=float))
                assignment_history.append(prev_assignment.copy())
                communication_ratio.append(np.ones(self.scenario.n_evaders, dtype=float))
                estimate_errors.append(np.zeros(self.scenario.n_evaders, dtype=float))
                continue

            p_before = p.copy()
            e_before = e.copy()
            p, e, step = self._step(
                pursuer_states=p,
                evader_states=e,
                graph=graph,
                weights=weights,
                rng=rng,
                exploration_std=0.0,
                dynamic_graph=dynamic_graph,
                visibility_mode=visibility_mode,
                step_idx=step_idx,
            )

            p_traj.append(p.copy())
            e_traj.append(e.copy())
            p_u.append(step.pursuer_u.copy())
            e_u.append(step.evader_u.copy())
            p_u_tanh.append(step.pursuer_u_tanh.copy())
            e_u_tanh.append(step.evader_u_tanh.copy())
            team_errors.append(float(step.total_team_error))
            group_errors.append(step.group_errors.copy())
            assigned_errors.append(step.assigned_errors.copy())
            assignment_history.append(step.assignment.copy())
            communication_ratio.append(step.communication_ratio.copy())
            estimate_errors.append(step.estimate_errors.copy())

            switched = bool(np.any(step.assignment != prev_assignment))
            changed_pairs: list[dict[str, int]] = []
            if switched:
                for pursuer_idx in range(self.scenario.n_pursuers):
                    old_tgt = int(prev_assignment[pursuer_idx])
                    new_tgt = int(step.assignment[pursuer_idx])
                    if old_tgt != new_tgt:
                        changed_pairs.append(
                            {
                                "pursuer": pursuer_idx + 1,
                                "from_evader": old_tgt + 1,
                                "to_evader": new_tgt + 1,
                            }
                        )
            prev_assignment = step.assignment.copy()

            if step_logs is not None:
                step_logs.append(
                    {
                        "step": int(step_idx),
                        "time_s": float(step_idx * self.learning.dt),
                        "dynamic_graph": bool(dynamic_graph),
                        "visibility_mode": visibility_mode,
                        "switch_triggered": switched,
                        "switch_pairs": changed_pairs,
                        "pursuer_states_before": np.round(p_before, 6).tolist(),
                        "evader_states_before": np.round(e_before, 6).tolist(),
                        "pursuer_states_after": np.round(p, 6).tolist(),
                        "evader_states_after": np.round(e, 6).tolist(),
                        "assigned_targets": (step.assignment + 1).tolist(),
                        "assigned_errors": np.round(step.assigned_errors, 6).tolist(),
                        "team_error": float(step.total_team_error),
                        "group_errors": np.round(step.group_errors, 6).tolist(),
                        "communication_ratio": np.round(step.communication_ratio, 6).tolist(),
                        "estimate_errors": np.round(step.estimate_errors, 6).tolist(),
                        "group_members": [[int(v) + 1 for v in group] for group in step.group_members],
                        "communication_matrices": [mat.tolist() for mat in step.communication_matrices],
                        "u_p": np.round(step.pursuer_u, 6).tolist(),
                        "u_e": np.round(step.evader_u, 6).tolist(),
                        "group_weight_norms": np.round(self._weight_norms(weights), 6).tolist(),
                    }
                )

            if capture_time is None and float(np.max(step.assigned_errors)) <= self.scenario.capture_radius:
                capture_time = float(step_idx) * self.learning.dt
                if zero_tail_after_capture:
                    terminal_mode = True
                elif stop_on_capture:
                    break

        return TeamCommMultiEvalResult(
            pursuer_traj=np.asarray(p_traj, dtype=float),
            evader_traj=np.asarray(e_traj, dtype=float),
            pursuer_u=np.asarray(p_u, dtype=float),
            evader_u=np.asarray(e_u, dtype=float),
            pursuer_u_tanh=np.asarray(p_u_tanh, dtype=float),
            evader_u_tanh=np.asarray(e_u_tanh, dtype=float),
            team_errors=np.asarray(team_errors, dtype=float),
            group_errors=np.asarray(group_errors, dtype=float),
            assigned_errors=np.asarray(assigned_errors, dtype=float),
            assignment_history=np.asarray(assignment_history, dtype=int),
            communication_ratio=np.asarray(communication_ratio, dtype=float),
            estimate_errors=np.asarray(estimate_errors, dtype=float),
            capture_time=capture_time,
            communication_windows=self._windows_payload(self.schedule.params.windows),
            step_logs=step_logs,
        )
