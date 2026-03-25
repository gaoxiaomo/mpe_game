from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .comm_graph import CommunicationGraph
from .config import AircraftParams, CommParams, ControlParams, FeatureParams, LearningParams, ScenarioConfig
from .controller import CommVSNACController
from .dynamics import AircraftDynamics
from .features import SigmoidFeatureMap
from .graph_switch import DynamicTargetGraph
from .offpolicy_ls import ReplayLeastSquares


@dataclass
class StepRecord:
    phi_t: List[np.ndarray]
    phi_tp1: List[np.ndarray]
    stage_costs: np.ndarray
    team_error: float
    pairwise_errors: np.ndarray
    assigned_targets: np.ndarray
    assigned_errors: np.ndarray
    initial_target_errors: np.ndarray
    u_p: np.ndarray
    u_e: np.ndarray
    u_p_tanh: np.ndarray
    u_e_tanh: np.ndarray
    d_min: float
    formation_error_norms: np.ndarray


@dataclass
class TrainResult:
    weights: List[np.ndarray]
    weight_history: np.ndarray
    delta_history: np.ndarray
    residual_history: np.ndarray
    sample_history: np.ndarray
    exploration_history: np.ndarray


@dataclass
class EvalResult:
    pursuer_traj: np.ndarray
    evader_traj: np.ndarray
    pursuer_u: np.ndarray
    evader_u: np.ndarray
    pursuer_u_tanh: np.ndarray
    evader_u_tanh: np.ndarray
    pairwise_errors: np.ndarray
    assigned_targets: np.ndarray
    assigned_errors: np.ndarray
    initial_target_errors: np.ndarray
    team_errors: np.ndarray
    capture_time: Optional[float]
    d_min_history: np.ndarray
    formation_errors: np.ndarray
    step_logs: Optional[List[dict[str, Any]]] = None


class MPECommSimulator:
    """Communication-augmented MPE simulator.

    Extends the base MPESimulator with an inter-pursuer communication
    graph that modifies the error state used by the V-SNAC controller.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        aircraft_params: AircraftParams,
        control_params: ControlParams,
        learning_params: LearningParams,
        feature_params: FeatureParams,
        comm_params: CommParams,
    ) -> None:
        self.scenario = scenario
        self.learning = learning_params
        self.comm_params = comm_params

        self.dynamics = AircraftDynamics(aircraft_params)
        self.features = SigmoidFeatureMap(feature_params, state_dim=6)
        self.controller = CommVSNACController(
            dynamics=self.dynamics,
            features=self.features,
            q=control_params.q,
            r1=control_params.r1,
            r2=control_params.r2,
            u_bar_p=control_params.u_bar_p,
            u_bar_e=control_params.u_bar_e,
            u_bar_p_actor=control_params.u_bar_p_actor,
            u_bar_e_actor=control_params.u_bar_e_actor,
            policy_gain=control_params.policy_gain,
            k_pos_p=control_params.k_pos_p,
            k_vel_p=control_params.k_vel_p,
            k_pos_e=control_params.k_pos_e,
            k_vel_e=control_params.k_vel_e,
            state_cost_scale=self.features.state_scale,
        )

        self.comm_graph = CommunicationGraph(
            n_p=scenario.n_pursuers,
            comm_mode=comm_params.comm_mode,
            formation_ref_dist=comm_params.formation_ref_dist,
        )

        self.displacements = scenario.displacement_matrix.copy()
        self.nu_display = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.nu_graph = self.nu_display.copy()
        self.initial_assignment = scenario.initial_assignment.copy()

    def _compute_d_min(self, pursuer_states: np.ndarray) -> float:
        """Minimum pairwise pursuer distance (position only)."""
        n_p = pursuer_states.shape[0]
        if n_p < 2:
            return float("inf")
        pos = pursuer_states[:, :3]
        d_min = float("inf")
        for j in range(n_p):
            for k in range(j + 1, n_p):
                d = float(np.linalg.norm(pos[j] - pos[k]))
                if d < d_min:
                    d_min = d
        return d_min

    def _compute_formation_error_norms(
        self,
        pursuer_states: np.ndarray,
        A_p: np.ndarray,
        delta_matrix: np.ndarray,
    ) -> np.ndarray:
        """Per-pursuer formation error norm: ||sum_k a_{jk} * [(x_j - x_k) - Delta_{jk}]||."""
        n_p = pursuer_states.shape[0]
        norms = np.zeros(n_p, dtype=float)
        for j in range(n_p):
            err = np.zeros(6, dtype=float)
            for k in range(n_p):
                if k == j or A_p[j, k] <= 0.0:
                    continue
                err += A_p[j, k] * ((pursuer_states[j] - pursuer_states[k]) - delta_matrix[j, k])
            norms[j] = float(np.linalg.norm(err))
        return norms

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
        drift = np.array(
            [
                0.25 * amp[0] * np.cos(phase + 0.4),
                0.25 * amp[1] * np.sin(phase + 1.2),
                0.0,
            ],
            dtype=float,
        )
        u = u * np.exp(-decay * t) + drift
        return np.clip(u, -self.controller.u_bar_e, self.controller.u_bar_e)

    def _reactive_evader_input(
        self,
        step_idx: int,
        evader_idx: int,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        u_p: np.ndarray,
    ) -> np.ndarray:
        pursuers_targeting = np.where(assignment == evader_idx)[0]
        if pursuers_targeting.size == 0:
            pursuers_targeting = np.arange(pursuer_states.shape[0])

        p_sel = pursuer_states[pursuers_targeting]
        e_state = evader_states[evader_idx]

        rel_pos = np.mean(e_state[:3] - p_sel[:, :3], axis=0)
        rel_vel = np.mean(e_state[3:6] - p_sel[:, 3:6], axis=0)
        guide = rel_pos + float(self.scenario.evader_reactive_vel_gain) * rel_vel
        norm = float(np.linalg.norm(guide))
        if norm < 1e-8:
            guide = np.array([1.0, 0.0, 0.0], dtype=float)
            norm = 1.0
        direction = guide / norm

        p_effort = float(np.mean(np.linalg.norm(u_p[pursuers_targeting], axis=1)))
        t = float(step_idx) * self.learning.dt
        decay = np.exp(-float(self.scenario.evader_reactive_decay) * t)
        floor = float(np.clip(self.scenario.evader_reactive_floor, 0.0, 1.0))
        amp = (
            floor * self.controller.u_bar_e
            + decay
            * (
                float(self.scenario.evader_reactive_base) * self.controller.u_bar_e
                + float(self.scenario.evader_reactive_response) * p_effort
            )
        )
        u = amp * direction
        return np.clip(u, -self.controller.u_bar_e, self.controller.u_bar_e)

    def _applied_evader_inputs(
        self,
        step_idx: int,
        u_e_virtual: np.ndarray,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        assignment: np.ndarray,
        u_p: np.ndarray,
    ) -> np.ndarray:
        mode = self.scenario.evader_motion_mode
        if mode == "policy":
            return u_e_virtual
        u_e_applied = u_e_virtual.copy()
        mix = float(np.clip(self.scenario.evader_script_mix, 0.0, 1.0))
        if mode == "scripted":
            for i in range(u_e_applied.shape[0]):
                u_script = self._scripted_evader_input(step_idx, i)
                u_e_applied[i] = np.clip(
                    (1.0 - mix) * u_e_virtual[i] + mix * u_script,
                    -self.controller.u_bar_e,
                    self.controller.u_bar_e,
                )
            return u_e_applied

        if mode == "reactive":
            for i in range(u_e_applied.shape[0]):
                u_reactive = self._reactive_evader_input(
                    step_idx=step_idx,
                    evader_idx=i,
                    pursuer_states=pursuer_states,
                    evader_states=evader_states,
                    assignment=assignment,
                    u_p=u_p,
                )
                u_e_applied[i] = np.clip(
                    (1.0 - mix) * u_e_virtual[i] + mix * u_reactive,
                    -self.controller.u_bar_e,
                    self.controller.u_bar_e,
                )
            return u_e_applied

        return u_e_applied

    def _new_graph(self) -> DynamicTargetGraph:
        return DynamicTargetGraph(
            n_p=self.scenario.n_pursuers,
            n_e=self.scenario.n_evaders,
            initial_assignment=self.initial_assignment,
        )

    def _init_weights(self, rng: np.random.Generator) -> List[np.ndarray]:
        return [rng.uniform(0.30, 0.38, size=self.features.n_features) for _ in range(self.scenario.n_pursuers)]

    def _reset_states(
        self, rng: np.random.Generator, perturb_scale: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.scenario.pursuer_init.copy()
        e = self.scenario.evader_init.copy()
        if perturb_scale > 0.0:
            p += rng.normal(0.0, perturb_scale, size=p.shape) * np.maximum(np.abs(p), 1.0)
            e += rng.normal(0.0, perturb_scale, size=e.shape) * np.maximum(np.abs(e), 1.0)
        return p, e

    def _pairwise_errors(self, pursuer_states: np.ndarray, evader_states: np.ndarray) -> np.ndarray:
        diff = pursuer_states[:, None, :] - evader_states[None, :, :] + self.displacements
        return np.linalg.norm(diff * self.nu_display[None, None, :], axis=2)

    def _graph_states_for_update(
        self, pursuer_states: np.ndarray, evader_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        tau = float(max(self.scenario.swap_lookahead_time, 0.0))
        if tau <= 0.0:
            return pursuer_states, evader_states
        p_pred = pursuer_states.copy()
        e_pred = evader_states.copy()
        p_pred[:, :3] = p_pred[:, :3] + tau * p_pred[:, 3:6]
        e_pred[:, :3] = e_pred[:, :3] + tau * e_pred[:, 3:6]
        return p_pred, e_pred

    def _step(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        graph: DynamicTargetGraph,
        weights: List[np.ndarray],
        rng: np.random.Generator,
        exploration_std: float,
        dynamic_graph: bool,
        step_idx: int,
        gamma: float,
        comm_graph: CommunicationGraph,
        dropout_rng: np.random.Generator | None = None,
        dropout_prob: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, StepRecord]:
        start_step = max(int(self.learning.graph_update_start_step), 0)
        interval = max(int(self.learning.graph_update_interval), 1)
        if dynamic_graph and (step_idx >= start_step) and ((step_idx - start_step) % interval == 0):
            p_for_graph, e_for_graph = self._graph_states_for_update(pursuer_states, evader_states)
            graph.update(
                pursuer_states=p_for_graph,
                evader_states=e_for_graph,
                displacements=self.displacements,
                switch_threshold=self.scenario.swap_threshold,
                max_switch_worsening=self.scenario.max_switch_worsening,
                nu=self.nu_graph,
            )

        assigned = graph.assignment.copy()

        # Build communication adjacency and compute formation data
        A_p = comm_graph.build_adjacency(pursuer_states, assigned)
        if dropout_prob > 0.0 and dropout_rng is not None:
            A_p = comm_graph.apply_dropout(A_p, dropout_rng, dropout_prob)
        degree_vector, _ = comm_graph.laplacian_and_degree(A_p)
        delta_matrix = comm_graph.compute_delta_matrix(self.displacements, assigned)

        u_p, u_e_virtual, x_err, u_p_tanh, u_e_tanh = self.controller.policy(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            assignment=assigned,
            displacements=self.displacements,
            weights=weights,
            rng=rng,
            exploration_std=exploration_std,
            gamma=gamma,
            A_p=A_p,
            delta_matrix=delta_matrix,
            formation_ref_dist=comm_graph.formation_ref_dist if comm_graph is not None else 500.0,
        )
        u_e_applied = self._applied_evader_inputs(
            step_idx=step_idx,
            u_e_virtual=u_e_virtual,
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            assignment=assigned,
            u_p=u_p,
        )
        stage_costs = self.controller.stage_costs(
            x_err=x_err,
            u_p=u_p,
            u_e=u_e_virtual,
            assignment=assigned,
        )

        phi_t = [self.features.phi(x_err[j]) for j in range(self.scenario.n_pursuers)]

        next_p = pursuer_states.copy()
        next_e = evader_states.copy()
        for j in range(self.scenario.n_pursuers):
            next_p[j] = self.dynamics.rk4_step(pursuer_states[j], u_p[j], self.learning.dt)
        for i in range(self.scenario.n_evaders):
            next_e[i] = self.dynamics.rk4_step(evader_states[i], u_e_applied[i], self.learning.dt)

        # Compute next-step augmented errors for critic update
        next_assigned = graph.assignment.copy()
        next_A_p = comm_graph.build_adjacency(next_p, next_assigned)
        if dropout_prob > 0.0 and dropout_rng is not None:
            next_A_p = comm_graph.apply_dropout(next_A_p, dropout_rng, dropout_prob)
        next_delta_matrix = comm_graph.compute_delta_matrix(self.displacements, next_assigned)

        next_x_err = np.zeros((self.scenario.n_pursuers, 6), dtype=float)
        for j in range(self.scenario.n_pursuers):
            i = int(next_assigned[j])
            next_x_err[j] = self.controller.individual_error(
                pursuer_idx=j,
                evader_idx=i,
                pursuer_states=next_p,
                evader_states=next_e,
                displacements=self.displacements,
            )
        phi_tp1 = [self.features.phi(next_x_err[j]) for j in range(self.scenario.n_pursuers)]

        pairwise = self._pairwise_errors(pursuer_states, evader_states)
        assigned_errors = np.array([pairwise[j, assigned[j]] for j in range(self.scenario.n_pursuers)])
        initial_errors = np.array(
            [pairwise[j, self.initial_assignment[j]] for j in range(self.scenario.n_pursuers)]
        )
        team_error = graph.team_error(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            displacements=self.displacements,
            nu=self.nu_display,
            pairwise_costs=pairwise,
        )

        d_min = self._compute_d_min(pursuer_states)
        formation_error_norms = self._compute_formation_error_norms(pursuer_states, A_p, delta_matrix)

        rec = StepRecord(
            phi_t=phi_t,
            phi_tp1=phi_tp1,
            stage_costs=stage_costs,
            team_error=team_error,
            pairwise_errors=pairwise,
            assigned_targets=assigned,
            assigned_errors=assigned_errors,
            initial_target_errors=initial_errors,
            u_p=u_p,
            u_e=u_e_applied,
            u_p_tanh=u_p_tanh,
            u_e_tanh=u_e_tanh,
            d_min=d_min,
            formation_error_norms=formation_error_norms,
        )
        return next_p, next_e, rec

    def train_policy(self, seed: int = 0, dynamic_graph: bool = True) -> TrainResult:
        """Train with standard V-SNAC (gamma=0).  Communication coupling is
        added only at evaluation time via the formation gradient."""
        rng = np.random.default_rng(seed)
        n_critics = self.scenario.n_pursuers
        gamma = 0.0  # Training always uses standard tracking error
        comm_graph = self.comm_graph

        weights = self._init_weights(rng)
        replay = ReplayLeastSquares(
            n_evaders=n_critics,
            n_features=self.features.n_features,
            capacity=self.learning.replay_capacity,
        )

        weight_hist = [np.vstack(weights)]
        delta_hist = []
        residual_hist = []
        sample_hist = []
        exploration_hist = []

        total_iters = self.learning.policy_iterations
        for s in range(total_iters):
            frac = 1.0 if total_iters == 1 else s / (total_iters - 1)
            explore = self.learning.exploration_std_start + frac * (
                self.learning.exploration_std_end - self.learning.exploration_std_start
            )
            exploration_hist.append(explore)

            p, e = self._reset_states(rng, self.learning.random_perturb_scale)
            graph = self._new_graph()

            for k in range(self.learning.rollout_steps):
                p, e, step = self._step(
                    pursuer_states=p,
                    evader_states=e,
                    graph=graph,
                    weights=weights,
                    rng=rng,
                    exploration_std=explore,
                    dynamic_graph=dynamic_graph,
                    step_idx=k,
                    gamma=gamma,
                    comm_graph=comm_graph,
                )
                for j in range(n_critics):
                    replay.add_sample(
                        evader_idx=j,
                        phi_t=step.phi_t[j],
                        phi_tp1=step.phi_tp1[j],
                        stage_cost=step.stage_costs[j],
                        dt=self.learning.dt,
                    )

            solved_weights, stats = replay.solve(
                current_weights=weights,
                ridge_lambda=self.learning.ridge_lambda,
                min_samples=self.learning.min_samples_per_evader,
            )

            alpha_base = float(np.clip(self.learning.critic_learning_rate, 0.0, 1.0))
            alpha = alpha_base * (float(self.learning.critic_lr_decay) ** s)
            updated = []
            for j in range(n_critics):
                w = weights[j] + alpha * (solved_weights[j] - weights[j])
                updated.append(w)

            deltas = np.array([np.linalg.norm(updated[j] - weights[j]) for j in range(n_critics)], dtype=float)
            weights = updated
            weight_hist.append(np.vstack(weights))
            delta_hist.append(deltas)
            residual_hist.append(np.array([st.residual_rms for st in stats], dtype=float))
            sample_hist.append(np.array([st.sample_count for st in stats], dtype=float))

            if np.nanmax(deltas) < self.learning.convergence_tol and all(
                st.sample_count >= self.learning.min_samples_per_evader for st in stats
            ):
                break

        return TrainResult(
            weights=weights,
            weight_history=np.asarray(weight_hist),
            delta_history=np.asarray(delta_hist),
            residual_history=np.asarray(residual_hist),
            sample_history=np.asarray(sample_hist),
            exploration_history=np.asarray(exploration_hist),
        )

    def evaluate_policy(
        self,
        weights: List[np.ndarray],
        seed: int = 1,
        dynamic_graph: bool = True,
        stop_on_capture: bool = True,
        zero_tail_after_capture: bool = False,
        record_logs: bool = False,
        comm_params_override: CommParams | None = None,
    ) -> EvalResult:
        """Evaluate with optional comm_params override for different eval modes."""
        rng = np.random.default_rng(seed)
        p, e = self._reset_states(rng, perturb_scale=0.0)
        graph = self._new_graph()

        # Determine comm parameters for this evaluation
        cp = comm_params_override if comm_params_override is not None else self.comm_params
        eval_gamma = cp.gamma
        eval_comm_graph = CommunicationGraph(
            n_p=self.scenario.n_pursuers,
            comm_mode=cp.comm_mode,
            formation_ref_dist=cp.formation_ref_dist,
        )
        eval_dropout_prob = cp.dropout_prob
        dropout_rng = np.random.default_rng(cp.dropout_seed) if eval_dropout_prob > 0.0 else None

        steps = int(self.scenario.t_final / self.learning.dt)

        p_traj = [p.copy()]
        e_traj = [e.copy()]
        p_u = []
        e_u = []
        p_u_tanh = []
        e_u_tanh = []
        team_err = []
        pairwise = []
        assigned = []
        assigned_err = []
        initial_err = []
        d_min_hist = []
        formation_err_hist = []
        step_logs: Optional[List[dict[str, Any]]] = [] if record_logs else None

        capture_time: Optional[float] = None
        prev_assigned = self.initial_assignment.copy()
        terminal_mode = False
        for t in range(steps):
            p_before = p.copy()
            e_before = e.copy()
            if terminal_mode:
                pairwise_now = self._pairwise_errors(p, e)
                assigned_now = graph.assignment.copy()
                assigned_errors_now = np.array(
                    [pairwise_now[j, assigned_now[j]] for j in range(self.scenario.n_pursuers)],
                    dtype=float,
                )
                initial_errors_now = np.array(
                    [pairwise_now[j, self.initial_assignment[j]] for j in range(self.scenario.n_pursuers)],
                    dtype=float,
                )
                team_error_now = graph.team_error(
                    pursuer_states=p,
                    evader_states=e,
                    displacements=self.displacements,
                    nu=self.nu_display,
                    pairwise_costs=pairwise_now,
                )
                d_min_now = self._compute_d_min(p)
                # Build adjacency for formation error computation even in terminal mode
                A_p_now = eval_comm_graph.build_adjacency(p, assigned_now)
                delta_now = eval_comm_graph.compute_delta_matrix(self.displacements, assigned_now)
                form_err_now = self._compute_formation_error_norms(p, A_p_now, delta_now)

                rec = StepRecord(
                    phi_t=[np.zeros(self.features.n_features, dtype=float) for _ in range(self.scenario.n_pursuers)],
                    phi_tp1=[np.zeros(self.features.n_features, dtype=float) for _ in range(self.scenario.n_pursuers)],
                    stage_costs=np.zeros(self.scenario.n_pursuers, dtype=float),
                    team_error=team_error_now,
                    pairwise_errors=pairwise_now,
                    assigned_targets=assigned_now,
                    assigned_errors=assigned_errors_now,
                    initial_target_errors=initial_errors_now,
                    u_p=np.zeros((self.scenario.n_pursuers, 3), dtype=float),
                    u_e=np.zeros((self.scenario.n_evaders, 3), dtype=float),
                    u_p_tanh=np.zeros((self.scenario.n_pursuers, 3), dtype=float),
                    u_e_tanh=np.zeros((self.scenario.n_evaders, 3), dtype=float),
                    d_min=d_min_now,
                    formation_error_norms=form_err_now,
                )
            else:
                p, e, rec = self._step(
                    pursuer_states=p,
                    evader_states=e,
                    graph=graph,
                    weights=weights,
                    rng=rng,
                    exploration_std=0.0,
                    dynamic_graph=dynamic_graph,
                    step_idx=t,
                    gamma=eval_gamma,
                    comm_graph=eval_comm_graph,
                    dropout_rng=dropout_rng,
                    dropout_prob=eval_dropout_prob,
                )
            p_traj.append(p.copy())
            e_traj.append(e.copy())
            p_u.append(rec.u_p.copy())
            e_u.append(rec.u_e.copy())
            p_u_tanh.append(rec.u_p_tanh.copy())
            e_u_tanh.append(rec.u_e_tanh.copy())
            team_err.append(rec.team_error)
            pairwise.append(rec.pairwise_errors.copy())
            assigned.append(rec.assigned_targets.copy())
            assigned_err.append(rec.assigned_errors.copy())
            initial_err.append(rec.initial_target_errors.copy())
            d_min_hist.append(rec.d_min)
            formation_err_hist.append(rec.formation_error_norms.copy())

            switched = bool(np.any(rec.assigned_targets != prev_assigned))
            changed_pairs: List[dict[str, int]] = []
            if switched:
                for j in range(self.scenario.n_pursuers):
                    old_tgt = int(prev_assigned[j])
                    new_tgt = int(rec.assigned_targets[j])
                    if old_tgt != new_tgt:
                        changed_pairs.append(
                            {
                                "pursuer": j + 1,
                                "from_evader": old_tgt + 1,
                                "to_evader": new_tgt + 1,
                            }
                        )
            prev_assigned = rec.assigned_targets.copy()

            if step_logs is not None:
                step_logs.append(
                    {
                        "step": int(t),
                        "time_s": float(t * self.learning.dt),
                        "dynamic_graph": bool(dynamic_graph),
                        "switch_triggered": switched,
                        "switch_pairs": changed_pairs,
                        "pursuer_states_before": np.round(p_before, 6).tolist(),
                        "evader_states_before": np.round(e_before, 6).tolist(),
                        "pursuer_states_after": np.round(p, 6).tolist(),
                        "evader_states_after": np.round(e, 6).tolist(),
                        "assigned_targets": (rec.assigned_targets + 1).tolist(),
                        "pairwise_errors": np.round(rec.pairwise_errors, 6).tolist(),
                        "assigned_errors": np.round(rec.assigned_errors, 6).tolist(),
                        "initial_target_errors": np.round(rec.initial_target_errors, 6).tolist(),
                        "team_error": float(rec.team_error),
                        "stage_costs": np.round(rec.stage_costs, 6).tolist(),
                        "u_p": np.round(rec.u_p, 6).tolist(),
                        "u_e": np.round(rec.u_e, 6).tolist(),
                        "u_p_tanh": np.round(rec.u_p_tanh, 6).tolist(),
                        "u_e_tanh": np.round(rec.u_e_tanh, 6).tolist(),
                        "critic_weights": [np.round(w, 6).tolist() for w in weights],
                        "d_min": float(rec.d_min),
                        "formation_error_norms": np.round(rec.formation_error_norms, 6).tolist(),
                        "comm_mode": cp.comm_mode,
                        "gamma": float(eval_gamma),
                    }
                )

            if capture_time is None and float(np.max(rec.assigned_errors)) <= self.scenario.capture_radius:
                capture_time = t * self.learning.dt
                if stop_on_capture:
                    break
                if zero_tail_after_capture:
                    terminal_mode = True

        return EvalResult(
            pursuer_traj=np.asarray(p_traj),
            evader_traj=np.asarray(e_traj),
            pursuer_u=np.asarray(p_u),
            evader_u=np.asarray(e_u),
            pursuer_u_tanh=np.asarray(p_u_tanh),
            evader_u_tanh=np.asarray(e_u_tanh),
            pairwise_errors=np.asarray(pairwise),
            assigned_targets=np.asarray(assigned),
            assigned_errors=np.asarray(assigned_err),
            initial_target_errors=np.asarray(initial_err),
            team_errors=np.asarray(team_err),
            capture_time=capture_time,
            d_min_history=np.asarray(d_min_hist),
            formation_errors=np.asarray(formation_err_hist),
            step_logs=step_logs,
        )
