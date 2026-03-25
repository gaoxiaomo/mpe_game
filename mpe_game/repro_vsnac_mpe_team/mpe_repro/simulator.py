from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from .config import (
    AircraftParams,
    CommMode,
    ControlParams,
    FeatureParams,
    LearningParams,
    ScenarioConfig,
    TeamParams,
)
from .controller import TeamVSNACController
from .dynamics import AircraftDynamics
from .features import SigmoidFeatureMap
from .graph_switch import DynamicTargetGraph
from .offpolicy_ls import ReplayLeastSquares


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

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


@dataclass
class TrainResult:
    weights: List[np.ndarray]
    weight_history: np.ndarray
    delta_history: np.ndarray
    residual_history: np.ndarray
    sample_history: np.ndarray
    exploration_history: np.ndarray


@dataclass
class CoordMetrics:
    d_min: np.ndarray
    angular_coverage: np.ndarray
    path_overlap: np.ndarray


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
    coord_metrics: Optional[CoordMetrics] = None


# ---------------------------------------------------------------------------
# Coordination metric helpers
# ---------------------------------------------------------------------------

def _min_inter_pursuer_distance(pursuer_states: np.ndarray) -> float:
    n = pursuer_states.shape[0]
    if n < 2:
        return float("inf")
    pos = pursuer_states[:, :3]
    d_min = float("inf")
    for j in range(n):
        for k in range(j + 1, n):
            d = float(np.linalg.norm(pos[j] - pos[k]))
            if d < d_min:
                d_min = d
    return d_min


def _angular_coverage(pursuer_states: np.ndarray, evader_state: np.ndarray) -> float:
    n = pursuer_states.shape[0]
    if n < 2:
        return 1.0
    rel = pursuer_states[:, :2] - evader_state[:2]
    norms = np.linalg.norm(rel, axis=1)
    valid = norms > 1e-6
    if np.sum(valid) < 2:
        return 1.0
    angles = np.arctan2(rel[valid, 1], rel[valid, 0])
    angles = np.sort(angles)
    gaps = np.diff(angles)
    gaps = np.append(gaps, angles[0] + 2.0 * np.pi - angles[-1])
    max_gap = float(np.max(gaps))
    return 1.0 - max_gap / (2.0 * np.pi)


def _path_overlap_fraction(pursuer_states: np.ndarray, d_thresh: float = 300.0) -> float:
    n = pursuer_states.shape[0]
    if n < 2:
        return 0.0
    count = 0
    total = n * (n - 1) // 2
    pos = pursuer_states[:, :3]
    for j in range(n):
        for k in range(j + 1, n):
            if float(np.linalg.norm(pos[j] - pos[k])) < d_thresh:
                count += 1
    return count / max(total, 1)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class TeamMPESimulator:
    """V-SNAC simulator with communication-enabled separation gradient.

    Training uses standard 6-feature V-SNAC (proven convergence).
    At control time, the ``use_separation`` flag adds an analytical
    potential-based repulsive gradient, which requires communication
    (teammate absolute states).
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        aircraft_params: AircraftParams,
        control_params: ControlParams,
        learning_params: LearningParams,
        feature_params: FeatureParams,
        team_params: TeamParams,
    ) -> None:
        self.scenario = scenario
        self.learning = learning_params
        self.team = team_params

        self.dynamics = AircraftDynamics(aircraft_params)
        self.features = SigmoidFeatureMap(feature_params, state_dim=6)
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
            policy_gain=control_params.policy_gain,
            k_pos_p=control_params.k_pos_p,
            k_vel_p=control_params.k_vel_p,
            k_pos_e=control_params.k_pos_e,
            k_vel_e=control_params.k_vel_e,
            state_cost_scale=self.features.state_scale,
            sep_alpha=team_params.gamma_sep,
            sep_d_ref=team_params.d_ref,
        )

        self.displacements = scenario.displacement_matrix.copy()
        self.nu_display = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.nu_graph = self.nu_display.copy()
        self.initial_assignment = scenario.initial_assignment.copy()

    # ------------------------------------------------------------------
    # Evader motion
    # ------------------------------------------------------------------
    def _scripted_evader_input(self, step_idx: int, evader_idx: int) -> np.ndarray:
        t = float(step_idx) * self.learning.dt
        amp = np.asarray(self.scenario.evader_script_amp, dtype=float)
        omega = float(self.scenario.evader_script_omega)
        decay = float(self.scenario.evader_script_decay)
        phase = 0.9 * float(evader_idx)
        u = np.array([
            amp[0] * np.sin(omega * t + phase),
            amp[1] * np.sin(1.1 * omega * t + phase + 0.8),
            amp[2] * np.sin(0.9 * omega * t + phase + 1.6),
        ], dtype=float)
        drift = np.array([
            0.25 * amp[0] * np.cos(phase + 0.4),
            0.25 * amp[1] * np.sin(phase + 1.2),
            0.0,
        ], dtype=float)
        u = u * np.exp(-decay * t) + drift
        return np.clip(u, -self.controller.u_bar_e, self.controller.u_bar_e)

    def _reactive_evader_input(
        self, step_idx, evader_idx, pursuer_states, evader_states, assignment, u_p,
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
            guide = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        direction = guide / norm
        p_effort = float(np.mean(np.linalg.norm(u_p[pursuers_targeting], axis=1)))
        t = float(step_idx) * self.learning.dt
        decay = np.exp(-float(self.scenario.evader_reactive_decay) * t)
        floor = float(np.clip(self.scenario.evader_reactive_floor, 0.0, 1.0))
        amp = (
            floor * self.controller.u_bar_e
            + decay * (
                float(self.scenario.evader_reactive_base) * self.controller.u_bar_e
                + float(self.scenario.evader_reactive_response) * p_effort
            )
        )
        return np.clip(amp * direction, -self.controller.u_bar_e, self.controller.u_bar_e)

    def _applied_evader_inputs(self, step_idx, u_e_virtual, pursuer_states, evader_states, assignment, u_p):
        mode = self.scenario.evader_motion_mode
        if mode == "policy":
            return u_e_virtual
        u_out = u_e_virtual.copy()
        mix = float(np.clip(self.scenario.evader_script_mix, 0.0, 1.0))
        for i in range(u_out.shape[0]):
            if mode == "scripted":
                u_s = self._scripted_evader_input(step_idx, i)
            elif mode == "reactive":
                u_s = self._reactive_evader_input(step_idx, i, pursuer_states, evader_states, assignment, u_p)
            else:
                continue
            u_out[i] = np.clip(
                (1.0 - mix) * u_e_virtual[i] + mix * u_s,
                -self.controller.u_bar_e, self.controller.u_bar_e,
            )
        return u_out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _new_graph(self) -> DynamicTargetGraph:
        return DynamicTargetGraph(
            n_p=self.scenario.n_pursuers,
            n_e=self.scenario.n_evaders,
            initial_assignment=self.initial_assignment,
        )

    def _init_weights(self, rng: np.random.Generator) -> List[np.ndarray]:
        return [rng.uniform(0.30, 0.38, size=self.features.n_features)
                for _ in range(self.scenario.n_pursuers)]

    def _reset_states(self, rng, perturb_scale=0.0):
        p = self.scenario.pursuer_init.copy()
        e = self.scenario.evader_init.copy()
        if perturb_scale > 0.0:
            p += rng.normal(0.0, perturb_scale, size=p.shape) * np.maximum(np.abs(p), 1.0)
            e += rng.normal(0.0, perturb_scale, size=e.shape) * np.maximum(np.abs(e), 1.0)
        return p, e

    def _pairwise_errors(self, p, e):
        diff = p[:, None, :] - e[None, :, :] + self.displacements
        return np.linalg.norm(diff * self.nu_display[None, None, :], axis=2)

    def _graph_states(self, p, e):
        tau = float(max(self.scenario.swap_lookahead_time, 0.0))
        if tau <= 0.0:
            return p, e
        pp = p.copy()
        ep = e.copy()
        pp[:, :3] += tau * pp[:, 3:6]
        ep[:, :3] += tau * ep[:, 3:6]
        return pp, ep

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _step(self, p, e, graph, weights, rng, explore, dynamic_graph, step_idx,
              use_separation=False, teammate_cache=None):
        start = max(int(self.learning.graph_update_start_step), 0)
        interval = max(int(self.learning.graph_update_interval), 1)
        if dynamic_graph and step_idx >= start and (step_idx - start) % interval == 0:
            pg, eg = self._graph_states(p, e)
            graph.update(pg, eg, self.displacements,
                         self.scenario.swap_threshold, self.scenario.max_switch_worsening, self.nu_graph)

        assigned = graph.assignment.copy()
        u_p, u_e_v, x_err, u_p_tanh, u_e_tanh = self.controller.policy(
            p, e, assigned, self.displacements, weights, rng, explore,
            use_separation=use_separation, teammate_cache=teammate_cache,
        )
        u_e_app = self._applied_evader_inputs(step_idx, u_e_v, p, e, assigned, u_p)
        stage = self.controller.stage_costs(x_err, u_p, u_e_v, assigned)

        phi_t = [self.features.phi(x_err[j]) for j in range(self.scenario.n_pursuers)]

        next_p = p.copy()
        next_e = e.copy()
        for j in range(self.scenario.n_pursuers):
            next_p[j] = self.dynamics.rk4_step(p[j], u_p[j], self.learning.dt)
        for i in range(self.scenario.n_evaders):
            next_e[i] = self.dynamics.rk4_step(e[i], u_e_app[i], self.learning.dt)

        next_err = np.zeros((self.scenario.n_pursuers, 6))
        for j in range(self.scenario.n_pursuers):
            i_t = int(assigned[j])
            next_err[j] = self.controller.individual_error(j, i_t, next_p, next_e, self.displacements)
        phi_tp1 = [self.features.phi(next_err[j]) for j in range(self.scenario.n_pursuers)]

        pw = self._pairwise_errors(p, e)
        ae = np.array([pw[j, assigned[j]] for j in range(self.scenario.n_pursuers)])
        ie = np.array([pw[j, self.initial_assignment[j]] for j in range(self.scenario.n_pursuers)])
        te = graph.team_error(p, e, self.displacements, self.nu_display, pw)

        rec = StepRecord(phi_t, phi_tp1, stage, te, pw, assigned, ae, ie, u_p, u_e_app, u_p_tanh, u_e_tanh)
        return next_p, next_e, rec

    # ------------------------------------------------------------------
    # Training (standard 6-feature V-SNAC, with optional separation during rollouts)
    # ------------------------------------------------------------------
    def train_policy(self, seed=0, dynamic_graph=True, use_separation=False) -> TrainResult:
        rng = np.random.default_rng(seed)
        n_c = self.scenario.n_pursuers
        weights = self._init_weights(rng)
        replay = ReplayLeastSquares(n_c, self.features.n_features, self.learning.replay_capacity)

        w_hist = [np.vstack(weights)]
        d_hist, r_hist, s_hist, e_hist = [], [], [], []

        iters = self.learning.policy_iterations
        for s in range(iters):
            frac = 1.0 if iters == 1 else s / (iters - 1)
            explore = self.learning.exploration_std_start + frac * (
                self.learning.exploration_std_end - self.learning.exploration_std_start)
            e_hist.append(explore)

            p, e = self._reset_states(rng, self.learning.random_perturb_scale)
            graph = self._new_graph()

            for k in range(self.learning.rollout_steps):
                p, e, rec = self._step(
                    p, e, graph, weights, rng, explore, dynamic_graph, k,
                    use_separation=use_separation,
                )
                for j in range(n_c):
                    replay.add_sample(j, rec.phi_t[j], rec.phi_tp1[j], rec.stage_costs[j], self.learning.dt)

            solved, stats = replay.solve(weights, self.learning.ridge_lambda, self.learning.min_samples_per_critic)
            alpha = float(np.clip(self.learning.critic_learning_rate, 0.0, 1.0))
            updated = [weights[j] + alpha * (solved[j] - weights[j]) for j in range(n_c)]
            deltas = np.array([np.linalg.norm(updated[j] - weights[j]) for j in range(n_c)])
            weights = updated
            w_hist.append(np.vstack(weights))
            d_hist.append(deltas)
            r_hist.append(np.array([st.residual_rms for st in stats]))
            s_hist.append(np.array([st.sample_count for st in stats]))

            if np.nanmax(deltas) < self.learning.convergence_tol and all(
                st.sample_count >= self.learning.min_samples_per_critic for st in stats
            ):
                break

        return TrainResult(weights, np.asarray(w_hist), np.asarray(d_hist),
                           np.asarray(r_hist), np.asarray(s_hist), np.asarray(e_hist))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_policy(
        self, weights, seed=1, dynamic_graph=True,
        use_separation=False, comm_mode: CommMode = "full",
        dropout_start_s=10.0, dropout_end_s=20.0,
        stop_on_capture=True,
    ) -> EvalResult:
        rng = np.random.default_rng(seed)
        p, e = self._reset_states(rng, 0.0)
        graph = self._new_graph()
        steps = int(self.scenario.t_final / self.learning.dt)
        dt = self.learning.dt

        p_traj, e_traj = [p.copy()], [e.copy()]
        pu, eu, pu_t, eu_t = [], [], [], []
        te_list, pw_list, as_list, ae_list, ie_list = [], [], [], [], []
        dm, ac, ol = [], [], []
        cap_time = None
        frozen = None

        for t_s in range(steps):
            cur_t = t_s * dt
            sep = use_separation
            tc = None
            if comm_mode == "local":
                sep = False
            elif comm_mode == "dropout" and use_separation:
                if dropout_start_s <= cur_t < dropout_end_s:
                    if frozen is None:
                        frozen = p.copy()
                    tc = frozen
                else:
                    frozen = None

            p, e, rec = self._step(
                p, e, graph, weights, rng, 0.0, dynamic_graph, t_s,
                use_separation=sep, teammate_cache=tc,
            )
            p_traj.append(p.copy())
            e_traj.append(e.copy())
            pu.append(rec.u_p); eu.append(rec.u_e)
            pu_t.append(rec.u_p_tanh); eu_t.append(rec.u_e_tanh)
            te_list.append(rec.team_error)
            pw_list.append(rec.pairwise_errors)
            as_list.append(rec.assigned_targets)
            ae_list.append(rec.assigned_errors)
            ie_list.append(rec.initial_target_errors)

            dm.append(_min_inter_pursuer_distance(p))
            if self.scenario.n_evaders == 1:
                ac.append(_angular_coverage(p, e[0]))
            else:
                a = rec.assigned_targets
                cnts = np.bincount(a.astype(int), minlength=self.scenario.n_evaders)
                me = int(np.argmax(cnts))
                mp = np.where(a == me)[0]
                ac.append(_angular_coverage(p[mp], e[me]) if mp.size >= 2 else 1.0)
            ol.append(_path_overlap_fraction(p, 300.0))

            if cap_time is None and float(np.max(rec.assigned_errors)) <= self.scenario.capture_radius:
                cap_time = cur_t
                if stop_on_capture:
                    break

        coord = CoordMetrics(np.asarray(dm), np.asarray(ac), np.asarray(ol))
        n_p, n_e = self.scenario.n_pursuers, self.scenario.n_evaders
        return EvalResult(
            np.asarray(p_traj), np.asarray(e_traj),
            np.asarray(pu) if pu else np.zeros((0, n_p, 3)),
            np.asarray(eu) if eu else np.zeros((0, n_e, 3)),
            np.asarray(pu_t) if pu_t else np.zeros((0, n_p, 3)),
            np.asarray(eu_t) if eu_t else np.zeros((0, n_e, 3)),
            np.asarray(pw_list) if pw_list else np.zeros((0, n_p, n_e)),
            np.asarray(as_list) if as_list else np.zeros((0, n_p), dtype=int),
            np.asarray(ae_list) if ae_list else np.zeros((0, n_p)),
            np.asarray(ie_list) if ie_list else np.zeros((0, n_p)),
            np.asarray(te_list),
            cap_time, coord,
        )
