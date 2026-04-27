from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mpe_repro.ac_equivalent import TraditionalACEquivalent
from mpe_repro.comm_graph import CommunicationGraph
from mpe_repro.config import AircraftParams, CommParams
from mpe_repro.general_scenarios import build_general_scenario
from mpe_repro.plotting import (
    plot_assigned_residual_norm,
    plot_assigned_state_errors,
    plot_assignment_timeline,
    plot_control_input_deltas,
    plot_control_inputs,
    plot_d_min_history,
    plot_formation_error,
    plot_old_new_errors,
    plot_trajectory_3d,
    plot_trajectory_multiview,
    plot_weight_convergence,
)
from mpe_repro.report import dump_json, dump_markdown, eval_summary, train_summary
from mpe_repro.simulator import EvalResult, MPECommSimulator
from run_comm import (
    GeneralCase,
    _collect_cases,
    _comm_params,
    _control_params,
    _feature_params,
    _learning_params,
    _scenario_spec,
)

matplotlib.use("Agg")


@dataclass(frozen=True)
class PolicyTransition:
    pursuer_states: np.ndarray
    evader_states: np.ndarray
    assignment: np.ndarray
    next_pursuer_states: np.ndarray
    next_evader_states: np.ndarray
    next_assignment: np.ndarray


@dataclass
class TraditionalEvalArtifacts:
    result: EvalResult
    actor_weight_norm_sq_history: np.ndarray
    critic_weight_norm_sq_history: np.ndarray
    warmup_wall_time_s: float
    eval_wall_time_s: float


def _plot_weight_norm_history(
    history: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(history.shape[0])
    for idx in range(history.shape[1]):
        ax.plot(steps, history[:, idx], linewidth=1.4, label=labels[idx])
    ax.set_xlabel("Online step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_two_method_metric(
    series_a: np.ndarray,
    series_b: np.ndarray,
    dt: float,
    label_a: str,
    label_b: str,
    ylabel: str,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = min(series_a.shape[0], series_b.shape[0])
    t = np.arange(n) * dt
    ax.plot(t, series_a[:n], linewidth=2.0, label=label_a, color="#1f77b4")
    ax.plot(t, series_b[:n], linewidth=2.0, label=label_b, color="#d62728")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if n > 0:
        ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_runtime_compare(
    vsnac_ms_per_step: float,
    traditional_ms_per_step: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = ["V-SNAC full comm", "Traditional AC full comm"]
    values = [vsnac_ms_per_step, traditional_ms_per_step]
    colors = ["#1f77b4", "#d62728"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("ms / step")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _network_labels(n_p: int, n_e: int) -> list[str]:
    labels = [f"P{i+1}" for i in range(n_p)]
    labels.extend(f"E{j+1}" for j in range(n_e))
    return labels


def _emit_eval_plots(
    out_dir: Path,
    case_name: str,
    eval_result: EvalResult,
    dt: float,
    displacements: np.ndarray,
    d_safe: float,
    title_prefix: str,
) -> None:
    plot_trajectory_3d(eval_result, out_dir / "fig_trajectory_xy.png", f"{case_name} {title_prefix}")
    plot_trajectory_multiview(eval_result, out_dir / "fig_trajectory_multiview.png", f"{case_name} {title_prefix}")
    plot_assigned_state_errors(
        eval_result,
        dt,
        out_dir / "fig_assigned_errors.png",
        f"{case_name} assigned state errors ({title_prefix})",
    )
    plot_assigned_residual_norm(
        eval_result,
        dt,
        displacements,
        out_dir / "fig_assigned_residual_norm.png",
        f"{case_name} residual norm ({title_prefix})",
    )
    plot_control_inputs(
        eval_result,
        dt,
        out_dir / "fig_control_inputs.png",
        f"{case_name} control inputs ({title_prefix})",
        component_idx=0,
        y_lim=(-30, 30),
        use_tanh=False,
    )
    plot_control_input_deltas(
        eval_result,
        dt,
        out_dir / "fig_control_input_deltas.png",
        f"{case_name} control input delta ({title_prefix})",
        use_tanh=False,
    )
    plot_d_min_history(
        eval_result,
        dt,
        out_dir / "fig_d_min.png",
        f"{case_name} d_min ({title_prefix})",
        d_min_threshold=d_safe,
    )
    plot_formation_error(
        eval_result,
        dt,
        out_dir / "fig_formation_error.png",
        f"{case_name} formation error ({title_prefix})",
    )
    plot_old_new_errors(
        eval_result,
        dt,
        out_dir / "fig_old_new_errors.png",
        f"{case_name} old/new assigned errors ({title_prefix})",
    )
    plot_assignment_timeline(
        eval_result,
        dt,
        out_dir / "fig_assignment_timeline.png",
        f"{case_name} assignment timeline ({title_prefix})",
    )


def _collect_full_comm_transitions(
    sim: MPECommSimulator,
    weights: list[np.ndarray],
    seed: int,
    dynamic_graph: bool,
    eval_comm_params: CommParams,
) -> list[PolicyTransition]:
    rng = np.random.default_rng(seed)
    p, e = sim._reset_states(rng, perturb_scale=0.0)
    graph = sim._new_graph()
    eval_graph = CommunicationGraph(
        n_p=sim.scenario.n_pursuers,
        comm_mode=eval_comm_params.comm_mode,
        formation_ref_dist=eval_comm_params.formation_ref_dist,
        d_safe=eval_comm_params.d_safe,
    )
    policy_rng = np.random.default_rng(seed + 12345)
    steps = int(sim.scenario.t_final / sim.learning.dt)

    snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for step_idx in range(steps):
        start_step = max(int(sim.learning.graph_update_start_step), 0)
        interval = max(int(sim.learning.graph_update_interval), 1)
        if dynamic_graph and (step_idx >= start_step) and ((step_idx - start_step) % interval == 0):
            p_for_graph, e_for_graph = sim._graph_states_for_update(p, e)
            graph.update(
                pursuer_states=p_for_graph,
                evader_states=e_for_graph,
                displacements=sim.displacements,
                switch_threshold=sim.scenario.swap_threshold,
                max_switch_worsening=sim.scenario.max_switch_worsening,
                nu=sim.nu_graph,
            )

        assigned = graph.assignment.copy()
        snapshots.append((p.copy(), e.copy(), assigned.copy()))
        A_p, delta_matrix = sim._comm_structures(eval_graph, assigned)
        _, coordination_grads = sim.controller.coordination_terms(
            pursuer_states=p,
            A_p=A_p,
            delta_matrix=delta_matrix,
            gamma=eval_comm_params.gamma,
            formation_ref_dist=eval_graph.formation_ref_dist,
            d_safe=eval_graph.d_safe,
        )
        u_p, u_e_virtual, _, _, _ = sim.controller.policy(
            pursuer_states=p,
            evader_states=e,
            assignment=assigned,
            displacements=sim.displacements,
            weights=weights,
            rng=policy_rng,
            exploration_std=0.0,
            gamma=eval_comm_params.gamma,
            A_p=A_p,
            delta_matrix=delta_matrix,
            formation_ref_dist=eval_graph.formation_ref_dist,
            d_safe=eval_graph.d_safe,
            coordination_gradients=coordination_grads,
        )
        u_e_applied = sim._applied_evader_inputs(
            step_idx=step_idx,
            u_e_virtual=u_e_virtual,
            pursuer_states=p,
            evader_states=e,
            assignment=assigned,
            u_p=u_p,
        )
        p = sim.dynamics.rk4_step_batch(p, u_p, sim.learning.dt)
        e = sim.dynamics.rk4_step_batch(e, u_e_applied, sim.learning.dt)

    transitions: list[PolicyTransition] = []
    for idx in range(max(len(snapshots) - 1, 0)):
        cur_p, cur_e, cur_assignment = snapshots[idx]
        next_p, next_e, next_assignment = snapshots[idx + 1]
        transitions.append(
            PolicyTransition(
                pursuer_states=cur_p,
                evader_states=cur_e,
                assignment=cur_assignment,
                next_pursuer_states=next_p,
                next_evader_states=next_e,
                next_assignment=next_assignment,
            )
        )
    return transitions


def _benchmark_runtime(
    transitions: list[PolicyTransition],
    fn,
    warmup_passes: int,
    timed_passes: int,
) -> dict[str, float]:
    if not transitions:
        return {"total_s": 0.0, "ms_per_step": 0.0, "steps_per_s": 0.0}

    for _ in range(max(warmup_passes, 0)):
        for transition in transitions:
            fn(transition)

    t0 = time.perf_counter()
    for _ in range(max(timed_passes, 1)):
        for transition in transitions:
            fn(transition)
    total_s = time.perf_counter() - t0
    total_steps = len(transitions) * max(timed_passes, 1)
    return {
        "total_s": float(total_s),
        "ms_per_step": float(1000.0 * total_s / max(total_steps, 1)),
        "steps_per_s": float(total_steps / max(total_s, 1e-12)),
    }


def _actor_norm_sq(bank: TraditionalACEquivalent) -> np.ndarray:
    joined = np.vstack([bank.pursuer_actor.reshape(bank.n_p, -1), bank.evader_actor.reshape(bank.n_e, -1)])
    return np.sum(joined * joined, axis=1)


def _critic_norm_sq(bank: TraditionalACEquivalent) -> np.ndarray:
    joined = np.vstack([bank.pursuer_critic, bank.evader_critic])
    return np.sum(joined * joined, axis=1)


def _traditional_warmup(
    sim: MPECommSimulator,
    bank: TraditionalACEquivalent,
    seed: int,
    dynamic_graph: bool,
    episodes: int,
) -> float:
    if episodes <= 0:
        return 0.0
    steps = int(sim.scenario.t_final / sim.learning.dt)
    t0 = time.perf_counter()
    for ep in range(episodes):
        rng = np.random.default_rng(seed + 100 * ep)
        p, e = sim._reset_states(rng, perturb_scale=sim.learning.random_perturb_scale)
        graph = sim._new_graph()
        for step_idx in range(steps):
            start_step = max(int(sim.learning.graph_update_start_step), 0)
            interval = max(int(sim.learning.graph_update_interval), 1)
            if dynamic_graph and (step_idx >= start_step) and ((step_idx - start_step) % interval == 0):
                p_for_graph, e_for_graph = sim._graph_states_for_update(p, e)
                graph.update(
                    pursuer_states=p_for_graph,
                    evader_states=e_for_graph,
                    displacements=sim.displacements,
                    switch_threshold=sim.scenario.swap_threshold,
                    max_switch_worsening=sim.scenario.max_switch_worsening,
                    nu=sim.nu_graph,
                )
            assignment = graph.assignment.copy()
            u_p, u_e_virtual = bank.policy_only(
                pursuer_states=p,
                evader_states=e,
                assignment=assignment,
                displacements=sim.displacements,
            )
            u_e_applied = sim._applied_evader_inputs(
                step_idx=step_idx,
                u_e_virtual=u_e_virtual,
                pursuer_states=p,
                evader_states=e,
                assignment=assignment,
                u_p=u_p,
            )
            next_p = sim.dynamics.rk4_step_batch(p, u_p, sim.learning.dt)
            next_e = sim.dynamics.rk4_step_batch(e, u_e_applied, sim.learning.dt)
            bank.online_qlearning_step(
                pursuer_states=p,
                evader_states=e,
                assignment=assignment,
                next_pursuer_states=next_p,
                next_evader_states=next_e,
                next_assignment=assignment,
                displacements=sim.displacements,
                dt=sim.learning.dt,
            )
            p, e = next_p, next_e
    return time.perf_counter() - t0


def _traditional_evaluate(
    sim: MPECommSimulator,
    bank: TraditionalACEquivalent,
    seed: int,
    dynamic_graph: bool,
    comm_graph: CommunicationGraph,
    stop_on_capture: bool = True,
) -> tuple[EvalResult, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    p, e = sim._reset_states(rng, perturb_scale=0.0)
    graph = sim._new_graph()
    dt = sim.learning.dt
    steps = int(sim.scenario.t_final / dt)

    p_traj = [p.copy()]
    e_traj = [e.copy()]
    p_u = []
    e_u = []
    p_u_tanh = []
    e_u_tanh = []
    pairwise = []
    assigned_hist = []
    assigned_err = []
    initial_err = []
    team_err = []
    d_min_hist = []
    formation_hist = []

    actor_hist = [_actor_norm_sq(bank)]
    critic_hist = [_critic_norm_sq(bank)]
    capture_time: float | None = None

    t0 = time.perf_counter()
    for step_idx in range(steps):
        start_step = max(int(sim.learning.graph_update_start_step), 0)
        interval = max(int(sim.learning.graph_update_interval), 1)
        if dynamic_graph and (step_idx >= start_step) and ((step_idx - start_step) % interval == 0):
            p_for_graph, e_for_graph = sim._graph_states_for_update(p, e)
            graph.update(
                pursuer_states=p_for_graph,
                evader_states=e_for_graph,
                displacements=sim.displacements,
                switch_threshold=sim.scenario.swap_threshold,
                max_switch_worsening=sim.scenario.max_switch_worsening,
                nu=sim.nu_graph,
            )

        assignment = graph.assignment.copy()
        A_p, delta_matrix = sim._comm_structures(comm_graph, assignment)

        u_p_now, u_e_virtual = bank.policy_only(
            pursuer_states=p,
            evader_states=e,
            assignment=assignment,
            displacements=sim.displacements,
        )
        u_e_applied = sim._applied_evader_inputs(
            step_idx=step_idx,
            u_e_virtual=u_e_virtual,
            pursuer_states=p,
            evader_states=e,
            assignment=assignment,
            u_p=u_p_now,
        )

        pairwise_now = sim._pairwise_errors(p, e)
        assigned_errors_now = pairwise_now[np.arange(sim.scenario.n_pursuers), assignment]
        initial_errors_now = pairwise_now[np.arange(sim.scenario.n_pursuers), sim.initial_assignment]
        team_error_now = graph.team_error(
            pursuer_states=p,
            evader_states=e,
            displacements=sim.displacements,
            nu=sim.nu_display,
            pairwise_costs=pairwise_now,
        )
        d_min_now = sim._compute_d_min(p)
        formation_now = sim._compute_formation_error_norms(p, A_p, delta_matrix)

        next_p = sim.dynamics.rk4_step_batch(p, u_p_now, dt)
        next_e = sim.dynamics.rk4_step_batch(e, u_e_applied, dt)
        bank.online_qlearning_step(
            pursuer_states=p,
            evader_states=e,
            assignment=assignment,
            next_pursuer_states=next_p,
            next_evader_states=next_e,
            next_assignment=assignment,
            displacements=sim.displacements,
            dt=dt,
        )

        p_traj.append(next_p.copy())
        e_traj.append(next_e.copy())
        p_u.append(u_p_now.copy())
        e_u.append(u_e_applied.copy())
        p_u_tanh.append(u_p_now.copy())
        e_u_tanh.append(u_e_virtual.copy())
        pairwise.append(pairwise_now.copy())
        assigned_hist.append(assignment.copy())
        assigned_err.append(assigned_errors_now.copy())
        initial_err.append(initial_errors_now.copy())
        team_err.append(float(team_error_now))
        d_min_hist.append(float(d_min_now))
        formation_hist.append(formation_now.copy())
        actor_hist.append(_actor_norm_sq(bank))
        critic_hist.append(_critic_norm_sq(bank))

        p, e = next_p, next_e
        if capture_time is None and float(np.max(assigned_errors_now)) <= sim.scenario.capture_radius:
            capture_time = step_idx * dt
            if stop_on_capture:
                break

    eval_wall_time_s = time.perf_counter() - t0
    result = EvalResult(
        pursuer_traj=np.asarray(p_traj),
        evader_traj=np.asarray(e_traj),
        pursuer_u=np.asarray(p_u),
        evader_u=np.asarray(e_u),
        pursuer_u_tanh=np.asarray(p_u_tanh),
        evader_u_tanh=np.asarray(e_u_tanh),
        pairwise_errors=np.asarray(pairwise),
        assigned_targets=np.asarray(assigned_hist),
        assigned_errors=np.asarray(assigned_err),
        initial_target_errors=np.asarray(initial_err),
        team_errors=np.asarray(team_err),
        capture_time=capture_time,
        d_min_history=np.asarray(d_min_hist),
        formation_errors=np.asarray(formation_hist),
        step_logs=None,
    )
    return result, np.asarray(actor_hist), np.asarray(critic_hist), eval_wall_time_s


def _case_report_markdown(summary: dict[str, Any]) -> str:
    timing = summary["timing_compare"]
    lines = [
        f"# Full-Communication Head-to-Head: {summary['case']['name']}",
        "",
        "## Scope",
        "- Only the full-communication setting is compared.",
        "- Both methods produce rollout figures on the same scenario and same dynamic target-assignment logic.",
        "- The traditional baseline is the paper-inspired online actor-critic/Q-learning implementation.",
        "",
        "## Metrics",
        f"- V-SNAC capture time (s): {summary['vsnac_full_comm']['eval']['capture_time_s']}",
        f"- Traditional AC capture time (s): {summary['traditional_full_comm']['eval']['capture_time_s']}",
        f"- V-SNAC mean assigned error: {summary['vsnac_full_comm']['eval']['mean_assigned_error']:.3f}",
        f"- Traditional AC mean assigned error: {summary['traditional_full_comm']['eval']['mean_assigned_error']:.3f}",
        f"- V-SNAC min d_min: {summary['vsnac_full_comm']['eval']['min_d_min']:.3f}",
        f"- Traditional AC min d_min: {summary['traditional_full_comm']['eval']['min_d_min']:.3f}",
        "",
        "## Runtime",
        f"- V-SNAC full-comm policy ms/step: {timing['vsnac_full_comm_policy']['ms_per_step']:.6f}",
        f"- Traditional AC full-comm online ms/step: {timing['traditional_full_comm_online']['ms_per_step']:.6f}",
        f"- Traditional / V-SNAC ratio: {timing['traditional_vs_vsnac_ratio']:.3f}x",
        f"- V-SNAC eval wall time (s): {summary['vsnac_full_comm']['eval_wall_time_s']:.3f}",
        f"- Traditional AC eval wall time (s): {summary['traditional_full_comm']['eval_wall_time_s']:.3f}",
        f"- Traditional AC warmup wall time (s): {summary['traditional_full_comm']['warmup_wall_time_s']:.3f}",
        "",
        "## Output Folders",
        f"- V-SNAC plots: `{summary['paths']['vsnac_dir']}`",
        f"- Traditional AC plots: `{summary['paths']['traditional_dir']}`",
    ]
    return "\n".join(lines)


def _run_single_case(
    case: GeneralCase,
    out_dir: Path,
    quick: bool,
    gamma: float,
    d_safe: float,
    warmup_passes: int,
    timed_passes: int,
    traditional_warmup_episodes: int,
    skip_plots: bool,
) -> dict[str, Any]:
    scenario = build_general_scenario(_scenario_spec(case, quick))
    control = _control_params()
    learning = _learning_params(case.n_pursuers, case.n_evaders, quick)
    feature = _feature_params()

    sim = MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        comm_params=_comm_params(gamma=gamma, comm_mode="full", d_safe=d_safe),
    )
    dynamic_graph = scenario.n_evaders > 1
    full_comm_cfg = CommParams(gamma=gamma, comm_mode="full", dropout_prob=0.0, d_safe=d_safe)
    full_comm_graph = CommunicationGraph(
        n_p=scenario.n_pursuers,
        comm_mode="full",
        formation_ref_dist=full_comm_cfg.formation_ref_dist,
        d_safe=full_comm_cfg.d_safe,
    )

    case_dir = out_dir / case.name
    vsnac_dir = case_dir / "vsnac_full_comm"
    traditional_dir = case_dir / "traditional_ac_full_comm"
    compare_dir = case_dir / "compare"
    vsnac_dir.mkdir(parents=True, exist_ok=True)
    traditional_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    train_t0 = time.perf_counter()
    train = sim.train_policy(seed=case.seed, dynamic_graph=dynamic_graph)
    train_wall_time_s = time.perf_counter() - train_t0

    eval_t0 = time.perf_counter()
    vsnac_eval = sim.evaluate_policy(
        weights=train.weights,
        seed=case.seed + 1000,
        dynamic_graph=dynamic_graph,
        stop_on_capture=False,
        record_logs=False,
        comm_params_override=full_comm_cfg,
    )
    vsnac_eval_wall_time_s = time.perf_counter() - eval_t0

    traditional_bank = TraditionalACEquivalent(
        state_scale=sim.features.state_scale,
        n_p=scenario.n_pursuers,
        n_e=scenario.n_evaders,
        u_bar_p=control.u_bar_p,
        u_bar_e=control.u_bar_e,
        q_diag=np.diag(control.q),
        r1_diag=np.diag(control.r1),
        r2_diag=np.diag(control.r2),
        seed=case.seed + 2024,
    )
    warmup_wall_time_s = _traditional_warmup(
        sim=sim,
        bank=traditional_bank,
        seed=case.seed + 2000,
        dynamic_graph=dynamic_graph,
        episodes=traditional_warmup_episodes,
    )
    timing_bank = traditional_bank.copy()
    traditional_eval, actor_hist, critic_hist, traditional_eval_wall_time_s = _traditional_evaluate(
        sim=sim,
        bank=traditional_bank,
        seed=case.seed + 3000,
        dynamic_graph=dynamic_graph,
        comm_graph=full_comm_graph,
        stop_on_capture=False,
    )

    reference_transitions = _collect_full_comm_transitions(
        sim=sim,
        weights=train.weights,
        seed=case.seed + 4000,
        dynamic_graph=dynamic_graph,
        eval_comm_params=full_comm_cfg,
    )
    policy_rng = np.random.default_rng(case.seed + 4242)

    def vsnac_full_step(transition: PolicyTransition) -> None:
        A_p, delta_matrix = sim._comm_structures(full_comm_graph, transition.assignment)
        _, coordination_grads = sim.controller.coordination_terms(
            pursuer_states=transition.pursuer_states,
            A_p=A_p,
            delta_matrix=delta_matrix,
            gamma=gamma,
            formation_ref_dist=full_comm_cfg.formation_ref_dist,
            d_safe=full_comm_cfg.d_safe,
        )
        sim.controller.policy(
            pursuer_states=transition.pursuer_states,
            evader_states=transition.evader_states,
            assignment=transition.assignment,
            displacements=sim.displacements,
            weights=train.weights,
            rng=policy_rng,
            exploration_std=0.0,
            gamma=gamma,
            A_p=A_p,
            delta_matrix=delta_matrix,
            formation_ref_dist=full_comm_cfg.formation_ref_dist,
            d_safe=full_comm_cfg.d_safe,
            coordination_gradients=coordination_grads,
        )

    def traditional_full_step(transition: PolicyTransition) -> None:
        timing_bank.online_qlearning_step(
            pursuer_states=transition.pursuer_states,
            evader_states=transition.evader_states,
            assignment=transition.assignment,
            next_pursuer_states=transition.next_pursuer_states,
            next_evader_states=transition.next_evader_states,
            next_assignment=transition.next_assignment,
            displacements=sim.displacements,
            dt=learning.dt,
        )

    timing_summary = {
        "vsnac_full_comm_policy": _benchmark_runtime(reference_transitions, vsnac_full_step, warmup_passes, timed_passes),
        "traditional_full_comm_online": _benchmark_runtime(
            reference_transitions, traditional_full_step, warmup_passes, timed_passes
        ),
    }
    timing_summary["traditional_vs_vsnac_ratio"] = float(
        timing_summary["traditional_full_comm_online"]["ms_per_step"]
        / max(timing_summary["vsnac_full_comm_policy"]["ms_per_step"], 1e-12)
    )

    if not skip_plots:
        _emit_eval_plots(
            out_dir=vsnac_dir,
            case_name=case.name,
            eval_result=vsnac_eval,
            dt=learning.dt,
            displacements=sim.displacements,
            d_safe=d_safe,
            title_prefix="V-SNAC full comm",
        )
        plot_weight_convergence(
            train,
            vsnac_dir / "fig_weight_convergence.png",
            f"{case.name} critic convergence (V-SNAC full comm)",
        )

        _emit_eval_plots(
            out_dir=traditional_dir,
            case_name=case.name,
            eval_result=traditional_eval,
            dt=learning.dt,
            displacements=sim.displacements,
            d_safe=d_safe,
            title_prefix="traditional AC full comm",
        )
        labels = _network_labels(scenario.n_pursuers, scenario.n_evaders)
        _plot_weight_norm_history(
            critic_hist,
            labels,
            traditional_dir / "fig_critic_weight_norm.png",
            f"{case.name} critic weight norm (traditional AC)",
            ylabel=r"$\|\hat W\|^2$",
        )
        _plot_weight_norm_history(
            actor_hist,
            labels,
            traditional_dir / "fig_actor_weight_norm.png",
            f"{case.name} actor weight norm (traditional AC)",
            ylabel=r"$\|\hat \Gamma\|^2$",
        )

        _plot_two_method_metric(
            vsnac_eval.team_errors,
            traditional_eval.team_errors,
            learning.dt,
            "V-SNAC full comm",
            "Traditional AC full comm",
            "Team state error",
            compare_dir / "fig_team_error_compare.png",
            f"{case.name} team error comparison",
        )
        _plot_two_method_metric(
            vsnac_eval.d_min_history,
            traditional_eval.d_min_history,
            learning.dt,
            "V-SNAC full comm",
            "Traditional AC full comm",
            "Min inter-pursuer distance (m)",
            compare_dir / "fig_d_min_compare.png",
            f"{case.name} d_min comparison",
        )
        _plot_runtime_compare(
            timing_summary["vsnac_full_comm_policy"]["ms_per_step"],
            timing_summary["traditional_full_comm_online"]["ms_per_step"],
            compare_dir / "fig_runtime_compare.png",
            f"{case.name} full-comm runtime comparison",
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case": {
            "name": case.name,
            **asdict(case),
        },
        "scenario": {
            "name": scenario.name,
            "capture_radius": float(scenario.capture_radius),
            "t_final": float(scenario.t_final),
            "initial_assignment": scenario.initial_assignment.tolist(),
        },
        "vsnac_full_comm": {
            "train": train_summary(train),
            "train_wall_time_s": float(train_wall_time_s),
            "eval": eval_summary(vsnac_eval, comm_mode="full_comm"),
            "eval_wall_time_s": float(vsnac_eval_wall_time_s),
        },
        "traditional_full_comm": {
            "eval": eval_summary(traditional_eval, comm_mode="full_comm"),
            "eval_wall_time_s": float(traditional_eval_wall_time_s),
            "warmup_wall_time_s": float(warmup_wall_time_s),
            "warmup_episodes": int(traditional_warmup_episodes),
            "network_summary": asdict(traditional_bank.parameter_summary()),
        },
        "timing_compare": timing_summary,
        "paths": {
            "vsnac_dir": str(vsnac_dir),
            "traditional_dir": str(traditional_dir),
            "compare_dir": str(compare_dir),
        },
    }
    dump_json(case_dir / "summary.json", summary)
    dump_markdown(case_dir / "REPORT.md", _case_report_markdown(summary))
    return summary


def _batch_report(out_dir: Path, case_summaries: list[dict[str, Any]], total_wall_time_s: float) -> None:
    batch = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_wall_time_s": float(total_wall_time_s),
        "cases": case_summaries,
    }
    dump_json(out_dir / "batch_summary.json", batch)

    lines = [
        "# Full-Communication Head-to-Head Batch Report",
        "",
        "- Only the full-communication condition is compared.",
        "- V-SNAC uses the communication-aware structured value function.",
        "- Traditional AC uses the paper-inspired online actor-critic/Q-learning baseline.",
        "",
        "| case | cap(vsnac) | cap(ac) | err(vsnac) | err(ac) | d_min(vsnac) | d_min(ac) | V-SNAC ms/step | AC ms/step | AC / V-SNAC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in case_summaries:
        ve = item["vsnac_full_comm"]["eval"]
        te = item["traditional_full_comm"]["eval"]
        timing = item["timing_compare"]
        lines.append(
            "| {case} | {cap_v} | {cap_t} | {err_v:.3f} | {err_t:.3f} | {d_v:.1f} | {d_t:.1f} | {ms_v:.6f} | {ms_t:.6f} | {ratio:.3f}x |".format(
                case=item["case"]["name"],
                cap_v="-" if ve["capture_time_s"] is None else f"{ve['capture_time_s']:.2f}",
                cap_t="-" if te["capture_time_s"] is None else f"{te['capture_time_s']:.2f}",
                err_v=ve["mean_assigned_error"],
                err_t=te["mean_assigned_error"],
                d_v=ve["min_d_min"],
                d_t=te["min_d_min"],
                ms_v=timing["vsnac_full_comm_policy"]["ms_per_step"],
                ms_t=timing["traditional_full_comm_online"]["ms_per_step"],
                ratio=timing["traditional_vs_vsnac_ratio"],
            )
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "- This report no longer mixes in `no_comm` or `dropout` cases.",
            "- The timing comparison is full-comm V-SNAC policy evaluation versus full-comm traditional AC online update.",
            "- Each case folder contains aligned V-SNAC and traditional AC figure sets for direct visual comparison.",
        ]
    )
    dump_markdown(out_dir / "REPORT.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-communication head-to-head comparison between V-SNAC and traditional actor-critic.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Use shorter training schedules")
    parser.add_argument("--case", type=str, nargs="*", default=None, help="Case tokens such as 3v1 3v3 5v3 6v3 8v4")
    parser.add_argument("--assignment-mode", type=str, default="shifted", choices=["zero", "cyclic", "shifted", "nearest", "random"])
    parser.add_argument("--layout-mode", type=str, default="structured", choices=["structured", "random"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--d-safe", type=float, default=100.0)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--timed-passes", type=int, default=12)
    parser.add_argument("--traditional-warmup-episodes", type=int, default=2)
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    cases = _collect_cases(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"full_comm_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    case_summaries = []
    for case in cases:
        case_summaries.append(
            _run_single_case(
                case=case,
                out_dir=out_dir,
                quick=bool(args.quick),
                gamma=float(args.gamma),
                d_safe=float(args.d_safe),
                warmup_passes=int(args.warmup_passes),
                timed_passes=int(args.timed_passes),
                traditional_warmup_episodes=int(args.traditional_warmup_episodes),
                skip_plots=bool(args.skip_plots),
            )
        )
    total_wall_time_s = time.perf_counter() - t0
    _batch_report(out_dir, case_summaries, total_wall_time_s)
    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
