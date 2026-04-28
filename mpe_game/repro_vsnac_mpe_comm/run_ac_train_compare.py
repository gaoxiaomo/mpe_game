"""Train and compare the traditional Actor-Critic / Q-learning baseline
against V-SNAC on a unified scenario.

Unlike ``run_ac_speed_compare.py`` (which only measures wall-clock per step
without ever running learning updates), this driver actually trains the
Actor-Critic bank by rolling out trajectories and applying the online
Q-learning update at every step. After convergence it produces:

- Convergence curves (per-episode critic / actor weight norms, mean
  assigned tracking error, capture-time proxy).
- Wall-time comparison against V-SNAC (ms/step for forward policy, ms/step
  for online learning step).
- Network parameter inventory contrasting the AC bank's
  ``2 * (N_p + N_e)`` networks against V-SNAC's ``N_p`` critics.

The default scenario is a 6v3 medium-size case so the comparison reflects
realistic team-level cost; ``--case`` accepts shorthand tokens like
``3v1`` / ``8v4``.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpe_repro.ac_equivalent import TraditionalACEquivalent
from mpe_repro.config import (
    AircraftParams,
    CommParams,
    ControlParams,
    FeatureParams,
    LearningParams,
)
from mpe_repro.general_scenarios import GeneralScenarioSpec, build_general_scenario
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.simulator import MPECommSimulator
from run_comm import (
    GeneralCase,
    _control_params,
    _feature_params,
    _learning_params,
    _scenario_spec,
)


# ---------------------------------------------------------------------------
# Local utilities
# ---------------------------------------------------------------------------


def _assigned_errors(
    pursuer_states: np.ndarray,
    evader_states: np.ndarray,
    assignment: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Per-pursuer assigned tracking error norm."""
    n_p = pursuer_states.shape[0]
    diff = pursuer_states - evader_states[assignment] + displacements[np.arange(n_p), assignment]
    return np.linalg.norm(diff, axis=1)


def _evader_inputs_scripted(
    sim: MPECommSimulator,
    step_idx: int,
    n_e: int,
    fallback: np.ndarray,
) -> np.ndarray:
    """Reuse the simulator's evader script when the scenario is scripted; else
    return ``fallback`` (typically the actor-output)."""
    if sim.scenario.evader_motion_mode != "scripted":
        return fallback
    out = np.zeros((n_e, 3), dtype=float)
    for i in range(n_e):
        out[i] = sim._scripted_evader_input(step_idx=step_idx, evader_idx=i)
    mix = float(np.clip(sim.scenario.evader_script_mix, 0.0, 1.0))
    return np.clip(
        (1.0 - mix) * fallback + mix * out,
        -sim.controller.u_bar_e,
        sim.controller.u_bar_e,
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _train_ac_bank(
    sim: MPECommSimulator,
    ac_bank: TraditionalACEquivalent,
    n_episodes: int,
    n_steps_per_episode: int,
    perturb_scale: float,
    seed: int,
) -> list[dict[str, float]]:
    """Run on-policy AC rollouts and apply the Q-learning update each step.

    Returns one record per episode with critic / actor norms and mean
    tracking error.
    """
    rng = np.random.default_rng(seed)
    dt = sim.learning.dt
    n_e = sim.scenario.n_evaders
    history: list[dict[str, float]] = []

    initial_assignment = sim.initial_assignment.copy()

    for ep in range(n_episodes):
        # Reset state with small random perturbation
        p, e = sim._reset_states(rng, perturb_scale=perturb_scale)
        assignment = initial_assignment.copy()
        ep_errors: list[float] = []

        for t in range(n_steps_per_episode):
            u_p, u_e_actor = ac_bank.policy_only(p, e, assignment, sim.displacements)

            # If the scenario uses a scripted evader, override the actor's
            # evader output with the script (otherwise the AC's evader actor
            # tries to learn against a non-stationary moving target while
            # never observing the script).
            u_e = _evader_inputs_scripted(sim, t, n_e, u_e_actor)

            next_p = sim.dynamics.rk4_step_batch(p, u_p, dt)
            next_e = sim.dynamics.rk4_step_batch(e, u_e, dt)

            # Online Q-learning + actor update on this transition.
            ac_bank.online_qlearning_step(
                pursuer_states=p,
                evader_states=e,
                assignment=assignment,
                next_pursuer_states=next_p,
                next_evader_states=next_e,
                next_assignment=assignment,
                displacements=sim.displacements,
                dt=dt,
            )

            err = _assigned_errors(p, e, assignment, sim.displacements)
            ep_errors.append(float(np.mean(err)))
            p, e = next_p, next_e

        history.append(
            {
                "episode": int(ep),
                "mean_assigned_error": float(np.mean(ep_errors)) if ep_errors else float("nan"),
                "final_assigned_error": float(ep_errors[-1]) if ep_errors else float("nan"),
                "pursuer_critic_norm": float(np.linalg.norm(ac_bank.pursuer_critic)),
                "evader_critic_norm": float(np.linalg.norm(ac_bank.evader_critic)),
                "pursuer_actor_norm": float(np.linalg.norm(ac_bank.pursuer_actor)),
                "evader_actor_norm": float(np.linalg.norm(ac_bank.evader_actor)),
            }
        )
    return history


def _evaluate_policy_only(
    sim: MPECommSimulator,
    ac_bank: TraditionalACEquivalent,
    seed: int,
    n_steps: int,
) -> dict[str, Any]:
    """Roll out with frozen weights (no learning) and report tracking error
    + min pairwise distance."""
    rng = np.random.default_rng(seed)
    p, e = sim._reset_states(rng, perturb_scale=0.0)
    assignment = sim.initial_assignment.copy()
    dt = sim.learning.dt
    n_e = sim.scenario.n_evaders

    err_hist: list[float] = []
    d_min_hist: list[float] = []
    capture_time: float | None = None

    for t in range(n_steps):
        u_p, u_e_actor = ac_bank.policy_only(p, e, assignment, sim.displacements)
        u_e = _evader_inputs_scripted(sim, t, n_e, u_e_actor)
        next_p = sim.dynamics.rk4_step_batch(p, u_p, dt)
        next_e = sim.dynamics.rk4_step_batch(e, u_e, dt)

        err = _assigned_errors(p, e, assignment, sim.displacements)
        err_hist.append(float(np.mean(err)))
        d_min_hist.append(float(sim._compute_d_min(p)))
        if capture_time is None and float(np.max(err)) <= sim.scenario.capture_radius:
            capture_time = float(t * dt)

        p, e = next_p, next_e

    return {
        "mean_assigned_error": float(np.mean(err_hist)) if err_hist else float("nan"),
        "final_assigned_error": float(err_hist[-1]) if err_hist else float("nan"),
        "min_d_min_m": float(np.min(d_min_hist)) if d_min_hist else float("nan"),
        "capture_time_s": capture_time,
        "error_history": err_hist,
        "d_min_history": d_min_hist,
    }


def _benchmark_ms_per_step(
    sim: MPECommSimulator,
    ac_bank: TraditionalACEquivalent,
    n_steps: int,
    dt: float,
) -> dict[str, float]:
    """Measure ms/step for AC policy_only and AC online_qlearning_step on a
    single deterministic rollout."""
    p = sim.scenario.pursuer_init.copy()
    e = sim.scenario.evader_init.copy()
    assignment = sim.initial_assignment.copy()
    displacements = sim.displacements

    # Pre-compute next states for the online-step bench (we want to measure
    # learning-step cost without dynamics noise).
    transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    cur_p = p.copy()
    cur_e = e.copy()
    for _ in range(n_steps):
        u_p, u_e = ac_bank.policy_only(cur_p, cur_e, assignment, displacements)
        next_p = sim.dynamics.rk4_step_batch(cur_p, u_p, dt)
        next_e = sim.dynamics.rk4_step_batch(cur_e, u_e, dt)
        transitions.append((cur_p.copy(), cur_e.copy(), next_p.copy(), next_e.copy()))
        cur_p, cur_e = next_p, next_e

    # AC policy_only timing
    t0 = time.perf_counter()
    for cp, ce, _, _ in transitions:
        ac_bank.policy_only(cp, ce, assignment, displacements)
    policy_total = time.perf_counter() - t0

    # AC online step timing (uses a copy so we don't mutate the trained net)
    bench_bank = ac_bank.copy()
    t0 = time.perf_counter()
    for cp, ce, np_next, ne_next in transitions:
        bench_bank.online_qlearning_step(
            pursuer_states=cp,
            evader_states=ce,
            assignment=assignment,
            next_pursuer_states=np_next,
            next_evader_states=ne_next,
            next_assignment=assignment,
            displacements=displacements,
            dt=dt,
        )
    online_total = time.perf_counter() - t0

    return {
        "ac_policy_ms_per_step": 1000.0 * policy_total / max(n_steps, 1),
        "ac_online_ms_per_step": 1000.0 * online_total / max(n_steps, 1),
    }


def _benchmark_vsnac(
    sim: MPECommSimulator,
    weights: list[np.ndarray],
    gamma: float,
    n_steps: int,
) -> dict[str, float]:
    """Measure ms/step for V-SNAC controller.policy and a full _step (with
    Phi)."""
    rng = np.random.default_rng(0)
    p = sim.scenario.pursuer_init.copy()
    e = sim.scenario.evader_init.copy()
    assignment = sim.initial_assignment.copy()

    # Roll forward to collect a stationary trajectory snapshot
    snapshots: list[tuple[np.ndarray, np.ndarray]] = []
    cur_p = p.copy()
    cur_e = e.copy()
    dt = sim.learning.dt
    for t in range(n_steps):
        snapshots.append((cur_p.copy(), cur_e.copy()))
        u_p, u_e, _, _, _ = sim.controller.policy(
            pursuer_states=cur_p,
            evader_states=cur_e,
            assignment=assignment,
            displacements=sim.displacements,
            weights=weights,
            rng=rng,
            exploration_std=0.0,
            gamma=0.0,
            A_p=None,
            delta_matrix=None,
            coordination_gradients=None,
        )
        cur_p = sim.dynamics.rk4_step_batch(cur_p, u_p, dt)
        cur_e = sim.dynamics.rk4_step_batch(cur_e, u_e, dt)

    # V-SNAC policy_only (no comm)
    t0 = time.perf_counter()
    for sp, se in snapshots:
        sim.controller.policy(
            pursuer_states=sp,
            evader_states=se,
            assignment=assignment,
            displacements=sim.displacements,
            weights=weights,
            rng=rng,
            exploration_std=0.0,
            gamma=0.0,
            A_p=None,
            delta_matrix=None,
            coordination_gradients=None,
        )
    no_comm_total = time.perf_counter() - t0

    # V-SNAC with full coordination
    A_base, delta_matrix = sim._comm_structures(sim.comm_graph, assignment)
    t0 = time.perf_counter()
    for sp, se in snapshots:
        _, coord_grads = sim.controller.coordination_terms(
            pursuer_states=sp,
            A_p=A_base,
            delta_matrix=delta_matrix,
            gamma=gamma,
            formation_ref_dist=sim.comm_graph.formation_ref_dist,
            d_safe=sim.comm_graph.d_safe,
        )
        sim.controller.policy(
            pursuer_states=sp,
            evader_states=se,
            assignment=assignment,
            displacements=sim.displacements,
            weights=weights,
            rng=rng,
            exploration_std=0.0,
            gamma=gamma,
            A_p=A_base,
            delta_matrix=delta_matrix,
            coordination_gradients=coord_grads,
        )
    full_comm_total = time.perf_counter() - t0

    return {
        "vsnac_no_comm_ms_per_step": 1000.0 * no_comm_total / max(n_steps, 1),
        "vsnac_full_comm_ms_per_step": 1000.0 * full_comm_total / max(n_steps, 1),
    }


# ---------------------------------------------------------------------------
# Warm initialisation
# ---------------------------------------------------------------------------


def _warm_init_actor_pd(ac_bank: TraditionalACEquivalent, pos_gain: float = 0.7, vel_gain: float = 1.2) -> None:
    """Initialise the pursuer actor to a proportional-derivative controller
    that drives the (normalised) tracking error to zero.

    Without this warm-start the random-init actor produces chaotic actions,
    pursuers fly off into divergent trajectories before the critic has any
    informative samples, and the actor target ``q_uu^{-1} q_ux x`` has no
    physical meaning. Initialising to a PD controller gives the AC bank a
    sensible starting policy that the critic can then refine, mirroring the
    standard practice of warm-starting actor-critic methods with a
    reasonable prior. Evader actor is left at its small random init since
    the scenario evader is scripted in this study.
    """
    n_p = ac_bank.n_p
    state_dim = ac_bank.state_dim
    action_dim = ac_bank.action_dim
    pursuer = np.zeros((n_p, action_dim, state_dim), dtype=float)
    # action axis 0/1/2 -> position/velocity components 0/1/2
    for axis in range(action_dim):
        pursuer[:, axis, axis] = -float(pos_gain)
        pursuer[:, axis, axis + 3] = -float(vel_gain)
    ac_bank.pursuer_actor = np.clip(pursuer, -ac_bank.actor_weight_clip, ac_bank.actor_weight_clip)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_convergence(history: list[dict[str, float]], path: Path) -> None:
    eps = np.array([h["episode"] for h in history], dtype=float)
    mean_err = np.array([h["mean_assigned_error"] for h in history], dtype=float)
    p_critic = np.array([h["pursuer_critic_norm"] for h in history], dtype=float)
    p_actor = np.array([h["pursuer_actor_norm"] for h in history], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    axes[0].plot(eps, mean_err, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Mean assigned error (m)")
    axes[0].set_title("Traditional AC — convergence")
    axes[0].grid(alpha=0.3)
    axes[1].plot(eps, p_critic, color="#d62728", linewidth=1.6, label="‖pursuer critic‖")
    axes[1].plot(eps, p_actor, color="#2ca02c", linewidth=1.6, label="‖pursuer actor‖")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Weight norm")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train traditional AC baseline and compare against V-SNAC.")
    parser.add_argument("--case", type=str, default="6v3", help="Scenario token, e.g. 3v1, 6v3, 8v4")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--episodes", type=int, default=80, help="AC training episodes")
    parser.add_argument("--steps", type=int, default=120, help="Steps per episode (rollout length)")
    parser.add_argument("--quick", action="store_true", help="Use shorter V-SNAC training schedule for the comparison")
    parser.add_argument("--gamma", type=float, default=0.3, help="V-SNAC coordination strength")
    parser.add_argument("--d-safe", type=float, default=150.0)
    parser.add_argument("--bench-steps", type=int, default=240, help="Steps used for ms/step measurement")
    parser.add_argument("--critic-lr", type=float, default=0.05, help="AC critic learning rate")
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=0.0,
        help=(
            "AC actor learning rate. End-to-end actor+critic learning diverges in this 6-DOF "
            "nonlinear setting because the indirect actor target -q_uu^-1 q_ux x is meaningful "
            "only when Q is exactly quadratic in u. We therefore default to actor_lr=0 (frozen "
            "PD-init actor + learned action-dependent critic), which is the stable variant; the "
            "divergent variant can be reproduced by setting --actor-lr > 0."
        ),
    )
    parser.add_argument(
        "--no-warm-init-pd",
        dest="warm_init_pd",
        action="store_false",
        help="Disable the default PD warm-init (random init only). Demonstrates the divergence baseline.",
    )
    parser.set_defaults(warm_init_pd=True)
    parser.add_argument("--pd-pos-gain", type=float, default=0.7)
    parser.add_argument("--pd-vel-gain", type=float, default=1.2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    case = GeneralCase(
        n_pursuers=int(args.case.split("v")[0]),
        n_evaders=int(args.case.split("v")[1]),
        seed=args.seed,
    )

    spec = _scenario_spec(case, bool(args.quick))
    scenario = build_general_scenario(spec)
    control = _control_params()
    learning = _learning_params(case.n_pursuers, case.n_evaders, bool(args.quick))
    feature = _feature_params()

    sim = MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        comm_params=CommParams(gamma=float(args.gamma), comm_mode="full", d_safe=float(args.d_safe)),
    )

    # Train V-SNAC first (used both as a fair-baseline reference and as a
    # source for the wall-time comparison).
    print(f"[ac-train] training V-SNAC on {case.name} (quick={args.quick})...")
    t0 = time.perf_counter()
    vsnac_train = sim.train_policy(seed=case.seed, dynamic_graph=case.n_evaders > 1)
    vsnac_wall = time.perf_counter() - t0
    print(f"[ac-train] V-SNAC trained in {vsnac_wall:.1f}s")

    # Build AC bank and train.
    ac_bank = TraditionalACEquivalent(
        state_scale=sim.features.state_scale,
        n_p=case.n_pursuers,
        n_e=case.n_evaders,
        u_bar_p=control.u_bar_p,
        u_bar_e=control.u_bar_e,
        q_diag=np.diag(control.q),
        r1_diag=np.diag(control.r1),
        r2_diag=np.diag(control.r2),
        seed=case.seed + 2024,
        critic_lr=float(args.critic_lr),
        actor_lr=float(args.actor_lr),
    )
    if bool(args.warm_init_pd):
        _warm_init_actor_pd(ac_bank, pos_gain=float(args.pd_pos_gain), vel_gain=float(args.pd_vel_gain))
        print(
            f"[ac-train] PD warm-init applied (pos_gain={args.pd_pos_gain}, vel_gain={args.pd_vel_gain})"
        )

    print(f"[ac-train] training Traditional AC for {args.episodes} episodes x {args.steps} steps...")
    t0 = time.perf_counter()
    history = _train_ac_bank(
        sim=sim,
        ac_bank=ac_bank,
        n_episodes=int(args.episodes),
        n_steps_per_episode=int(args.steps),
        perturb_scale=float(learning.random_perturb_scale),
        seed=case.seed + 5000,
    )
    ac_train_wall = time.perf_counter() - t0
    print(f"[ac-train] AC trained in {ac_train_wall:.1f}s")

    last_episode = history[-1] if history else {}
    print(
        "[ac-train] last episode: mean_err={mean:.1f}m critic_norm={cn:.2f} actor_norm={an:.2f}".format(
            mean=last_episode.get("mean_assigned_error", float("nan")),
            cn=last_episode.get("pursuer_critic_norm", float("nan")),
            an=last_episode.get("pursuer_actor_norm", float("nan")),
        )
    )

    # Frozen-policy evaluation
    eval_steps = int(scenario.t_final / sim.learning.dt)
    eval_result = _evaluate_policy_only(sim, ac_bank, seed=case.seed + 7000, n_steps=eval_steps)
    print(
        "[ac-train] frozen-policy eval: mean_err={mean:.1f}m, min_d_min={dmin:.1f}m, capture={cap}".format(
            mean=eval_result["mean_assigned_error"],
            dmin=eval_result["min_d_min_m"],
            cap=eval_result["capture_time_s"],
        )
    )

    # Wall-time benchmark
    print(f"[ac-train] running ms/step benchmark over {args.bench_steps} transitions...")
    ac_timings = _benchmark_ms_per_step(sim, ac_bank, n_steps=int(args.bench_steps), dt=sim.learning.dt)
    vsnac_timings = _benchmark_vsnac(sim, weights=vsnac_train.weights, gamma=float(args.gamma), n_steps=int(args.bench_steps))

    # Output
    out_dir = Path(args.output) if args.output else Path(__file__).resolve().parent / "outputs" / f"ac_train_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case": case.name,
        "seed": case.seed,
        "scenario": {
            "name": scenario.name,
            "n_pursuers": case.n_pursuers,
            "n_evaders": case.n_evaders,
            "t_final_s": float(scenario.t_final),
        },
        "ac_hyperparams": {
            "episodes": int(args.episodes),
            "steps": int(args.steps),
            "critic_lr": float(args.critic_lr),
            "actor_lr": float(args.actor_lr),
        },
        "network_inventory": asdict(ac_bank.parameter_summary()),
        "vsnac_critics": int(case.n_pursuers),
        "vsnac_critic_features": int(sim.features.n_features),
        "vsnac_critic_params_total": int(case.n_pursuers * sim.features.n_features),
        "training_walls_s": {
            "vsnac": float(vsnac_wall),
            "ac": float(ac_train_wall),
        },
        "ac_eval": {
            "mean_assigned_error": eval_result["mean_assigned_error"],
            "min_d_min_m": eval_result["min_d_min_m"],
            "capture_time_s": eval_result["capture_time_s"],
        },
        "ac_history": history,
        "timings_ms_per_step": {**ac_timings, **vsnac_timings},
    }

    dump_json(out_dir / "summary.json", summary)
    _plot_convergence(history, out_dir / "fig_ac_convergence.png")

    md_lines = [
        f"# AC Train Compare: {case.name}",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- case: {case.name}",
        f"- seed: {case.seed}",
        f"- AC hyperparams: critic_lr={args.critic_lr}, actor_lr={args.actor_lr}, episodes={args.episodes}, steps={args.steps}",
        "",
        "## Network inventory",
        f"- Traditional AC total networks: 2 * (N_p + N_e) = {summary['network_inventory']['total_networks']}",
        f"- AC actor params per network: {summary['network_inventory']['actor_parameters_per_network']}",
        f"- AC critic params per network: {summary['network_inventory']['critic_parameters_per_network']} (quadratic basis on z = [x, u, d])",
        f"- AC total scalar parameters: {summary['network_inventory']['total_scalar_parameters']}",
        f"- V-SNAC critics: {summary['vsnac_critics']} (one per pursuer)",
        f"- V-SNAC features per critic: {summary['vsnac_critic_features']}",
        f"- V-SNAC total scalar parameters: {summary['vsnac_critic_params_total']}",
        "",
        "## Training wall time",
        f"- V-SNAC: {summary['training_walls_s']['vsnac']:.1f} s",
        f"- Traditional AC: {summary['training_walls_s']['ac']:.1f} s",
        "",
        "## Frozen-policy evaluation (after AC training)",
        f"- mean assigned error: {eval_result['mean_assigned_error']:.1f} m",
        f"- min(d_min) over episode: {eval_result['min_d_min_m']:.1f} m",
        f"- capture time (s): {eval_result['capture_time_s']}",
        "",
        "## ms / step",
        f"- AC policy_only:  {ac_timings['ac_policy_ms_per_step']:.4f}",
        f"- AC online step:  {ac_timings['ac_online_ms_per_step']:.4f}",
        f"- V-SNAC no_comm:  {vsnac_timings['vsnac_no_comm_ms_per_step']:.4f}",
        f"- V-SNAC full_comm:{vsnac_timings['vsnac_full_comm_ms_per_step']:.4f}",
    ]
    dump_markdown(out_dir / "REPORT.md", "\n".join(md_lines))

    print(f"[ac-train] results saved to: {out_dir}")


if __name__ == "__main__":
    main()
