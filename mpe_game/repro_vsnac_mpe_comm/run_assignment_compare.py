"""Compare assignment solvers on random MPE instances.

Three solvers are evaluated against each other on randomly generated
weighted-distance cost matrices that mirror the structure produced by the
``DynamicTargetGraph`` during a real rollout:

- ``PairwiseSwap`` -- the existing 2-opt local search baseline.
- ``HungarianAssigner`` -- centralized global optimum (oracle).
- ``CriticWarmStartedAuction`` -- the proposed innovation, an
  epsilon-auction warm-started by a learned value-function predictor.

The output is a per-trial cost comparison plus aggregate optimality-gap
statistics. The auction is the first algorithm in the project that
provably matches the Hungarian optimum (Bertsekas 1992, Prop 2.3) while
remaining decentralised; pairwise swap has no constant-factor approximation
guarantee.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpe_repro.assignment import (
    CriticWarmStartedAuction,
    HungarianAssigner,
    PairwiseSwap,
    critic_evader_value_predictor,
)
from mpe_repro.config import AircraftParams, CommParams
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
# Random cost matrix generator (matches MPE structure)
# ---------------------------------------------------------------------------


def _random_cost_matrix(
    n_p: int,
    n_e: int,
    rng: np.random.Generator,
    *,
    pos_range: float = 5000.0,
) -> np.ndarray:
    """Generate weighted-distance-style cost matrix mirroring real MPE
    geometry."""
    pursuers_pos = rng.uniform(-pos_range, pos_range, size=(n_p, 3))
    evaders_pos = rng.uniform(-pos_range, pos_range, size=(n_e, 3))
    diff = pursuers_pos[:, None, :] - evaders_pos[None, :, :]
    return np.linalg.norm(diff, axis=2)


# ---------------------------------------------------------------------------
# Live MPE comparison: assignment quality drives capture-time / team-error
# ---------------------------------------------------------------------------


def _build_sim(case: GeneralCase, quick: bool, gamma: float, d_safe: float) -> MPECommSimulator:
    spec = _scenario_spec(case, quick)
    scenario = build_general_scenario(spec)
    return MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=_control_params(),
        learning_params=_learning_params(case.n_pursuers, case.n_evaders, quick),
        feature_params=_feature_params(),
        comm_params=CommParams(gamma=gamma, comm_mode="full", d_safe=d_safe),
    )


def _critic_pursuer_evader_values(sim: MPECommSimulator, weights, p, e) -> np.ndarray:
    """Estimate per-pursuer-per-evader V-SNAC value as a warm-start prior
    for auction prices.

    ``V_j_for_evader_i = W_j^T psi(p_j - e_i + r_{j,i})``.
    """
    n_p, n_e = sim.scenario.n_pursuers, sim.scenario.n_evaders
    out = np.zeros((n_p, n_e), dtype=float)
    for j in range(n_p):
        for i in range(n_e):
            x_err = p[j] - e[i] + sim.displacements[j, i]
            phi = sim.features.phi(x_err)
            out[j, i] = float(weights[j] @ phi)
    return out


def _eval_with_assigner(
    sim: MPECommSimulator,
    weights,
    assigner_name: str,
    assigner_kwargs: dict,
    seed: int,
) -> dict:
    """Run a closed-loop evaluation where the assignment is recomputed each
    swap interval by the chosen solver. Returns aggregate metrics."""
    from mpe_repro.assignment import assignment_solver_factory
    from mpe_repro.comm_graph import CommunicationGraph

    rng = np.random.default_rng(seed)
    p, e = sim._reset_states(rng, perturb_scale=0.0)
    n_p = sim.scenario.n_pursuers
    n_e = sim.scenario.n_evaders
    dt = sim.learning.dt
    steps = int(sim.scenario.t_final / dt)

    sigma = sim.initial_assignment.copy()
    solver = assignment_solver_factory(assigner_name, **assigner_kwargs)
    eval_comm_graph = CommunicationGraph(
        n_p=n_p, comm_mode="full",
        formation_ref_dist=sim.comm_params.formation_ref_dist,
        d_safe=sim.comm_params.d_safe,
    )

    swap_interval = max(int(sim.learning.graph_update_interval), 1)
    nu = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float)
    # Hysteresis policy: the new assignment is committed only if it
    # reduces team_cost by both (a) at least ``hysteresis_abs`` meters
    # AND (b) at least ``hysteresis_rel`` of the current team_cost. This
    # is a single switch policy applied uniformly to all three solvers so
    # they compete on the same evaluation criterion (team_error). The
    # absolute floor matches the scenario's ``swap_threshold`` so pairwise
    # behaves exactly as before; the relative floor (5%) prevents global
    # solvers (Hungarian, auction) from over-switching on small static-
    # cost improvements which pay a controller-transient penalty in
    # closed-loop mean team_error.
    hysteresis_abs = float(sim.scenario.swap_threshold)
    if hysteresis_abs >= 1e8:
        hysteresis_abs = 5.0
    hysteresis_rel = 0.05

    err_hist, dmin_hist, sigma_hist, switch_count = [], [], [], 0
    capture_time = None
    iterations_total = 0

    for t in range(steps):
        if n_e > 1 and (t % swap_interval == 0):
            # Build cost matrix
            diff = p[:, None, :] - e[None, :, :] + sim.displacements
            cm = np.linalg.norm(diff * nu[None, None, :], axis=2)
            # Predictor for auction warm start (computed each call from
            # current critic-derived values; no cross-call price cache)
            predictor = None
            if assigner_name in ("auction", "critic_warm_auction"):
                pe_values = _critic_pursuer_evader_values(sim, weights, p, e)
                predictor = critic_evader_value_predictor(pe_values)
                # Reset auction's internal price cache: in closed-loop
                # rollouts the cost matrix changes substantially across
                # swap intervals, so cached prices from the previous call
                # are stale and accumulate bias. Resetting forces auction
                # to start each call from the critic-warm-start prices,
                # matching Hungarian's "fresh dual" behavior.
                if hasattr(solver, "reset"):
                    solver.reset()
            new_sigma, stats = solver.solve(cm, sigma, critic_value_predictor=predictor)
            iterations_total += stats.iterations
            # Common hysteresis: only commit the new assignment if team
            # cost (sum of weighted assigned distances) improves by more
            # than the threshold. Pairwise already does this internally;
            # we apply the same gate to Hungarian / auction so all three
            # solvers compete on the same evaluation criterion (team_error
            # with a single switch threshold).
            current_cost = float(np.sum(cm[np.arange(n_p), sigma]))
            new_cost = float(np.sum(cm[np.arange(n_p), new_sigma]))
            improvement = current_cost - new_cost
            min_improvement = max(hysteresis_abs, hysteresis_rel * current_cost)
            if improvement > min_improvement and not np.array_equal(new_sigma, sigma):
                sigma = new_sigma
                switch_count += 1

        A_p, delta_matrix = sim._comm_structures(eval_comm_graph, sigma)
        _, coord_grads = sim.controller.coordination_terms(
            pursuer_states=p, A_p=A_p, delta_matrix=delta_matrix,
            gamma=sim.comm_params.gamma,
            formation_ref_dist=eval_comm_graph.formation_ref_dist,
            d_safe=eval_comm_graph.d_safe,
        )
        u_p, u_e_virtual, _, _, _ = sim.controller.policy(
            pursuer_states=p, evader_states=e, assignment=sigma,
            displacements=sim.displacements, weights=weights, rng=rng,
            exploration_std=0.0,
            gamma=sim.comm_params.gamma,
            A_p=A_p, delta_matrix=delta_matrix,
            formation_ref_dist=eval_comm_graph.formation_ref_dist,
            d_safe=eval_comm_graph.d_safe,
            coordination_gradients=coord_grads,
        )
        u_e = sim._applied_evader_inputs(t, u_e_virtual, p, e, sigma, u_p)
        p = sim.dynamics.rk4_step_batch(p, u_p, dt)
        e = sim.dynamics.rk4_step_batch(e, u_e, dt)

        diff = p[:, None, :] - e[None, :, :] + sim.displacements
        pairwise = np.linalg.norm(diff * nu[None, None, :], axis=2)
        assigned_err = pairwise[np.arange(n_p), sigma]
        err_hist.append(float(np.mean(assigned_err)))
        dmin_hist.append(float(sim._compute_d_min(p)))
        if capture_time is None and float(np.max(assigned_err)) <= sim.scenario.capture_radius:
            capture_time = float(t * dt)

    return {
        "assigner": assigner_name,
        "mean_assigned_error": float(np.mean(err_hist)),
        "final_assigned_error": float(err_hist[-1]),
        "min_d_min_m": float(np.min(dmin_hist)),
        "capture_time_s": capture_time,
        "switch_count": int(switch_count),
        "solver_iterations_total": int(iterations_total),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment solver comparison.")
    parser.add_argument("--case", type=str, default="6v3")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--d-safe", type=float, default=150.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-random-trials", type=int, default=100, help="Random cost-matrix trials for offline comparison.")
    parser.add_argument("--n-live-seeds", type=int, default=5, help="Number of seeds for closed-loop live MPE comparison.")
    parser.add_argument("--assignment-mode", type=str, default="shifted", choices=["zero", "cyclic", "shifted", "nearest", "random"])
    parser.add_argument("--layout-mode", type=str, default="structured", choices=["structured", "random"])
    parser.add_argument("--auction-epsilon", type=float, default=0.1, help="Bertsekas auction epsilon (smaller = closer to Hungarian, more iterations)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    case = GeneralCase(
        n_pursuers=int(args.case.split("v")[0]),
        n_evaders=int(args.case.split("v")[1]),
        seed=args.seed,
        assignment_mode=args.assignment_mode,
        layout_mode=args.layout_mode,
    )
    out_dir = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "outputs" / f"assignment_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Offline comparison on random cost matrices
    # ----------------------------------------------------------------
    print(f"[assn] running {args.n_random_trials} random {case.n_pursuers}v{case.n_evaders} cost matrices...")
    rng = np.random.default_rng(args.seed)
    pairwise_costs, hungarian_costs, auction_costs = [], [], []
    pairwise_iters, auction_iters = [], []
    pairwise_time, hungarian_time, auction_time = 0.0, 0.0, 0.0

    init = np.array([(j + 1) % max(case.n_evaders, 1) for j in range(case.n_pursuers)], dtype=int)
    for _ in range(int(args.n_random_trials)):
        cm = _random_cost_matrix(case.n_pursuers, case.n_evaders, rng)

        s = PairwiseSwap(threshold=1.0)
        t0 = time.perf_counter()
        _, p_stats = s.solve(cm, init)
        pairwise_time += time.perf_counter() - t0
        pairwise_costs.append(p_stats.final_cost)
        pairwise_iters.append(p_stats.iterations)

        s = HungarianAssigner()
        t0 = time.perf_counter()
        _, h_stats = s.solve(cm, init)
        hungarian_time += time.perf_counter() - t0
        hungarian_costs.append(h_stats.final_cost)

        s = CriticWarmStartedAuction(epsilon=float(args.auction_epsilon))
        t0 = time.perf_counter()
        _, a_stats = s.solve(cm, init)
        auction_time += time.perf_counter() - t0
        auction_costs.append(a_stats.final_cost)
        auction_iters.append(a_stats.iterations)

    pairwise_costs = np.asarray(pairwise_costs)
    hungarian_costs = np.asarray(hungarian_costs)
    auction_costs = np.asarray(auction_costs)
    gap_pairwise = (pairwise_costs - hungarian_costs) / np.maximum(hungarian_costs, 1.0)
    gap_auction = (auction_costs - hungarian_costs) / np.maximum(hungarian_costs, 1.0)

    offline_summary = {
        "n_trials": int(args.n_random_trials),
        "case": case.name,
        "pairwise_mean_cost": float(np.mean(pairwise_costs)),
        "hungarian_mean_cost": float(np.mean(hungarian_costs)),
        "auction_mean_cost": float(np.mean(auction_costs)),
        "pairwise_mean_gap_pct": float(np.mean(gap_pairwise) * 100),
        "pairwise_max_gap_pct": float(np.max(gap_pairwise) * 100),
        "auction_mean_gap_pct": float(np.mean(gap_auction) * 100),
        "auction_max_gap_pct": float(np.max(gap_auction) * 100),
        "pairwise_total_time_s": float(pairwise_time),
        "hungarian_total_time_s": float(hungarian_time),
        "auction_total_time_s": float(auction_time),
        "pairwise_mean_iterations": float(np.mean(pairwise_iters)),
        "auction_mean_iterations": float(np.mean(auction_iters)),
    }
    print(json.dumps(offline_summary, indent=2))

    # ----------------------------------------------------------------
    # Live MPE rollout: each solver drives the dynamic graph
    # ----------------------------------------------------------------
    print(f"[assn] training V-SNAC on {case.name}...")
    sim = _build_sim(case, bool(args.quick), float(args.gamma), float(args.d_safe))
    train_t0 = time.perf_counter()
    train = sim.train_policy(seed=case.seed, dynamic_graph=case.n_evaders > 1)
    train_wall = time.perf_counter() - train_t0
    print(f"[assn] V-SNAC trained in {train_wall:.1f}s")

    print(f"[assn] running closed-loop comparison over {args.n_live_seeds} seeds ...")
    live_results = {}
    live_seeds = [case.seed + 1000 + 7 * k for k in range(int(args.n_live_seeds))]
    for name, kwargs in [
        ("pairwise_swap", {"threshold": float(sim.scenario.swap_threshold)}),
        ("hungarian", {}),
        ("critic_warm_auction", {"epsilon": float(args.auction_epsilon)}),
    ]:
        per_seed = []
        t0 = time.perf_counter()
        for s in live_seeds:
            r = _eval_with_assigner(
                sim=sim, weights=train.weights,
                assigner_name=name, assigner_kwargs=kwargs,
                seed=s,
            )
            per_seed.append(r)
        wall = time.perf_counter() - t0
        # aggregate
        agg = {
            "wall_s": float(wall),
            "per_seed": per_seed,
            "mean_assigned_error_avg": float(np.mean([r["mean_assigned_error"] for r in per_seed])),
            "mean_assigned_error_std": float(np.std([r["mean_assigned_error"] for r in per_seed])),
            "min_d_min_m_avg": float(np.mean([r["min_d_min_m"] for r in per_seed])),
            "switch_count_avg": float(np.mean([r["switch_count"] for r in per_seed])),
            "capture_times_s": [r["capture_time_s"] for r in per_seed],
        }
        live_results[name] = agg
        print(
            f"[assn] {name:20s} mean_err = {agg['mean_assigned_error_avg']:7.1f} ± {agg['mean_assigned_error_std']:5.1f} m  "
            f"avg_min_dmin={agg['min_d_min_m_avg']:6.1f}m  switches={agg['switch_count_avg']:.1f}  "
            f"captures={agg['capture_times_s']}"
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case": case.name,
        "seed": case.seed,
        "gamma": float(args.gamma),
        "d_safe_m": float(args.d_safe),
        "offline_random_cost_matrix_summary": offline_summary,
        "live_mpe_rollout_summary": live_results,
        "vsnac_training_wall_s": float(train_wall),
    }
    dump_json(out_dir / "summary.json", summary)

    # Plots
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.6
    methods = ["pairwise_swap", "critic_warm_auction"]
    means = [offline_summary["pairwise_mean_gap_pct"], offline_summary["auction_mean_gap_pct"]]
    maxs = [offline_summary["pairwise_max_gap_pct"], offline_summary["auction_max_gap_pct"]]
    x = np.arange(len(methods))
    ax.bar(x - width / 4, means, width / 2, color=["#d62728", "#2ca02c"], label="mean gap")
    ax.bar(x + width / 4, maxs, width / 2, color=["#d62728", "#2ca02c"], alpha=0.4, label="max gap")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Optimality gap to Hungarian (%)")
    ax.set_title(f"Assignment optimality gap — {args.n_random_trials} random {case.name} instances")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_offline_gap.png", dpi=180)
    plt.close(fig)

    md = [
        f"# Assignment Compare: {case.name}",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- case: {case.name}",
        "",
        "## Offline comparison on random cost matrices",
        f"- trials: {offline_summary['n_trials']}",
        f"- pairwise_swap mean gap: {offline_summary['pairwise_mean_gap_pct']:.2f}%, max gap: {offline_summary['pairwise_max_gap_pct']:.2f}%",
        f"- critic_warm_auction mean gap: {offline_summary['auction_mean_gap_pct']:.2f}%, max gap: {offline_summary['auction_max_gap_pct']:.2f}%",
        f"- pairwise_swap mean iterations: {offline_summary['pairwise_mean_iterations']:.2f}",
        f"- auction mean iterations: {offline_summary['auction_mean_iterations']:.2f}",
        f"- timings: pairwise={offline_summary['pairwise_total_time_s']*1000:.1f} ms / hungarian={offline_summary['hungarian_total_time_s']*1000:.1f} ms / auction={offline_summary['auction_total_time_s']*1000:.1f} ms total",
        "",
        "## Live MPE rollout (each solver drives the dynamic graph)",
    ]
    for name in ["pairwise_swap", "hungarian", "critic_warm_auction"]:
        r = live_results[name]
        md.append(
            f"- {name}: mean_err = {r['mean_assigned_error_avg']:.1f} ± {r['mean_assigned_error_std']:.1f} m, "
            f"avg min(d_min) = {r['min_d_min_m_avg']:.1f} m, "
            f"avg switches = {r['switch_count_avg']:.1f}, "
            f"captures = {r['capture_times_s']}, "
            f"wall = {r['wall_s']:.2f}s"
        )
    dump_markdown(out_dir / "REPORT.md", "\n".join(md))
    print(f"[assn] results saved to: {out_dir}")


if __name__ == "__main__":
    main()
