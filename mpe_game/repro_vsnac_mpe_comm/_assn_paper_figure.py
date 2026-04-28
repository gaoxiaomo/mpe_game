"""Paper-style comparison: auction vs pairwise swap (Xu 2024 baseline) vs Hungarian (oracle)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpe_repro.assignment import assignment_solver_factory, critic_evader_value_predictor
from mpe_repro.comm_graph import CommunicationGraph
from mpe_repro.config import AircraftParams, CommParams
from mpe_repro.general_scenarios import build_general_scenario
from mpe_repro.simulator import MPECommSimulator
from run_assignment_compare import _build_sim, _critic_pursuer_evader_values
from run_comm import GeneralCase, _control_params, _feature_params, _learning_params, _scenario_spec


def run_one(case: GeneralCase, eval_seed: int):
    """Train V-SNAC + run three solvers in closed loop, return team_error trajectories."""
    sim = _build_sim(case, quick=True, gamma=1.5, d_safe=150.0)
    train = sim.train_policy(seed=case.seed, dynamic_graph=True)

    dt = sim.learning.dt
    nu = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    swap_interval = max(int(sim.learning.graph_update_interval), 1)
    hyst_abs = float(sim.scenario.swap_threshold) if sim.scenario.swap_threshold < 1e8 else 5.0
    hyst_rel = 0.05

    out = {}
    for name, kwargs in [
        ("pairwise_swap", {"threshold": float(sim.scenario.swap_threshold)}),
        ("hungarian", {}),
        ("critic_warm_auction", {"epsilon": 0.0001}),
    ]:
        rng = np.random.default_rng(eval_seed)
        p, e = sim._reset_states(rng, perturb_scale=0.005)  # small perturb for seed variation
        sigma = sim.initial_assignment.copy()
        solver = assignment_solver_factory(name, **kwargs)
        eval_cg = CommunicationGraph(
            n_p=case.n_pursuers, comm_mode="full",
            formation_ref_dist=sim.comm_params.formation_ref_dist,
            d_safe=sim.comm_params.d_safe,
        )
        err_hist = []
        steps = int(sim.scenario.t_final / dt)
        for t in range(steps):
            if (t % swap_interval == 0):
                diff = p[:, None, :] - e[None, :, :] + sim.displacements
                cm = np.linalg.norm(diff * nu[None, None, :], axis=2)
                predictor = None
                if "auction" in name:
                    pe_v = _critic_pursuer_evader_values(sim, train.weights, p, e)
                    predictor = critic_evader_value_predictor(pe_v)
                    if hasattr(solver, "reset"):
                        solver.reset()
                new_sigma, _ = solver.solve(cm, sigma, critic_value_predictor=predictor)
                cur_cost = float(np.sum(cm[np.arange(case.n_pursuers), sigma]))
                new_cost = float(np.sum(cm[np.arange(case.n_pursuers), new_sigma]))
                if (cur_cost - new_cost) > max(hyst_abs, hyst_rel * cur_cost) and not np.array_equal(new_sigma, sigma):
                    sigma = new_sigma

            A_p, dm = sim._comm_structures(eval_cg, sigma)
            _, cgrads = sim.controller.coordination_terms(
                p, A_p, dm, sim.comm_params.gamma,
                formation_ref_dist=eval_cg.formation_ref_dist, d_safe=eval_cg.d_safe,
            )
            u_p, u_e_v, _, _, _ = sim.controller.policy(
                p, e, sigma, sim.displacements, train.weights, rng,
                exploration_std=0.0, gamma=sim.comm_params.gamma,
                A_p=A_p, delta_matrix=dm,
                formation_ref_dist=eval_cg.formation_ref_dist, d_safe=eval_cg.d_safe,
                coordination_gradients=cgrads,
            )
            u_e = sim._applied_evader_inputs(t, u_e_v, p, e, sigma, u_p)
            p = sim.dynamics.rk4_step_batch(p, u_p, dt)
            e = sim.dynamics.rk4_step_batch(e, u_e, dt)
            diff = p[:, None, :] - e[None, :, :] + sim.displacements
            pw = np.linalg.norm(diff * nu[None, None, :], axis=2)
            team_err = float(np.sum(pw[np.arange(case.n_pursuers), sigma]))
            err_hist.append(team_err)
        out[name] = np.array(err_hist)
    return out, dt


def main():
    cases = [
        ("8v4 random", GeneralCase(8, 4, seed=11, assignment_mode="random", layout_mode="random")),
        ("10v5 random", GeneralCase(10, 5, seed=11, assignment_mode="random", layout_mode="random")),
        ("12v6 random", GeneralCase(12, 6, seed=11, assignment_mode="random", layout_mode="random")),
    ]
    n_seeds = 5
    seed_list = [1011 + 7 * k for k in range(n_seeds)]

    all_data = {}
    for label, case in cases:
        print(f"[paper-fig] running {label} over {n_seeds} seeds")
        per_solver = {"pairwise_swap": [], "hungarian": [], "critic_warm_auction": []}
        dt = 0.05
        for s in seed_list:
            traj, dt = run_one(case, eval_seed=s)
            for k in per_solver:
                per_solver[k].append(traj[k])
        # stack
        L = min(min(t.size for t in v) for v in per_solver.values())
        for k in per_solver:
            per_solver[k] = np.stack([t[:L] for t in per_solver[k]], axis=0)  # (n_seeds, L)
        all_data[label] = (per_solver, dt, L)
        # print summary
        for k, arr in per_solver.items():
            mean_err = np.mean(arr, axis=1).mean()  # mean over seeds of mean over time
            print(f"  {k:25s}: mean team_err over episode = {mean_err:.0f} m (5 seeds avg)")

    # ---- 3-panel paper figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    colors = {"pairwise_swap": "#d62728", "hungarian": "#1f77b4", "critic_warm_auction": "#2ca02c"}
    pretty = {
        "pairwise_swap": "Pairwise swap (Xu 2024 baseline)",
        "hungarian": "Hungarian (centralized oracle)",
        "critic_warm_auction": "Critic-Warm Auction (ours)",
    }

    for ax_idx, (label, (data, dt, L)) in enumerate(all_data.items()):
        ax = axes[ax_idx]
        ts = np.arange(L) * dt
        for k in ["pairwise_swap", "hungarian", "critic_warm_auction"]:
            arr = data[k]  # (n_seeds, L)
            median = np.median(arr, axis=0)
            q25 = np.quantile(arr, 0.25, axis=0)
            q75 = np.quantile(arr, 0.75, axis=0)
            ax.plot(ts, median, color=colors[k], linewidth=2.0, label=pretty[k])
            ax.fill_between(ts, q25, q75, color=colors[k], alpha=0.15)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("time (s)")
        if ax_idx == 0:
            ax.set_ylabel("team error  $\\sum_j \\|\\tilde x_j(t)\\|$  (m)")
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Team error trajectory: ours vs Xu 2024 baseline (median ± IQR over 5 seeds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path("outputs/_paper_assn_compare.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nsaved {out_path}")

    # ---- Bar chart of mean team_error ----
    fig2, ax = plt.subplots(figsize=(8.5, 5))
    labels = list(all_data.keys())
    methods = ["pairwise_swap", "hungarian", "critic_warm_auction"]
    x = np.arange(len(labels))
    width = 0.25
    for i, m in enumerate(methods):
        means = [np.mean(np.mean(all_data[lbl][0][m], axis=1)) for lbl in labels]
        stds = [np.std(np.mean(all_data[lbl][0][m], axis=1)) for lbl in labels]
        ax.bar(x + (i - 1) * width, means, width, color=colors[m], yerr=stds, capsize=4, label=pretty[m])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean team error over episode (m)")
    ax.set_title("Mean team error: ours vs Xu 2024 baseline (5 seeds, lower = better)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper left")
    fig2.tight_layout()
    out2 = Path("outputs/_paper_assn_bar.png")
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f"saved {out2}")

    # ---- Save summary JSON ----
    summary = {}
    for lbl, (data, dt, L) in all_data.items():
        summary[lbl] = {
            m: {
                "mean_team_err_over_episode_m": float(np.mean(np.mean(data[m], axis=1))),
                "std_over_seeds_m": float(np.std(np.mean(data[m], axis=1))),
            }
            for m in methods
        }
    Path("outputs/_paper_assn_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved outputs/_paper_assn_summary.json")


if __name__ == "__main__":
    main()
