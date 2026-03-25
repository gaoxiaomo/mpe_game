"""Team V-SNAC: Two-stage communication-coupled value function experiments.

Stage 1: Standard V-SNAC (individual W_j, proven convergence)
Stage 2: Fix W_j, train shared coupling W_c (9-dim: 6 state + 3 control)

Evaluates: team (full comm) vs local (no comm) vs state-only vs dropout.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

from mpe_repro.config import (
    AircraftParams,
    ControlParams,
    FeatureParams,
    LearningParams,
    TeamParams,
)
from mpe_repro.general_scenarios import GeneralScenarioSpec, build_general_scenario
from mpe_repro.plotting import (
    plot_angular_coverage,
    plot_assigned_errors,
    plot_assignment_timeline,
    plot_comparison_bars,
    plot_d_min,
    plot_path_overlap,
    plot_team_error_compare,
    plot_trajectory_xy,
    plot_weight_convergence,
)
from mpe_repro.report import dump_json, dump_markdown, eval_summary, network_summary, train_summary
from mpe_repro.simulator import TeamMPESimulator, TrainResult


# ------------------------------------------------------------------
# Parameter factories
# ------------------------------------------------------------------

def _feature_params() -> FeatureParams:
    return FeatureParams(
        state_scale=(3200.0, 3200.0, 2200.0, 160.0, 160.0, 160.0),
        feature_gain=2.7,
    )


def _control_params() -> ControlParams:
    return ControlParams(
        q_diag=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        r1_diag=(1.0, 1.0, 1.0),
        r2_diag=(1.0, 1.0, 1.0),
        u_bar_p=25.0,
        u_bar_e=15.0,
        u_bar_p_policy=35.0,
        u_bar_e_policy=15.0,
        policy_gain=28.0,
        k_pos_p=0.020,
        k_vel_p=0.160,
        k_pos_e=0.010,
        k_vel_e=0.080,
    )


def _learning_params(n_p: int, n_e: int) -> LearningParams:
    size = max(n_p, n_e)
    return LearningParams(
        policy_iterations=55 + 3 * max(0, size - 3),
        rollout_steps=200 + 20 * size,
        min_samples_per_critic=400 + 60 * size,
        graph_update_interval=1,
        graph_update_start_step=0,
        critic_learning_rate=0.01,
        convergence_tol=5e-3,
        random_perturb_scale=0.02,
    )


def _team_params() -> TeamParams:
    return TeamParams(
        comm_mode="full",
        coupling_gain=2.0,
        gamma_sep=0.5,
        d_ref=2000.0,
        dropout_start_s=12.0,
        dropout_end_s=22.0,
        coupling_w_min=-0.20,
        coupling_w_max=0.20,
    )


def _scenario_spec(n_p: int, n_e: int, seed: int) -> GeneralScenarioSpec:
    return GeneralScenarioSpec(
        n_pursuers=n_p,
        n_evaders=n_e,
        seed=seed,
        assignment_mode="shifted" if n_e > 1 else "zero",
        layout_mode="structured",
        t_final=90.0 if n_e == 1 else 140.0,
        capture_radius=180.0 if n_e == 1 else 220.0,
        swap_threshold=5.0 if n_e > 1 else 1.0e9,
        max_switch_worsening=0.0,
        evader_motion_mode="scripted",
        evader_script_amp=(10.0, 8.0, 4.0) if n_e == 1 else (8.0, 8.0, 3.0),
        evader_script_omega=0.42 if n_e == 1 else 0.16,
        evader_script_decay=0.030 if n_e == 1 else 0.010,
        evader_script_mix=0.60 if n_e == 1 else 0.35,
        swap_lookahead_time=0.0 if n_e == 1 else 0.5,
    )


# ------------------------------------------------------------------
# Run a single case
# ------------------------------------------------------------------

def run_case(
    n_p: int,
    n_e: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    case_name = f"{n_p}v{n_e}"
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{case_name}] starting ...")

    scenario = build_general_scenario(_scenario_spec(n_p, n_e, seed))
    control = _control_params()
    learning = _learning_params(n_p, n_e)
    feature = _feature_params()
    team = _team_params()
    dynamic_graph = scenario.n_evaders > 1
    size = max(n_p, n_e)

    # ---- Build simulator ----
    sim = TeamMPESimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        team_params=team,
    )

    # ---- Stage 1: Standard V-SNAC ----
    t0 = time.perf_counter()
    train1 = sim.train_policy(seed=seed, dynamic_graph=dynamic_graph)
    t1 = time.perf_counter() - t0
    print(f"  [{case_name}] Stage 1 done ({t1:.1f}s, {len(train1.exploration_history)} iters)")

    # ---- Stage 2: Coupling W_c ----
    t0 = time.perf_counter()
    train2 = sim.train_policy_stage2(
        stage1_weights=train1.weights,
        seed=seed,
        dynamic_graph=dynamic_graph,
        stage2_iters=30 + 2 * max(0, size - 3),
        stage2_rollout_steps=220 + 15 * size,
        stage2_lr=0.03,
        stage2_explore_start=0.5,
        stage2_explore_end=0.05,
        stage2_tol=3e-3,
    )
    t2 = time.perf_counter() - t0
    wc_iters = len(train2.w_c_delta_history) if train2.w_c_delta_history is not None else 0
    print(f"  [{case_name}] Stage 2 done ({t2:.1f}s, {wc_iters} iters, W_c={np.round(train2.w_c, 4).tolist()})")

    eval_seed = seed + 1000
    weights = train1.weights
    w_c = train2.w_c

    # ---- Evaluate: TEAM (full comm) ----
    eval_team = sim.evaluate_policy(
        weights=weights, seed=eval_seed, dynamic_graph=dynamic_graph,
        w_c=w_c, comm_mode="full", stop_on_capture=False,
    )

    # ---- Evaluate: LOCAL (no comm) ----
    eval_local = sim.evaluate_policy(
        weights=weights, seed=eval_seed, dynamic_graph=dynamic_graph,
        w_c=None, comm_mode="local", stop_on_capture=False,
    )

    # ---- Evaluate: STATE-ONLY (state coupling only, no control coupling) ----
    eval_state = sim.evaluate_policy(
        weights=weights, seed=eval_seed, dynamic_graph=dynamic_graph,
        w_c=w_c, comm_mode="state_only", stop_on_capture=False,
    )

    # ---- Evaluate: DROPOUT ----
    eval_dropout = sim.evaluate_policy(
        weights=weights, seed=eval_seed, dynamic_graph=dynamic_graph,
        w_c=w_c, comm_mode="dropout",
        dropout_start_s=team.dropout_start_s, dropout_end_s=team.dropout_end_s,
        stop_on_capture=False,
    )

    print(f"  [{case_name}] evaluation done")

    # ---- Summaries ----
    dt = learning.dt
    sum_team = eval_summary(eval_team, "team_full_comm")
    sum_local = eval_summary(eval_local, "local_no_comm")
    sum_state = eval_summary(eval_state, "team_state_only")
    sum_dropout = eval_summary(eval_dropout, "team_dropout")

    # ---- Plots ----
    results = {
        "Team (full)": eval_team,
        "Local": eval_local,
        "State-only": eval_state,
        "Dropout": eval_dropout,
    }

    plot_trajectory_xy(eval_team, case_dir / "fig_trajectory_team.png", f"{case_name} Team trajectory")
    plot_trajectory_xy(eval_local, case_dir / "fig_trajectory_local.png", f"{case_name} Local trajectory")

    plot_assigned_errors(eval_team, dt, case_dir / "fig_errors_team.png", f"{case_name} Team assigned errors")
    plot_assigned_errors(eval_local, dt, case_dir / "fig_errors_local.png", f"{case_name} Local assigned errors")

    plot_weight_convergence(train1, case_dir / "fig_weight_convergence.png", f"{case_name} Stage 1 weights")

    plot_team_error_compare(results, dt, case_dir / "fig_team_error_compare.png", f"{case_name} Team Error Comparison")
    plot_d_min(results, dt, case_dir / "fig_d_min_compare.png", f"{case_name} Min Inter-Pursuer Distance")
    plot_angular_coverage(results, dt, case_dir / "fig_angular_coverage.png", f"{case_name} Angular Coverage")
    plot_path_overlap(results, dt, case_dir / "fig_path_overlap.png", f"{case_name} Path Overlap")

    if scenario.n_evaders > 1:
        plot_assignment_timeline(eval_team, dt, case_dir / "fig_assignment_team.png", f"{case_name} Team assignments")

    plot_comparison_bars(
        {"Team": sum_team, "Local": sum_local, "State": sum_state, "Dropout": sum_dropout},
        case_dir / "fig_bars.png",
        f"{case_name} Metric Comparison",
    )

    # ---- JSON ----
    summary = {
        "case": {"name": case_name, "n_pursuers": n_p, "n_evaders": n_e, "seed": seed},
        "train": train_summary(train1),
        "w_c": np.round(w_c, 6).tolist() if w_c is not None else None,
        "w_c_iters": wc_iters,
        "eval_team": sum_team,
        "eval_local": sum_local,
        "eval_state_only": sum_state,
        "eval_dropout": sum_dropout,
        "network": network_summary(n_p, n_e, sim.features.n_features),
        "timing": {"stage1_s": t1, "stage2_s": t2},
    }
    dump_json(case_dir / "summary.json", summary)

    # ---- Markdown report ----
    md = [
        f"# Team V-SNAC Report: {case_name}",
        "",
        "## Capture Time",
        "| Mode | Capture (s) |",
        "|---|---:|",
        f"| Team (full comm) | {sum_team['capture_time_s']} |",
        f"| Local (no comm) | {sum_local['capture_time_s']} |",
        f"| State-only | {sum_state['capture_time_s']} |",
        f"| Dropout | {sum_dropout['capture_time_s']} |",
        "",
        "## Coordination Metrics (mean)",
        "| Metric | Team | Local | State-only | Dropout |",
        "|---|---:|---:|---:|---:|",
        "| d_min (m) | {:.1f} | {:.1f} | {:.1f} | {:.1f} |".format(
            sum_team.get('d_min_mean', 0), sum_local.get('d_min_mean', 0),
            sum_state.get('d_min_mean', 0), sum_dropout.get('d_min_mean', 0)),
        "| angular coverage | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            sum_team.get('angular_coverage_mean', 0), sum_local.get('angular_coverage_mean', 0),
            sum_state.get('angular_coverage_mean', 0), sum_dropout.get('angular_coverage_mean', 0)),
        "| path overlap | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            sum_team.get('path_overlap_mean', 0), sum_local.get('path_overlap_mean', 0),
            sum_state.get('path_overlap_mean', 0), sum_dropout.get('path_overlap_mean', 0)),
        "",
        f"## Coupling Weights W_c",
        f"```",
        f"{np.round(w_c, 4).tolist() if w_c is not None else 'None'}",
        f"```",
    ]
    dump_markdown(case_dir / "REPORT.md", "\n".join(md))

    print(f"  [{case_name}] done. Team cap={sum_team['capture_time_s']}s, Local cap={sum_local['capture_time_s']}s")
    return summary


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Team V-SNAC two-stage experiments")
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help="Case tokens like 3v1 4v1 5v1 4v2 6v3. Default: 3v1, 4v1, 5v1.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.cases:
        cases = []
        for tok in args.cases:
            tok = tok.lower().strip()
            for sep in ("v", "x"):
                if sep in tok:
                    left, right = tok.split(sep, 1)
                    cases.append((int(left), int(right)))
                    break
    else:
        cases = [(3, 1), (4, 1), (5, 1)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"team_vsnac_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir}")
    print(f"Cases: {cases}")

    all_summaries = []
    for idx, (n_p, n_e) in enumerate(cases):
        s = run_case(n_p, n_e, seed=args.seed + 10 * idx, out_dir=out_dir)
        all_summaries.append(s)

    # Batch summary
    batch = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": all_summaries,
    }
    dump_json(out_dir / "batch_summary.json", batch)

    # Batch report
    md = [
        "# Team V-SNAC Batch Report",
        "",
        "## Summary",
        "| Case | Team cap (s) | Local cap (s) | State cap (s) | Team d_min | Local d_min | Team ang_cov | Local ang_cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_summaries:
        md.append(
            "| {name} | {tc} | {lc} | {sc} | {td:.0f} | {ld:.0f} | {ta:.3f} | {la:.3f} |".format(
                name=s["case"]["name"],
                tc=s["eval_team"]["capture_time_s"] or "-",
                lc=s["eval_local"]["capture_time_s"] or "-",
                sc=s["eval_state_only"]["capture_time_s"] or "-",
                td=s["eval_team"].get("d_min_mean", 0),
                ld=s["eval_local"].get("d_min_mean", 0),
                ta=s["eval_team"].get("angular_coverage_mean", 0),
                la=s["eval_local"].get("angular_coverage_mean", 0),
            )
        )
    dump_markdown(out_dir / "REPORT.md", "\n".join(md))
    print(f"\nAll done. Results in {out_dir}")


if __name__ == "__main__":
    main()
