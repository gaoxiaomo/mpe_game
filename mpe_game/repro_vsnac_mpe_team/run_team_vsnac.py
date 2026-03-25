"""Team V-SNAC: Communication-coupled team value function experiments.

Runs team (full comm) vs local (no comm) vs dropout comparisons
for configurable m-vs-n pursuit-evasion scenarios.
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
        policy_iterations=36 + 3 * max(0, size - 3),
        rollout_steps=220 + 25 * size,
        min_samples_per_critic=500 + 80 * size,
        graph_update_interval=1,
        graph_update_start_step=0,
        critic_learning_rate=0.012,
        convergence_tol=8e-5,
        random_perturb_scale=0.02,
    )


def _team_params(comm_mode: str = "full") -> TeamParams:
    return TeamParams(
        comm_mode=comm_mode,
        coupling_gain=0.0,  # unused in revised approach
        gamma_sep=4.5,      # separation gradient alpha (competes with ∇V ~3.5)
        d_ref=380.0,        # range of repulsion (m) - focused on close proximity
        dropout_start_s=12.0,
        dropout_end_s=22.0,
        coupling_w_min=-0.35,
        coupling_w_max=0.35,
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
    team = _team_params("full")
    dynamic_graph = scenario.n_evaders > 1

    # ---- Build simulator ----
    sim = TeamMPESimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        team_params=team,
    )

    # ---- Train standard V-SNAC (no separation), proven convergence ----
    t0 = time.perf_counter()
    train_result = sim.train_policy(seed=seed, dynamic_graph=dynamic_graph, use_separation=False)
    train_s = time.perf_counter() - t0
    print(f"  [{case_name}] training done ({train_s:.1f}s, {len(train_result.exploration_history)} iters)")

    eval_seed = seed + 1000
    weights = train_result.weights

    # ---- Evaluate: TEAM (full comm, separation gradient ON) ----
    eval_team = sim.evaluate_policy(
        weights=weights, seed=eval_seed, dynamic_graph=dynamic_graph,
        use_separation=True, comm_mode="full", stop_on_capture=False,
    )

    # ---- Evaluate: LOCAL (no comm, separation gradient OFF) ----
    eval_local = sim.evaluate_policy(
        weights=weights, seed=eval_seed, dynamic_graph=dynamic_graph,
        use_separation=False, comm_mode="local", stop_on_capture=False,
    )

    # ---- Evaluate: DROPOUT (separation ON, but frozen states during dropout) ----
    eval_dropout = sim.evaluate_policy(
        weights=weights, seed=eval_seed, dynamic_graph=dynamic_graph,
        use_separation=True, comm_mode="dropout",
        dropout_start_s=team.dropout_start_s, dropout_end_s=team.dropout_end_s,
        stop_on_capture=False,
    )

    print(f"  [{case_name}] evaluation done")

    # ---- Summaries ----
    dt = learning.dt
    sum_team = eval_summary(eval_team, "team_full_comm")
    sum_local = eval_summary(eval_local, "local_no_comm")
    sum_dropout = eval_summary(eval_dropout, "team_dropout")

    # ---- Plots ----
    results = {
        "Team (full comm)": eval_team,
        "Local (no comm)": eval_local,
        "Team (dropout)": eval_dropout,
    }

    plot_trajectory_xy(eval_team, case_dir / "fig_trajectory_team.png", f"{case_name} Team trajectory")
    plot_trajectory_xy(eval_local, case_dir / "fig_trajectory_local.png", f"{case_name} Local trajectory")

    plot_assigned_errors(eval_team, dt, case_dir / "fig_errors_team.png", f"{case_name} Team assigned errors")
    plot_assigned_errors(eval_local, dt, case_dir / "fig_errors_local.png", f"{case_name} Local assigned errors")

    plot_weight_convergence(train_result, case_dir / "fig_weight_convergence.png", f"{case_name} weight convergence")

    plot_team_error_compare(results, dt, case_dir / "fig_team_error_compare.png", f"{case_name} Team Error Comparison")
    plot_d_min(results, dt, case_dir / "fig_d_min_compare.png", f"{case_name} Min Inter-Pursuer Distance")
    plot_angular_coverage(results, dt, case_dir / "fig_angular_coverage.png", f"{case_name} Angular Coverage")
    plot_path_overlap(results, dt, case_dir / "fig_path_overlap.png", f"{case_name} Path Overlap")

    if scenario.n_evaders > 1:
        plot_assignment_timeline(eval_team, dt, case_dir / "fig_assignment_team.png", f"{case_name} Team assignments")

    plot_comparison_bars(
        {"Team": sum_team, "Local": sum_local, "Dropout": sum_dropout},
        case_dir / "fig_bars.png",
        f"{case_name} Metric Comparison",
    )

    # ---- JSON ----
    summary = {
        "case": {"name": case_name, "n_pursuers": n_p, "n_evaders": n_e, "seed": seed},
        "train": train_summary(train_result),
        "eval_team": sum_team,
        "eval_local": sum_local,
        "eval_dropout": sum_dropout,
        "network": network_summary(n_p, n_e, sim.features.n_features),
        "timing": {"train_s": train_s},
    }
    dump_json(case_dir / "summary.json", summary)

    # ---- Markdown report ----
    md = [
        f"# Team V-SNAC Report: {case_name}",
        "",
        "## Capture Time",
        f"| Mode | Capture (s) |",
        f"|---|---:|",
        f"| Team (full comm) | {sum_team['capture_time_s']} |",
        f"| Local (no comm) | {sum_local['capture_time_s']} |",
        f"| Team (dropout) | {sum_dropout['capture_time_s']} |",
        "",
        "## Coordination Metrics (mean over trajectory)",
        f"| Metric | Team | Local | Dropout |",
        f"|---|---:|---:|---:|",
        f"| d_min (m) | {sum_team.get('d_min_mean',0):.1f} | {sum_local.get('d_min_mean',0):.1f} | {sum_dropout.get('d_min_mean',0):.1f} |",
        f"| angular coverage | {sum_team.get('angular_coverage_mean',0):.3f} | {sum_local.get('angular_coverage_mean',0):.3f} | {sum_dropout.get('angular_coverage_mean',0):.3f} |",
        f"| path overlap | {sum_team.get('path_overlap_mean',0):.3f} | {sum_local.get('path_overlap_mean',0):.3f} | {sum_dropout.get('path_overlap_mean',0):.3f} |",
        f"| team error (final) | {sum_team['final_team_error']:.1f} | {sum_local['final_team_error']:.1f} | {sum_dropout['final_team_error']:.1f} |",
    ]
    dump_markdown(case_dir / "REPORT.md", "\n".join(md))

    print(f"  [{case_name}] done. Team cap={sum_team['capture_time_s']}s, Local cap={sum_local['capture_time_s']}s")
    return summary


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Team V-SNAC experiments")
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help="Case tokens like 3v1 4v1 5v1 4v2 6v3. Default: 3v1, 4v1, 5v1, 4v2.",
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
        cases = [(3, 1), (4, 1), (5, 1), (4, 2)]

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
        "| Case | Team cap (s) | Local cap (s) | Dropout cap (s) | Team d_min | Local d_min | Team ang_cov | Local ang_cov |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_summaries:
        md.append(
            "| {name} | {tc} | {lc} | {dc} | {td:.0f} | {ld:.0f} | {ta:.3f} | {la:.3f} |".format(
                name=s["case"]["name"],
                tc=s["eval_team"]["capture_time_s"] or "-",
                lc=s["eval_local"]["capture_time_s"] or "-",
                dc=s["eval_dropout"]["capture_time_s"] or "-",
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
