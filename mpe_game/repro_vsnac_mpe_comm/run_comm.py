from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mpe_repro.config import AircraftParams, CommParams, ControlParams, FeatureParams, LearningParams
from mpe_repro.general_scenarios import GeneralScenarioSpec, build_general_scenario
from mpe_repro.plotting import (
    plot_assigned_residual_norm,
    plot_assigned_state_errors,
    plot_assignment_timeline,
    plot_comm_comparison,
    plot_control_input_deltas,
    plot_control_inputs,
    plot_d_min_history,
    plot_formation_error,
    plot_old_new_errors,
    plot_team_error_compare,
    plot_trajectory_3d,
    plot_trajectory_animation_3d,
    plot_trajectory_multiview,
    plot_weight_convergence,
)
from mpe_repro.report import dump_json, dump_markdown, eval_summary, network_summary, train_summary
from mpe_repro.simulator import EvalResult, MPECommSimulator, TrainResult

matplotlib.use("Agg")


@dataclass(frozen=True)
class GeneralCase:
    n_pursuers: int
    n_evaders: int
    seed: int
    assignment_mode: str = "shifted"
    layout_mode: str = "structured"

    @property
    def name(self) -> str:
        return f"{self.n_pursuers}v{self.n_evaders}"


def _parse_case_token(token: str, seed: int, assignment_mode: str, layout_mode: str) -> GeneralCase:
    token = token.lower().replace(" ", "")
    for sep in ("x", "v", "*", ":"):
        if sep in token:
            left, right = token.split(sep, 1)
            n_p = int(left)
            n_e = int(right)
            return GeneralCase(
                n_pursuers=n_p,
                n_evaders=n_e,
                seed=seed,
                assignment_mode=assignment_mode,
                layout_mode=layout_mode,
            )
    raise ValueError(f"unsupported case token: {token}")


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


def _learning_params(n_p: int, n_e: int, quick: bool) -> LearningParams:
    size = max(n_p, n_e)
    if quick:
        return LearningParams(
            policy_iterations=14,
            rollout_steps=80 + 10 * size,
            min_samples_per_evader=50,
            graph_update_interval=1,
            graph_update_start_step=0,
            critic_learning_rate=0.05,
            critic_lr_decay=0.90,
            convergence_tol=5e-4,
            random_perturb_scale=0.02,
        )
    return LearningParams(
        policy_iterations=32 + 2 * max(0, size - 3),
        rollout_steps=160 + 15 * size,
        min_samples_per_evader=50,
        graph_update_interval=1,
        graph_update_start_step=0,
        critic_learning_rate=0.05,
        critic_lr_decay=0.90,
        convergence_tol=1e-4,
        random_perturb_scale=0.02,
    )


def _comm_params(gamma: float, comm_mode: str = "full", dropout_prob: float = 0.0) -> CommParams:
    return CommParams(gamma=gamma, comm_mode=comm_mode, dropout_prob=dropout_prob)


def _scenario_spec(case: GeneralCase, quick: bool) -> GeneralScenarioSpec:
    n_p = case.n_pursuers
    n_e = case.n_evaders
    return GeneralScenarioSpec(
        n_pursuers=n_p,
        n_evaders=n_e,
        seed=case.seed,
        assignment_mode=case.assignment_mode,
        layout_mode=case.layout_mode,
        t_final=90.0 if n_e == 1 else (110.0 if quick else 140.0),
        capture_radius=180.0 if n_e == 1 else 220.0,
        swap_threshold=5.0 if n_e > 1 else 1.0e9,
        max_switch_worsening=0.0,
        evader_motion_mode="scripted",
        evader_script_amp=(8.0, 8.0, 3.0) if n_e > 1 else (10.0, 8.0, 4.0),
        evader_script_omega=0.16 if n_e > 1 else 0.42,
        evader_script_decay=0.010 if n_e > 1 else 0.030,
        evader_script_mix=0.35 if n_e > 1 else 0.60,
        swap_lookahead_time=0.5 if n_e > 1 else 0.0,
    )


def _switch_times(assigned_targets: np.ndarray, dt: float) -> list[float]:
    if assigned_targets.size == 0:
        return []
    steps: list[int] = []
    if assigned_targets.shape[0] > 1:
        changes = np.any(assigned_targets[1:] != assigned_targets[:-1], axis=1)
        steps.extend((np.where(changes)[0] + 1).tolist())
    return [float(step * dt) for step in steps]


def _runtime_metrics(
    train: TrainResult,
    eval_results: dict[str, EvalResult],
    learning: LearningParams,
    wall_s: float,
    train_wall_s: float,
    eval_wall_s: dict[str, float],
) -> dict[str, Any]:
    train_iterations = int(train.weight_history.shape[0] - 1)
    train_steps = train_iterations * int(learning.rollout_steps)
    eval_steps_total = sum(int(r.team_errors.shape[0]) for r in eval_results.values())
    total_steps = train_steps + eval_steps_total
    return {
        "wall_time_s": float(wall_s),
        "train_wall_time_s": float(train_wall_s),
        "eval_wall_times_s": {k: float(v) for k, v in eval_wall_s.items()},
        "train_iterations": train_iterations,
        "simulated_steps_total": int(total_steps),
        "ms_per_step_total": float(1000.0 * wall_s / max(total_steps, 1)),
        "ms_per_step_train": float(1000.0 * train_wall_s / max(train_steps, 1)),
    }


def _write_case_report(path: Path, summary: dict[str, Any]) -> None:
    case_name = summary["case"]["name"]
    md = [
        f"# Communication-Augmented MPE Case Report: {case_name}",
        "",
        "## Case",
        f"- Pursuers: {summary['case']['n_pursuers']}",
        f"- Evaders: {summary['case']['n_evaders']}",
        f"- Seed: {summary['case']['seed']}",
        f"- Assignment mode: `{summary['case']['assignment_mode']}`",
        f"- Gamma (training): {summary['gamma']}",
        "",
        "## Runtime",
        f"- Total wall time (s): {summary['runtime']['wall_time_s']:.3f}",
        f"- Total ms/step: {summary['runtime']['ms_per_step_total']:.4f}",
        f"- Train ms/step: {summary['runtime']['ms_per_step_train']:.4f}",
        "",
        "## Evaluation Modes",
    ]
    for mode_name in ["full_comm", "no_comm", "dropout"]:
        ev = summary.get(f"eval_{mode_name}")
        if ev is None:
            continue
        md.extend([
            f"### {mode_name}",
            f"- Capture time (s): {ev['capture_time_s']}",
            f"- Final Eteam: {ev['final_team_error']:.3f}",
            f"- Mean assigned error: {ev['mean_assigned_error']:.3f}",
            f"- Min d_min: {ev['min_d_min']:.3f}" if ev['min_d_min'] is not None else "- Min d_min: N/A",
            f"- Mean formation error: {ev['mean_formation_error']:.3f}" if ev['mean_formation_error'] is not None else "- Mean formation error: N/A",
            "",
        ])

    md.extend([
        "## Networks",
        f"- V-SNAC critics: {summary['network']['v_snac_networks']}",
        f"- AC estimated networks: {summary['network']['ac_networks_estimated']}",
        f"- Estimated reduction (%): {summary['network']['network_reduction_percent']:.2f}",
        "",
        "## Switch Info",
        f"- Switch count: {summary['dynamic_switch_count']}",
        f"- Switch times (s): {summary['dynamic_switch_times_s']}",
    ])

    dump_markdown(path, "\n".join(md))


def _run_single_case(job: dict[str, Any]) -> dict[str, Any]:
    case = GeneralCase(**job["case"])
    case_dir = Path(job["case_dir"])
    quick = bool(job["quick"])
    make_plots = bool(job.get("make_plots", True))
    gamma = float(job["gamma"])
    dropout_prob = float(job["dropout_prob"])

    scenario = build_general_scenario(_scenario_spec(case, quick))
    control = _control_params()
    learning = _learning_params(case.n_pursuers, case.n_evaders, quick)
    feature = _feature_params()

    dynamic_graph_train = scenario.n_evaders > 1
    dynamic_graph_eval = scenario.n_evaders > 1

    # Train with full communication
    train_comm = _comm_params(gamma=gamma, comm_mode="full")
    sim = MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        comm_params=train_comm,
    )

    wall_t0 = time.perf_counter()
    train_t0 = time.perf_counter()
    train = sim.train_policy(seed=case.seed, dynamic_graph=dynamic_graph_train)
    train_wall_s = time.perf_counter() - train_t0

    eval_seed = case.seed + 1000

    # Three evaluation modes using the SAME trained weights
    eval_configs = {
        "full_comm": CommParams(gamma=gamma, comm_mode="full", dropout_prob=0.0),
        "no_comm": CommParams(gamma=0.0, comm_mode="none", dropout_prob=0.0),
        "dropout": CommParams(gamma=gamma, comm_mode="full", dropout_prob=dropout_prob),
    }

    eval_results: dict[str, EvalResult] = {}
    eval_wall_s: dict[str, float] = {}

    for mode_name, mode_comm in eval_configs.items():
        t0 = time.perf_counter()
        result = sim.evaluate_policy(
            weights=train.weights,
            seed=eval_seed,
            dynamic_graph=dynamic_graph_eval,
            stop_on_capture=False,
            record_logs=False,
            comm_params_override=mode_comm,
        )
        eval_wall_s[mode_name] = time.perf_counter() - t0
        eval_results[mode_name] = result

    wall_s = time.perf_counter() - wall_t0

    case_dir.mkdir(parents=True, exist_ok=True)

    if make_plots:
        # Per-mode plots for full_comm (primary)
        eval_full = eval_results["full_comm"]
        plot_trajectory_3d(eval_full, case_dir / "fig_trajectory_xy.png", f"{case.name} trajectory (full comm)")
        plot_trajectory_multiview(
            eval_full,
            case_dir / "fig_trajectory_multiview.png",
            f"{case.name} trajectory multiview (full comm)",
        )
        plot_trajectory_animation_3d(
            eval_full,
            case_dir / "fig_trajectory_3d.gif",
            f"{case.name} pursuit-evasion (full comm)",
        )
        plot_assigned_state_errors(
            eval_full,
            learning.dt,
            case_dir / "fig_assigned_errors.png",
            f"{case.name} assigned target errors (full comm)",
        )
        plot_assigned_residual_norm(
            eval_full,
            learning.dt,
            scenario.displacement_matrix,
            case_dir / "fig_assigned_residual_norm.png",
            f"{case.name} assigned target residual norms (full comm)",
        )
        plot_control_inputs(
            eval_full,
            learning.dt,
            case_dir / "fig_control_inputs.png",
            f"{case.name} control inputs (full comm)",
            component_idx=0,
            y_lim=(-40.0, 40.0),
            use_tanh=False,
            paper_style_labels=False,
        )
        plot_control_input_deltas(
            eval_full,
            learning.dt,
            case_dir / "fig_control_input_deltas.png",
            f"{case.name} control input deltas (full comm)",
            use_tanh=False,
        )
        plot_weight_convergence(
            train,
            case_dir / "fig_weight_convergence.png",
            f"{case.name} critic convergence",
        )

        # Communication-specific plots
        plot_comm_comparison(
            eval_results,
            learning.dt,
            case_dir / "fig_comm_comparison.png",
            f"{case.name} team error: comm mode comparison",
        )
        plot_d_min_history(
            eval_full,
            learning.dt,
            case_dir / "fig_d_min_full.png",
            f"{case.name} d_min (full comm)",
        )
        plot_formation_error(
            eval_full,
            learning.dt,
            case_dir / "fig_formation_error_full.png",
            f"{case.name} formation error (full comm)",
        )
        plot_d_min_history(
            eval_results["no_comm"],
            learning.dt,
            case_dir / "fig_d_min_nocomm.png",
            f"{case.name} d_min (no comm)",
        )
        plot_formation_error(
            eval_results["dropout"],
            learning.dt,
            case_dir / "fig_formation_error_dropout.png",
            f"{case.name} formation error (dropout)",
        )

        # Dynamic vs fixed graph comparison (if multi-evader)
        if scenario.n_evaders > 1:
            eval_fixed_t0 = time.perf_counter()
            eval_fixed = sim.evaluate_policy(
                weights=train.weights,
                seed=eval_seed,
                dynamic_graph=False,
                stop_on_capture=False,
                record_logs=False,
                comm_params_override=CommParams(gamma=gamma, comm_mode="full", dropout_prob=0.0),
            )
            plot_team_error_compare(
                eval_full.team_errors,
                eval_fixed.team_errors,
                learning.dt,
                case_dir / "fig_team_error_compare.png",
                f"{case.name} dynamic vs fixed graph Eteam (full comm)",
            )
            plot_old_new_errors(
                eval_full,
                learning.dt,
                case_dir / "fig_old_new_errors.png",
                f"{case.name} old/new assigned errors (full comm)",
            )
            plot_assignment_timeline(
                eval_full,
                learning.dt,
                case_dir / "fig_assignment_timeline.png",
                f"{case.name} assignment timeline (full comm)",
            )

    # Build summary
    summary: dict[str, Any] = {
        "case": {
            "name": case.name,
            "n_pursuers": case.n_pursuers,
            "n_evaders": case.n_evaders,
            "seed": case.seed,
            "assignment_mode": case.assignment_mode,
            "layout_mode": case.layout_mode,
        },
        "gamma": gamma,
        "dropout_prob": dropout_prob,
        "scenario": {
            "name": scenario.name,
            "initial_assignment": scenario.initial_assignment.tolist(),
            "capture_radius": float(scenario.capture_radius),
            "swap_threshold": float(scenario.swap_threshold),
            "t_final": float(scenario.t_final),
        },
        "train": train_summary(train),
        "network": network_summary(scenario.n_pursuers, scenario.n_evaders),
        "runtime": _runtime_metrics(
            train=train,
            eval_results=eval_results,
            learning=learning,
            wall_s=wall_s,
            train_wall_s=train_wall_s,
            eval_wall_s=eval_wall_s,
        ),
    }

    for mode_name, result in eval_results.items():
        summary[f"eval_{mode_name}"] = eval_summary(result, comm_mode=mode_name)

    # Switch info from full_comm eval
    eval_full = eval_results["full_comm"]
    switch_times = _switch_times(eval_full.assigned_targets, learning.dt)
    if eval_full.assigned_targets.size > 0 and not np.array_equal(
        eval_full.assigned_targets[0], scenario.initial_assignment
    ):
        switch_times = [0.0] + switch_times
    summary["dynamic_switch_times_s"] = switch_times
    summary["dynamic_switch_count"] = len(switch_times)

    dump_json(case_dir / "summary.json", summary)
    _write_case_report(case_dir / "REPORT.md", summary)
    return summary


def _default_cases(seed: int, assignment_mode: str, layout_mode: str) -> list[GeneralCase]:
    return [
        GeneralCase(3, 1, seed=seed, assignment_mode="zero", layout_mode=layout_mode),
        GeneralCase(3, 3, seed=seed + 10, assignment_mode=assignment_mode, layout_mode=layout_mode),
        GeneralCase(5, 3, seed=seed + 20, assignment_mode=assignment_mode, layout_mode=layout_mode),
    ]


def _collect_cases(args: argparse.Namespace) -> list[GeneralCase]:
    if args.case:
        cases = []
        for idx, token in enumerate(args.case):
            assign_mode = "zero" if token.lower().replace(" ", "").endswith(("x1", "v1", "*1", ":1")) else args.assignment_mode
            cases.append(
                _parse_case_token(
                    token,
                    seed=int(args.seed) + 10 * idx,
                    assignment_mode=assign_mode,
                    layout_mode=args.layout_mode,
                )
            )
        return cases
    return _default_cases(seed=int(args.seed), assignment_mode=args.assignment_mode, layout_mode=args.layout_mode)


def _batch_report(
    out_dir: Path,
    case_summaries: list[dict[str, Any]],
    total_wall_s: float,
    parallel_workers: int,
) -> None:
    sum_case_wall_s = float(sum(item["runtime"]["wall_time_s"] for item in case_summaries))
    estimated_speedup = sum_case_wall_s / max(total_wall_s, 1e-12)
    batch = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parallel_workers": int(parallel_workers),
        "total_wall_time_s": float(total_wall_s),
        "sum_case_wall_time_s": sum_case_wall_s,
        "estimated_parallel_speedup_vs_serial": float(estimated_speedup),
        "cases": case_summaries,
    }
    dump_json(out_dir / "batch_summary.json", batch)

    md = [
        "# Communication-Augmented MPE Batch Report",
        "",
        f"- Output folder: `{out_dir}`",
        f"- Parallel workers: `{parallel_workers}`",
        f"- Total wall time (s): `{total_wall_s:.3f}`",
        f"- Sum of per-case wall times (s): `{sum_case_wall_s:.3f}`",
        "",
        "## Case Summary",
        "| case | cap(full) | cap(none) | cap(drop) | err(full) | err(none) | err(drop) | d_min(full) | d_min(none) | switches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in case_summaries:
        ef = item.get("eval_full_comm", {})
        en = item.get("eval_no_comm", {})
        ed = item.get("eval_dropout", {})

        def _cap(ev: dict) -> str:
            c = ev.get("capture_time_s")
            return "-" if c is None else f"{c:.2f}"

        def _err(ev: dict) -> str:
            e = ev.get("mean_assigned_error")
            return "-" if e is None else f"{e:.3f}"

        def _dmin(ev: dict) -> str:
            d = ev.get("min_d_min")
            return "-" if d is None else f"{d:.1f}"

        md.append(
            "| {case} | {cap_f} | {cap_n} | {cap_d} | {err_f} | {err_n} | {err_d} | {dm_f} | {dm_n} | {sw} |".format(
                case=item["case"]["name"],
                cap_f=_cap(ef),
                cap_n=_cap(en),
                cap_d=_cap(ed),
                err_f=_err(ef),
                err_n=_err(en),
                err_d=_err(ed),
                dm_f=_dmin(ef),
                dm_n=_dmin(en),
                sw=item.get("dynamic_switch_count", 0),
            )
        )

    md.extend([
        "",
        "## Conclusions",
        "- Training is performed ONCE with full communication (gamma > 0).",
        "- The same trained weights are evaluated under three modes: full_comm, no_comm, and dropout.",
        "- full_comm should provide the best team error; no_comm degrades to MN-equivalent behavior.",
        "- dropout tests robustness of the learned policy to intermittent communication failures.",
        "- d_min tracks minimum inter-pursuer distance to verify collision avoidance.",
    ])
    dump_markdown(out_dir / "REPORT.md", "\n".join(md))


def _plot_runtime_summary(case_summaries: list[dict[str, Any]], path: Path) -> None:
    labels = [item["case"]["name"] for item in case_summaries]
    values = [item["runtime"]["ms_per_step_total"] for item in case_summaries]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_xlabel("Case")
    ax.set_ylabel("ms / simulated step")
    ax.set_title("Communication-augmented MPE runtime per step")
    ax.grid(alpha=0.25, axis="y")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Communication-augmented m-pursuer / n-evader MPE runner.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run shorter experiments for validation")
    parser.add_argument(
        "--case",
        type=str,
        nargs="*",
        default=None,
        help="Case tokens such as 3v1 3v3 5v3. If omitted, defaults to 3v1/3v3/5v3 batch.",
    )
    parser.add_argument(
        "--assignment-mode",
        type=str,
        default="shifted",
        choices=["zero", "cyclic", "shifted", "nearest", "random"],
        help="Initial pursuer-to-evader assignment for n>1 scenarios.",
    )
    parser.add_argument(
        "--layout-mode",
        type=str,
        default="structured",
        choices=["structured", "random"],
        help="Initial pursuer/evader state layout mode.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base random seed")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel worker count for batch runs")
    parser.add_argument("--gamma", type=float, default=0.3, help="Formation coupling weight gamma")
    parser.add_argument("--dropout-prob", type=float, default=0.15, help="Edge dropout probability for dropout eval")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation. Useful for runtime benchmarking.",
    )
    args = parser.parse_args()

    cases = _collect_cases(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"comm_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "case": asdict(case),
            "case_dir": str(out_dir / case.name),
            "quick": bool(args.quick),
            "make_plots": not bool(args.skip_plots),
            "gamma": float(args.gamma),
            "dropout_prob": float(args.dropout_prob),
        }
        for case in cases
    ]

    t0 = time.perf_counter()
    workers = int(args.parallel_workers)
    if workers <= 0:
        workers = min(len(jobs), max(1, (os.cpu_count() or 1)))

    if len(jobs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            case_summaries = list(pool.map(_run_single_case, jobs))
    else:
        case_summaries = [_run_single_case(job) for job in jobs]
    total_wall_s = time.perf_counter() - t0

    case_summaries.sort(key=lambda item: (item["case"]["n_pursuers"], item["case"]["n_evaders"]))
    if not args.skip_plots:
        _plot_runtime_summary(case_summaries, out_dir / "fig_runtime_ms_per_step.png")
    _batch_report(out_dir, case_summaries, total_wall_s, workers)
    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
