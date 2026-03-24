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

from mpe_repro.config import AircraftParams, ControlParams, FeatureParams, LearningParams
from mpe_repro.general_scenarios import GeneralScenarioSpec, build_general_scenario
from mpe_repro.plotting import (
    plot_assignment_timeline,
    plot_assigned_residual_norm,
    plot_assigned_state_errors,
    plot_assigned_xyz_errors,
    plot_assigned_xyz_residuals,
    plot_control_input_deltas,
    plot_control_inputs,
    plot_old_new_errors,
    plot_team_error_compare,
    plot_trajectory_animation_3d,
    plot_trajectory_3d,
    plot_trajectory_multiview,
    plot_weight_convergence,
)
from mpe_repro.report import dump_json, dump_markdown, eval_summary, network_summary, train_summary
from mpe_repro.simulator import EvalResult, MPESimulator, TrainResult

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


def _parse_int_set(token: str) -> list[int]:
    token = token.strip()
    if not token:
        return []
    values: list[int] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            left, right = part.split(":", 1)
            a = int(left)
            b = int(right)
            if a <= b:
                values.extend(list(range(a, b + 1)))
            else:
                values.extend(list(range(a, b - 1, -1)))
        else:
            values.append(int(part))
    # preserve order but remove duplicates
    dedup: list[int] = []
    seen = set()
    for v in values:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


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
            policy_iterations=12,
            rollout_steps=70 + 10 * size,
            min_samples_per_evader=120,
            graph_update_interval=1,
            graph_update_start_step=0,
            critic_learning_rate=0.01,
            convergence_tol=5e-4,
            random_perturb_scale=0.02,
        )
    return LearningParams(
        policy_iterations=28 + 2 * max(0, size - 3),
        rollout_steps=150 + 15 * size,
        min_samples_per_evader=320 + 40 * size,
        graph_update_interval=1,
        graph_update_start_step=0,
        critic_learning_rate=0.01,
        convergence_tol=1e-4,
        random_perturb_scale=0.02,
    )


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
    initial = assigned_targets[0]
    if assigned_targets.shape[0] > 1:
        changes = np.any(assigned_targets[1:] != assigned_targets[:-1], axis=1)
        steps.extend((np.where(changes)[0] + 1).tolist())
    # An immediate reassignment at step 0 will not appear in the diff above.
    # We record it later by comparing against scenario.initial_assignment.
    return [float(step * dt) for step in steps]


def _runtime_metrics(
    train: TrainResult,
    eval_dynamic: EvalResult,
    eval_fixed: EvalResult | None,
    learning: LearningParams,
    wall_s: float,
    train_wall_s: float,
    eval_dynamic_wall_s: float,
    eval_fixed_wall_s: float,
) -> dict[str, Any]:
    train_iterations = int(train.weight_history.shape[0] - 1)
    train_steps = train_iterations * int(learning.rollout_steps)
    eval_dynamic_steps = int(eval_dynamic.team_errors.shape[0])
    eval_fixed_steps = int(eval_fixed.team_errors.shape[0]) if eval_fixed is not None else 0
    total_steps = train_steps + eval_dynamic_steps + eval_fixed_steps
    return {
        "wall_time_s": float(wall_s),
        "train_wall_time_s": float(train_wall_s),
        "eval_dynamic_wall_time_s": float(eval_dynamic_wall_s),
        "eval_fixed_wall_time_s": float(eval_fixed_wall_s),
        "train_iterations": train_iterations,
        "simulated_steps_total": int(total_steps),
        "ms_per_step_total": float(1000.0 * wall_s / max(total_steps, 1)),
        "ms_per_step_train": float(1000.0 * train_wall_s / max(train_steps, 1)),
        "ms_per_step_eval_dynamic": float(1000.0 * eval_dynamic_wall_s / max(eval_dynamic_steps, 1)),
        "ms_per_step_eval_fixed": (
            None if eval_fixed is None else float(1000.0 * eval_fixed_wall_s / max(eval_fixed_steps, 1))
        ),
    }


def _write_case_report(path: Path, summary: dict[str, Any]) -> None:
    case_name = summary["case"]["name"]
    md = [
        f"# Generalized MPE Case Report: {case_name}",
        "",
        "## Case",
        f"- Pursuers: {summary['case']['n_pursuers']}",
        f"- Evaders: {summary['case']['n_evaders']}",
        f"- Seed: {summary['case']['seed']}",
        f"- Assignment mode: `{summary['case']['assignment_mode']}`",
        "",
        "## Runtime",
        f"- Total wall time (s): {summary['runtime']['wall_time_s']:.3f}",
        f"- Total ms/step: {summary['runtime']['ms_per_step_total']:.4f}",
        f"- Train ms/step: {summary['runtime']['ms_per_step_train']:.4f}",
        f"- Eval(dynamic) ms/step: {summary['runtime']['ms_per_step_eval_dynamic']:.4f}",
    ]
    if summary["runtime"]["ms_per_step_eval_fixed"] is not None:
        md.append(f"- Eval(fixed) ms/step: {summary['runtime']['ms_per_step_eval_fixed']:.4f}")

    md.extend(
        [
            "",
        "## Dynamic Evaluation",
        f"- Capture time (s): {summary['dynamic_eval']['capture_time_s']}",
        f"- Final Eteam at common horizon: {summary['dynamic_eval']['final_team_error']:.3f}",
        f"- Mean assigned error: {summary['dynamic_eval']['mean_assigned_error']:.3f}",
        f"- Final max assigned error: {summary['dynamic_eval']['final_max_assigned_error']:.3f}",
        f"- Switch count: {summary['dynamic_switch_count']}",
        f"- Switch times (s): {summary['dynamic_switch_times_s']}",
            "",
            "## Networks",
            f"- V-SNAC critics: {summary['network']['v_snac_networks']}",
            f"- AC estimated networks: {summary['network']['ac_networks_estimated']}",
            f"- Estimated reduction (%): {summary['network']['network_reduction_percent']:.2f}",
        ]
    )
    if summary["fixed_eval"] is not None:
        better = summary["dynamic_eval"]["final_team_error"] < summary["fixed_eval"]["final_team_error"]
        md.extend(
            [
                "",
                "## Fixed-Graph Baseline",
                f"- Capture time (s): {summary['fixed_eval']['capture_time_s']}",
                f"- Final Eteam at common horizon: {summary['fixed_eval']['final_team_error']:.3f}",
                f"- Mean assigned error: {summary['fixed_eval']['mean_assigned_error']:.3f}",
                f"- Dynamic graph better on final Eteam: {better}",
            ]
        )

    md.extend(
        [
            "",
            "## Files",
            "- `summary.json`",
            "- `fig_trajectory_xy.png`",
            "- `fig_trajectory_3d.gif`",
            "- `fig_trajectory_multiview.png`",
            "- `fig_assigned_errors.png`",
            "- `fig_assigned_residual_norm.png`",
            "- `fig_control_inputs.png`",
            "- `fig_control_input_deltas.png`",
            "- `fig_weight_convergence.png`",
        ]
    )
    if summary["fixed_eval"] is not None:
        md.extend(
            [
                "- `fig_team_error_compare.png`",
                "- `fig_old_new_errors.png`",
                "- `fig_assignment_timeline.png`",
            ]
        )

    dump_markdown(path, "\n".join(md))


def _plot_runtime_summary(case_summaries: list[dict[str, Any]], path: Path) -> None:
    labels = [item["case"]["name"] for item in case_summaries]
    values = [item["runtime"]["ms_per_step_total"] for item in case_summaries]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_xlabel("Case")
    ax.set_ylabel("ms / simulated step")
    ax.set_title("Generalized MPE runtime per step")
    ax.grid(alpha=0.25, axis="y")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _run_single_case(job: dict[str, Any]) -> dict[str, Any]:
    case = GeneralCase(**job["case"])
    case_dir = Path(job["case_dir"])
    quick = bool(job["quick"])
    make_plots = bool(job.get("make_plots", True))

    scenario = build_general_scenario(_scenario_spec(case, quick))
    control = _control_params()
    learning = _learning_params(case.n_pursuers, case.n_evaders, quick)
    feature = _feature_params()

    dynamic_graph_train = scenario.n_evaders > 1
    dynamic_graph_eval = scenario.n_evaders > 1

    sim = MPESimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
    )

    wall_t0 = time.perf_counter()
    train_t0 = time.perf_counter()
    train = sim.train_policy(seed=case.seed, dynamic_graph=dynamic_graph_train)
    train_wall_s = time.perf_counter() - train_t0

    eval_seed = case.seed + 1000
    eval_dynamic_t0 = time.perf_counter()
    eval_dynamic = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        dynamic_graph=dynamic_graph_eval,
        stop_on_capture=False,
        record_logs=False,
    )
    eval_dynamic_wall_s = time.perf_counter() - eval_dynamic_t0

    eval_fixed = None
    eval_fixed_wall_s = 0.0
    if scenario.n_evaders > 1:
        eval_fixed_t0 = time.perf_counter()
        eval_fixed = sim.evaluate_policy(
            weights=train.weights,
            seed=eval_seed,
            dynamic_graph=False,
            stop_on_capture=False,
            record_logs=False,
        )
        eval_fixed_wall_s = time.perf_counter() - eval_fixed_t0

    wall_s = time.perf_counter() - wall_t0

    case_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plot_trajectory_3d(eval_dynamic, case_dir / "fig_trajectory_xy.png", f"{case.name} trajectory")
        plot_trajectory_multiview(
            eval_dynamic,
            case_dir / "fig_trajectory_multiview.png",
            f"{case.name} trajectory multiview",
        )
        plot_trajectory_animation_3d(
            eval_dynamic,
            case_dir / "fig_trajectory_3d.gif",
            f"{case.name} pursuit-evasion",
        )
        plot_assigned_state_errors(
            eval_dynamic,
            learning.dt,
            case_dir / "fig_assigned_errors.png",
            f"{case.name} assigned target errors",
        )
        plot_assigned_residual_norm(
            eval_dynamic,
            learning.dt,
            scenario.displacement_matrix,
            case_dir / "fig_assigned_residual_norm.png",
            f"{case.name} assigned target residual norms (with r)",
        )
        plot_control_inputs(
            eval_dynamic,
            learning.dt,
            case_dir / "fig_control_inputs.png",
            f"{case.name} control inputs",
            component_idx=0,
            y_lim=(-40.0, 40.0),
            use_tanh=False,
            paper_style_labels=False,
        )
        plot_control_input_deltas(
            eval_dynamic,
            learning.dt,
            case_dir / "fig_control_input_deltas.png",
            f"{case.name} control input deltas",
            use_tanh=False,
        )
        plot_weight_convergence(
            train,
            case_dir / "fig_weight_convergence.png",
            f"{case.name} critic convergence",
        )
        if eval_fixed is not None:
            plot_team_error_compare(
                eval_dynamic.team_errors,
                eval_fixed.team_errors,
                learning.dt,
                case_dir / "fig_team_error_compare.png",
                f"{case.name} dynamic vs fixed graph Eteam",
            )
            plot_old_new_errors(
                eval_dynamic,
                learning.dt,
                case_dir / "fig_old_new_errors.png",
                f"{case.name} old/new assigned errors",
            )
            plot_assignment_timeline(
                eval_dynamic,
                learning.dt,
                case_dir / "fig_assignment_timeline.png",
                f"{case.name} assignment timeline",
            )

    summary = {
        "case": {
            "name": case.name,
            "n_pursuers": case.n_pursuers,
            "n_evaders": case.n_evaders,
            "seed": case.seed,
            "assignment_mode": case.assignment_mode,
            "layout_mode": case.layout_mode,
        },
        "scenario": {
            "name": scenario.name,
            "initial_assignment": scenario.initial_assignment.tolist(),
            "capture_radius": float(scenario.capture_radius),
            "swap_threshold": float(scenario.swap_threshold),
            "t_final": float(scenario.t_final),
        },
        "train": train_summary(train),
        "dynamic_eval": eval_summary(eval_dynamic),
        "fixed_eval": None if eval_fixed is None else eval_summary(eval_fixed),
        "network": network_summary(scenario.n_pursuers, scenario.n_evaders),
        "runtime": _runtime_metrics(
            train=train,
            eval_dynamic=eval_dynamic,
            eval_fixed=eval_fixed,
            learning=learning,
            wall_s=wall_s,
            train_wall_s=train_wall_s,
            eval_dynamic_wall_s=eval_dynamic_wall_s,
            eval_fixed_wall_s=eval_fixed_wall_s,
        ),
        "dynamic_switch_count": 0,  # filled below
        "dynamic_switch_times_s": [],  # filled below
        "same_weights_same_seed_baseline": bool(eval_fixed is not None),
    }
    switch_times = _switch_times(eval_dynamic.assigned_targets, learning.dt)
    if eval_dynamic.assigned_targets.size > 0 and not np.array_equal(
        eval_dynamic.assigned_targets[0], scenario.initial_assignment
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
    if args.sweep_pursuers or args.sweep_evaders:
        if not args.sweep_pursuers or not args.sweep_evaders:
            raise ValueError("--sweep-pursuers and --sweep-evaders must be provided together")
        p_vals = _parse_int_set(args.sweep_pursuers)
        e_vals = _parse_int_set(args.sweep_evaders)
        cases: list[GeneralCase] = []
        idx = 0
        for n_p in p_vals:
            for n_e in e_vals:
                if args.only_m_gt_n and n_p <= n_e:
                    continue
                assign_mode = "zero" if n_e == 1 else args.assignment_mode
                cases.append(
                    GeneralCase(
                        n_pursuers=int(n_p),
                        n_evaders=int(n_e),
                        seed=int(args.seed) + 10 * idx,
                        assignment_mode=assign_mode,
                        layout_mode=args.layout_mode,
                    )
                )
                idx += 1
        if not cases:
            raise ValueError("the requested sweep produced no valid cases")
        return cases
    if args.n_pursuers is not None or args.n_evaders is not None:
        if args.n_pursuers is None or args.n_evaders is None:
            raise ValueError("--n-pursuers and --n-evaders must be provided together")
        return [
            GeneralCase(
                n_pursuers=int(args.n_pursuers),
                n_evaders=int(args.n_evaders),
                seed=int(args.seed),
                assignment_mode="zero" if int(args.n_evaders) == 1 else args.assignment_mode,
                layout_mode=args.layout_mode,
            )
        ]
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
    parallel_helped = estimated_speedup >= 1.0
    batch = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parallel_workers": int(parallel_workers),
        "total_wall_time_s": float(total_wall_s),
        "sum_case_wall_time_s": sum_case_wall_s,
        "estimated_parallel_speedup_vs_serial": float(estimated_speedup),
        "parallel_helped": bool(parallel_helped),
        "cases": case_summaries,
    }
    dump_json(out_dir / "batch_summary.json", batch)

    md = [
        "# Generalized MPE Batch Report",
        "",
        f"- Output folder: `{out_dir}`",
        f"- Parallel workers: `{parallel_workers}`",
        f"- Total wall time (s): `{total_wall_s:.3f}`",
        f"- Sum of per-case wall times (s): `{sum_case_wall_s:.3f}`",
    ]
    if parallel_helped:
        md.append(f"- Observed parallel speedup vs summed per-case wall times: `{estimated_speedup:.3f}x`")
    else:
        md.append(
            f"- Parallel overhead note: current batch has throughput ratio `{estimated_speedup:.3f}x`; "
            "for these relatively short runs, process startup and plotting overhead dominate, so this value is not used as a performance claim."
        )
    md.extend(
        [
            "",
            "## Case Summary",
            "| case | capture(dynamic) | mean assigned(dynamic) | final Eteam@horizon(dynamic) | capture(fixed) | mean assigned(fixed) | final Eteam@horizon(fixed) | switches | ms/step |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in case_summaries:
        fixed_eval = item["fixed_eval"]
        md.append(
            "| {case} | {cap_d} | {mean_d:.3f} | {eteam_d:.3f} | {cap_f} | {mean_f} | {eteam_f} | {sw} | {msp:.4f} |".format(
                case=item["case"]["name"],
                cap_d="-" if item["dynamic_eval"]["capture_time_s"] is None else f"{item['dynamic_eval']['capture_time_s']:.2f}",
                mean_d=item["dynamic_eval"]["mean_assigned_error"],
                eteam_d=item["dynamic_eval"]["final_team_error"],
                cap_f="-" if fixed_eval is None or fixed_eval["capture_time_s"] is None else f"{fixed_eval['capture_time_s']:.2f}",
                mean_f="-" if fixed_eval is None else f"{fixed_eval['mean_assigned_error']:.3f}",
                eteam_f="-" if fixed_eval is None else f"{fixed_eval['final_team_error']:.3f}",
                sw=item["dynamic_switch_count"],
                msp=item["runtime"]["ms_per_step_total"],
            )
        )

    md.extend(
        [
            "",
        "## Conclusions",
            "- The generalized runner supports arbitrary `m` pursuers and `n` evaders through a shared scenario generator.",
            "- A sweep mode is provided to enumerate a whole family of `m > n` scenarios and run them in parallel.",
            "- For `n > 1`, the same trained weights are evaluated under dynamic graph and fixed graph using the same initial state, same seed, and the same simulation horizon.",
            "- In the current validation set, dynamic graph improves capture time and lowers mean assigned error for the multi-evader cases, while final horizon Eteam remains close and case-dependent.",
            "- Therefore, capture time and mean assigned error are treated as the primary effectiveness indicators for the generalized `m`-vs.-`n` extension.",
            "- Runtime metrics are reported per case to support computer-oriented discussion on scalable simulation and parallel execution.",
        ]
    )
    dump_markdown(out_dir / "REPORT.md", "\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized m-pursuer / n-evader MPE runner.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run shorter experiments for validation")
    parser.add_argument("--n-pursuers", type=int, default=None, help="Single-case number of pursuers")
    parser.add_argument("--n-evaders", type=int, default=None, help="Single-case number of evaders")
    parser.add_argument(
        "--case",
        type=str,
        nargs="*",
        default=None,
        help="Case tokens such as 3x1 3x3 5x3. If omitted, defaults to 3x1/3x3/5x3 batch.",
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
    parser.add_argument(
        "--sweep-pursuers",
        type=str,
        default=None,
        help="Pursuer counts for sweep, e.g. 4:8 or 4,5,6,8",
    )
    parser.add_argument(
        "--sweep-evaders",
        type=str,
        default=None,
        help="Evader counts for sweep, e.g. 1:4 or 2,3",
    )
    parser.add_argument(
        "--only-m-gt-n",
        action="store_true",
        help="When sweep mode is used, keep only cases with m > n.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation. Useful for runtime benchmarking.",
    )
    args = parser.parse_args()

    cases = _collect_cases(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"generalized_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "case": asdict(case),
            "case_dir": str(out_dir / case.name),
            "quick": bool(args.quick),
            "make_plots": not bool(args.skip_plots),
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
