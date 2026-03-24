from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mpe_repro.config import AircraftParams, ControlParams, LearningParams
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.team_comm_config import TeamFeatureParams
from mpe_repro.team_comm_multi_config import (
    GeneralTeamCommScenarioSpec,
    TeamCommMultiScenario,
    build_team_comm_multi_scenario,
)
from mpe_repro.team_comm_multi_plotting import (
    plot_assigned_error_summary,
    plot_assignment_timeline,
    plot_comm_comparison,
    plot_comm_ratio,
    plot_convergence_diagnostics,
    plot_group_weight_convergence,
    plot_multiteam_errors,
    plot_multiteam_trajectory_gif,
    plot_multiteam_trajectory_xy,
)
from mpe_repro.team_comm_multi_simulator import (
    MultiTeamCommunicationSimulator,
    TeamCommMultiEvalResult,
    TeamCommMultiTrainResult,
)


@dataclass(frozen=True)
class TeamCommGeneralCase:
    n_pursuers: int
    n_evaders: int
    seed: int
    assignment_mode: str = "shifted"
    layout_mode: str = "structured"

    @property
    def name(self) -> str:
        return f"{self.n_pursuers}v{self.n_evaders}"


def _parse_case_token(token: str, seed: int, assignment_mode: str, layout_mode: str) -> TeamCommGeneralCase:
    token = token.lower().replace(" ", "")
    for sep in ("x", "v", "*", ":"):
        if sep in token:
            left, right = token.split(sep, 1)
            return TeamCommGeneralCase(
                n_pursuers=int(left),
                n_evaders=int(right),
                seed=seed,
                assignment_mode=assignment_mode,
                layout_mode=layout_mode,
            )
    raise ValueError(f"unsupported case token: {token}")


def _feature_params() -> TeamFeatureParams:
    return TeamFeatureParams(
        state_scale=(2600.0, 2600.0, 2600.0, 150.0, 150.0, 150.0),
        local_gain=2.2,
        cross_gain=0.8,
    )


def _control_params() -> ControlParams:
    return ControlParams(
        q_diag=(80.0, 80.0, 25.0, 30.0, 30.0, 12.0),
        r1_diag=(1.0, 1.0, 1.0),
        r2_diag=(1.0, 1.0, 1.0),
        u_bar_p=25.0,
        u_bar_e=15.0,
        u_bar_p_policy=None,
        u_bar_e_policy=None,
    )


def _learning_params(n_p: int, n_e: int, quick: bool, mode: str = "comm") -> LearningParams:
    size = max(n_p, n_e)
    # Feature count depends on max group size (cross features grow quadratically).
    max_group = int(np.ceil(n_p / max(n_e, 1)))
    if mode == "summed":
        n_feat = 6  # summed mode always uses 6 features
    else:
        n_feat = 6 * max_group + 3 * (max_group * (max_group - 1) // 2)
    if quick:
        return LearningParams(
            dt=0.05,
            ridge_lambda=1e-3,
            replay_capacity=40000 + 1000 * n_feat,
            min_samples_per_evader=max(200, 10 * n_feat),
            policy_iterations=18 + max(0, size - 3),
            rollout_steps=120 + 15 * size,
            exploration_std_start=0.50,
            exploration_std_end=0.03,
            critic_learning_rate=0.06,
            convergence_tol=5e-4,
            random_perturb_scale=0.025,
            graph_update_interval=1,
            graph_update_start_step=0,
        )
    return LearningParams(
        dt=0.05,
        ridge_lambda=1e-3,
        replay_capacity=80000 + 1500 * n_feat,
        min_samples_per_evader=max(400, 14 * n_feat),
        policy_iterations=35 + max(0, size - 3) * 2,
        rollout_steps=220 + 20 * size,
        exploration_std_start=0.60,
        exploration_std_end=0.03,
        critic_learning_rate=0.06,
        convergence_tol=5e-4,
        random_perturb_scale=0.025,
        graph_update_interval=1,
        graph_update_start_step=0,
    )


def _scenario_spec(case: TeamCommGeneralCase, quick: bool) -> GeneralTeamCommScenarioSpec:
    n_p = case.n_pursuers
    n_e = case.n_evaders
    # Extra time after expected capture so the GIF shows post-capture behavior.
    t_final = 120.0 if quick else 150.0
    return GeneralTeamCommScenarioSpec(
        n_pursuers=n_p,
        n_evaders=n_e,
        seed=case.seed,
        assignment_mode=case.assignment_mode,
        layout_mode=case.layout_mode,
        t_final=t_final if n_e > 1 else min(t_final, 110.0),
        capture_radius=220.0 if n_e > 1 else 180.0,
        swap_threshold=5.0 if n_e > 1 else 1.0e9,
        max_switch_worsening=0.0,
        evader_motion_mode="scripted",
        evader_script_amp=(14.0, 12.0, 5.0) if n_e > 1 else (13.0, 11.0, 5.0),
        evader_script_omega=0.45 if n_e > 1 else 0.55,
        evader_script_decay=0.018 if n_e > 1 else 0.030,
        evader_script_mix=1.0,
        swap_lookahead_time=0.5 if n_e > 1 else 0.0,
        comm_windows_per_group=1 if quick else 2,
        comm_dropout_fraction=0.45,
        comm_full_group_prob=0.35,
    )


def _switch_times(assignment_history: np.ndarray, dt: float, initial_assignment: np.ndarray | None = None) -> list[float]:
    if assignment_history.size == 0:
        return []
    steps: list[int] = []
    if initial_assignment is not None and np.any(assignment_history[0] != initial_assignment):
        steps.append(0)
    changes = np.any(assignment_history[1:] != assignment_history[:-1], axis=1)
    steps.extend((np.where(changes)[0] + 1).tolist())
    return [float(step * dt) for step in steps]


def _rollouts_per_iter(learning: LearningParams) -> int:
    return max(3, int(math.ceil(learning.min_samples_per_evader / max(learning.rollout_steps, 1))))


def _write_case_report(
    path: Path,
    case: TeamCommGeneralCase,
    scenario: TeamCommMultiScenario,
    train: TeamCommMultiTrainResult,
    eval_result: TeamCommMultiEvalResult,
    learning: LearningParams,
    runtime_s: float,
    switch_times: list[float],
) -> None:
    lines = [
        f"# Generalized Team Communication Report: {case.name}",
        "",
        "## Scenario",
        f"- Pursuers: `{case.n_pursuers}`",
        f"- Evaders: `{case.n_evaders}`",
        f"- Group sizes: `{scenario.group_sizes.tolist()}`",
        f"- Assignment mode: `{case.assignment_mode}`",
        f"- Layout mode: `{case.layout_mode}`",
        f"- Reference communication windows: `{[asdict(win) for win in scenario.communication.windows]}`",
        f"- Evaluation communication windows: `{eval_result.communication_windows}`",
        "",
        "## Results",
        f"- Capture time: `{None if eval_result.capture_time is None else round(float(eval_result.capture_time), 3)} s`",
        f"- Final total Eteam: `{float(eval_result.team_errors[-1]):.6f}`",
        f"- Final max assigned error: `{float(np.max(eval_result.assigned_errors[-1])):.6f}`",
        f"- Final group errors: `{np.round(eval_result.group_errors[-1], 6).tolist()}`",
        f"- Final group communication ratios: `{np.round(eval_result.communication_ratio[-1], 6).tolist()}`",
        f"- Final group estimate errors: `{np.round(eval_result.estimate_errors[-1], 6).tolist()}`",
        f"- Switch count: `{len(switch_times)}`",
        f"- Switch times (s): `{switch_times}`",
        "",
        "## Training",
        f"- Iterations executed: `{train.weight_norm_history.shape[0] - 1}`",
        f"- Best iteration: `{train.best_iteration}`",
        f"- Final delta per group: `{np.round(train.delta_history[-1], 8).tolist() if train.delta_history.size else []}`",
        f"- Final residual rms per group: `{np.round(train.residual_history[-1], 8).tolist() if train.residual_history.size else []}`",
        f"- Final weight norms: `{np.round(train.weight_norm_history[-1], 8).tolist()}`",
        "",
        "## Runtime",
        f"- Total wall time (s): `{runtime_s:.3f}`",
        "",
        "## Interpretation",
        "- Training uses a curriculum: early iterations use full intra-group communication, later iterations fine-tune under random per-group outages.",
        "- Only pursuers assigned to the same evader communicate; inter-group coordination is handled only by the dynamic slot-swap graph.",
        "- Evaluation uses paper-style zero tail after capture, so curves continue to 0 once all assigned errors enter the capture radius.",
        "",
        "## Files",
        "- `summary.json`",
        "- `fig_trajectory_xy.png`",
        "- `fig_team_errors.png`",
        "- `fig_assignment_timeline.png`",
        "- `fig_comm_ratio.png`",
        "- `fig_weight_convergence.png`",
        "- `fig_assigned_errors.png`",
    ]
    dump_markdown(path, "\n".join(lines))


def _run_single_case(job: dict[str, Any]) -> dict[str, Any]:
    case = TeamCommGeneralCase(**job["case"])
    case_dir = Path(job["case_dir"])
    quick = bool(job["quick"])
    make_plots = bool(job.get("make_plots", True))
    mode = str(job.get("mode", "comm"))
    summed_mode = mode == "summed"

    scenario = build_team_comm_multi_scenario(_scenario_spec(case, quick))
    learning = _learning_params(case.n_pursuers, case.n_evaders, quick, mode=mode)
    sim = MultiTeamCommunicationSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=_control_params(),
        learning_params=learning,
        feature_params=_feature_params(),
        summed_mode=summed_mode,
    )

    dynamic_graph_eval = scenario.n_evaders > 1
    t0 = time.perf_counter()
    if summed_mode:
        # Summed mode: no communication, train with full visibility
        train = sim.train_policy(seed=case.seed, dynamic_graph=False, visibility_mode="full")
    else:
        train = sim.train_policy(seed=case.seed, dynamic_graph=False, visibility_mode="curriculum")
    # Evaluation with communication dropout (realistic) -- summed mode forces full internally
    eval_result = sim.evaluate_policy(
        weights=train.weights,
        seed=case.seed + 1000,
        dynamic_graph=dynamic_graph_eval,
        visibility_mode="dropout",
        stop_on_capture=False,
        zero_tail_after_capture=False,
        record_logs=False,
    )
    # Evaluation with full communication (baseline for comparison)
    eval_full = sim.evaluate_policy(
        weights=train.weights,
        seed=case.seed + 1000,
        dynamic_graph=dynamic_graph_eval,
        visibility_mode="full",
        stop_on_capture=False,
        zero_tail_after_capture=False,
        record_logs=False,
    )
    wall_s = time.perf_counter() - t0

    rpi = _rollouts_per_iter(learning)
    case_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        plot_multiteam_trajectory_xy(eval_result, case_dir / "fig_trajectory_xy.png", f"{case.name} trajectory")
        plot_multiteam_errors(eval_result, learning.dt, case_dir / "fig_team_errors.png", f"{case.name} Eteam")
        plot_assignment_timeline(
            eval_result,
            learning.dt,
            case_dir / "fig_assignment_timeline.png",
            f"{case.name} dynamic assignments",
        )
        plot_comm_ratio(eval_result, learning.dt, case_dir / "fig_comm_ratio.png", f"{case.name} communication ratio")
        plot_group_weight_convergence(
            train,
            case_dir / "fig_weight_convergence.png",
            f"{case.name} group critic convergence",
            dt=learning.dt,
            rollout_steps=int(learning.rollout_steps),
            rollouts_per_iter=rpi,
        )
        plot_assigned_error_summary(
            eval_result,
            learning.dt,
            case_dir / "fig_assigned_errors.png",
            f"{case.name} assigned error summary",
        )
        plot_convergence_diagnostics(
            train,
            case_dir / "fig_convergence_diagnostics.png",
            f"{case.name} convergence diagnostics",
            dt=learning.dt,
            rollout_steps=int(learning.rollout_steps),
            rollouts_per_iter=rpi,
        )
        plot_comm_comparison(
            eval_full,
            eval_result,
            learning.dt,
            case_dir / "fig_comm_comparison.png",
            f"{case.name} full comm vs dropout",
        )
        # GIF disabled for speed — uncomment to re-enable:
        # plot_multiteam_trajectory_gif(
        #     eval_result,
        #     learning.dt,
        #     case_dir / "trajectory.gif",
        #     f"{case.name} pursuit-evasion",
        #     capture_radius=float(scenario.capture_radius),
        # )

    switch_times = _switch_times(eval_result.assignment_history, learning.dt, scenario.initial_assignment)
    train_iterations = int(train.weight_norm_history.shape[0] - 1)
    total_steps = train_iterations * _rollouts_per_iter(learning) * int(learning.rollout_steps) + int(eval_result.team_errors.shape[0])

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
            "group_sizes": scenario.group_sizes.tolist(),
            "capture_radius": float(scenario.capture_radius),
            "swap_threshold": float(scenario.swap_threshold),
            "drop_windows": [asdict(win) for win in scenario.communication.windows],
        },
        "eval": {
            "capture_time_s": None if eval_result.capture_time is None else float(eval_result.capture_time),
            "final_team_error": float(eval_result.team_errors[-1]),
            "final_max_assigned_error": float(np.max(eval_result.assigned_errors[-1])),
            "mean_assigned_error": float(np.mean(eval_result.assigned_errors)),
            "final_group_errors": np.round(eval_result.group_errors[-1], 6).tolist(),
            "final_group_comm_ratio": np.round(eval_result.communication_ratio[-1], 6).tolist(),
            "min_group_comm_ratio": np.round(np.min(eval_result.communication_ratio, axis=0), 6).tolist(),
            "final_group_estimate_errors": np.round(eval_result.estimate_errors[-1], 6).tolist(),
            "random_drop_windows": eval_result.communication_windows,
        },
        "switching": {
            "switch_count": int(len(switch_times)),
            "switch_times_s": switch_times,
        },
        "train": {
            "iterations": train_iterations,
            "best_iteration": int(train.best_iteration),
            "final_delta": np.round(train.delta_history[-1], 8).tolist() if train.delta_history.size else [],
            "final_residual": np.round(train.residual_history[-1], 8).tolist() if train.residual_history.size else [],
            "final_weight_norms": np.round(train.weight_norm_history[-1], 8).tolist(),
            "validation_history": np.round(train.validation_history[:, 0], 6).tolist() if train.validation_history.size else [],
        },
        "runtime": {
            "wall_time_s": float(wall_s),
            "simulated_steps_total": int(total_steps),
            "ms_per_step_total": float(1000.0 * wall_s / max(total_steps, 1)),
        },
    }
    dump_json(case_dir / "summary.json", summary)
    _write_case_report(case_dir / "REPORT.md", case, scenario, train, eval_result, learning, wall_s, switch_times)
    return summary


def _write_batch_report(out_dir: Path, summaries: list[dict[str, Any]], total_wall_s: float, workers: int) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parallel_workers": int(workers),
        "total_wall_time_s": float(total_wall_s),
        "cases": summaries,
    }
    dump_json(out_dir / "batch_summary.json", payload)

    lines = [
        "# Generalized Team Communication Batch Summary",
        "",
        f"- Output folder: `{out_dir}`",
        f"- Parallel workers: `{workers}`",
        f"- Total wall time (s): `{total_wall_s:.3f}`",
        "",
        "## Cases",
        "| case | group sizes | capture (s) | final Eteam | final max assigned | switch count | min comm ratio | ms/step |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {case} | {sizes} | {capture} | {eteam:.3f} | {amax:.3f} | {switches} | {ratio} | {msp:.4f} |".format(
                case=item["case"]["name"],
                sizes=item["scenario"]["group_sizes"],
                capture="-" if item["eval"]["capture_time_s"] is None else f"{item['eval']['capture_time_s']:.2f}",
                eteam=item["eval"]["final_team_error"],
                amax=item["eval"]["final_max_assigned_error"],
                switches=item["switching"]["switch_count"],
                ratio=item["eval"]["min_group_comm_ratio"],
                msp=item["runtime"]["ms_per_step_total"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Each evader owns a fixed number of communication slots, so dynamic reassignment swaps slot occupants instead of changing group size.",
            "- Every group uses a communication-aware team critic with masked decentralized execution, and only same-target pursuers share information.",
            "- Random communication outages are resampled independently for each tracking group in every rollout and evaluation episode.",
            "- Runtime is reduced by vectorized slot-cost computation plus optional process-level parallel case execution.",
            "- Reported horizon-end errors use a zero tail after capture so the convergence plots visibly settle to 0 once capture is achieved.",
        ]
    )
    dump_markdown(out_dir / "REPORT.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generalized multi-team communication-aware pursuit-evasion cases.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Short debug run")
    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=["4v2", "5v2", "6v3"],
        help="Case tokens like 4v2 5v2 6v3",
    )
    parser.add_argument("--assignment-mode", type=str, default="shifted", help="zero/cyclic/shifted/nearest/random")
    parser.add_argument("--layout-mode", type=str, default="structured", help="structured/random")
    parser.add_argument("--seed-base", type=int, default=29, help="Base random seed")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel worker count")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure generation")
    parser.add_argument(
        "--mode",
        type=str,
        default="comm",
        choices=["comm", "summed"],
        help="Controller mode: 'comm' (stacked, communication-aware) or 'summed' (original paper's summed-state)",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    mode_suffix = f"_{args.mode}" if args.mode != "comm" else ""
    out_dir = Path(args.output) if args.output else root / "outputs" / f"team_comm_generalized{mode_suffix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        _parse_case_token(token, seed=int(args.seed_base) + 10 * idx, assignment_mode=args.assignment_mode, layout_mode=args.layout_mode)
        for idx, token in enumerate(args.cases)
    ]
    mode = str(args.mode)
    jobs = [
        {
            "case": asdict(case),
            "case_dir": str(out_dir / case.name),
            "quick": bool(args.quick),
            "make_plots": not bool(args.skip_plots),
            "mode": mode,
        }
        for case in cases
    ]

    t0 = time.perf_counter()
    workers = int(args.parallel_workers)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(_run_single_case, jobs))
    else:
        summaries = [_run_single_case(job) for job in jobs]
    total_wall_s = time.perf_counter() - t0

    summaries.sort(key=lambda item: (item["case"]["n_pursuers"], item["case"]["n_evaders"], item["case"]["seed"]))
    _write_batch_report(out_dir, summaries, total_wall_s, workers)

    print(json.dumps({"output_dir": str(out_dir), "cases": [item["case"]["name"] for item in summaries]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
