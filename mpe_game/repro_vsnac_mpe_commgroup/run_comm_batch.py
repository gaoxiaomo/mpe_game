from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mpe_repro.config import AircraftParams, ControlParams, LearningParams
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.team_comm_config import TeamFeatureParams, communication_many_pursuer_one_evader_scenario
from mpe_repro.team_comm_plotting import (
    plot_control_inputs_compare,
    plot_team_error_compare,
    plot_team_trajectory_multiview,
    plot_team_weight_convergence,
)
from mpe_repro.team_comm_simulator import TeamCommunicationSimulator, TeamCommEvalResult, TeamCommTrainResult


@dataclass(frozen=True)
class TeamCommCase:
    n_pursuers: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.n_pursuers}v1"


def _learning_params(n_pursuers: int, quick: bool) -> LearningParams:
    if quick:
        return LearningParams(
            dt=0.05,
            ridge_lambda=1e-3,
            replay_capacity=40000,
            min_samples_per_evader=200,
            policy_iterations=10 + max(0, n_pursuers - 3),
            rollout_steps=90 + 15 * n_pursuers,
            exploration_std_start=0.35,
            exploration_std_end=0.02,
            critic_learning_rate=0.10,
            convergence_tol=5e-4,
            random_perturb_scale=0.02,
        )
    return LearningParams(
        dt=0.05,
        ridge_lambda=1e-3,
        replay_capacity=60000,
        min_samples_per_evader=420,
        policy_iterations=18 + max(0, n_pursuers - 3),
        rollout_steps=170 + 20 * n_pursuers,
        exploration_std_start=0.35,
        exploration_std_end=0.02,
        critic_learning_rate=0.10,
        convergence_tol=2e-4,
        random_perturb_scale=0.02,
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
    )


def _feature_params() -> TeamFeatureParams:
    return TeamFeatureParams(
        state_scale=(3000.0, 3000.0, 2600.0, 160.0, 160.0, 160.0),
        local_gain=2.8,
        cross_gain=1.35,
    )


def _sample_eteam_at_time(team_errors: np.ndarray, dt: float, time_s: float | None) -> float | None:
    if time_s is None or team_errors.size == 0:
        return None
    idx = int(round(time_s / dt))
    idx = max(0, min(idx, team_errors.shape[0] - 1))
    return float(team_errors[idx])


def _write_case_report(
    path: Path,
    case: TeamCommCase,
    train: TeamCommTrainResult,
    no_drop: TeamCommEvalResult,
    drop: TeamCommEvalResult,
    learning: LearningParams,
    output_dir: Path,
    runtime_s: float,
    scenario_windows: list[dict[str, Any]],
) -> None:
    matched_time_s = None if no_drop.capture_time is None else float(no_drop.capture_time)
    no_drop_matched = _sample_eteam_at_time(no_drop.team_errors, learning.dt, matched_time_s)
    drop_matched = _sample_eteam_at_time(drop.team_errors, learning.dt, matched_time_s)
    peak_est = np.max(drop.estimate_errors, axis=0).tolist() if drop.estimate_errors.size else []
    lines = [
        f"# Team Communication Batch Report: {case.name}",
        "",
        "## Scenario",
        f"- Pursuers: `{case.n_pursuers}`",
        "- Evaders: `1`",
        "- Comparison is fair: same trained weights, same initial state, same eval seed; only communication availability changes.",
        "- Drop windows are deterministic random windows generated from the case seed.",
        f"- Drop windows: `{scenario_windows}`",
        "",
        "## Core Metric",
        "- `Eteam = sum_j || nu * x_tilde_j ||`, with `nu = [1,1,1,0,0,0]`.",
        "",
        "## Results",
        f"- No-drop capture time: `{None if no_drop.capture_time is None else round(float(no_drop.capture_time), 3)} s`",
        f"- Drop/recovery capture time: `{None if drop.capture_time is None else round(float(drop.capture_time), 3)} s`",
        f"- Final no-drop Eteam: `{float(no_drop.team_errors[-1]):.6f}`",
        f"- Final drop/recovery Eteam: `{float(drop.team_errors[-1]):.6f}`",
        f"- Matched comparison time: `{None if matched_time_s is None else round(float(matched_time_s), 3)} s`",
        f"- No-drop Eteam at matched time: `{None if no_drop_matched is None else round(float(no_drop_matched), 6)}`",
        f"- Drop/recovery Eteam at matched time: `{None if drop_matched is None else round(float(drop_matched), 6)}`",
        f"- Peak estimate mismatch during rollout: `{np.round(np.asarray(peak_est), 6).tolist()}`",
        f"- Final estimate mismatch after recovery: `{np.round(drop.estimate_errors[-1], 6).tolist()}`",
        "",
        "## Training",
        f"- Iterations executed: `{train.weight_history.shape[0] - 1}`",
        f"- Final delta: `{np.round(train.delta_history[-1], 8).tolist() if train.delta_history.size else []}`",
        f"- Weight norm squared: `{float(np.sum(train.weight_history[-1] ** 2)):.6f}`",
        "",
        "## Runtime",
        f"- Total wall time (s): `{runtime_s:.3f}`",
        "",
        "## Files",
        f"- `{(output_dir / 'fig_team_traj_no_drop.png').name}`",
        f"- `{(output_dir / 'fig_team_traj_drop.png').name}`",
        f"- `{(output_dir / 'fig_eteam_compare.png').name}`",
        f"- `{(output_dir / 'fig_control_inputs.png').name}`",
        f"- `{(output_dir / 'fig_weight_convergence.png').name}`",
    ]
    dump_markdown(path, "\n".join(lines))


def _run_case(job: dict[str, Any]) -> dict[str, Any]:
    case = TeamCommCase(**job["case"])
    case_dir = Path(job["case_dir"])
    quick = bool(job["quick"])

    scenario = communication_many_pursuer_one_evader_scenario(case.n_pursuers, seed=case.seed)
    control = _control_params()
    learning = _learning_params(case.n_pursuers, quick)
    features = _feature_params()

    sim = TeamCommunicationSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=features,
    )

    t0 = time.perf_counter()
    train = sim.train_policy(seed=case.seed)
    eval_seed = case.seed + 1000
    no_drop = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        use_dropout=False,
        stop_on_capture=True,
        record_logs=False,
    )
    drop = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        use_dropout=True,
        stop_on_capture=True,
        record_logs=False,
    )
    wall_s = time.perf_counter() - t0

    case_dir.mkdir(parents=True, exist_ok=True)
    plot_team_trajectory_multiview(no_drop, learning.dt, case_dir / "fig_team_traj_no_drop.png", f"{case.name} no-drop")
    plot_team_trajectory_multiview(drop, learning.dt, case_dir / "fig_team_traj_drop.png", f"{case.name} drop/recovery")
    plot_team_error_compare(no_drop, drop, learning.dt, case_dir / "fig_eteam_compare.png", f"{case.name} Eteam")
    plot_control_inputs_compare(no_drop, drop, learning.dt, case_dir / "fig_control_inputs.png", f"{case.name} control inputs", component_idx=0)
    plot_team_weight_convergence(train, case_dir / "fig_weight_convergence.png", f"{case.name} team critic convergence")

    matched_time_s = None if no_drop.capture_time is None else float(no_drop.capture_time)
    matched_eteam_no_drop = _sample_eteam_at_time(no_drop.team_errors, learning.dt, matched_time_s)
    matched_eteam_drop = _sample_eteam_at_time(drop.team_errors, learning.dt, matched_time_s)
    peak_estimate_error = np.max(drop.estimate_errors, axis=0).tolist() if drop.estimate_errors.size else []
    windows = [
        {
            "start_s": float(win.start_s),
            "end_s": float(win.end_s),
            "isolated_agents": list(win.isolated_agents),
        }
        for win in scenario.communication.windows
    ]
    summary = {
        "case": {
            "name": case.name,
            "n_pursuers": case.n_pursuers,
            "n_evaders": 1,
            "seed": case.seed,
        },
        "scenario": {
            "name": scenario.name,
            "capture_radius": float(scenario.capture_radius),
            "drop_windows": windows,
        },
        "no_drop": {
            "capture_time_s": None if no_drop.capture_time is None else float(no_drop.capture_time),
            "final_eteam": float(no_drop.team_errors[-1]),
            "final_block_errors": np.round(no_drop.block_errors[-1], 6).tolist(),
        },
        "drop_recovery": {
            "capture_time_s": None if drop.capture_time is None else float(drop.capture_time),
            "final_eteam": float(drop.team_errors[-1]),
            "final_block_errors": np.round(drop.block_errors[-1], 6).tolist(),
            "final_estimate_errors": np.round(drop.estimate_errors[-1], 6).tolist(),
            "peak_estimate_errors": np.round(np.asarray(peak_estimate_error), 6).tolist(),
            "min_comm_ratio": float(np.min(drop.communication_ratio)) if drop.communication_ratio.size else None,
        },
        "matched_time_compare": {
            "time_s": matched_time_s,
            "no_drop_eteam": matched_eteam_no_drop,
            "drop_recovery_eteam": matched_eteam_drop,
        },
        "train": {
            "iterations": int(train.weight_history.shape[0] - 1),
            "best_iteration": int(train.best_iteration),
            "final_delta": np.round(train.delta_history[-1], 8).tolist() if train.delta_history.size else [],
            "final_residual": np.round(train.residual_history[-1], 8).tolist() if train.residual_history.size else [],
            "weight_norm_sq": float(np.sum(train.weight_history[-1] ** 2)),
        },
        "runtime": {
            "wall_time_s": float(wall_s),
        },
        "same_weights_same_seed_baseline": True,
    }
    dump_json(case_dir / "summary.json", summary)
    _write_case_report(case_dir / "REPORT.md", case, train, no_drop, drop, learning, case_dir, wall_s, windows)
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
        "# Team Communication Batch Summary",
        "",
        f"- Output folder: `{out_dir}`",
        f"- Parallel workers: `{workers}`",
        f"- Total wall time (s): `{total_wall_s:.3f}`",
        "",
        "## Cases",
        "| case | no-drop capture (s) | drop capture (s) | no-drop Eteam@matched | drop Eteam@matched | final no-drop Eteam | final drop Eteam | min comm ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {case} | {cap_no} | {cap_drop} | {matched_no:.3f} | {matched_drop:.3f} | {final_no:.3f} | {final_drop:.3f} | {ratio:.3f} |".format(
                case=item["case"]["name"],
                cap_no="-" if item["no_drop"]["capture_time_s"] is None else f"{item['no_drop']['capture_time_s']:.2f}",
                cap_drop="-" if item["drop_recovery"]["capture_time_s"] is None else f"{item['drop_recovery']['capture_time_s']:.2f}",
                matched_no=item["matched_time_compare"]["no_drop_eteam"],
                matched_drop=item["matched_time_compare"]["drop_recovery_eteam"],
                final_no=item["no_drop"]["final_eteam"],
                final_drop=item["drop_recovery"]["final_eteam"],
                ratio=item["drop_recovery"]["min_comm_ratio"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- All cases are fair paired comparisons: same trained weights, same initial states, same evaluation seed.",
            "- `no-drop` is the ideal full-communication execution limit.",
            "- `drop/recovery` injects deterministic random communication windows while keeping the same evader motion.",
            "- The primary comparison is the matched-time Eteam: we compare both runs at the no-drop capture time.",
            "- If drop capture time is later and matched-time Eteam is larger, communication loss slows down the many-pursuer-one-evader team.",
            "- Final-horizon Eteam is reported for completeness, but it is secondary because the two runs may finish capture at different times.",
        ]
    )
    dump_markdown(out_dir / "REPORT.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for generalized n-pursuer one-evader communication cases.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Shorter experiments")
    parser.add_argument("--pursuer-cases", type=int, nargs="*", default=[3, 4, 5], help="Pursuer counts to evaluate")
    parser.add_argument("--seed-base", type=int, default=17, help="Base random seed")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel worker count")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"comm_batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [TeamCommCase(n_pursuers=int(n), seed=int(args.seed_base) + 10 * idx) for idx, n in enumerate(args.pursuer_cases)]
    jobs = [{"case": asdict(case), "case_dir": str(out_dir / case.name), "quick": bool(args.quick)} for case in cases]

    t0 = time.perf_counter()
    workers = int(args.parallel_workers)
    if workers <= 0:
        workers = min(len(jobs), max(1, 2))

    if len(jobs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(_run_case, jobs))
    else:
        summaries = [_run_case(job) for job in jobs]
    total_wall_s = time.perf_counter() - t0

    summaries.sort(key=lambda item: item["case"]["n_pursuers"])
    _write_batch_report(out_dir, summaries, total_wall_s, workers)
    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
