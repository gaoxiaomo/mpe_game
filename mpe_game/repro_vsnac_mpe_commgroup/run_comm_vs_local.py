from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from mpe_repro.config import AircraftParams, ControlParams, LearningParams
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.team_comm_config import TeamFeatureParams, communication_many_pursuer_one_evader_scenario
from mpe_repro.team_comm_plotting import (
    plot_control_inputs_compare,
    plot_team_error_multi,
    plot_team_trajectory_multiview,
    plot_team_weight_convergence,
)
from mpe_repro.team_comm_simulator import TeamCommunicationSimulator, TeamCommEvalResult, TeamCommTrainResult


@dataclass(frozen=True)
class TeamCommVsLocalCase:
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
            min_samples_per_evader=220,
            policy_iterations=10 + max(0, n_pursuers - 3),
            rollout_steps=100 + 15 * n_pursuers,
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
    case: TeamCommVsLocalCase,
    full_train: TeamCommTrainResult,
    local_train: TeamCommTrainResult,
    full_eval: TeamCommEvalResult,
    local_eval: TeamCommEvalResult,
    drop_eval: TeamCommEvalResult,
    learning: LearningParams,
    output_dir: Path,
    scenario_windows: list[dict],
) -> None:
    matched_time = None if full_eval.capture_time is None else float(full_eval.capture_time)
    full_matched = _sample_eteam_at_time(full_eval.team_errors, learning.dt, matched_time)
    local_matched = _sample_eteam_at_time(local_eval.team_errors, learning.dt, matched_time)
    drop_matched = _sample_eteam_at_time(drop_eval.team_errors, learning.dt, matched_time)
    lines = [
        f"# Communication vs Local Baseline Report: {case.name}",
        "",
        "## Scenario",
        f"- Pursuers: `{case.n_pursuers}`",
        "- Evaders: `1`",
        "- `full-comm`: team critic trained and executed with full team visibility.",
        "- `local-only`: same team-state formulation, but each pursuer can only observe its own block (self mask only) during both training and execution.",
        "- `drop/recovery`: uses the full-communication-trained critic, but injects a finite blackout window during execution.",
        f"- Drop windows: `{scenario_windows}`",
        "",
        "## Interpretation",
        "- If communication is useful, then the full-communication policy should reduce `Eteam` faster and/or capture earlier than the local-only baseline.",
        "- The local-only baseline represents the previous non-communication many-pursuer-one-evader setting.",
        "",
        "## Results",
        f"- Full-communication capture time: `{None if full_eval.capture_time is None else round(float(full_eval.capture_time), 3)} s`",
        f"- Local-only capture time: `{None if local_eval.capture_time is None else round(float(local_eval.capture_time), 3)} s`",
        f"- Drop/recovery capture time: `{None if drop_eval.capture_time is None else round(float(drop_eval.capture_time), 3)} s`",
        f"- Matched comparison time (full-comm capture): `{None if matched_time is None else round(float(matched_time), 3)} s`",
        f"- Full-comm Eteam at matched time: `{None if full_matched is None else round(float(full_matched), 6)}`",
        f"- Local-only Eteam at matched time: `{None if local_matched is None else round(float(local_matched), 6)}`",
        f"- Drop/recovery Eteam at matched time: `{None if drop_matched is None else round(float(drop_matched), 6)}`",
        f"- Final estimate mismatch after recovery: `{np.round(drop_eval.estimate_errors[-1], 6).tolist()}`",
        "",
        "## Training",
        f"- Full-comm iterations: `{full_train.weight_history.shape[0] - 1}`",
        f"- Local-only iterations: `{local_train.weight_history.shape[0] - 1}`",
        f"- Full-comm weight norm squared: `{float(np.sum(full_train.weight_history[-1] ** 2)):.6f}`",
        f"- Local-only weight norm squared: `{float(np.sum(local_train.weight_history[-1] ** 2)):.6f}`",
        "",
        "## Files",
        f"- `{(output_dir / 'fig_team_traj_full.png').name}`",
        f"- `{(output_dir / 'fig_team_traj_local.png').name}`",
        f"- `{(output_dir / 'fig_team_traj_drop.png').name}`",
        f"- `{(output_dir / 'fig_eteam_compare_all.png').name}`",
        f"- `{(output_dir / 'fig_control_inputs_full_vs_local.png').name}`",
        f"- `{(output_dir / 'fig_weight_convergence_full.png').name}`",
        f"- `{(output_dir / 'fig_weight_convergence_local.png').name}`",
    ]
    dump_markdown(path, "\n".join(lines))


def _run_case(job: dict) -> dict:
    case = TeamCommVsLocalCase(**job["case"])
    case_dir = Path(job["case_dir"])
    quick = bool(job["quick"])

    scenario = communication_many_pursuer_one_evader_scenario(case.n_pursuers, seed=case.seed)
    sim = TeamCommunicationSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=_control_params(),
        learning_params=_learning_params(case.n_pursuers, quick),
        feature_params=_feature_params(),
    )
    learning = _learning_params(case.n_pursuers, quick)

    full_train = sim.train_policy(seed=case.seed, visibility_mode="full")
    local_train = sim.train_policy(seed=case.seed, visibility_mode="self_only")

    eval_seed = case.seed + 1000
    full_eval = sim.evaluate_policy(full_train.weights, seed=eval_seed, visibility_mode="full", stop_on_capture=True, record_logs=False)
    local_eval = sim.evaluate_policy(local_train.weights, seed=eval_seed, visibility_mode="self_only", stop_on_capture=True, record_logs=False)
    drop_eval = sim.evaluate_policy(full_train.weights, seed=eval_seed, visibility_mode="dropout", stop_on_capture=True, record_logs=False)

    case_dir.mkdir(parents=True, exist_ok=True)
    plot_team_trajectory_multiview(full_eval, learning.dt, case_dir / "fig_team_traj_full.png", f"{case.name} full communication")
    plot_team_trajectory_multiview(local_eval, learning.dt, case_dir / "fig_team_traj_local.png", f"{case.name} local-only baseline")
    plot_team_trajectory_multiview(drop_eval, learning.dt, case_dir / "fig_team_traj_drop.png", f"{case.name} drop/recovery")
    plot_team_error_multi(
        [
            ("Full communication", full_eval),
            ("Local only", local_eval),
            ("Drop / recovery", drop_eval),
        ],
        learning.dt,
        case_dir / "fig_eteam_compare_all.png",
        f"{case.name} Eteam comparison",
    )
    plot_control_inputs_compare(
        full_eval,
        local_eval,
        learning.dt,
        case_dir / "fig_control_inputs_full_vs_local.png",
        f"{case.name} control inputs: full vs local",
        component_idx=0,
    )
    plot_team_weight_convergence(full_train, case_dir / "fig_weight_convergence_full.png", f"{case.name} full-comm critic convergence")
    plot_team_weight_convergence(local_train, case_dir / "fig_weight_convergence_local.png", f"{case.name} local-only critic convergence")

    matched_time = None if full_eval.capture_time is None else float(full_eval.capture_time)
    windows = [
        {
            "start_s": float(win.start_s),
            "end_s": float(win.end_s),
            "isolated_agents": list(win.isolated_agents),
        }
        for win in scenario.communication.windows
    ]

    summary = {
        "case": {"name": case.name, "n_pursuers": case.n_pursuers, "n_evaders": 1, "seed": case.seed},
        "scenario": {"name": scenario.name, "capture_radius": float(scenario.capture_radius), "drop_windows": windows},
        "full_comm": {
            "capture_time_s": None if full_eval.capture_time is None else float(full_eval.capture_time),
            "final_eteam": float(full_eval.team_errors[-1]),
        },
        "local_only": {
            "capture_time_s": None if local_eval.capture_time is None else float(local_eval.capture_time),
            "final_eteam": float(local_eval.team_errors[-1]),
        },
        "drop_recovery": {
            "capture_time_s": None if drop_eval.capture_time is None else float(drop_eval.capture_time),
            "final_eteam": float(drop_eval.team_errors[-1]),
            "final_estimate_errors": np.round(drop_eval.estimate_errors[-1], 6).tolist(),
        },
        "matched_time_compare": {
            "time_s": matched_time,
            "full_comm_eteam": _sample_eteam_at_time(full_eval.team_errors, learning.dt, matched_time),
            "local_only_eteam": _sample_eteam_at_time(local_eval.team_errors, learning.dt, matched_time),
            "drop_recovery_eteam": _sample_eteam_at_time(drop_eval.team_errors, learning.dt, matched_time),
        },
        "train": {
            "full_comm_iterations": int(full_train.weight_history.shape[0] - 1),
            "local_only_iterations": int(local_train.weight_history.shape[0] - 1),
            "full_comm_weight_norm_sq": float(np.sum(full_train.weight_history[-1] ** 2)),
            "local_only_weight_norm_sq": float(np.sum(local_train.weight_history[-1] ** 2)),
        },
    }

    dump_json(case_dir / "summary.json", summary)
    _write_case_report(
        case_dir / "REPORT.md",
        case,
        full_train,
        local_train,
        full_eval,
        local_eval,
        drop_eval,
        learning,
        case_dir,
        windows,
    )
    return summary


def _write_batch_report(out_dir: Path, summaries: list[dict], workers: int, total_wall_s: float) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parallel_workers": int(workers),
        "total_wall_time_s": float(total_wall_s),
        "cases": summaries,
    }
    dump_json(out_dir / "batch_summary.json", payload)

    lines = [
        "# Communication vs Local Baseline Summary",
        "",
        f"- Output folder: `{out_dir}`",
        f"- Parallel workers: `{workers}`",
        f"- Total wall time (s): `{total_wall_s:.3f}`",
        "",
        "## Cases",
        "| case | full capture (s) | local capture (s) | drop capture (s) | full Eteam@matched | local Eteam@matched | drop Eteam@matched |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {case} | {full_cap} | {local_cap} | {drop_cap} | {full_e:.3f} | {local_e:.3f} | {drop_e:.3f} |".format(
                case=item["case"]["name"],
                full_cap="-" if item["full_comm"]["capture_time_s"] is None else f"{item['full_comm']['capture_time_s']:.2f}",
                local_cap="-" if item["local_only"]["capture_time_s"] is None else f"{item['local_only']['capture_time_s']:.2f}",
                drop_cap="-" if item["drop_recovery"]["capture_time_s"] is None else f"{item['drop_recovery']['capture_time_s']:.2f}",
                full_e=item["matched_time_compare"]["full_comm_eteam"],
                local_e=item["matched_time_compare"]["local_only_eteam"],
                drop_e=item["matched_time_compare"]["drop_recovery_eteam"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `full communication` uses the richest team information set.",
            "- `local only` is the non-communication baseline: each pursuer can only observe its own block.",
            "- `drop / recovery` quantifies robustness loss when the full-communication policy suffers a finite blackout window.",
            "- If communication is beneficial, then full-communication should capture earlier and/or yield lower matched-time Eteam than the local-only baseline.",
        ]
    )
    dump_markdown(out_dir / "REPORT.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair communication-vs-local comparison for many-pursuer one-evader team pursuit.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Shorter experiments")
    parser.add_argument("--pursuer-cases", type=int, nargs="*", default=[3, 4, 5], help="Pursuer counts to evaluate")
    parser.add_argument("--seed-base", type=int, default=17, help="Base random seed")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Parallel worker count")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"comm_vs_local_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [TeamCommVsLocalCase(n_pursuers=int(n), seed=int(args.seed_base) + 10 * idx) for idx, n in enumerate(args.pursuer_cases)]
    jobs = [{"case": asdict(case), "case_dir": str(out_dir / case.name), "quick": bool(args.quick)} for case in cases]

    workers = int(args.parallel_workers)
    if workers <= 0:
        workers = min(len(jobs), max(1, 2))

    t0 = datetime.now()
    if len(jobs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            summaries = list(pool.map(_run_case, jobs))
    else:
        summaries = [_run_case(job) for job in jobs]
    total_wall_s = (datetime.now() - t0).total_seconds()

    summaries.sort(key=lambda item: item["case"]["n_pursuers"])
    _write_batch_report(out_dir, summaries, workers, total_wall_s)
    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
