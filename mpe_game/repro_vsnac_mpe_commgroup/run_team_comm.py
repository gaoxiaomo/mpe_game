from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from mpe_repro.config import AircraftParams, ControlParams, LearningParams
from mpe_repro.report import dump_markdown
from mpe_repro.team_comm_config import TeamFeatureParams, communication_three_pursuer_one_evader_scenario
from mpe_repro.team_comm_plotting import (
    plot_control_inputs_compare,
    plot_team_error_compare,
    plot_team_trajectory_multiview,
    plot_team_weight_convergence,
)
from mpe_repro.team_comm_simulator import TeamCommunicationSimulator


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sample_eteam_at_time(team_errors: np.ndarray, dt: float, time_s: float | None) -> float | None:
    if time_s is None or team_errors.size == 0:
        return None
    idx = int(round(time_s / dt))
    idx = max(0, min(idx, team_errors.shape[0] - 1))
    return float(team_errors[idx])


def write_report(path: Path, train, no_drop, drop, output_dir: Path, learning: LearningParams, matched_time_s: float | None) -> None:
    no_drop_matched = sample_eteam_at_time(no_drop.team_errors, learning.dt, matched_time_s)
    drop_matched = sample_eteam_at_time(drop.team_errors, learning.dt, matched_time_s)
    peak_est = np.max(drop.estimate_errors, axis=0).tolist() if drop.estimate_errors.size else []
    lines = [
        "# Team Communication 3v1 Report",
        "",
        "## Scenario",
        "- New communication version is applied only to the three-pursuer-one-evader case.",
        "- The three-pursuer-three-evader case remains the original one-to-one dynamic target graph interpretation.",
        "- Training uses full communication; evaluation compares `no-drop` and `drop/recovery` under the same trained team critic and the same bounded evader maneuver profile.",
        "- The evader maneuver is kept identical across the two evaluations so the performance gap mainly reflects pursuer-team communication, not a different evader policy.",
        "",
        "## Core Metric",
        "- `Eteam = sum_j || nu * x_tilde_j ||`, with `nu = [1,1,1,0,0,0]`.",
        "- This keeps the evaluation focused on x/y/h relative capture quality, consistent with the original project.",
        "",
        "## Results",
        f"- No-drop capture time: `{None if no_drop.capture_time is None else round(float(no_drop.capture_time), 3)} s`",
        f"- Drop/recovery capture time: `{None if drop.capture_time is None else round(float(drop.capture_time), 3)} s`",
        f"- Final no-drop Eteam (at its stop time): `{float(no_drop.team_errors[-1]):.6f}`",
        f"- Final drop/recovery Eteam (at its stop time): `{float(drop.team_errors[-1]):.6f}`",
        f"- Matched comparison time: `{None if matched_time_s is None else round(float(matched_time_s), 3)} s`",
        f"- No-drop Eteam at matched time: `{None if no_drop_matched is None else round(float(no_drop_matched), 6)}`",
        f"- Drop/recovery Eteam at matched time: `{None if drop_matched is None else round(float(drop_matched), 6)}`",
        f"- Final estimate mismatch after recovery: `{np.round(drop.estimate_errors[-1], 6).tolist()}`",
        f"- Peak estimate mismatch during rollout: `{np.round(np.asarray(peak_est), 6).tolist()}`",
        "",
        "## Training",
        f"- Iterations executed: `{train.weight_history.shape[0] - 1}`",
        f"- Converged iteration: `{train.best_iteration}`",
        f"- Final delta: `{np.round(train.delta_history[-1], 8).tolist() if train.delta_history.size else []}`",
        f"- Final residual rms: `{np.round(train.residual_history[-1], 8).tolist() if train.residual_history.size else []}`",
        "",
        "## Output Files",
        f"- Team trajectory (no-drop): `{(output_dir / 'fig_team_traj_no_drop.png').name}`",
        f"- Team trajectory (drop/recovery): `{(output_dir / 'fig_team_traj_drop.png').name}`",
        f"- Eteam compare: `{(output_dir / 'fig_eteam_compare.png').name}`",
        f"- Control inputs: `{(output_dir / 'fig_control_inputs.png').name}`",
        f"- Weight convergence: `{(output_dir / 'fig_weight_convergence.png').name}`",
        "",
        "## Interpretation",
        "- `no-drop` is the ideal centralized-information execution limit.",
        "- `drop/recovery` keeps the same team critic and same initial state, but each pursuer uses zero-filled teammate blocks together with a visibility mask while the communication window is disabled.",
        "- The matched-time Eteam comparison is the fair comparison for performance degradation.",
        "- The later capture time in `drop/recovery` means communication loss slows the team down, while recovery still allows the pursuit to converge.",
        "- Training now uses team-specific projected LS, fresh current-policy samples, and damped backtracking updates, so the raw critic weights themselves converge instead of drifting in later iterations.",
        "",
        "## Notes",
        f"- Evaluation time step: `{learning.dt}` s",
        "- This folder contains only the new multi-pursuer-one-evader communication innovation. The original reproduction remains untouched.",
    ]
    dump_markdown(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the communication-aware 3v1 team-critic experiment.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Short debug run")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    output_dir = Path(args.output) if args.output else root / "outputs" / f"team_comm_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = communication_three_pursuer_one_evader_scenario()
    control = ControlParams(
        q_diag=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        r1_diag=(1.0, 1.0, 1.0),
        r2_diag=(1.0, 1.0, 1.0),
        u_bar_p=25.0,
        u_bar_e=15.0,
        u_bar_p_policy=35.0,
        u_bar_e_policy=15.0,
    )
    learning = LearningParams(
        dt=0.05,
        ridge_lambda=1e-3,
        replay_capacity=30000,
        min_samples_per_evader=160 if args.quick else 500,
        policy_iterations=12 if args.quick else 30,
        rollout_steps=90 if args.quick else 220,
        exploration_std_start=0.35,
        exploration_std_end=0.02,
        critic_learning_rate=0.15,
        convergence_tol=5e-4,
        random_perturb_scale=0.02,
    )
    features = TeamFeatureParams(
        state_scale=(2600.0, 2600.0, 2600.0, 150.0, 150.0, 150.0),
        local_gain=2.8,
        cross_gain=1.35,
    )

    sim = TeamCommunicationSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=features,
    )

    train = sim.train_policy(seed=17)
    eval_seed = 117
    no_drop = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        use_dropout=False,
        stop_on_capture=True,
        record_logs=True,
    )
    drop = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        use_dropout=True,
        stop_on_capture=True,
        record_logs=True,
    )

    plot_team_trajectory_multiview(no_drop, learning.dt, output_dir / "fig_team_traj_no_drop.png", "3v1 team communication (no-drop)")
    plot_team_trajectory_multiview(drop, learning.dt, output_dir / "fig_team_traj_drop.png", "3v1 team communication (drop/recovery)")
    plot_team_error_compare(
        no_drop,
        drop,
        learning.dt,
        output_dir / "fig_eteam_compare.png",
        "Eteam (paper-style zero tail after capture)",
    )
    plot_control_inputs_compare(
        no_drop,
        drop,
        learning.dt,
        output_dir / "fig_control_inputs.png",
        "Control inputs (x-channel)",
        component_idx=0,
    )
    plot_team_weight_convergence(train, output_dir / "fig_weight_convergence.png", "Team critic convergence")

    matched_time_s = None if no_drop.capture_time is None else float(no_drop.capture_time)
    matched_eteam_no_drop = sample_eteam_at_time(no_drop.team_errors, learning.dt, matched_time_s)
    matched_eteam_drop = sample_eteam_at_time(drop.team_errors, learning.dt, matched_time_s)
    peak_estimate_error = np.max(drop.estimate_errors, axis=0).tolist() if drop.estimate_errors.size else []

    summary = {
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
    }
    write_json(output_dir / "summary.json", summary)

    if no_drop.step_logs is not None:
        dump_markdown(
            output_dir / "eval_log_no_drop.md",
            "# No-drop evaluation log\n\n```json\n"
            + json.dumps(no_drop.step_logs, ensure_ascii=False, indent=2)
            + "\n```",
        )
    if drop.step_logs is not None:
        dump_markdown(
            output_dir / "eval_log_drop_recovery.md",
            "# Drop/recovery evaluation log\n\n```json\n"
            + json.dumps(drop.step_logs, ensure_ascii=False, indent=2)
            + "\n```",
        )

    write_report(output_dir / "REPORT.md", train, no_drop, drop, output_dir, learning, matched_time_s)

    print(f"output_dir={output_dir}")
    print(
        "no_drop_capture_time_s=",
        None if no_drop.capture_time is None else round(float(no_drop.capture_time), 3),
    )
    print(
        "drop_recovery_capture_time_s=",
        None if drop.capture_time is None else round(float(drop.capture_time), 3),
    )
    print(
        "matched_eteam=",
        None if matched_eteam_no_drop is None else round(float(matched_eteam_no_drop), 3),
        None if matched_eteam_drop is None else round(float(matched_eteam_drop), 3),
    )


if __name__ == "__main__":
    main()
