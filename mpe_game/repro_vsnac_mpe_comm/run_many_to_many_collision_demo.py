from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from mpe_repro.config import (
    AircraftParams,
    CommParams,
    ControlParams,
    FeatureParams,
    LearningParams,
    ScenarioConfig,
)
from mpe_repro.plotting import (
    plot_assigned_state_errors,
    plot_d_min_history,
    plot_trajectory_animation_3d,
    plot_trajectory_multiview,
)
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.simulator import EvalResult, MPECommSimulator


def _demo_control_params() -> ControlParams:
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


def _demo_feature_params() -> FeatureParams:
    return FeatureParams(
        state_scale=(3200.0, 3200.0, 2200.0, 160.0, 160.0, 160.0),
        feature_gain=2.7,
    )


def _demo_learning_params() -> LearningParams:
    return LearningParams(
        policy_iterations=22,
        rollout_steps=140,
        min_samples_per_evader=50,
        critic_learning_rate=0.05,
        critic_lr_decay=0.85,
        convergence_tol=5e-4,
        random_perturb_scale=0.01,
    )


def _build_6v3_crossing_scenario(
    *,
    evader_x_sep: float = 2600.0,
    start_x: float = 1300.0,
    target_x: float = 280.0,
    y_start: float = -1400.0,
    y_evader: float = 2600.0,
    speed: float = 80.0,
) -> ScenarioConfig:
    """Construct a fixed-assignment 6v3 crossing geometry.

    There are 3 pursuit groups. Each group has 2 pursuers assigned to the same
    evader. The two pursuers start on opposite sides of their eventual desired
    left/right offsets, so the no-analytic controller must perform a lane swap
    near the evader, which creates a strong near-collision tendency.
    """

    n_p = 6
    n_e = 3
    evaders = np.zeros((n_e, 6), dtype=float)
    pursuers = np.zeros((n_p, 6), dtype=float)
    displacements = np.zeros((n_p, n_e, 6), dtype=float)
    assignment = np.array([0, 0, 1, 1, 2, 2], dtype=int)

    evader_x = np.array([-evader_x_sep, 0.0, evader_x_sep], dtype=float)
    for i in range(n_e):
        evaders[i, :3] = [evader_x[i], y_evader + 120.0 * np.sin(float(i)), 300.0]
        evaders[i, 3:] = [0.0, 58.0, 0.0]

    # Desired physical offsets around each evader: one left, one right.
    desired_offsets = [
        np.array([-target_x, -80.0, 0.0], dtype=float),
        np.array([target_x, -80.0, 0.0], dtype=float),
    ]
    # Tracking error is x_p - x_e + r = 0, hence desired position is x_e - r.
    r_pair = [-desired_offsets[0], -desired_offsets[1]]

    idx = 0
    for i in range(n_e):
        start_pair = [
            np.array([evader_x[i] + start_x, y_start - 140.0 * i, 300.0], dtype=float),
            np.array([evader_x[i] - start_x, y_start - 140.0 * i - 70.0, 300.0], dtype=float),
        ]
        for local in range(2):
            j = idx + local
            displacements[j, :, :3] = r_pair[local]
            pursuers[j, :3] = start_pair[local]
            desired_position = evaders[i, :3] - displacements[j, i, :3]
            guide = desired_position - pursuers[j, :3]
            guide = guide / max(np.linalg.norm(guide), 1e-8)
            pursuers[j, 3:] = speed * guide
        idx += 2

    return ScenarioConfig(
        name="crossing_6v3_fixed_groups",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,
        max_switch_worsening=0.0,
        initial_assignment=assignment,
        t_final=80.0,
        capture_radius=180.0,
        evader_motion_mode="scripted",
        evader_script_amp=(1.5, 1.5, 0.3),
        evader_script_omega=0.12,
        evader_script_decay=0.02,
        evader_script_mix=0.15,
        swap_lookahead_time=0.0,
    )


def _run_method(
    scenario: ScenarioConfig,
    *,
    gamma: float,
    d_safe: float,
    train_seed: int,
    eval_seed: int,
) -> tuple[EvalResult, float]:
    comm_mode = "none" if gamma <= 0.0 else "full"
    cp = CommParams(gamma=gamma, comm_mode=comm_mode, d_safe=d_safe)
    sim = MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=_demo_control_params(),
        learning_params=_demo_learning_params(),
        feature_params=_demo_feature_params(),
        comm_params=cp,
    )
    t0 = time.perf_counter()
    train = sim.train_policy(seed=train_seed, dynamic_graph=False)
    wall = time.perf_counter() - t0
    eval_result = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        dynamic_graph=False,
        stop_on_capture=False,
        comm_params_override=cp,
    )
    return eval_result, wall


def _unsafe_metrics(d_min_history: np.ndarray, threshold: float, dt: float) -> dict[str, float]:
    unsafe = d_min_history < threshold
    total = float(np.sum(unsafe) * dt)
    longest = 0
    current = 0
    for flag in unsafe.tolist():
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "min_d_min": float(np.min(d_min_history)),
        "time_below_threshold_s": total,
        "longest_unsafe_streak_s": float(longest * dt),
    }


def _sustained_capture_time(assigned_errors: np.ndarray, capture_radius: float, dt: float) -> float | None:
    """Earliest time from which capture remains satisfied for the rest of the rollout."""
    if assigned_errors.size == 0:
        return None
    max_err = np.max(assigned_errors, axis=1)
    ok = max_err <= float(capture_radius)
    suffix_ok = np.logical_and.accumulate(ok[::-1])[::-1]
    idx = np.flatnonzero(suffix_ok)
    if idx.size == 0:
        return None
    return float(idx[0] * dt)


def _plot_curve_compare(
    t: np.ndarray,
    y_no: np.ndarray,
    y_yes: np.ndarray,
    *,
    ylabel: str,
    title: str,
    path: Path,
    threshold: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, y_no, linewidth=2.1, color="#d62728", linestyle="--", label="No analytic term")
    ax.plot(t, y_yes, linewidth=2.1, color="#2ca02c", label="Smooth analytic term")
    if threshold is not None:
        ax.axhline(threshold, color="black", linewidth=1.2, linestyle=":", label=f"$d_{{safe}}={threshold:.0f}$ m")
        ax.axhspan(0.0, threshold, color="#d62728", alpha=0.08)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_xy_compare(
    eval_no: EvalResult,
    eval_yes: EvalResult,
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)
    p_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    e_colors = ["#111111", "#444444", "#777777"]

    def draw(ax: plt.Axes, result: EvalResult, panel_title: str) -> None:
        for j in range(result.pursuer_traj.shape[1]):
            traj = result.pursuer_traj[:, j, :2]
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.8, color=p_colors[j], label=f"P{j+1}")
            ax.scatter(traj[0, 0], traj[0, 1], color=p_colors[j], s=42, marker="o", edgecolors="black", linewidths=0.5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=p_colors[j], s=34, marker="s")
        for i in range(result.evader_traj.shape[1]):
            traj = result.evader_traj[:, i, :2]
            ax.plot(traj[:, 0], traj[:, 1], linewidth=2.0, color=e_colors[i], linestyle="--", label=f"E{i+1}")
            ax.scatter(traj[0, 0], traj[0, 1], color=e_colors[i], s=48, marker="x")
        ax.set_title(panel_title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, ncol=3, loc="upper left")

    draw(axes[0], eval_no, "No analytic term")
    draw(axes[1], eval_yes, "Smooth analytic term")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _build_report(summary: dict[str, Any]) -> str:
    no = summary["no_analytic"]
    yes = summary["smooth_analytic"]
    lines = [
        "# Many-to-Many 6v3 Collision-Crossing Demo",
        "",
        "## Scenario",
        "- Three evaders move in parallel with fixed assignment.",
        "- Each evader is pursued by exactly two pursuers.",
        "- In every group, the two pursuers start on opposite sides of their desired left/right offsets, so reaching the target offsets requires an explicit lane swap near the evader.",
        "",
        "## Metrics",
        f"- Unsafe-band threshold: {summary['d_safe_m']:.1f} m",
        f"- Smooth analytic gamma: {summary['gamma']:.3f}",
        f"- No analytic term: min d_min = {no['min_d_min']:.3f} m, time below d_safe = {no['time_below_threshold_s']:.3f} s, longest unsafe streak = {no['longest_unsafe_streak_s']:.3f} s, sustained capture time = {no['capture_time_s']}, first-hit time = {no['first_hit_time_s']}",
        f"- Smooth analytic term: min d_min = {yes['min_d_min']:.3f} m, time below d_safe = {yes['time_below_threshold_s']:.3f} s, longest unsafe streak = {yes['longest_unsafe_streak_s']:.3f} s, sustained capture time = {yes['capture_time_s']}, first-hit time = {yes['first_hit_time_s']}",
        "",
        "## Interpretation",
        "- This is a many-to-many stress case, but the collision pressure is intentionally concentrated inside each pursuit group so that the effect of the analytic pairwise term is directly visible.",
        "- Without the analytic term, each pair performs a near-collision lane swap while tracking its own offset.",
        "- With the smooth analytic term, the group-level crossing is still completed, but the minimum inter-pursuer distance stays outside the chosen unsafe band.",
        "- The safer behavior comes with slower convergence, so the demo should be presented as a safety-versus-aggressiveness tradeoff, not as a universal performance gain.",
        "- Because the rollout continues after first entry into the capture radius, the report distinguishes between first-hit time and sustained capture time; the latter is used as the stricter comparison metric.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigorous 6v3 many-to-many collision-crossing demo.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--gamma", type=float, default=0.3, help="Analytic-term coupling weight")
    parser.add_argument("--d-safe", type=float, default=50.0, help="Unsafe-band threshold and smooth factor reference distance")
    parser.add_argument("--seed", type=int, default=9, help="Training seed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else root / "outputs" / f"many_to_many_collision_6v3_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = _build_6v3_crossing_scenario()
    dt = _demo_learning_params().dt

    eval_no, wall_no = _run_method(
        scenario,
        gamma=0.0,
        d_safe=0.0,
        train_seed=int(args.seed),
        eval_seed=int(args.seed) + 1000,
    )
    eval_yes, wall_yes = _run_method(
        scenario,
        gamma=float(args.gamma),
        d_safe=float(args.d_safe),
        train_seed=int(args.seed),
        eval_seed=int(args.seed) + 1000,
    )

    no_dir = out_dir / "no_analytic"
    yes_dir = out_dir / "smooth_analytic"
    no_dir.mkdir(parents=True, exist_ok=True)
    yes_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectory_multiview(eval_no, no_dir / "fig_trajectory_multiview.png", "6v3 crossing (no analytic term)")
    plot_trajectory_multiview(eval_yes, yes_dir / "fig_trajectory_multiview.png", "6v3 crossing (smooth analytic term)")
    plot_trajectory_animation_3d(eval_no, no_dir / "fig_trajectory_3d.gif", "6v3 crossing (no analytic term)", fps=12, max_frames=180)
    plot_trajectory_animation_3d(eval_yes, yes_dir / "fig_trajectory_3d.gif", "6v3 crossing (smooth analytic term)", fps=12, max_frames=180)
    plot_assigned_state_errors(eval_no, dt, no_dir / "fig_assigned_errors.png", "6v3 assigned errors (no analytic term)")
    plot_assigned_state_errors(eval_yes, dt, yes_dir / "fig_assigned_errors.png", "6v3 assigned errors (smooth analytic term)")
    plot_d_min_history(eval_no, dt, no_dir / "fig_d_min.png", "6v3 d_min (no analytic term)", d_min_threshold=float(args.d_safe))
    plot_d_min_history(eval_yes, dt, yes_dir / "fig_d_min.png", "6v3 d_min (smooth analytic term)", d_min_threshold=float(args.d_safe))

    t = np.arange(eval_no.d_min_history.shape[0]) * dt
    _plot_curve_compare(
        t,
        eval_no.d_min_history,
        eval_yes.d_min_history,
        ylabel="Min inter-pursuer distance (m)",
        title="6v3 crossing: d_min comparison",
        path=out_dir / "fig_dmin_compare.png",
        threshold=float(args.d_safe),
    )
    _plot_curve_compare(
        t,
        np.mean(eval_no.assigned_errors, axis=1),
        np.mean(eval_yes.assigned_errors, axis=1),
        ylabel="Mean assigned tracking error",
        title="6v3 crossing: mean assigned tracking error",
        path=out_dir / "fig_mean_error_compare.png",
        threshold=None,
    )
    _plot_xy_compare(
        eval_no,
        eval_yes,
        out_dir / "fig_xy_compare.png",
        "6v3 many-to-many crossing geometry",
    )

    no_metrics = {
        **_unsafe_metrics(eval_no.d_min_history, float(args.d_safe), dt),
        "capture_time_s": _sustained_capture_time(eval_no.assigned_errors, scenario.capture_radius, dt),
        "first_hit_time_s": None if eval_no.capture_time is None else float(eval_no.capture_time),
        "mean_assigned_error": float(np.mean(eval_no.assigned_errors)),
        "train_wall_time_s": float(wall_no),
    }
    yes_metrics = {
        **_unsafe_metrics(eval_yes.d_min_history, float(args.d_safe), dt),
        "capture_time_s": _sustained_capture_time(eval_yes.assigned_errors, scenario.capture_radius, dt),
        "first_hit_time_s": None if eval_yes.capture_time is None else float(eval_yes.capture_time),
        "mean_assigned_error": float(np.mean(eval_yes.assigned_errors)),
        "train_wall_time_s": float(wall_yes),
    }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": {
            "name": scenario.name,
            "n_pursuers": int(scenario.n_pursuers),
            "n_evaders": int(scenario.n_evaders),
            "initial_assignment": scenario.initial_assignment.tolist(),
            "t_final_s": float(scenario.t_final),
            "capture_radius_m": float(scenario.capture_radius),
            "description": "Each evader has two pursuers that start on opposite sides of their desired left/right offsets, forcing a lane-swap crossing near the evader.",
        },
        "gamma": float(args.gamma),
        "d_safe_m": float(args.d_safe),
        "no_analytic": no_metrics,
        "smooth_analytic": yes_metrics,
    }
    dump_json(out_dir / "summary.json", summary)
    dump_markdown(out_dir / "REPORT.md", _build_report(summary))
    print(f"6v3 many-to-many collision demo saved to: {out_dir}")


if __name__ == "__main__":
    main()
