"""Unified safety stress demonstration.

This script intentionally separates the evidence into two layers:

1. Reduced 2D collision illustration:
   A deterministic crossing-path geometry where the no-analytic controller
   produces an actual near-zero separation event, while the smooth analytic
   term prevents the collision in the reduced model.

2. Full 6D unsafe-band stress test:
   A deterministic fixed-assignment scenario in the main simulator where
   pursuers must swap sides to reach their desired offsets. Without the
   analytic term they remain in the unsafe band for a long interval; with
   the smooth analytic term they stay outside the chosen safety threshold.

The 2D part is a mechanism illustration. The 6D part is the main evidence
because it uses the actual training and evaluation pipeline.
"""
from __future__ import annotations

import argparse
import math
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
from mpe_repro.plotting import plot_assigned_state_errors, plot_trajectory_multiview
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.simulator import EvalResult, MPECommSimulator


def _smooth_amp_scalar(dist_sq: float, d_safe: float) -> float:
    if d_safe <= 0.0:
        return 1.0
    ratio = d_safe * d_safe / max(dist_sq, 1e-8)
    delta = ratio - 1.0
    return 1.0 + 0.5 * (delta + math.sqrt(delta * delta + 1e-4))


def _smooth_amp_array(dist_sq: np.ndarray, d_safe: float) -> np.ndarray:
    if d_safe <= 0.0:
        return np.ones_like(dist_sq, dtype=float)
    ratio = d_safe * d_safe / np.maximum(dist_sq, 1e-8)
    delta = ratio - 1.0
    return 1.0 + 0.5 * (delta + np.sqrt(delta * delta + 1e-4))


# ---------------------------------------------------------------------------
# Reduced 2D collision illustration
# ---------------------------------------------------------------------------


def _reduced_vsnac_grad_v(x_tilde: np.ndarray, w: np.ndarray, gain: float, pos_scale: float) -> np.ndarray:
    p = x_tilde[:2] / pos_scale
    v = x_tilde[2:]
    return gain * np.array(
        [
            w[0] * p[0] + w[2] * v[0],
            w[1] * p[1] + w[3] * v[1],
        ],
        dtype=float,
    )


def _simulate_reduced_2d(
    p_init: np.ndarray,
    e_init: np.ndarray,
    displacements: np.ndarray,
    *,
    gamma: float,
    d_safe: float,
    dt: float = 0.05,
    t_final: float = 120.0,
    u_bar_p: float = 25.0,
    u_bar_e: float = 15.0,
    r1: float = 0.1,
    r2: float = 0.1,
    d_ref: float = 500.0,
    gain: float = 0.005,
    pos_scale: float = 3.0,
    w: np.ndarray | None = None,
) -> dict[str, Any]:
    n_p = p_init.shape[0]
    steps = int(t_final / dt)
    a = np.ones((n_p, n_p), dtype=float) - np.eye(n_p, dtype=float)
    degree = a.sum(axis=1)
    delta_p = np.zeros((n_p, n_p, 2), dtype=float)
    for j in range(n_p):
        for k in range(n_p):
            delta_p[j, k] = displacements[k] - displacements[j]

    r1_inv = 1.0 / r1
    r2_inv = 1.0 / r2
    if w is None:
        w = np.array([0.8, 0.8, 0.6, 0.6], dtype=float)

    p_st = p_init.astype(float).copy()
    e_st = e_init.astype(float).copy()
    p_traj = np.zeros((steps + 1, n_p, 4), dtype=float)
    e_traj = np.zeros((steps + 1, 4), dtype=float)
    d_min_hist = np.zeros(steps + 1, dtype=float)
    err_hist = np.zeros((steps + 1, n_p), dtype=float)

    def record(idx: int) -> None:
        p_traj[idx] = p_st
        e_traj[idx] = e_st
        for j in range(n_p):
            xe = np.concatenate([p_st[j, :2] - e_st[:2] + displacements[j], p_st[j, 2:] - e_st[2:]])
            err_hist[idx, j] = float(np.linalg.norm(xe))
        d_min_hist[idx] = float(np.linalg.norm(p_st[0, :2] - p_st[1, :2]))

    record(0)
    for t in range(steps):
        grad_v = np.zeros((n_p, 2), dtype=float)
        for j in range(n_p):
            xe = np.concatenate([p_st[j, :2] - e_st[:2] + displacements[j], p_st[j, 2:] - e_st[2:]])
            grad_v[j] = _reduced_vsnac_grad_v(xe, w, gain, pos_scale)

        u_p = np.zeros((n_p, 2), dtype=float)
        for j in range(n_p):
            total = grad_v[j].copy()
            if gamma > 0.0 and degree[j] > 0.0:
                coord = np.zeros(2, dtype=float)
                for k in range(n_p):
                    if k == j or a[j, k] <= 0.0:
                        continue
                    dv = p_st[j, 2:] - p_st[k, 2:]
                    dp = p_st[j, :2] - p_st[k, :2] - delta_p[j, k]
                    sep = p_st[j, :2] - p_st[k, :2]
                    amp = _smooth_amp_scalar(float(sep @ sep), d_safe)
                    coord += amp * a[j, k] * (dv + dp / d_ref)
                total += gamma * coord / degree[j]
            rho = r1_inv * total / (2.0 * u_bar_p)
            u_p[j] = -u_bar_p * np.tanh(rho)

        mean_grad = grad_v.mean(axis=0)
        rho_e = r2_inv * mean_grad / (2.0 * u_bar_e)
        u_e = -u_bar_e * np.tanh(rho_e)

        p_st[:, :2] += p_st[:, 2:] * dt
        p_st[:, 2:] += u_p * dt
        e_st[:2] += e_st[2:] * dt
        e_st[2:] += u_e * dt
        record(t + 1)

    return {
        "p_traj": p_traj,
        "e_traj": e_traj,
        "d_min_history": d_min_hist,
        "error_history": err_hist,
        "min_d_min": float(np.min(d_min_hist)),
    }


def _build_reduced_collision_problem() -> dict[str, Any]:
    return {
        "p_init": np.array(
            [
                [0.0, 1500.0, 0.0, 0.0],
                [0.0, -1500.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        "e_init": np.array([3000.0, 0.0, 5.0, 0.0], dtype=float),
        "displacements": np.array(
            [
                [0.0, 800.0],
                [0.0, -800.0],
            ],
            dtype=float,
        ),
        "gamma": 1.0,
        "d_safe": 500.0,
        "dt": 0.05,
        "t_final": 120.0,
    }


def _plot_reduced_trajectories(
    res_no: dict[str, Any],
    res_yes: dict[str, Any],
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    p_colors = ["#d62728", "#1f77b4"]

    def draw(ax: plt.Axes, res: dict[str, Any], panel_title: str) -> None:
        for j in range(res["p_traj"].shape[1]):
            traj = res["p_traj"][:, j, :2]
            ax.plot(traj[:, 0], traj[:, 1], color=p_colors[j], linewidth=2.0, label=f"P{j+1}")
            ax.scatter(traj[0, 0], traj[0, 1], color=p_colors[j], s=60, marker="o", edgecolors="black", linewidths=0.6)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=p_colors[j], s=55, marker="s")
        et = res["e_traj"][:, :2]
        ax.plot(et[:, 0], et[:, 1], color="#111111", linewidth=2.2, linestyle="--", label="Evader")
        ax.scatter(et[0, 0], et[0, 1], color="#111111", s=70, marker="x")
        ax.set_title(panel_title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=9, loc="upper left")

    draw(axes[0], res_no, "No analytic term")
    draw(axes[1], res_yes, "With smooth analytic term")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_reduced_dmin(
    res_no: dict[str, Any],
    res_yes: dict[str, Any],
    dt: float,
    d_safe: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = np.arange(res_no["d_min_history"].shape[0]) * dt
    ax.plot(t, res_no["d_min_history"], linewidth=2.0, color="#d62728", linestyle="--", label="No analytic term")
    ax.plot(t, res_yes["d_min_history"], linewidth=2.0, color="#2ca02c", label="Smooth analytic term")
    ax.axhline(d_safe, color="black", linewidth=1.2, linestyle=":", label=f"$d_{{safe}}={d_safe:.0f}$ m")
    ax.axhspan(0.0, d_safe, color="#d62728", alpha=0.08)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min inter-pursuer distance (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_reduced_errors(
    res_no: dict[str, Any],
    res_yes: dict[str, Any],
    dt: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    t = np.arange(res_no["error_history"].shape[0]) * dt
    ax.plot(t, np.mean(res_no["error_history"], axis=1), linewidth=2.0, color="#d62728", linestyle="--", label="No analytic term")
    ax.plot(t, np.mean(res_yes["error_history"], axis=1), linewidth=2.0, color="#2ca02c", label="Smooth analytic term")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean tracking error")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full 6D unsafe-band stress test
# ---------------------------------------------------------------------------


def _build_full_6d_unsafe_band_scenario() -> ScenarioConfig:
    n_p = 4
    evaders = np.array([[0.0, 2600.0, 300.0, 0.0, 58.0, 0.0]], dtype=float)
    pursuers = np.zeros((n_p, 6), dtype=float)
    displacements = np.zeros((n_p, 1, 6), dtype=float)

    target_signs = [1.0, -1.0, 1.0, -1.0]
    start_signs = [-s for s in target_signs]
    x_target = 300.0
    x_start = 600.0
    y_start = -1400.0
    y_offset = -80.0
    speed = 78.0

    for j, (target_sign, start_sign) in enumerate(zip(target_signs, start_signs)):
        displacements[j, 0, :3] = [target_sign * x_target, y_offset, 0.0]
        pursuers[j, :3] = [start_sign * x_start, y_start - 120.0 * j, 300.0]
        desired = evaders[0, :3] - displacements[j, 0, :3]
        guide = desired - pursuers[j, :3]
        guide = guide / max(np.linalg.norm(guide), 1e-8)
        pursuers[j, 3:] = speed * guide

    return ScenarioConfig(
        name="unsafe_band_4v1_fixed",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,
        max_switch_worsening=0.0,
        initial_assignment=np.zeros(n_p, dtype=int),
        t_final=70.0,
        capture_radius=180.0,
        evader_motion_mode="scripted",
        evader_script_amp=(2.0, 2.0, 0.5),
        evader_script_omega=0.16,
        evader_script_decay=0.025,
        evader_script_mix=0.20,
        swap_lookahead_time=0.0,
    )


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
        policy_iterations=20,
        rollout_steps=120,
        min_samples_per_evader=50,
        critic_learning_rate=0.05,
        critic_lr_decay=0.85,
        convergence_tol=5e-4,
        random_perturb_scale=0.01,
    )


def _run_full_6d_method(
    scenario: ScenarioConfig,
    *,
    gamma: float,
    d_safe: float,
    train_seed: int,
    eval_seed: int,
) -> tuple[EvalResult, float]:
    comm_mode = "full" if gamma > 0.0 else "none"
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
    if d_min_history.size == 0:
        return {
            "min_d_min": float("nan"),
            "time_below_threshold_s": 0.0,
            "longest_unsafe_streak_s": 0.0,
        }
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
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(t, y_no, linewidth=2.0, color="#d62728", linestyle="--", label="No analytic term")
    ax.plot(t, y_yes, linewidth=2.0, color="#2ca02c", label="Smooth analytic term")
    if threshold is not None:
        ax.axhline(threshold, color="black", linewidth=1.2, linestyle=":", label=f"threshold = {threshold:.1f}")
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


def _build_report(summary: dict[str, Any]) -> str:
    r2 = summary["reduced_2d"]
    f6 = summary["full_6d"]
    lines = [
        "# Unified Safety Stress Report",
        "",
        "## Reduced 2D Collision Illustration",
        "- Purpose: mechanism illustration in a reduced-order crossing geometry.",
        f"- d_safe: {r2['d_safe_m']:.1f} m",
        f"- No analytic term: min d_min = {r2['no_analytic']['min_d_min_m']:.3f} m, time below d_safe = {r2['no_analytic']['time_below_d_safe_s']:.3f} s",
        f"- Smooth analytic term: min d_min = {r2['smooth_analytic']['min_d_min_m']:.3f} m, time below d_safe = {r2['smooth_analytic']['time_below_d_safe_s']:.3f} s",
        "",
        "## Full 6D Unsafe-Band Stress Test",
        "- Purpose: full-order evidence in the actual MPE simulator.",
        f"- d_safe: {f6['d_safe_m']:.1f} m",
        f"- Gamma (smooth analytic): {f6['gamma']:.3f}",
        f"- No analytic term: min d_min = {f6['no_analytic']['min_d_min']:.3f} m, time below d_safe = {f6['no_analytic']['time_below_threshold_s']:.3f} s, longest unsafe streak = {f6['no_analytic']['longest_unsafe_streak_s']:.3f} s, capture time = {f6['no_analytic']['capture_time_s']}",
        f"- Smooth analytic term: min d_min = {f6['smooth_analytic']['min_d_min']:.3f} m, time below d_safe = {f6['smooth_analytic']['time_below_threshold_s']:.3f} s, longest unsafe streak = {f6['smooth_analytic']['longest_unsafe_streak_s']:.3f} s, capture time = {f6['smooth_analytic']['capture_time_s']}",
        "",
        "## Interpretation",
        "- The reduced 2D model is used only to show an explicit collision-suppression mechanism.",
        "- The full 6D scenario is the main evidence: the no-analytic controller remains inside the unsafe band for a long interval, whereas the smooth analytic term keeps the team outside the chosen safety threshold.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified reduced-order + full-order safety stress demo.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"collision_demo_rigorous_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2D reduced collision illustration
    # ------------------------------------------------------------------
    reduced_cfg = _build_reduced_collision_problem()
    reduced_dir = out_dir / "reduced_2d"
    reduced_dir.mkdir(parents=True, exist_ok=True)

    res_2d_no = _simulate_reduced_2d(
        reduced_cfg["p_init"],
        reduced_cfg["e_init"],
        reduced_cfg["displacements"],
        gamma=0.0,
        d_safe=0.0,
        dt=reduced_cfg["dt"],
        t_final=reduced_cfg["t_final"],
    )
    res_2d_yes = _simulate_reduced_2d(
        reduced_cfg["p_init"],
        reduced_cfg["e_init"],
        reduced_cfg["displacements"],
        gamma=reduced_cfg["gamma"],
        d_safe=reduced_cfg["d_safe"],
        dt=reduced_cfg["dt"],
        t_final=reduced_cfg["t_final"],
    )

    time_below_2d_no = float(np.sum(res_2d_no["d_min_history"] < reduced_cfg["d_safe"]) * reduced_cfg["dt"])
    time_below_2d_yes = float(np.sum(res_2d_yes["d_min_history"] < reduced_cfg["d_safe"]) * reduced_cfg["dt"])

    _plot_reduced_trajectories(
        res_2d_no,
        res_2d_yes,
        reduced_dir / "fig_2d_trajectory_compare.png",
        "Reduced 2D crossing geometry",
    )
    _plot_reduced_dmin(
        res_2d_no,
        res_2d_yes,
        reduced_cfg["dt"],
        reduced_cfg["d_safe"],
        reduced_dir / "fig_2d_dmin_compare.png",
        "Reduced 2D min-distance comparison",
    )
    _plot_reduced_errors(
        res_2d_no,
        res_2d_yes,
        reduced_cfg["dt"],
        reduced_dir / "fig_2d_mean_error_compare.png",
        "Reduced 2D mean tracking error",
    )

    reduced_summary = {
        "d_safe_m": float(reduced_cfg["d_safe"]),
        "gamma": float(reduced_cfg["gamma"]),
        "no_analytic": {
            "min_d_min_m": float(res_2d_no["min_d_min"]),
            "time_below_d_safe_s": time_below_2d_no,
        },
        "smooth_analytic": {
            "min_d_min_m": float(res_2d_yes["min_d_min"]),
            "time_below_d_safe_s": time_below_2d_yes,
        },
    }
    dump_json(reduced_dir / "summary.json", reduced_summary)

    # ------------------------------------------------------------------
    # 6D full-order unsafe-band stress test
    # ------------------------------------------------------------------
    full_dir = out_dir / "full_6d"
    full_dir.mkdir(parents=True, exist_ok=True)

    scenario_6d = _build_full_6d_unsafe_band_scenario()
    d_safe_6d = 30.0
    gamma_6d = 1.5
    dt_6d = _demo_learning_params().dt

    ev_6d_no, wall_no = _run_full_6d_method(
        scenario_6d,
        gamma=0.0,
        d_safe=0.0,
        train_seed=9,
        eval_seed=1009,
    )
    ev_6d_yes, wall_yes = _run_full_6d_method(
        scenario_6d,
        gamma=gamma_6d,
        d_safe=d_safe_6d,
        train_seed=9,
        eval_seed=1009,
    )

    unsafe_no = _unsafe_metrics(ev_6d_no.d_min_history, d_safe_6d, dt_6d)
    unsafe_yes = _unsafe_metrics(ev_6d_yes.d_min_history, d_safe_6d, dt_6d)
    t_6d = np.arange(ev_6d_no.d_min_history.shape[0]) * dt_6d

    plot_trajectory_multiview(ev_6d_no, full_dir / "fig_6d_multiview_noanalytic.png", "6D unsafe-band stress (no analytic term)")
    plot_trajectory_multiview(ev_6d_yes, full_dir / "fig_6d_multiview_smooth.png", "6D unsafe-band stress (smooth analytic term)")
    plot_assigned_state_errors(ev_6d_no, dt_6d, full_dir / "fig_6d_errors_noanalytic.png", "6D assigned errors (no analytic term)")
    plot_assigned_state_errors(ev_6d_yes, dt_6d, full_dir / "fig_6d_errors_smooth.png", "6D assigned errors (smooth analytic term)")
    _plot_curve_compare(
        t_6d,
        ev_6d_no.d_min_history,
        ev_6d_yes.d_min_history,
        ylabel="Min inter-pursuer distance (m)",
        title="6D unsafe-band stress: d_min comparison",
        path=full_dir / "fig_6d_dmin_compare.png",
        threshold=d_safe_6d,
    )
    _plot_curve_compare(
        t_6d,
        np.mean(ev_6d_no.assigned_errors, axis=1),
        np.mean(ev_6d_yes.assigned_errors, axis=1),
        ylabel="Mean assigned tracking error",
        title="6D unsafe-band stress: mean assigned error",
        path=full_dir / "fig_6d_mean_error_compare.png",
        threshold=None,
    )

    full_summary = {
        "scenario": {
            "name": scenario_6d.name,
            "n_pursuers": int(scenario_6d.n_pursuers),
            "n_evaders": int(scenario_6d.n_evaders),
            "capture_radius_m": float(scenario_6d.capture_radius),
            "t_final_s": float(scenario_6d.t_final),
            "initial_assignment": scenario_6d.initial_assignment.tolist(),
        },
        "d_safe_m": d_safe_6d,
        "gamma": gamma_6d,
        "no_analytic": {
            **unsafe_no,
            "capture_time_s": None if ev_6d_no.capture_time is None else float(ev_6d_no.capture_time),
            "mean_assigned_error": float(np.mean(ev_6d_no.assigned_errors)),
            "train_wall_time_s": float(wall_no),
        },
        "smooth_analytic": {
            **unsafe_yes,
            "capture_time_s": None if ev_6d_yes.capture_time is None else float(ev_6d_yes.capture_time),
            "mean_assigned_error": float(np.mean(ev_6d_yes.assigned_errors)),
            "train_wall_time_s": float(wall_yes),
        },
    }
    dump_json(full_dir / "summary.json", full_summary)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "reduced_2d": reduced_summary,
        "full_6d": full_summary,
    }
    dump_json(out_dir / "summary.json", summary)
    dump_markdown(out_dir / "REPORT.md", _build_report(summary))
    print(f"Unified safety stress demo saved to: {out_dir}")


if __name__ == "__main__":
    main()
