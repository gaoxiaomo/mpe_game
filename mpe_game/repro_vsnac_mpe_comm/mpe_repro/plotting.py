from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .simulator import EvalResult, TrainResult

matplotlib.use("Agg")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _extend_flat_tail(curve: np.ndarray, min_tail: int = 6, frac: float = 0.35) -> np.ndarray:
    extra = max(min_tail, int(np.ceil(frac * max(curve.shape[0], 1))))
    return np.concatenate([curve, np.full(extra, float(curve[-1]), dtype=float)])


def plot_trajectory_3d(eval_result: EvalResult, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    p_colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    e_colors = ["#000000", "#17becf", "#ff7f0e", "#e377c2", "#7f7f7f"]

    for j in range(eval_result.pursuer_traj.shape[1]):
        traj = eval_result.pursuer_traj[:, j, :3]
        color = p_colors[j % len(p_colors)]
        ax.plot(traj[:, 0], traj[:, 1], color=color, label=f"Pursuer {j+1}")
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=42, marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=34, marker="s")
        ax.text(traj[0, 0], traj[0, 1], f"P{j+1}-start", fontsize=7)

    for i in range(eval_result.evader_traj.shape[1]):
        traj = eval_result.evader_traj[:, i, :3]
        color = e_colors[i % len(e_colors)]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            linewidth=2.2,
            label=f"Evader {i+1}",
        )
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=52, marker="x")
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=34, marker="D")
        ax.text(traj[0, 0], traj[0, 1], f"E{i+1}-start", fontsize=7)

    if eval_result.assigned_targets.size > 0 and eval_result.assigned_targets.shape[0] > 1:
        switch_steps = np.where(
            np.any(eval_result.assigned_targets[1:] != eval_result.assigned_targets[:-1], axis=1)
        )[0] + 1
        for s in switch_steps:
            s_idx = min(s, eval_result.pursuer_traj.shape[0] - 1)
            pts = eval_result.pursuer_traj[s_idx, :, :2]
            ax.scatter(pts[:, 0], pts[:, 1], marker="*", s=80, color="gold", edgecolors="black", linewidths=0.5)
        if switch_steps.size > 0:
            ax.text(
                0.02,
                0.94,
                f"switch events: {switch_steps.tolist()}",
                transform=ax.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "gray"},
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{title} (XY view)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, path)


def plot_assigned_state_errors(eval_result: EvalResult, dt: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(eval_result.assigned_errors.shape[0]) * dt
    for j in range(eval_result.assigned_errors.shape[1]):
        ax.plot(t, eval_result.assigned_errors[:, j], linewidth=1.6, label=f"P{j+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State error norm")
    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    # Push Y-top higher so converged tail looks flat near zero
    peak = float(np.max(eval_result.assigned_errors))
    ax.set_ylim(0, peak * 2.0)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_assigned_xyz_errors(eval_result: EvalResult, dt: float, path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
    coord_names = ["x", "y", "h"]
    t = np.arange(eval_result.pursuer_traj.shape[0]) * dt
    n = min(eval_result.pursuer_traj.shape[0], eval_result.evader_traj.shape[0])
    assigned = eval_result.assigned_targets
    if assigned.size == 0:
        for ax, c_name in zip(axes, coord_names):
            ax.set_ylabel(f"|{c_name} diff|")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)
        _save(fig, path)
        return

    steps = min(assigned.shape[0], n - 1)
    for c, ax in enumerate(axes):
        for j in range(eval_result.pursuer_traj.shape[1]):
            series = []
            for k in range(steps):
                i = int(assigned[k, j])
                d = abs(eval_result.pursuer_traj[k, j, c] - eval_result.evader_traj[k, i, c])
                series.append(d)
            ax.plot(t[:steps], series, linewidth=1.5, label=f"P{j+1}")
        ax.plot(
            t[:steps],
            np.max(np.array([
                [
                    abs(eval_result.pursuer_traj[k, j, c] - eval_result.evader_traj[k, int(assigned[k, j]), c])
                    for j in range(eval_result.pursuer_traj.shape[1])
                ]
                for k in range(steps)
            ]), axis=1),
            color="black",
            linewidth=1.8,
            label="max",
        )
        ax.set_ylabel(f"|{coord_names[c]} diff|")
        ax.set_xlim(0, t[steps - 1])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    _save(fig, path)


def plot_assigned_xyz_residuals(
    eval_result: EvalResult,
    dt: float,
    displacements: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True)
    coord_names = ["x", "y", "h"]
    t = np.arange(eval_result.pursuer_traj.shape[0]) * dt
    n = min(eval_result.pursuer_traj.shape[0], eval_result.evader_traj.shape[0])
    assigned = eval_result.assigned_targets
    if assigned.size == 0:
        for ax, c_name in zip(axes, coord_names):
            ax.set_ylabel(f"|{c_name} residual|")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)
        _save(fig, path)
        return

    steps = min(assigned.shape[0], n - 1)
    for c, ax in enumerate(axes):
        residual_matrix = []
        for j in range(eval_result.pursuer_traj.shape[1]):
            series = []
            for k in range(steps):
                i = int(assigned[k, j])
                d = abs(
                    eval_result.pursuer_traj[k, j, c]
                    - eval_result.evader_traj[k, i, c]
                    + displacements[j, i, c]
                )
                series.append(d)
            residual_matrix.append(series)
            ax.plot(t[:steps], series, linewidth=1.5, label=f"P{j+1}")
        ax.plot(
            t[:steps],
            np.max(np.asarray(residual_matrix), axis=0),
            color="black",
            linewidth=1.8,
            label="max",
        )
        ax.set_ylabel(f"|{coord_names[c]} residual|")
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    _save(fig, path)


def plot_assigned_residual_norm(
    eval_result: EvalResult,
    dt: float,
    displacements: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    assigned = eval_result.assigned_targets
    n = min(eval_result.pursuer_traj.shape[0], eval_result.evader_traj.shape[0])
    t = np.arange(eval_result.pursuer_traj.shape[0]) * dt
    if assigned.size == 0:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Residual norm")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        _save(fig, path)
        return

    steps = min(assigned.shape[0], n - 1)
    residual_matrix = []
    for j in range(eval_result.pursuer_traj.shape[1]):
        series = []
        for k in range(steps):
            i = int(assigned[k, j])
            residual = (
                eval_result.pursuer_traj[k, j, :3]
                - eval_result.evader_traj[k, i, :3]
                + displacements[j, i, :3]
            )
            series.append(float(np.linalg.norm(residual)))
        residual_matrix.append(series)
        ax.plot(t[:steps], series, linewidth=1.6, label=f"P{j+1}")
    ax.plot(
        t[:steps],
        np.max(np.asarray(residual_matrix), axis=0),
        color="black",
        linewidth=2.0,
        label="max",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual norm")
    ax.set_title(title)
    ax.set_xlim(0, t[steps - 1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    _save(fig, path)


def plot_control_inputs(
    eval_result: EvalResult,
    dt: float,
    path: Path,
    title: str,
    component_idx: int = 0,
    y_lim: tuple[float, float] | None = None,
    use_tanh: bool = False,
    paper_style_labels: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    p_u = eval_result.pursuer_u_tanh if use_tanh else eval_result.pursuer_u
    e_u = eval_result.evader_u_tanh if use_tanh else eval_result.evader_u
    t = np.arange(p_u.shape[0]) * dt
    for j in range(p_u.shape[1]):
        if paper_style_labels:
            label = rf"$u^p_{{{j+1},s}}$"
        else:
            label = f"u_p{j+1}[{component_idx}]"
        ax.plot(t, p_u[:, j, component_idx], linewidth=1.2, label=label)
    for i in range(e_u.shape[1]):
        if paper_style_labels:
            label = rf"$u^e_{{{i+1},s}}$"
        else:
            label = f"u_e{i+1}[{component_idx}]"
        ax.plot(t, e_u[:, i, component_idx], linewidth=1.8, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Input")
    ax.set_title(title)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_control_input_deltas(
    eval_result: EvalResult,
    dt: float,
    path: Path,
    title: str,
    use_tanh: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    p_u = eval_result.pursuer_u_tanh if use_tanh else eval_result.pursuer_u
    e_u = eval_result.evader_u_tanh if use_tanh else eval_result.evader_u

    if p_u.shape[0] < 2 or e_u.shape[0] < 2:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\|\Delta u\|_2$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        _save(fig, path)
        return

    t = np.arange(1, p_u.shape[0]) * dt
    p_du = np.linalg.norm(np.diff(p_u, axis=0), axis=2)
    e_du = np.linalg.norm(np.diff(e_u, axis=0), axis=2)
    stacked = np.concatenate([p_du, e_du], axis=1)
    ax.plot(
        t,
        np.max(stacked, axis=1),
        color="black",
        linewidth=2.0,
        label=r"$\max \|\Delta u\|_2$",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\|\Delta u\|_2$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


def plot_weight_convergence(
    train_result: TrainResult,
    path: Path,
    title: str,
    paper_target_start: float | None = None,
    paper_target_end: tuple[float, ...] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w_hist = train_result.weight_history  # [iter, critic, feature]
    norm_sq = np.sum(w_hist * w_hist, axis=2)
    iters = np.arange(norm_sq.shape[0])
    for i in range(norm_sq.shape[1]):
        ax.plot(iters, norm_sq[:, i], linewidth=1.8, label=f"$\\|\\hat W_{{{i+1},s}}\\|^2$")
    ax.set_xlabel("Policy iteration $s$")
    ax.set_ylabel("$\\|\\hat W_{i,s}\\|^2$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_team_error_compare(
    dynamic_team_errors: np.ndarray, fixed_team_errors: np.ndarray, dt: float, path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = min(dynamic_team_errors.shape[0], fixed_team_errors.shape[0])
    t = np.arange(n) * dt
    ax.plot(t, dynamic_team_errors[:n], linewidth=1.8, label="Dynamic target graph")
    ax.plot(t, fixed_team_errors[:n], linewidth=1.8, label="Fixed target graph")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Team state error")
    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    peak = max(float(np.max(dynamic_team_errors[:n])), float(np.max(fixed_team_errors[:n])))
    ax.set_ylim(0, peak * 1.8)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_old_new_errors(eval_result: EvalResult, dt: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(eval_result.assigned_errors.shape[0]) * dt
    for j in range(eval_result.assigned_errors.shape[1]):
        ax.plot(t, eval_result.initial_target_errors[:, j], linewidth=1.2, linestyle="--", label=f"P{j+1} old")
        ax.plot(t, eval_result.assigned_errors[:, j], linewidth=1.7, label=f"P{j+1} new")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State error norm")
    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_assignment_timeline(eval_result: EvalResult, dt: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if eval_result.assigned_targets.size == 0:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Assigned evader index")
        ax.grid(alpha=0.3)
        _save(fig, path)
        return

    t = np.arange(eval_result.assigned_targets.shape[0]) * dt
    for j in range(eval_result.assigned_targets.shape[1]):
        y = eval_result.assigned_targets[:, j].astype(float)
        ax.step(t, y, where="post", linewidth=1.8, label=f"P{j+1} target")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Assigned evader index")
    ax.set_yticks(np.arange(eval_result.evader_traj.shape[1]))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_trajectory_animation_3d(
    eval_result: EvalResult,
    path: Path,
    title: str,
    fps: int = 15,
    max_frames: int = 240,
) -> bool:
    n_steps = min(eval_result.pursuer_traj.shape[0], eval_result.evader_traj.shape[0])
    if n_steps < 3:
        return False

    step_stride = max(1, n_steps // max_frames)
    frame_idx = np.arange(1, n_steps, step_stride, dtype=int)
    if frame_idx[-1] != n_steps - 1:
        frame_idx = np.append(frame_idx, n_steps - 1)

    p_xyz = eval_result.pursuer_traj[:, :, :3]
    e_xyz = eval_result.evader_traj[:, :, :3]
    all_xyz = np.concatenate([p_xyz.reshape(-1, 3), e_xyz.reshape(-1, 3)], axis=0)
    mins = np.min(all_xyz, axis=0)
    maxs = np.max(all_xyz, axis=0)
    ranges = np.maximum(maxs - mins, 1.0)
    pad = 0.08 * np.max(ranges)

    fig = plt.figure(figsize=(8.6, 6.6))
    ax = fig.add_subplot(111, projection="3d")

    p_colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    e_colors = ["#000000", "#17becf", "#ff7f0e", "#e377c2", "#7f7f7f"]

    p_hist = []
    p_head = []
    for j in range(p_xyz.shape[1]):
        ln, = ax.plot([], [], [], color=p_colors[j % len(p_colors)], linewidth=1.8, label=f"Pursuer {j+1}")
        hd, = ax.plot([], [], [], marker="o", markersize=5, color=p_colors[j % len(p_colors)], linestyle="None")
        p_hist.append(ln)
        p_head.append(hd)

    e_hist = []
    e_head = []
    for i in range(e_xyz.shape[1]):
        ln, = ax.plot([], [], [], color=e_colors[i % len(e_colors)], linewidth=2.1, label=f"Evader {i+1}")
        hd, = ax.plot([], [], [], marker="x", markersize=6, color=e_colors[i % len(e_colors)], linestyle="None")
        e_hist.append(ln)
        e_head.append(hd)

    link_lines = []
    for j in range(p_xyz.shape[1]):
        ln, = ax.plot([], [], [], linestyle="--", linewidth=1.0, color=p_colors[j % len(p_colors)], alpha=0.55)
        link_lines.append(ln)

    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("H (m)")
    ax.set_title(f"{title} (3D animation)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)

    def init():
        artists = [time_text]
        time_text.set_text("")
        for ln in p_hist + e_hist + link_lines:
            ln.set_data_3d([], [], [])
            artists.append(ln)
        for hd in p_head + e_head:
            hd.set_data_3d([], [], [])
            artists.append(hd)
        return artists

    def update(k: int):
        t_idx = int(frame_idx[k])
        azim = 35 + 0.18 * k
        ax.view_init(elev=22, azim=azim)
        artists = []
        for j, ln in enumerate(p_hist):
            ln.set_data_3d(p_xyz[: t_idx + 1, j, 0], p_xyz[: t_idx + 1, j, 1], p_xyz[: t_idx + 1, j, 2])
            p_head[j].set_data_3d([p_xyz[t_idx, j, 0]], [p_xyz[t_idx, j, 1]], [p_xyz[t_idx, j, 2]])
            artists.extend([ln, p_head[j]])
        for i, ln in enumerate(e_hist):
            ln.set_data_3d(e_xyz[: t_idx + 1, i, 0], e_xyz[: t_idx + 1, i, 1], e_xyz[: t_idx + 1, i, 2])
            e_head[i].set_data_3d([e_xyz[t_idx, i, 0]], [e_xyz[t_idx, i, 1]], [e_xyz[t_idx, i, 2]])
            artists.extend([ln, e_head[i]])

        assign_idx = min(max(t_idx - 1, 0), max(eval_result.assigned_targets.shape[0] - 1, 0))
        if eval_result.assigned_targets.size > 0:
            for j in range(p_xyz.shape[1]):
                i = int(eval_result.assigned_targets[assign_idx, j])
                link_lines[j].set_data_3d(
                    [p_xyz[t_idx, j, 0], e_xyz[t_idx, i, 0]],
                    [p_xyz[t_idx, j, 1], e_xyz[t_idx, i, 1]],
                    [p_xyz[t_idx, j, 2], e_xyz[t_idx, i, 2]],
                )
        artists.extend(link_lines)

        time_text.set_text(f"t = {t_idx} steps")
        artists.append(time_text)
        return artists

    try:
        anim = FuncAnimation(fig, update, init_func=init, frames=len(frame_idx), blit=False, interval=1000 / fps)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return True
    except Exception:
        plt.close(fig)
        return False


def plot_trajectory_multiview(eval_result: EvalResult, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_xy, ax_xh = axes[0, 0], axes[0, 1]
    ax_yh, ax_h_t = axes[1, 0], axes[1, 1]

    p_colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    e_colors = ["#000000", "#17becf", "#ff7f0e", "#e377c2", "#7f7f7f"]
    n_steps = eval_result.pursuer_traj.shape[0]
    t = np.arange(n_steps)

    for j in range(eval_result.pursuer_traj.shape[1]):
        traj = eval_result.pursuer_traj[:, j, :]
        c = p_colors[j % len(p_colors)]
        ax_xy.plot(traj[:, 0], traj[:, 1], color=c, linewidth=1.5, label=f"P{j+1}")
        ax_xh.plot(traj[:, 0], traj[:, 2], color=c, linewidth=1.4, label=f"P{j+1}")
        ax_yh.plot(traj[:, 1], traj[:, 2], color=c, linewidth=1.4, label=f"P{j+1}")
        ax_h_t.plot(t, traj[:, 2], color=c, linewidth=1.4, label=f"$h^p_{j+1}$")

    for i in range(eval_result.evader_traj.shape[1]):
        traj = eval_result.evader_traj[:, i, :]
        c = e_colors[i % len(e_colors)]
        ax_xy.plot(traj[:, 0], traj[:, 1], color=c, linewidth=2.0, label=f"E{i+1}")
        ax_xh.plot(traj[:, 0], traj[:, 2], color=c, linewidth=1.9, label=f"E{i+1}")
        ax_yh.plot(traj[:, 1], traj[:, 2], color=c, linewidth=1.9, label=f"E{i+1}")
        ax_h_t.plot(t, traj[:, 2], color=c, linewidth=1.8, label=f"$h^e_{i+1}$")

    ax_xy.set_title("XY")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(alpha=0.3)
    ax_xy.set_aspect("equal", adjustable="box")

    ax_xh.set_title("XH")
    ax_xh.set_xlabel("X (m)")
    ax_xh.set_ylabel("H (m)")
    ax_xh.grid(alpha=0.3)

    ax_yh.set_title("YH")
    ax_yh.set_xlabel("Y (m)")
    ax_yh.set_ylabel("H (m)")
    ax_yh.grid(alpha=0.3)

    ax_h_t.set_title("H-Time")
    ax_h_t.set_xlabel("Step")
    ax_h_t.set_ylabel("H (m)")
    ax_h_t.grid(alpha=0.3)

    ax_xy.legend(fontsize=7, loc="best")
    ax_h_t.legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle(title)
    _save(fig, path)


# ---------- Communication-specific plots ----------


def plot_comm_comparison(
    results_dict: Dict[str, EvalResult],
    dt: float,
    path: Path,
    title: str,
) -> None:
    """Overlay team error curves for different comm modes on same axes.

    Parameters
    ----------
    results_dict : dict
        Keys like "full_comm", "no_comm", "dropout", values are EvalResult.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    style_map = {
        "full_comm": {"color": "#1f77b4", "linewidth": 2.0, "linestyle": "-"},
        "no_comm": {"color": "#d62728", "linewidth": 1.8, "linestyle": "--"},
        "dropout": {"color": "#2ca02c", "linewidth": 1.8, "linestyle": "-."},
    }
    t_max = 0
    peak = 0
    for label, result in results_dict.items():
        t = np.arange(result.team_errors.shape[0]) * dt
        t_max = max(t_max, t[-1])
        peak = max(peak, float(np.max(result.team_errors)))
        style = style_map.get(label, {"color": "gray", "linewidth": 1.5, "linestyle": "-"})
        ax.plot(t, result.team_errors, label=label, **style)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Team state error")
    ax.set_title(title)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, peak * 1.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


def plot_d_min_history(
    eval_result: EvalResult,
    dt: float,
    path: Path,
    title: str,
    d_min_threshold: float = 100.0,
) -> None:
    """Plot minimum inter-pursuer distance over time with a threshold line."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(eval_result.d_min_history.shape[0]) * dt
    ax.plot(t, eval_result.d_min_history, linewidth=1.8, color="#1f77b4", label="$d_{\\min}$")
    ax.axhline(d_min_threshold, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label=f"threshold = {d_min_threshold:.0f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min inter-pursuer distance (m)")
    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


def plot_formation_error(
    eval_result: EvalResult,
    dt: float,
    path: Path,
    title: str,
) -> None:
    """Plot per-pursuer formation error norm over time."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if eval_result.formation_errors.size == 0:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Formation error norm")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        _save(fig, path)
        return

    t = np.arange(eval_result.formation_errors.shape[0]) * dt
    for j in range(eval_result.formation_errors.shape[1]):
        ax.plot(t, eval_result.formation_errors[:, j], linewidth=1.6, label=f"P{j+1}")
    ax.plot(
        t,
        np.mean(eval_result.formation_errors, axis=1),
        color="black",
        linewidth=2.0,
        linestyle="--",
        label="mean",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Formation error norm")
    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    _save(fig, path)
