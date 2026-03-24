from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .team_comm_multi_simulator import TeamCommMultiEvalResult, TeamCommMultiTrainResult

matplotlib.use("Agg")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _pad_series(values: np.ndarray, total_steps: int, fill_after: float = 0.0) -> np.ndarray:
    out = np.full(total_steps, fill_after, dtype=float)
    n = min(int(values.shape[0]), total_steps)
    out[:n] = values[:n]
    return out


def plot_multiteam_trajectory_xy(result: TeamCommMultiEvalResult, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    cmap_p = plt.get_cmap("tab20")
    cmap_e = plt.get_cmap("Dark2")

    for pursuer_idx in range(result.pursuer_traj.shape[1]):
        traj = result.pursuer_traj[:, pursuer_idx, :]
        ax.plot(traj[:, 0], traj[:, 1], color=cmap_p(pursuer_idx % 20), linewidth=1.4, label=f"P{pursuer_idx + 1}")

    for evader_idx in range(result.evader_traj.shape[1]):
        traj = result.evader_traj[:, evader_idx, :]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=cmap_e(evader_idx % 8),
            linewidth=2.2,
            linestyle="--",
            label=f"E{evader_idx + 1}",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_multiteam_errors(result: TeamCommMultiEvalResult, dt: float, path: Path, title: str) -> None:
    steps = int(result.team_errors.shape[0])
    t = np.arange(steps) * dt
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)

    axes[0].plot(t, result.team_errors, linewidth=2.1, color="#2f5597", label="Total Eteam")
    axes[0].set_ylabel("Eteam")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    for group_idx in range(result.group_errors.shape[1]):
        axes[1].plot(
            t,
            result.group_errors[:, group_idx],
            linewidth=1.8,
            label=f"Group {group_idx + 1}",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Group error")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_assignment_timeline(result: TeamCommMultiEvalResult, dt: float, path: Path, title: str) -> None:
    if result.assignment_history.size == 0:
        return
    steps = int(result.assignment_history.shape[0])
    t = np.arange(steps) * dt
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    cmap = plt.get_cmap("tab10")
    for pursuer_idx in range(result.assignment_history.shape[1]):
        y = result.assignment_history[:, pursuer_idx] + 1
        ax.step(t, y, where="post", linewidth=1.6, color=cmap(pursuer_idx % 10), label=f"P{pursuer_idx + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Assigned evader")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_comm_ratio(result: TeamCommMultiEvalResult, dt: float, path: Path, title: str) -> None:
    if result.communication_ratio.size == 0:
        return
    steps = int(result.communication_ratio.shape[0])
    t = np.arange(steps) * dt
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for group_idx in range(result.communication_ratio.shape[1]):
        ax.plot(
            t,
            result.communication_ratio[:, group_idx],
            linewidth=1.8,
            label=f"Group {group_idx + 1}",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Communication ratio")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_group_weight_convergence(
    train: TeamCommMultiTrainResult,
    path: Path,
    title: str,
    dt: float = 0.0,
    rollout_steps: int = 0,
    rollouts_per_iter: int = 0,
) -> None:
    norms_sq = train.weight_norm_history * train.weight_norm_history
    n_iters = norms_sq.shape[0]
    # Use wall-clock-equivalent time if rollout info is available.
    if dt > 0 and rollout_steps > 0 and rollouts_per_iter > 0:
        iter_time = dt * rollout_steps * rollouts_per_iter
        x = np.arange(n_iters, dtype=float) * iter_time
        xlabel = "Training time (s)"
    else:
        x = np.arange(n_iters, dtype=float)
        xlabel = "Policy iteration"
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for group_idx in range(norms_sq.shape[1]):
        ax.plot(x, norms_sq[:, group_idx], linewidth=1.9, label=f"Group {group_idx + 1}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$||\hat W_i||^2$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, path)


def plot_convergence_diagnostics(train: TeamCommMultiTrainResult, path: Path, title: str) -> None:
    """Plot weight delta and Bellman residual per iteration to diagnose convergence."""
    if train.delta_history.size == 0:
        return
    n_iters = train.delta_history.shape[0]
    iters = np.arange(n_iters, dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.5), sharex=True)

    for group_idx in range(train.delta_history.shape[1]):
        axes[0].plot(iters, train.delta_history[:, group_idx], linewidth=1.6, label=f"Group {group_idx + 1}")
    axes[0].set_ylabel("Weight delta ||dW||")
    axes[0].set_title(title)
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    for group_idx in range(train.residual_history.shape[1]):
        axes[1].plot(iters, train.residual_history[:, group_idx], linewidth=1.6, label=f"Group {group_idx + 1}")
    axes[1].set_ylabel("Bellman residual RMS")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)

    val = train.validation_history[:, 0] if train.validation_history.ndim > 1 else train.validation_history
    valid_mask = ~np.isnan(val)
    if np.any(valid_mask):
        axes[2].plot(np.arange(len(val))[valid_mask], val[valid_mask], "o-", linewidth=1.6, markersize=3, label="Validation metric")
    if train.best_iteration < len(val):
        axes[2].axvline(train.best_iteration, color="red", linestyle="--", alpha=0.6, label=f"Best iter={train.best_iteration}")
    axes[2].set_xlabel("Policy iteration")
    axes[2].set_ylabel("Validation metric")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    _save(fig, path)


def plot_comm_comparison(
    result_full: TeamCommMultiEvalResult,
    result_dropout: TeamCommMultiEvalResult,
    dt: float,
    path: Path,
    title: str,
) -> None:
    """Compare team errors under full communication vs dropout."""
    steps_f = int(result_full.team_errors.shape[0])
    steps_d = int(result_dropout.team_errors.shape[0])
    t_f = np.arange(steps_f) * dt
    t_d = np.arange(steps_d) * dt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    # Total team error
    axes[0].plot(t_f, result_full.team_errors, linewidth=2.0, color="#2f5597", label="Full comm")
    axes[0].plot(t_d, result_dropout.team_errors, linewidth=2.0, color="#c0392b", linestyle="--", label="With dropout")
    axes[0].set_ylabel("Total Eteam")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # Per-group max assigned error
    n_groups = result_full.group_errors.shape[1]
    cmap = plt.get_cmap("tab10")
    for g in range(n_groups):
        c = cmap(g % 10)
        axes[1].plot(t_f, result_full.group_errors[:, g], linewidth=1.6, color=c, label=f"G{g+1} full")
        axes[1].plot(t_d, result_dropout.group_errors[:, g], linewidth=1.6, color=c, linestyle="--", alpha=0.7, label=f"G{g+1} dropout")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Group error")
    axes[1].grid(alpha=0.3)
    axes[1].legend(ncol=2, fontsize=7)
    _save(fig, path)


def plot_assigned_error_summary(result: TeamCommMultiEvalResult, dt: float, path: Path, title: str) -> None:
    if result.assigned_errors.size == 0:
        return
    steps = int(result.assigned_errors.shape[0])
    t = np.arange(steps) * dt
    mean_err = np.mean(result.assigned_errors, axis=1)
    max_err = np.max(result.assigned_errors, axis=1)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(t, mean_err, linewidth=2.0, label="Mean assigned error")
    ax.plot(t, max_err, linewidth=2.0, label="Max assigned error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_multiteam_trajectory_gif(
    result: TeamCommMultiEvalResult,
    dt: float,
    path: Path,
    title: str,
    capture_radius: float = 220.0,
    fps: int = 20,
    frame_skip: int = 4,
    tail_steps: int = 60,
) -> None:
    """Generate an animated GIF of the multi-team pursuit-evasion trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_p = result.pursuer_traj.shape[1]
    n_e = result.evader_traj.shape[1]
    total_steps = result.pursuer_traj.shape[0]
    frame_indices = list(range(0, total_steps, frame_skip))
    if frame_indices[-1] != total_steps - 1:
        frame_indices.append(total_steps - 1)

    cmap_p = plt.get_cmap("tab10")
    cmap_e = plt.get_cmap("Set1")
    # Assign distinct colors per evader group for pursuers
    evader_colors = [cmap_e(i % 9) for i in range(n_e)]

    # Compute axis limits from full trajectories with padding
    all_x = np.concatenate([result.pursuer_traj[:, :, 0].ravel(), result.evader_traj[:, :, 0].ravel()])
    all_y = np.concatenate([result.pursuer_traj[:, :, 1].ravel(), result.evader_traj[:, :, 1].ravel()])
    pad = max(np.ptp(all_x), np.ptp(all_y)) * 0.08
    xlim = (float(np.min(all_x) - pad), float(np.max(all_x) + pad))
    ylim = (float(np.min(all_y) - pad), float(np.max(all_y) + pad))

    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(alpha=0.2)

    # Create plot elements
    pursuer_trails = [ax.plot([], [], linewidth=1.2, alpha=0.5)[0] for _ in range(n_p)]
    pursuer_dots = [ax.plot([], [], "o", markersize=7)[0] for _ in range(n_p)]
    evader_trails = [ax.plot([], [], linewidth=2.0, linestyle="--", alpha=0.6)[0] for _ in range(n_e)]
    evader_dots = [ax.plot([], [], "s", markersize=10)[0] for _ in range(n_e)]
    assign_lines = [ax.plot([], [], linewidth=0.8, alpha=0.3, linestyle=":")[0] for _ in range(n_p)]
    # Capture circles around evaders
    capture_circles = [
        plt.Circle((0, 0), capture_radius, fill=False, linestyle="--", linewidth=0.8, alpha=0.3)
        for _ in range(n_e)
    ]
    for c in capture_circles:
        ax.add_patch(c)

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=11, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    title_text = ax.set_title(title)

    has_assignment = result.assignment_history.size > 0

    def _update(frame_num: int) -> list:
        idx = frame_indices[frame_num]
        t_s = idx * dt

        # Get assignment at this step
        if has_assignment and idx < result.assignment_history.shape[0]:
            assignment = result.assignment_history[idx]
        elif has_assignment:
            assignment = result.assignment_history[-1]
        else:
            assignment = np.zeros(n_p, dtype=int)

        # Trail window
        trail_start = max(0, idx - tail_steps * frame_skip)

        for j in range(n_p):
            evader_target = int(assignment[j])
            color = evader_colors[evader_target % len(evader_colors)]
            px = result.pursuer_traj[trail_start:idx + 1, j, 0]
            py = result.pursuer_traj[trail_start:idx + 1, j, 1]
            pursuer_trails[j].set_data(px, py)
            pursuer_trails[j].set_color(color)
            pursuer_dots[j].set_data([result.pursuer_traj[idx, j, 0]], [result.pursuer_traj[idx, j, 1]])
            pursuer_dots[j].set_color(color)
            # Assignment line from pursuer to its target evader
            ex = result.evader_traj[min(idx, result.evader_traj.shape[0] - 1), evader_target, 0]
            ey = result.evader_traj[min(idx, result.evader_traj.shape[0] - 1), evader_target, 1]
            assign_lines[j].set_data(
                [result.pursuer_traj[idx, j, 0], ex],
                [result.pursuer_traj[idx, j, 1], ey],
            )
            assign_lines[j].set_color(color)

        for i in range(n_e):
            color = evader_colors[i % len(evader_colors)]
            darker = tuple(c * 0.6 for c in color[:3]) + (1.0,)
            eidx = min(idx, result.evader_traj.shape[0] - 1)
            ex_trail = result.evader_traj[trail_start:eidx + 1, i, 0]
            ey_trail = result.evader_traj[trail_start:eidx + 1, i, 1]
            evader_trails[i].set_data(ex_trail, ey_trail)
            evader_trails[i].set_color(darker)
            evader_dots[i].set_data([result.evader_traj[eidx, i, 0]], [result.evader_traj[eidx, i, 1]])
            evader_dots[i].set_color(darker)
            capture_circles[i].center = (result.evader_traj[eidx, i, 0], result.evader_traj[eidx, i, 1])

        cap_str = ""
        if result.capture_time is not None and t_s >= result.capture_time:
            cap_str = f"  [CAPTURED at {result.capture_time:.1f}s]"
        time_text.set_text(f"t = {t_s:.1f}s{cap_str}")

        return pursuer_trails + pursuer_dots + evader_trails + evader_dots + assign_lines + [time_text]

    anim = FuncAnimation(fig, _update, frames=len(frame_indices), interval=1000 // fps, blit=False)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)

