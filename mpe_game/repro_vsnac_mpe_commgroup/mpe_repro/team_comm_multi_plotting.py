from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


def plot_group_weight_convergence(train: TeamCommMultiTrainResult, path: Path, title: str) -> None:
    norms_sq = train.weight_norm_history * train.weight_norm_history
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for group_idx in range(norms_sq.shape[1]):
        ax.plot(
            np.arange(norms_sq.shape[0], dtype=float),
            norms_sq[:, group_idx],
            linewidth=1.9,
            label=f"Group {group_idx + 1}",
        )
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel(r"$||\hat W_i||^2$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
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

