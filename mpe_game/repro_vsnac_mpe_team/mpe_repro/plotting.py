from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .simulator import EvalResult, TrainResult

matplotlib.use("Agg")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
# Trajectory
# ------------------------------------------------------------------

def plot_trajectory_xy(result: EvalResult, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    p_colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    e_colors = ["#000000", "#17becf", "#ff7f0e"]
    for j in range(result.pursuer_traj.shape[1]):
        t = result.pursuer_traj[:, j, :3]
        c = p_colors[j % len(p_colors)]
        ax.plot(t[:, 0], t[:, 1], color=c, linewidth=1.5, label=f"P{j+1}")
        ax.scatter(t[0, 0], t[0, 1], color=c, s=40, marker="o", edgecolors="k", linewidths=0.4)
        ax.scatter(t[-1, 0], t[-1, 1], color=c, s=28, marker="s")
    for i in range(result.evader_traj.shape[1]):
        t = result.evader_traj[:, i, :3]
        c = e_colors[i % len(e_colors)]
        ax.plot(t[:, 0], t[:, 1], color=c, linewidth=2.2, label=f"E{i+1}")
        ax.scatter(t[0, 0], t[0, 1], color=c, s=48, marker="x")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _save(fig, path)


# ------------------------------------------------------------------
# Assigned errors
# ------------------------------------------------------------------

def plot_assigned_errors(result: EvalResult, dt: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(result.assigned_errors.shape[0]) * dt
    for j in range(result.assigned_errors.shape[1]):
        ax.plot(t, result.assigned_errors[:, j], linewidth=1.5, label=f"P{j+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Assigned error norm")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


# ------------------------------------------------------------------
# Weight convergence
# ------------------------------------------------------------------

def plot_weight_convergence(train: TrainResult, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w_hist = train.weight_history
    norm_sq = np.sum(w_hist * w_hist, axis=2)
    for i in range(norm_sq.shape[1]):
        ax.plot(norm_sq[:, i], linewidth=1.6, label=f"critic {i+1}")
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel("||W||^2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


# ------------------------------------------------------------------
# Team error comparison
# ------------------------------------------------------------------

def plot_team_error_compare(
    results: dict[str, EvalResult],
    dt: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (label, res) in enumerate(results.items()):
        t = np.arange(res.team_errors.shape[0]) * dt
        ax.plot(t, res.team_errors, linewidth=1.8, color=colors[idx % len(colors)], label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Team error")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


# ------------------------------------------------------------------
# d_min (minimum inter-pursuer distance)
# ------------------------------------------------------------------

def plot_d_min(
    results: dict[str, EvalResult],
    dt: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (label, res) in enumerate(results.items()):
        if res.coord_metrics is None:
            continue
        t = np.arange(res.coord_metrics.d_min.shape[0]) * dt
        ax.plot(t, res.coord_metrics.d_min, linewidth=1.6, color=colors[idx % len(colors)], label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min inter-pursuer distance (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


# ------------------------------------------------------------------
# Angular coverage
# ------------------------------------------------------------------

def plot_angular_coverage(
    results: dict[str, EvalResult],
    dt: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (label, res) in enumerate(results.items()):
        if res.coord_metrics is None:
            continue
        t = np.arange(res.coord_metrics.angular_coverage.shape[0]) * dt
        ax.plot(t, res.coord_metrics.angular_coverage, linewidth=1.6, color=colors[idx % len(colors)], label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular coverage [0, 1]")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


# ------------------------------------------------------------------
# Path overlap
# ------------------------------------------------------------------

def plot_path_overlap(
    results: dict[str, EvalResult],
    dt: float,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (label, res) in enumerate(results.items()):
        if res.coord_metrics is None:
            continue
        t = np.arange(res.coord_metrics.path_overlap.shape[0]) * dt
        ax.plot(t, res.coord_metrics.path_overlap, linewidth=1.6, color=colors[idx % len(colors)], label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Path overlap fraction")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, path)


# ------------------------------------------------------------------
# Assignment timeline
# ------------------------------------------------------------------

def plot_assignment_timeline(result: EvalResult, dt: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if result.assigned_targets.size == 0:
        ax.set_title(title)
        _save(fig, path)
        return
    t = np.arange(result.assigned_targets.shape[0]) * dt
    for j in range(result.assigned_targets.shape[1]):
        ax.step(t, result.assigned_targets[:, j].astype(float), where="post", linewidth=1.6, label=f"P{j+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Target evader idx")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


# ------------------------------------------------------------------
# Comparison bar chart
# ------------------------------------------------------------------

def plot_comparison_bars(
    summaries: dict[str, dict],
    path: Path,
    title: str,
) -> None:
    """Bar chart comparing capture time, d_min, angular coverage across modes."""
    labels = list(summaries.keys())
    cap_times = [s.get("capture_time_s", None) for s in summaries.values()]
    d_mins = [s.get("d_min_mean", 0) for s in summaries.values()]
    ang_covs = [s.get("angular_coverage_mean", 0) for s in summaries.values()]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax = axes[0]
    vals = [v if v is not None else 0 for v in cap_times]
    ax.bar(labels, vals, color=colors[:len(labels)])
    ax.set_ylabel("Capture time (s)")
    ax.set_title("Capture Time")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(labels, d_mins, color=colors[:len(labels)])
    ax.set_ylabel("Mean d_min (m)")
    ax.set_title("Mean Min Inter-Pursuer Dist")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(labels, ang_covs, color=colors[:len(labels)])
    ax.set_ylabel("Mean angular coverage")
    ax.set_title("Angular Coverage")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(title)
    _save(fig, path)
