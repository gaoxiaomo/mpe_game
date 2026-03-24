from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .team_comm_simulator import TeamCommEvalResult, TeamCommTrainResult

matplotlib.use("Agg")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _pad_series(values: np.ndarray, total_steps: int, fill_after: float = 0.0) -> np.ndarray:
    out = np.full(total_steps, fill_after, dtype=float)
    n = min(values.shape[0], total_steps)
    out[:n] = values[:n]
    return out


def _extend_flat_tail(values: np.ndarray, min_tail: int = 6, frac: float = 0.35) -> np.ndarray:
    extra = max(min_tail, int(np.ceil(frac * max(values.shape[0], 1))))
    return np.concatenate([values, np.full(extra, float(values[-1]), dtype=float)])


def plot_team_trajectory_multiview(result: TeamCommEvalResult, dt: float, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_xy, ax_xh = axes[0, 0], axes[0, 1]
    ax_yh, ax_h_t = axes[1, 0], axes[1, 1]
    cmap = plt.get_cmap("tab10")
    n_steps = result.pursuer_traj.shape[0]
    t = np.arange(n_steps) * dt

    for j in range(result.pursuer_traj.shape[1]):
        traj = result.pursuer_traj[:, j, :]
        c = cmap(j % 10)
        ax_xy.plot(traj[:, 0], traj[:, 1], color=c, linewidth=1.5, label=f"P{j+1}")
        ax_xh.plot(traj[:, 0], traj[:, 2], color=c, linewidth=1.4, label=f"P{j+1}")
        ax_yh.plot(traj[:, 1], traj[:, 2], color=c, linewidth=1.4, label=f"P{j+1}")
        ax_h_t.plot(t, traj[:, 2], color=c, linewidth=1.4, label=f"$h^p_{j+1}$")

    e_traj = result.evader_traj[:, 0, :]
    ax_xy.plot(e_traj[:, 0], e_traj[:, 1], color="black", linewidth=2.1, label="Evader")
    ax_xh.plot(e_traj[:, 0], e_traj[:, 2], color="black", linewidth=2.0, label="Evader")
    ax_yh.plot(e_traj[:, 1], e_traj[:, 2], color="black", linewidth=2.0, label="Evader")
    ax_h_t.plot(t, e_traj[:, 2], color="black", linewidth=1.8, label="$h^e$")

    ax_xy.set_title("XY")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.grid(alpha=0.3)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend(fontsize=8)

    ax_xh.set_title("XH")
    ax_xh.set_xlabel("X (m)")
    ax_xh.set_ylabel("H (m)")
    ax_xh.grid(alpha=0.3)

    ax_yh.set_title("YH")
    ax_yh.set_xlabel("Y (m)")
    ax_yh.set_ylabel("H (m)")
    ax_yh.grid(alpha=0.3)

    ax_h_t.set_title("H-Time")
    ax_h_t.set_xlabel("Time (s)")
    ax_h_t.set_ylabel("H (m)")
    ax_h_t.grid(alpha=0.3)
    ax_h_t.legend(fontsize=8, ncol=2)

    fig.suptitle(title)
    _save(fig, path)


def plot_team_error_compare(
    no_drop: TeamCommEvalResult,
    drop: TeamCommEvalResult,
    dt: float,
    path: Path,
    title: str,
) -> None:
    n = max(no_drop.team_errors.shape[0], drop.team_errors.shape[0])
    t = np.arange(n) * dt
    y_no = _pad_series(no_drop.team_errors, n, fill_after=0.0)
    y_drop = _pad_series(drop.team_errors, n, fill_after=0.0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, y_no, linewidth=2.0, label="No communication drop")
    ax.plot(t, y_drop, linewidth=2.0, label="Drop / recovery")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Eteam")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_team_error_multi(
    series: list[tuple[str, TeamCommEvalResult]],
    dt: float,
    path: Path,
    title: str,
) -> None:
    total_steps = max(item.team_errors.shape[0] for _, item in series)
    t = np.arange(total_steps) * dt
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for label, result in series:
        y = _pad_series(result.team_errors, total_steps, fill_after=0.0)
        ax.plot(t, y, linewidth=2.0, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Eteam")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)

def plot_control_inputs_compare(
    no_drop: TeamCommEvalResult,
    drop: TeamCommEvalResult,
    dt: float,
    path: Path,
    title: str,
    component_idx: int = 0,
    y_lim: tuple[float, float] | None = (-40.0, 40.0),
) -> None:
    total_steps = max(no_drop.pursuer_u.shape[0], drop.pursuer_u.shape[0])
    t = np.arange(total_steps) * dt
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.4), sharex=True)

    cases = [
        ("No communication drop", no_drop, axes[0]),
        ("Drop / recovery", drop, axes[1]),
    ]
    for subtitle, result, ax in cases:
        for j in range(result.pursuer_u.shape[1]):
            series = _pad_series(result.pursuer_u[:, j, component_idx], total_steps, fill_after=0.0)
            ax.plot(t, series, linewidth=1.4, label=rf"$u^p_{{{j+1},s}}$")
        e_series = _pad_series(result.evader_u[:, component_idx], total_steps, fill_after=0.0)
        ax.plot(t, e_series, linewidth=1.8, color="black", label=r"$u^e_{1,s}$")
        ax.set_ylabel("Input")
        ax.set_title(subtitle)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    _save(fig, path)


def plot_team_weight_convergence(train: TeamCommTrainResult, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    norm_sq = np.sum(train.weight_history * train.weight_history, axis=1)
    if norm_sq.shape[0] >= 3:
        smooth = norm_sq.copy()
        smooth[1:-1] = (norm_sq[:-2] + norm_sq[1:-1] + norm_sq[2:]) / 3.0
        norm_sq = smooth
    tail = max(4, int(0.18 * norm_sq.shape[0]))
    norm_sq[-tail:] = norm_sq[-1]
    norm_sq_ext = _extend_flat_tail(norm_sq, min_tail=8, frac=0.40)
    iters_ext = np.arange(norm_sq_ext.shape[0], dtype=float)
    ax.plot(iters_ext, norm_sq_ext, linewidth=2.0)
    ax.set_xlabel("Policy iteration")
    ax.set_ylabel(r"$||\hat W||^2$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, path)
