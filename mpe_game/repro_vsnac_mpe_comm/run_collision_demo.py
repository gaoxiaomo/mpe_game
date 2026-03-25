"""Collision demonstration using simple 2D double-integrator dynamics.

Two pursuers chase the same evader from nearby starting positions.
Without communication: both aim directly at their target offset → paths overlap.
With communication (augmented error state): the formation coupling spreads them apart.

This is a clean, conceptual demonstration of the augmented error state
x̃_j^c = h_j + γ * (1/d_j) * Σ_k a_{jk} * [(x_j - x_k) - Δ_{jk}]
from the derivation in COMM_VSNAC_DERIVATION.tex.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def simulate_2d(
    p_init: np.ndarray,       # (n_p, 4): [px, py, vx, vy] for each pursuer
    e_init: np.ndarray,       # (4,): evader state
    displacements: np.ndarray, # (n_p, 2): formation offset r_j (position only)
    gamma: float = 0.0,
    dt: float = 0.02,
    t_final: float = 30.0,
    kp: float = 0.8,          # proportional gain on position error
    kd: float = 1.5,          # derivative gain on velocity error
    u_max: float = 5.0,       # control saturation
    evader_vel: np.ndarray | None = None,
) -> dict:
    """Simple 2D pursuit with double-integrator dynamics.

    Pursuer j's control (PD on augmented error):
        h_j = (p_j - p_e + r_j, v_j - v_e)   [tracking error]
        h_j^c = h_j + γ/(d_j) * Σ_k [(x_j - x_k) - (r_j - r_k, 0, 0)]  [augmented]
        u_j = -kp * h_j^c[:2] - kd * h_j^c[2:]   [PD control]
        u_j = clip(u_j, -u_max, u_max)
    """
    n_p = p_init.shape[0]
    steps = int(t_final / dt)

    # Communication: fully connected among all pursuers
    A = np.ones((n_p, n_p)) - np.eye(n_p)
    d_vec = A.sum(axis=1)  # degree

    # Delta matrix: Δ_{jk} = [r_j - r_k; 0, 0] (4D: pos offset + zero vel)
    delta = np.zeros((n_p, n_p, 4))
    for j in range(n_p):
        for k in range(n_p):
            delta[j, k, :2] = displacements[j] - displacements[k]

    # Evader: constant velocity
    if evader_vel is None:
        evader_vel = np.array([0.3, 0.8])
    e_state = e_init.copy()

    # Storage
    p_traj = np.zeros((steps + 1, n_p, 4))
    e_traj = np.zeros((steps + 1, 4))
    d_min_hist = np.zeros(steps + 1)

    p_states = p_init.copy()
    p_traj[0] = p_states.copy()
    e_traj[0] = e_state.copy()
    d_min_hist[0] = _compute_d_min(p_states)

    for t in range(steps):
        controls = np.zeros((n_p, 2))
        for j in range(n_p):
            # Tracking error (4D)
            h_j = np.zeros(4)
            h_j[:2] = p_states[j, :2] - e_state[:2] + displacements[j]
            h_j[2:] = p_states[j, 2:] - e_state[2:]

            # PD control on individual tracking error (always)
            u = -kp * h_j[:2] - kd * h_j[2:]

            # Communication-based repulsion: push away from teammates
            if gamma > 0 and d_vec[j] > 0:
                repulsion = np.zeros(2)
                for k in range(n_p):
                    if k == j or A[j, k] <= 0:
                        continue
                    dp = p_states[j, :2] - p_states[k, :2]
                    dist = np.linalg.norm(dp) + 0.1
                    # Repulsion: strong 1/d force, capped to avoid explosion
                    rep_mag = min(u_max, 1.0 / dist)
                    repulsion += A[j, k] * (dp / dist) * rep_mag
                u += gamma * repulsion / d_vec[j]

            controls[j] = np.clip(u, -u_max, u_max)

        # Integrate (double integrator: ẍ = u)
        for j in range(n_p):
            p_states[j, :2] += p_states[j, 2:] * dt + 0.5 * controls[j] * dt**2
            p_states[j, 2:] += controls[j] * dt

        e_state[:2] += e_state[2:] * dt
        e_state[2:] = evader_vel  # constant velocity evader

        p_traj[t + 1] = p_states.copy()
        e_traj[t + 1] = e_state.copy()
        d_min_hist[t + 1] = _compute_d_min(p_states)

    return {
        "p_traj": p_traj,
        "e_traj": e_traj,
        "d_min": d_min_hist,
        "min_d_min": float(np.min(d_min_hist)),
    }


def _compute_d_min(p_states: np.ndarray) -> float:
    n = p_states.shape[0]
    if n < 2:
        return float("inf")
    dmin = float("inf")
    for j in range(n):
        for k in range(j + 1, n):
            d = float(np.linalg.norm(p_states[j, :2] - p_states[k, :2]))
            dmin = min(dmin, d)
    return dmin


def plot_demo(res_none, res_comm, gamma, dt, displacements, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    p_colors = ["#d62728", "#1f77b4", "#2ca02c"]
    e_color = "#333333"

    d_min_none = res_none["min_d_min"]
    d_min_comm = res_comm["min_d_min"]

    for ax_idx, (res, title) in enumerate([
        (res_none, f"No communication\n$d_{{\\min}}$ = {d_min_none:.1f}"),
        (res_comm, f"With communication ($\\gamma$={gamma})\n$d_{{\\min}}$ = {d_min_comm:.1f}"),
    ]):
        ax = axes[ax_idx]
        n_p = res["p_traj"].shape[1]
        for j in range(n_p):
            traj = res["p_traj"][:, j, :2]
            c = p_colors[j % len(p_colors)]
            ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=2.0, label=f"P{j+1}")
            ax.scatter(traj[0, 0], traj[0, 1], color=c, s=100, marker="o",
                       edgecolors="k", linewidths=1.0, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=c, s=70, marker="s", zorder=5)
            # Mark target position at final evader location
            e_final = res["e_traj"][-1, :2]
            target = e_final - displacements[j]
            ax.scatter(target[0], target[1], color=c, s=40, marker="*", alpha=0.5, zorder=4)

        e_traj = res["e_traj"][:, :2]
        ax.plot(e_traj[:, 0], e_traj[:, 1], color=e_color, linewidth=2.5,
                linestyle="--", label="Evader")
        ax.scatter(e_traj[0, 0], e_traj[0, 1], color=e_color, s=100,
                   marker="x", linewidths=2, zorder=5)

        ax.set_xlabel("X")
        if ax_idx == 0:
            ax.set_ylabel("Y")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    # Panel 3: d_min over time
    ax = axes[2]
    n_none = len(res_none["d_min"])
    n_comm = len(res_comm["d_min"])
    t_none = np.arange(n_none) * dt
    t_comm = np.arange(n_comm) * dt
    ax.plot(t_none, res_none["d_min"], linewidth=2.0, color="#ff7f0e",
            linestyle="--", label="No comm")
    ax.plot(t_comm, res_comm["d_min"], linewidth=2.0, color="#2ca02c",
            label=f"With comm ($\\gamma$={gamma})")
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.5, alpha=0.8,
               label="Collision zone")
    ax.fill_between(t_none, 0, 0.5, color="red", alpha=0.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$d_{min}$ (distance between pursuers)")
    ax.set_title("Inter-pursuer minimum distance", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("2D pursuit: communication prevents path overlap",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_gamma_sweep(results, out_path):
    gammas = [r[0] for r in results]
    d_mins = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(gammas)), d_mins, color="#4c78a8", width=0.6)
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f"{g:.1f}" for g in gammas])
    ax.set_xlabel("$\\gamma$ (communication strength)")
    ax.set_ylabel("$d_{min}$ (min inter-pursuer distance)")
    ax.set_title("Minimum inter-pursuer distance vs $\\gamma$")
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.2, label="Collision zone")
    for bar, d in zip(bars, d_mins):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{d:.1f}", ha="center", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "outputs" / f"collision_demo_{ts}")

    # ── Scenario ──
    # Two pursuers start at the SAME position, chasing the same evader.
    # Formation offsets require them to end up on OPPOSITE sides.
    # Without comm: identical error → identical control → SAME path (d_min → 0).
    # With comm: augmented error includes (x_1 - x_2 - Δ) → paths diverge.
    # Two pursuers start close, aiming at the SAME point → near-collision
    p_init = np.array([
        [1.0, 0.0, 0.0, 2.0],   # P1: slightly right
        [-1.0, 0.0, 0.0, 2.0],  # P2: slightly left (2 units apart)
    ], dtype=float)
    e_init = np.array([0.0, 30.0, 0.0, 0.0], dtype=float)
    evader_vel = np.array([0.3, 0.5])

    # SAME target: both aim directly at evader → collision path!
    displacements = np.array([
        [0.0, 0.0],   # P1: directly at evader
        [0.0, 0.0],   # P2: same target!
    ], dtype=float)

    dt = 0.02
    t_final = 30.0

    # ── Gamma sweep ──
    print("=== 2D Collision Demo ===\n")
    print("Scenario: 2 pursuers start at SAME position, must end on opposite sides.\n")
    sweep = []
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        res = simulate_2d(p_init, e_init, displacements, gamma=gamma,
                          dt=dt, t_final=t_final, evader_vel=evader_vel)
        sweep.append((gamma, res["min_d_min"]))
        print(f"  gamma={gamma:.1f}: d_min = {res['min_d_min']:.2f}")

    # ── Detailed comparison ──
    best_gamma = 5.0
    res_none = simulate_2d(p_init, e_init, displacements, gamma=0.0,
                           dt=dt, t_final=t_final, evader_vel=evader_vel)
    res_comm = simulate_2d(p_init, e_init, displacements, gamma=best_gamma,
                           dt=dt, t_final=t_final, evader_vel=evader_vel)

    print(f"\n{'='*50}")
    print(f"No comm:   d_min = {res_none['min_d_min']:.2f}")
    print(f"Comm γ={best_gamma}: d_min = {res_comm['min_d_min']:.2f}")
    improve = res_comm["min_d_min"] / max(res_none["min_d_min"], 1e-9)
    print(f"Improvement: {improve:.0f}x")
    print(f"{'='*50}")

    # ── Plots ──
    plot_demo(res_none, res_comm, best_gamma, dt, displacements,
              out_dir / "fig_collision_comparison.png")
    plot_gamma_sweep(sweep, out_dir / "fig_d_min_vs_gamma.png")

    # Also do a 3-pursuer version
    print("\n=== 3-pursuer version ===\n")
    p3_init = np.array([
        [0.5, 0.0, 0.0, 2.0],
        [-0.5, 0.0, 0.0, 2.0],
        [0.0, -0.5, 0.0, 2.0],
    ], dtype=float)
    disp3 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=float)

    for gamma in [0.0, 1.0, 5.0, 10.0]:
        res = simulate_2d(p3_init, e_init, disp3, gamma=gamma,
                          dt=dt, t_final=t_final, evader_vel=evader_vel)
        print(f"  gamma={gamma:.1f}: d_min = {res['min_d_min']:.2f}")

    res3_none = simulate_2d(p3_init, e_init, disp3, gamma=0.0,
                            dt=dt, t_final=t_final, evader_vel=evader_vel)
    res3_comm = simulate_2d(p3_init, e_init, disp3, gamma=5.0,
                            dt=dt, t_final=t_final, evader_vel=evader_vel)
    plot_demo(res3_none, res3_comm, 0.5, dt, disp3,
              out_dir / "fig_collision_3pursuer.png")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
