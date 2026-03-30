"""Controlled scenario: paper-consistent V-SNAC + augmented value function.

Two pursuers chase the same evader from the same starting position using
a simplified 2D double-integrator model.  Each pursuer has a DIFFERENT
formation offset (r1 != r2), giving delta_jk != 0.  The control structures
exactly match the paper:

  - V-SNAC basis  psi = g * [p1*v1, p2*v2, v1^2/2, v2^2/2]    (eq.10)
  - Pursuer ctrl  u = -u_bar tanh(R^-1 g^T nabla V_aug / 2u_bar) (eq.18)
  - Coordination  nabla_v Phi = gamma/d sum[(dv)+kappa/d_ref*(dp-delta)]  (eq.13)
  - Evader ctrl   u_e = -u_bar_e tanh(R^-1 g^T nabla V / 2u_bar_e)      (eq.19)

Key insight: the PD consensus potential drives pursuers toward their
DESIRED relative positions (delta_jk).  With delta=0, it attracts
(same target).  With delta!=0, it separates (different formation offsets).
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# V-SNAC velocity-space gradient (matches paper eq.10 / features.py)
# ---------------------------------------------------------------------------

def vsnac_grad_v(x_tilde: np.ndarray, W: np.ndarray,
                 gain: float, pos_scale: float) -> np.ndarray:
    """grad_v V = gain * [W0*p1/s + W2*v1,  W1*p2/s + W3*v2]"""
    p = x_tilde[:2] / pos_scale
    v = x_tilde[2:]
    return gain * np.array([
        W[0] * p[0] + W[2] * v[0],
        W[1] * p[1] + W[3] * v[1],
    ])


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    p_init: np.ndarray,        # (n_p, 4)
    e_init: np.ndarray,        # (4,)
    displacements: np.ndarray, # (n_p, 2)
    *,
    gamma: float = 0.0,
    d_safe: float = 0.0,
    dt: float = 0.05,
    t_final: float = 90.0,
    u_bar_p: float = 25.0,
    u_bar_e: float = 15.0,
    R1: float = 0.1,
    R2: float = 0.1,
    kappa: float = 1.0,
    d_ref: float = 500.0,
    gain: float = 0.005,
    pos_scale: float = 3.0,
    W: np.ndarray | None = None,
) -> dict:
    n_p = p_init.shape[0]
    steps = int(t_final / dt)
    R1_inv, R2_inv = 1.0 / R1, 1.0 / R2

    A = np.ones((n_p, n_p)) - np.eye(n_p)
    d_vec = A.sum(axis=1)

    # delta_jk = desired (p_j - p_k) = (e-r_j)-(e-r_k) = r_k - r_j
    delta_p = np.zeros((n_p, n_p, 2))
    for j in range(n_p):
        for k in range(n_p):
            delta_p[j, k] = displacements[k] - displacements[j]

    if W is None:
        W = np.array([0.8, 0.8, 0.6, 0.6])

    p_st = p_init.copy().astype(float)
    e_st = e_init.copy().astype(float)
    p_traj = np.zeros((steps + 1, n_p, 4))
    e_traj = np.zeros((steps + 1, 4))
    err_hist = np.zeros((steps + 1, n_p))
    dmin_hist = np.zeros(steps + 1)

    def record(idx):
        p_traj[idx] = p_st
        e_traj[idx] = e_st
        for j in range(n_p):
            xe = np.concatenate([
                p_st[j, :2] - e_st[:2] + displacements[j],
                p_st[j, 2:] - e_st[2:]
            ])
            err_hist[idx, j] = np.linalg.norm(xe)
        dm = float("inf")
        for j in range(n_p):
            for k in range(j + 1, n_p):
                dm = min(dm, float(np.linalg.norm(p_st[j, :2] - p_st[k, :2])))
        dmin_hist[idx] = dm

    record(0)

    for t in range(steps):
        gv = np.zeros((n_p, 2))
        for j in range(n_p):
            xe = np.concatenate([
                p_st[j, :2] - e_st[:2] + displacements[j],
                p_st[j, 2:] - e_st[2:]
            ])
            gv[j] = vsnac_grad_v(xe, W, gain, pos_scale)

        u_p = np.zeros((n_p, 2))
        for j in range(n_p):
            total = gv[j].copy()
            if gamma > 0 and d_vec[j] > 0:
                coord = np.zeros(2)
                for k in range(n_p):
                    if k == j or A[j, k] <= 0:
                        continue
                    dv = p_st[j, 2:] - p_st[k, 2:]
                    dp = p_st[j, :2] - p_st[k, :2] - delta_p[j, k]
                    # Per-pair adaptive gamma amplification
                    if d_safe > 0:
                        sep = p_st[j, :2] - p_st[k, :2]
                        dist_sq = float(sep @ sep)
                        amp = max(1.0, d_safe * d_safe / max(dist_sq, 1e-8))
                    else:
                        amp = 1.0
                    coord += amp * A[j, k] * (dv + kappa / d_ref * dp)
                total += gamma / d_vec[j] * coord
            rho = R1_inv * total / (2.0 * u_bar_p)
            u_p[j] = -u_bar_p * np.tanh(rho)

        gv_mean = gv.mean(axis=0)
        rho_e = R2_inv * gv_mean / (2.0 * u_bar_e)
        u_e = -u_bar_e * np.tanh(rho_e)

        for j in range(n_p):
            p_st[j, :2] += p_st[j, 2:] * dt
            p_st[j, 2:] += u_p[j] * dt
        e_st[:2] += e_st[2:] * dt
        e_st[2:] += u_e * dt

        record(t + 1)

    return {
        "p_traj": p_traj, "e_traj": e_traj,
        "err_hist": err_hist, "d_min": dmin_hist,
        "min_d_min": float(np.min(dmin_hist)),
        "final_d": float(dmin_hist[-1]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_trajectory_panel(ax, res, label, cp, xlim, ylim, show_ylabel=True):
    n_p = res["p_traj"].shape[1]
    for j in range(n_p):
        tr = res["p_traj"][:, j, :2]
        ax.plot(tr[:, 0], tr[:, 1], c=cp[j], lw=2, label=f"P{j+1}")
        ax.scatter(tr[0, 0], tr[0, 1], c=cp[j], s=80, marker="o",
                   edgecolors="k", lw=0.8, zorder=5)
        ax.scatter(tr[-1, 0], tr[-1, 1], c=cp[j], s=60, marker="s", zorder=5)
    et = res["e_traj"][:, :2]
    ax.plot(et[:, 0], et[:, 1], c="#333", lw=2, ls="--", label="Evader")
    ax.scatter(et[0, 0], et[0, 1], c="#333", s=80, marker="x", lw=2, zorder=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("X (m)", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)


def plot_demo(res_none, res_comm, gamma_val, dt, out_path, d_safe=0.0):
    cp = ["#d62728", "#1f77b4"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared axis limits
    all_x, all_y = [], []
    for res in [res_none, res_comm]:
        n_p = res["p_traj"].shape[1]
        for j in range(n_p):
            all_x.extend(res["p_traj"][:, j, 0].tolist())
            all_y.extend(res["p_traj"][:, j, 1].tolist())
        all_x.extend(res["e_traj"][:, 0].tolist())
        all_y.extend(res["e_traj"][:, 1].tolist())
    pad_x = (max(all_x) - min(all_x)) * 0.05
    pad_y = (max(all_y) - min(all_y)) * 0.05
    xlim = (min(all_x) - pad_x, max(all_x) + pad_x)
    ylim = (min(all_y) - pad_y, max(all_y) + pad_y)

    # Figure 1: (a) and (b) trajectory panels side by side
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_trajectory_panel(axes1[0], res_none,
                           r"No communication ($\gamma$=0)", cp, xlim, ylim, True)
    _plot_trajectory_panel(axes1[1], res_comm,
                           rf"With communication ($\gamma$={gamma_val})", cp, xlim, ylim, False)
    fig1.tight_layout()
    traj_path = out_path.parent / out_path.name.replace(".png", "_traj.png")
    fig1.savefig(traj_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved {traj_path}")

    # Figure 2: (c) d_min time history standalone
    fig2, ax = plt.subplots(figsize=(7, 3.5))
    n_steps = len(res_none["d_min"])
    t = np.arange(n_steps) * dt
    ax.plot(t, res_none["d_min"] / 1000, lw=2, ls="--", c="#ff7f0e",
            label=r"$\gamma$=0")
    ax.plot(t, res_comm["d_min"] / 1000, lw=2, c="#2ca02c",
            label=rf"$\gamma$={gamma_val} (adaptive)")
    if d_safe > 0:
        ax.axhline(d_safe / 1000, color="red", linestyle=":", lw=1.5,
                   label=rf"$d_{{safe}}$ = {d_safe/1000:.1f} km")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(r"$d_{\min}$ (km)", fontsize=11)
    ax.set_title("Inter-pursuer distance", fontsize=12)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    fig2.tight_layout()
    dmin_path = out_path.parent / out_path.name.replace(".png", "_dmin.png")
    fig2.savefig(dmin_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {dmin_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else (
        base / "outputs" / f"collision_demo_{ts}")
    fig_dir = base / "paper_figures"

    # ── Scenario ──
    # True crossing-path: each pursuer starts on the OPPOSITE side of its
    # target and must fly THROUGH y=0 to reach it.
    #
    # P1 starts at y=+1200, target is y=-800 (below E) → must drop 2000m
    # P2 starts at y=-1200, target is y=+800 (above E) → must rise 2000m
    # Their paths physically cross near y=0.
    #
    # Without coordination: actual collision (d_min ≈ 0 m) at crossing.
    # With coordination: delta_jk drives early lateral separation.
    p_init = np.array([
        [0.0,  1200.0, 0.0, 0.0],  # P1: starts high, target is below E
        [0.0, -1200.0, 0.0, 0.0],  # P2: starts low,  target is above E
    ])
    e_init = np.array([4000.0, 0.0, 5.0, 0.0])

    # r1 = (0, +800) → target = E - r1 = (4000, -800)
    # r2 = (0, -800) → target = E - r2 = (4000, +800)
    # P1 must cross from y=1200 to y=-800, P2 from y=-1200 to y=+800
    displacements = np.array([
        [0.0, +800.0],
        [0.0, -800.0],
    ])

    dt = 0.05
    t_final = 120.0

    print("=== Controlled Scenario (paper-consistent V-SNAC) ===\n")
    print("  P1: starts high, targets below E.  P2: starts low, targets above E.")
    print(f"  delta_12 = r2-r1 = (0, {displacements[1,1]-displacements[0,1]:.0f}) m\n")

    # ── Adaptive gamma with d_safe ──
    d_safe = 500.0  # safety distance (m)
    best_gamma = 5.0

    print(f"  d_safe = {d_safe:.0f} m\n")

    # Sweep with adaptive gamma
    sweep = []
    for g in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        res = simulate(p_init, e_init, displacements,
                       gamma=g, d_safe=d_safe, dt=dt, t_final=t_final)
        sep_km = res["min_d_min"] / 1000.0
        sweep.append((g, sep_km))
        tag = " *" if res["min_d_min"] >= d_safe else ""
        print(f"  gamma={g:5.1f}:  min d_min = {res['min_d_min']:8.1f} m  "
              f"({sep_km:.2f} km){tag}")

    # ── Detailed comparison ──
    res_none = simulate(p_init, e_init, displacements,
                        gamma=0.0, dt=dt, t_final=t_final)
    res_comm = simulate(p_init, e_init, displacements,
                        gamma=best_gamma, d_safe=d_safe, dt=dt, t_final=t_final)

    print(f"\n  No comm:              min d_min = {res_none['min_d_min']:.1f} m")
    print(f"  Adaptive gamma={best_gamma}:  min d_min = {res_comm['min_d_min']:.1f} m")
    print(f"  d_safe guarantee:     {res_comm['min_d_min'] >= d_safe}")

    # ── Generate figures ──
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_demo(res_none, res_comm, best_gamma, dt,
              out_dir / "fig_collision_comparison.png", d_safe=d_safe)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_demo(res_none, res_comm, best_gamma, dt,
              fig_dir / "fig_collision_comparison.png", d_safe=d_safe)

    # ── Print Table I data ──
    print("\n  Table I (for paper):")
    print("  gamma | Sep (km)")
    print("  ------+---------")
    for g, s in sweep:
        print(f"  {g:5.1f} | {s:.2f}")

    print(f"\n  Output: {out_dir}")


if __name__ == "__main__":
    main()
