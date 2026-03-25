"""Collision-risk demonstration: pursuers must cross paths to reach targets.

P1 starts on the RIGHT, target displacement is on the LEFT of the evader.
P2 starts on the LEFT, target displacement is on the RIGHT.
Without communication, their optimal paths cross → near-collision.
With communication, the formation gradient spreads them apart.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
from mpe_repro.simulator import MPECommSimulator


def _build_cross_scenario() -> ScenarioConfig:
    """Two pursuers that must swap sides to capture the evader."""
    # Evader ahead, moving slowly
    evaders = np.array([[0.0, 3000.0, 300.0, 5.0, 12.0, 1.0]], dtype=float)

    # THREE pursuers start spread but all CONVERGE toward the evader.
    # Their approach paths naturally overlap in the middle zone.
    pursuers = np.array(
        [
            [600.0, -1200.0, 350.0, -15.0, 65.0, 5.0],   # P1: right
            [-600.0, -1200.0, 250.0, 15.0, 65.0, 3.0],   # P2: left
            [0.0, -1800.0, 300.0, 2.0, 70.0, 4.0],       # P3: behind
        ],
        dtype=float,
    )

    # Target displacements: P1 LEFT, P2 RIGHT, P3 ABOVE — triangular formation
    # Without comm they all take the same path; with comm they spread out
    displacements = np.zeros((3, 1, 6), dtype=float)
    displacements[0, 0, :3] = [-400.0, -50.0, 0.0]   # P1: left
    displacements[1, 0, :3] = [400.0, -50.0, 0.0]    # P2: right
    displacements[2, 0, :3] = [0.0, -50.0, 400.0]    # P3: above

    return ScenarioConfig(
        name="cross_2v1",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=1e9,
        max_switch_worsening=0.0,
        initial_assignment=np.array([0, 0, 0], dtype=int),
        t_final=80.0,
        capture_radius=200.0,
        evader_motion_mode="scripted",
        evader_script_amp=(4.0, 3.0, 1.5),
        evader_script_omega=0.3,
        evader_script_decay=0.04,
        evader_script_mix=0.8,
    )


def _plot_cross_comparison(
    ev_comm, ev_none, dt: float, path: Path, d_min_comm, d_min_none,
) -> None:
    """Side-by-side trajectory + d_min comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    p_colors = ["#d62728", "#1f77b4", "#2ca02c"]
    e_color = "#000000"

    # Panel 1: no communication trajectories
    ax = axes[0]
    ax.set_title(f"No communication (d_min = {d_min_none:.0f} m)", fontsize=11)
    for j in range(ev_none.pursuer_traj.shape[1]):
        traj = ev_none.pursuer_traj[:, j, :3]
        ax.plot(traj[:, 0], traj[:, 1], color=p_colors[j], linewidth=1.8, label=f"P{j+1}")
        ax.scatter(traj[0, 0], traj[0, 1], color=p_colors[j], s=80, marker="o", edgecolors="k", zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=p_colors[j], s=60, marker="s", zorder=5)
    traj_e = ev_none.evader_traj[:, 0, :3]
    ax.plot(traj_e[:, 0], traj_e[:, 1], color=e_color, linewidth=2.2, linestyle="--", label="Evader")
    ax.scatter(traj_e[0, 0], traj_e[0, 1], color=e_color, s=80, marker="x", zorder=5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # Panel 2: with communication trajectories
    ax = axes[1]
    ax.set_title(f"With communication (d_min = {d_min_comm:.0f} m)", fontsize=11)
    for j in range(ev_none.pursuer_traj.shape[1]):
        traj = ev_comm.pursuer_traj[:, j, :3]
        ax.plot(traj[:, 0], traj[:, 1], color=p_colors[j], linewidth=1.8, label=f"P{j+1}")
        ax.scatter(traj[0, 0], traj[0, 1], color=p_colors[j], s=80, marker="o", edgecolors="k", zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=p_colors[j], s=60, marker="s", zorder=5)
    traj_e = ev_comm.evader_traj[:, 0, :3]
    ax.plot(traj_e[:, 0], traj_e[:, 1], color=e_color, linewidth=2.2, linestyle="--", label="Evader")
    ax.scatter(traj_e[0, 0], traj_e[0, 1], color=e_color, s=80, marker="x", zorder=5)
    ax.set_xlabel("X (m)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    # Panel 3: d_min over time
    ax = axes[2]
    n_none = len(ev_none.d_min_history) if ev_none.d_min_history is not None else 0
    n_comm = len(ev_comm.d_min_history) if ev_comm.d_min_history is not None else 0
    if n_none > 0:
        t_none = np.arange(n_none) * dt
        ax.plot(t_none, ev_none.d_min_history, linewidth=1.8, color="#ff7f0e",
                linestyle="--", label="No comm")
    if n_comm > 0:
        t_comm = np.arange(n_comm) * dt
        ax.plot(t_comm, ev_comm.d_min_history, linewidth=1.8, color="#2ca02c",
                label="With comm")
    ax.axhline(100.0, color="red", linestyle=":", linewidth=1.2, alpha=0.7, label="Safety threshold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min inter-pursuer distance (m)")
    ax.set_title("d_min comparison", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Path-crossing scenario: collision avoidance via communication", fontsize=13, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "outputs" / f"cross_demo_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = _build_cross_scenario()
    ctrl = ControlParams(
        q_diag=(1, 1, 1, 1, 1, 1), r1_diag=(1, 1, 1), r2_diag=(1, 1, 1),
        u_bar_p=25, u_bar_e=15, u_bar_p_policy=35, u_bar_e_policy=15,
    )
    feat = FeatureParams(state_scale=(3200, 3200, 2200, 160, 160, 160), feature_gain=2.7)
    lp = LearningParams(
        policy_iterations=50, rollout_steps=200, min_samples_per_evader=50,
        critic_learning_rate=0.05, critic_lr_decay=0.85,
        convergence_tol=1e-5, random_perturb_scale=0.02,
    )

    # Train once (standard V-SNAC, gamma=0 during training)
    gamma = 0.5
    comm = CommParams(gamma=gamma, comm_mode="full", formation_ref_dist=500.0)
    sim = MPECommSimulator(
        scenario=scenario, aircraft_params=AircraftParams(),
        control_params=ctrl, learning_params=lp,
        feature_params=feat, comm_params=comm,
    )
    print("Training...")
    train = sim.train_policy(seed=42, dynamic_graph=False)

    # Eval: no communication
    print("Eval: no communication...")
    ev_none = sim.evaluate_policy(
        weights=train.weights, seed=1042, dynamic_graph=False,
        stop_on_capture=False,
        comm_params_override=CommParams(gamma=0.0, comm_mode="none"),
    )
    # Eval: with communication
    print("Eval: with communication...")
    ev_comm = sim.evaluate_policy(
        weights=train.weights, seed=1042, dynamic_graph=False,
        stop_on_capture=False,
        comm_params_override=CommParams(gamma=gamma, comm_mode="full", formation_ref_dist=500.0),
    )

    d_min_none = float(np.min(ev_none.d_min_history)) if ev_none.d_min_history is not None and len(ev_none.d_min_history) > 0 else 0.0
    d_min_comm = float(np.min(ev_comm.d_min_history)) if ev_comm.d_min_history is not None and len(ev_comm.d_min_history) > 0 else 0.0

    print(f"\n=== Results ===")
    print(f"No comm:   capture={ev_none.capture_time}, d_min={d_min_none:.0f} m")
    print(f"With comm: capture={ev_comm.capture_time}, d_min={d_min_comm:.0f} m")
    print(f"d_min improvement: {d_min_comm - d_min_none:+.0f} m ({(d_min_comm/max(d_min_none,1)-1)*100:+.0f}%)")

    _plot_cross_comparison(
        ev_comm, ev_none, lp.dt,
        out_dir / "fig_cross_comparison.png",
        d_min_comm, d_min_none,
    )

    # Also plot assigned errors for both modes
    from mpe_repro.plotting import plot_assigned_state_errors
    plot_assigned_state_errors(ev_none, lp.dt, out_dir / "fig_errors_nocomm.png", "cross 2v1 errors (no comm)")
    plot_assigned_state_errors(ev_comm, lp.dt, out_dir / "fig_errors_comm.png", "cross 2v1 errors (with comm)")

    print(f"\nPlots saved to: {out_dir}")


if __name__ == "__main__":
    main()
