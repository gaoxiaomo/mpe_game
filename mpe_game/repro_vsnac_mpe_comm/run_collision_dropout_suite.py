"""Collision-prone scenario suite under partial communication loss.

Runs each scenario in ``mpe_repro.collision_scenarios`` (C1..C6) under a
matrix of communication conditions:

- ``no_comm``        : gamma=0 baseline (collision floor)
- ``full_comm``      : gamma=gamma_default, no dropout (separation ceiling)
- ``iid_15/30/50``   : per-edge per-step Bernoulli dropout, 5 seeds each
- ``persistent_15``  : permanent random subset of edges removed, 5 seeds each
- ``persistent_30``  : permanent random subset of edges removed, 5 seeds each
- ``periodic_off25`` : duty-cycle outage, 25% off in 2s window
- ``periodic_off50`` : duty-cycle outage, 50% off in 2s window

For each (scenario, mode) pair the script records the d_min(t) curve, the
minimum d_min over the episode, the mean assigned tracking error, the
formation error, and the (sustained) capture time. Multi-seed results are
aggregated to median and inter-quartile range.

Outputs:
    summary.json          : full nested summary
    REPORT.md             : human-readable rollup
    fig_dmin_grid.png     : 6-panel d_min(t) per scenario
    fig_degradation.png   : x = effective dropout rate, y = min d_min
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from mpe_repro.collision_scenarios import COLLISION_SCENARIO_BUILDERS
from mpe_repro.comm_graph import (
    DropoutPattern,
    IIDBernoulliDropout,
    PeriodicOutageDropout,
    PersistentEdgeDropout,
)
from mpe_repro.config import (
    AircraftParams,
    CommParams,
    ControlParams,
    FeatureParams,
    LearningParams,
)
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.simulator import EvalResult, MPECommSimulator


# Two distinct distance scales appear in the analysis (the split is preserved
# as an abstraction so the two values can be tuned independently if needed):
#   * d_safe_report (NMAC, ~150 m) is the regulatory safety threshold used
#     for reporting "time below safety" and the plot reference line. It comes
#     from FAA AIM 7-7-3 (500 ft NMAC).
#   * d_safe_amp is the amplification scale inside the smooth pair-amp factor
#     of the coordination potential. Set equal to d_safe_report by default so
#     Phi is already amplifying as soon as pursuers enter the regulatory
#     unsafe band; this gives early lateral separation rather than waiting
#     for the inner-most close-approach zone.
DEFAULT_D_SAFE_REPORT = 150.0
DEFAULT_D_SAFE_AMP = 150.0
# Coordination strength. The paper sweeps gamma in [0.5, 10] for collision
# figures. gamma=1.5 is the smallest value that pushes the 2v1 / stacked-pair
# cases (C1, C3, C4) cleanly above d_safe = 150 m. Larger gamma over-saturates
# 2v1 geometries; smaller gamma compresses the full-comm vs dropout gap. For
# the larger 3v1 / 5v2 groups (C2, C5, C6), per-pair coordination is weaker
# due to degree normalisation in Phi, so absolute clearance is smaller while
# the qualitative full-comm > dropout > no-comm ordering is preserved.
DEFAULT_GAMMA = 1.5
DEFAULT_DROPOUT_SEEDS = (101, 211, 307, 419, 503)


# ---------------------------------------------------------------------------
# Hyperparameters (kept independent of run_comm.py to avoid coupling)
# ---------------------------------------------------------------------------


def _feature_params() -> FeatureParams:
    return FeatureParams(
        state_scale=(3200.0, 3200.0, 2200.0, 160.0, 160.0, 160.0),
        feature_gain=2.7,
    )


def _control_params() -> ControlParams:
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


def _learning_params(quick: bool) -> LearningParams:
    if quick:
        return LearningParams(
            policy_iterations=14,
            rollout_steps=120,
            min_samples_per_evader=50,
            graph_update_interval=1,
            graph_update_start_step=0,
            critic_learning_rate=0.05,
            critic_lr_decay=0.85,
            convergence_tol=5e-4,
            random_perturb_scale=0.01,
        )
    return LearningParams(
        policy_iterations=30,
        rollout_steps=180,
        min_samples_per_evader=50,
        graph_update_interval=1,
        graph_update_start_step=0,
        critic_learning_rate=0.05,
        critic_lr_decay=0.85,
        convergence_tol=2e-4,
        random_perturb_scale=0.015,
    )


# ---------------------------------------------------------------------------
# Mode definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeSpec:
    name: str            # short tag e.g. "iid_15"
    pretty: str          # human-readable label
    gamma: float         # 0 if no_comm, otherwise gamma_default
    comm_mode: str       # "full" or "none"
    pattern_kind: str    # "none", "iid", "persistent", "periodic"
    pattern_param: float # iid/persistent: prob; periodic: off_fraction
    n_seeds: int         # 1 for deterministic, multiple for stochastic
    effective_loss: float  # for plotting on the degradation curve


def _build_modes(gamma: float, n_stoch_seeds: int) -> list[ModeSpec]:
    """Define the evaluation matrix.

    The ``effective_loss`` column gives an interpretable scalar for the x-axis
    of the degradation curve. For iid Bernoulli p, that's just p. For
    persistent-edge p, it equals p (fraction of edges removed). For periodic
    off-fraction f, the time-averaged loss is f.
    """
    return [
        ModeSpec("no_comm",        "No comm (gamma=0)",          0.0,   "none", "none",        0.0,  1,              1.00),
        ModeSpec("full_comm",      "Full comm",                  gamma, "full", "none",        0.0,  1,              0.00),
        ModeSpec("iid_15",         "IID dropout 15%",            gamma, "full", "iid",         0.15, n_stoch_seeds,  0.15),
        ModeSpec("iid_30",         "IID dropout 30%",            gamma, "full", "iid",         0.30, n_stoch_seeds,  0.30),
        ModeSpec("iid_50",         "IID dropout 50%",            gamma, "full", "iid",         0.50, n_stoch_seeds,  0.50),
        ModeSpec("persistent_15",  "Persistent edge 15%",        gamma, "full", "persistent",  0.15, n_stoch_seeds,  0.15),
        ModeSpec("persistent_30",  "Persistent edge 30%",        gamma, "full", "persistent",  0.30, n_stoch_seeds,  0.30),
        ModeSpec("periodic_off25", "Periodic outage off=25%",    gamma, "full", "periodic",    0.25, 1,              0.25),
        ModeSpec("periodic_off50", "Periodic outage off=50%",    gamma, "full", "periodic",    0.50, 1,              0.50),
    ]


def _make_pattern(kind: str, param: float, seed: int) -> DropoutPattern | None:
    if kind == "none":
        return None
    if kind == "iid":
        return IIDBernoulliDropout(prob=param)
    if kind == "persistent":
        return PersistentEdgeDropout(prob=param, seed=seed)
    if kind == "periodic":
        return PeriodicOutageDropout(period_s=2.0, off_fraction=param)
    raise ValueError(f"unknown pattern kind: {kind}")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _sustained_capture_time(
    assigned_errors: np.ndarray,
    capture_radius: float,
    dt: float,
    streak_steps: int = 6,
) -> float | None:
    """Earliest time at which max assigned error stays <= capture_radius for
    ``streak_steps`` consecutive steps."""
    if assigned_errors.size == 0:
        return None
    max_err = np.max(assigned_errors, axis=1)
    inside = max_err <= capture_radius
    streak = 0
    for t in range(inside.size):
        if inside[t]:
            streak += 1
            if streak >= streak_steps:
                return float((t - streak_steps + 1) * dt)
        else:
            streak = 0
    return None


def _unsafe_metrics(d_min_hist: np.ndarray, d_safe: float, dt: float) -> dict[str, float]:
    if d_min_hist.size == 0:
        return {"min_d_min": float("nan"), "time_below_d_safe_s": 0.0, "longest_unsafe_s": 0.0}
    below = d_min_hist < d_safe
    total = float(np.sum(below) * dt)
    longest = 0
    cur = 0
    for flag in below.tolist():
        if flag:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return {
        "min_d_min": float(np.min(d_min_hist)),
        "time_below_d_safe_s": total,
        "longest_unsafe_s": float(longest * dt),
    }


def _evaluate_mode(
    sim: MPECommSimulator,
    weights: list[np.ndarray],
    spec: ModeSpec,
    seed_list: tuple[int, ...],
    d_safe_report: float,
    d_safe_amp: float,
) -> dict[str, Any]:
    """Run all seeds for one mode and aggregate."""
    seeds_to_run = seed_list if spec.n_seeds > 1 else (seed_list[0],)
    runs: list[dict[str, Any]] = []
    d_min_curves: list[np.ndarray] = []

    for seed_idx, seed in enumerate(seeds_to_run[: spec.n_seeds]):
        cp = CommParams(
            gamma=spec.gamma,
            comm_mode=spec.comm_mode,
            formation_ref_dist=500.0,
            dropout_prob=0.0,
            dropout_seed=seed,
            d_safe=d_safe_amp,
        )
        pattern = _make_pattern(spec.pattern_kind, spec.pattern_param, seed)
        eval_seed = 1000 + seed_idx
        result = sim.evaluate_policy(
            weights=weights,
            seed=eval_seed,
            dynamic_graph=sim.scenario.n_evaders > 1,
            stop_on_capture=False,
            zero_tail_after_capture=False,
            record_logs=False,
            comm_params_override=cp,
            dropout_pattern=pattern,
        )
        d_min_hist = np.asarray(result.d_min_history, dtype=float)
        d_min_curves.append(d_min_hist)
        unsafe = _unsafe_metrics(d_min_hist, d_safe_report, sim.learning.dt)
        capture_time = _sustained_capture_time(
            np.asarray(result.assigned_errors), sim.scenario.capture_radius, sim.learning.dt
        )
        runs.append(
            {
                "seed": int(seed),
                "min_d_min_m": unsafe["min_d_min"],
                "time_below_d_safe_s": unsafe["time_below_d_safe_s"],
                "longest_unsafe_s": unsafe["longest_unsafe_s"],
                "capture_time_s": capture_time,
                "mean_assigned_error": float(np.mean(result.assigned_errors)),
                "mean_formation_error": float(np.mean(result.formation_errors)),
            }
        )

    # Aggregate d_min curves to (median, q25, q75) on a common length grid
    L = min(c.size for c in d_min_curves)
    stacked = np.stack([c[:L] for c in d_min_curves], axis=0)
    median = np.median(stacked, axis=0)
    q25 = np.quantile(stacked, 0.25, axis=0)
    q75 = np.quantile(stacked, 0.75, axis=0)

    min_d_min_arr = np.array([r["min_d_min_m"] for r in runs], dtype=float)
    time_below_arr = np.array([r["time_below_d_safe_s"] for r in runs], dtype=float)

    return {
        "mode": spec.name,
        "pretty": spec.pretty,
        "effective_loss": spec.effective_loss,
        "n_seeds": int(spec.n_seeds),
        "runs": runs,
        "median_min_d_min_m": float(np.median(min_d_min_arr)),
        "iqr_min_d_min_m": [float(np.quantile(min_d_min_arr, 0.25)), float(np.quantile(min_d_min_arr, 0.75))],
        "median_time_below_d_safe_s": float(np.median(time_below_arr)),
        "d_min_curve": {
            "t_s": (np.arange(L) * sim.learning.dt).tolist(),
            "median": median.tolist(),
            "q25": q25.tolist(),
            "q75": q75.tolist(),
        },
    }


def _run_one_scenario(
    tag: str,
    *,
    out_dir: Path,
    quick: bool,
    gamma: float,
    d_safe_report: float,
    d_safe_amp: float,
    n_stoch_seeds: int,
    train_seed: int,
) -> dict[str, Any]:
    builder = COLLISION_SCENARIO_BUILDERS[tag]
    scenario = builder()
    control = _control_params()
    learning = _learning_params(quick)
    feature = _feature_params()

    train_cp = CommParams(
        gamma=gamma,
        comm_mode="full",
        formation_ref_dist=500.0,
        d_safe=d_safe_amp,
    )
    sim = MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        comm_params=train_cp,
    )

    train_t0 = time.perf_counter()
    train = sim.train_policy(seed=train_seed, dynamic_graph=scenario.n_evaders > 1)
    train_wall_s = time.perf_counter() - train_t0

    modes = _build_modes(gamma=gamma, n_stoch_seeds=n_stoch_seeds)
    mode_results = []
    for spec in modes:
        t0 = time.perf_counter()
        mr = _evaluate_mode(
            sim=sim,
            weights=train.weights,
            spec=spec,
            seed_list=DEFAULT_DROPOUT_SEEDS,
            d_safe_report=d_safe_report,
            d_safe_amp=d_safe_amp,
        )
        mr["wall_s"] = float(time.perf_counter() - t0)
        mode_results.append(mr)

    scenario_summary = {
        "tag": tag,
        "scenario_name": scenario.name,
        "n_pursuers": int(scenario.n_pursuers),
        "n_evaders": int(scenario.n_evaders),
        "t_final_s": float(scenario.t_final),
        "capture_radius_m": float(scenario.capture_radius),
        "d_safe_report_m": float(d_safe_report),
        "d_safe_amp_m": float(d_safe_amp),
        "gamma": float(gamma),
        "train_wall_s": float(train_wall_s),
        "modes": mode_results,
    }

    case_dir = out_dir / tag
    case_dir.mkdir(parents=True, exist_ok=True)
    dump_json(case_dir / "summary.json", scenario_summary)
    return scenario_summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


_MODE_COLORS = {
    "no_comm":        "#d62728",  # red
    "full_comm":      "#2ca02c",  # green
    "iid_15":         "#1f77b4",
    "iid_30":         "#1f77b4",
    "iid_50":         "#1f77b4",
    "persistent_15":  "#9467bd",
    "persistent_30":  "#9467bd",
    "periodic_off25": "#ff7f0e",
    "periodic_off50": "#ff7f0e",
}
_MODE_ALPHAS = {
    "iid_15":         0.85,
    "iid_30":         0.55,
    "iid_50":         0.30,
    "persistent_15":  0.85,
    "persistent_30":  0.55,
    "periodic_off25": 0.85,
    "periodic_off50": 0.55,
}
_MODE_LINESTYLES = {
    "no_comm":        ":",
    "full_comm":      "-",
    "iid_15":         "-",
    "iid_30":         "-",
    "iid_50":         "-",
    "persistent_15":  "--",
    "persistent_30":  "--",
    "periodic_off25": "-.",
    "periodic_off50": "-.",
}


def _plot_dmin_grid(scenarios: list[dict[str, Any]], path: Path, d_safe: float) -> None:
    n = len(scenarios)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows))
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for idx, scenario in enumerate(scenarios):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for mode in scenario["modes"]:
            tag = mode["mode"]
            color = _MODE_COLORS.get(tag, "#444444")
            alpha = _MODE_ALPHAS.get(tag, 1.0)
            linestyle = _MODE_LINESTYLES.get(tag, "-")
            curve = mode["d_min_curve"]
            t = np.asarray(curve["t_s"], dtype=float)
            median = np.asarray(curve["median"], dtype=float)
            q25 = np.asarray(curve["q25"], dtype=float)
            q75 = np.asarray(curve["q75"], dtype=float)
            ax.plot(t, median, color=color, alpha=alpha, linestyle=linestyle, linewidth=1.6, label=mode["pretty"])
            if mode["n_seeds"] > 1:
                ax.fill_between(t, q25, q75, color=color, alpha=0.10)

        ax.axhline(d_safe, color="black", linewidth=0.8, linestyle=":")
        ax.axhspan(0.0, d_safe, color="#d62728", alpha=0.05)
        ax.set_title(f"{scenario['tag']}: {scenario['scenario_name']}", fontsize=10)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("d_min (m)")
        ax.grid(alpha=0.3)

    # Hide extra panels
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    # Single shared legend at the bottom
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_degradation_curve(scenarios: list[dict[str, Any]], path: Path, d_safe: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cmap = plt.get_cmap("tab10")
    for idx, scenario in enumerate(scenarios):
        xs: list[float] = []
        ys: list[float] = []
        for mode in scenario["modes"]:
            xs.append(mode["effective_loss"])
            ys.append(mode["median_min_d_min_m"])
        order = np.argsort(xs)
        xs_sorted = np.asarray(xs, dtype=float)[order]
        ys_sorted = np.asarray(ys, dtype=float)[order]
        ax.plot(
            xs_sorted,
            ys_sorted,
            marker="o",
            linewidth=2.0,
            color=cmap(idx % 10),
            label=f"{scenario['tag']}: {scenario['scenario_name']}",
        )
    ax.axhline(d_safe, color="black", linewidth=1.0, linestyle=":", label=f"d_safe = {d_safe:.0f} m (NMAC)")
    ax.set_xlabel("Effective communication-loss rate")
    ax.set_ylabel("Median min(d_min) over episode (m)")
    ax.set_title("Graceful-degradation curve across collision scenarios")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_report_md(batch: dict[str, Any]) -> str:
    lines = [
        "# Collision-suite dropout robustness report",
        "",
        f"- generated_at: {batch['generated_at']}",
        f"- d_safe_report (NMAC, 500 ft): {batch['d_safe_report_m']:.1f} m",
        f"- d_safe_amp (smooth-amp scale): {batch['d_safe_amp_m']:.1f} m",
        f"- gamma: {batch['gamma']}",
        f"- stochastic seeds per random mode: {batch['n_stoch_seeds']}",
        "",
        "## Per-scenario summary",
        "",
        "| Tag | Scenario | mode | median min(d_min) [m] | median time below d_safe [s] |",
        "|---|---|---|---:|---:|",
    ]
    for sc in batch["scenarios"]:
        for mode in sc["modes"]:
            lines.append(
                "| {tag} | {name} | {mode} | {dmin:.1f} | {below:.2f} |".format(
                    tag=sc["tag"],
                    name=sc["scenario_name"],
                    mode=mode["mode"],
                    dmin=mode["median_min_d_min_m"],
                    below=mode["median_time_below_d_safe_s"],
                )
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Collision-suite dropout robustness sweep.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--scenarios", type=str, nargs="*", default=None, help="Subset of scenario tags (default: all C1..C6)")
    parser.add_argument("--quick", action="store_true", help="Use shorter training schedule")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Coordination strength")
    parser.add_argument(
        "--d-safe-report",
        type=float,
        default=DEFAULT_D_SAFE_REPORT,
        help="Reporting safety threshold (m); FAA NMAC = 152.4 m, default 150 m",
    )
    parser.add_argument(
        "--d-safe-amp",
        type=float,
        default=DEFAULT_D_SAFE_AMP,
        help="Amplification scale (m) inside the smooth pair-amplification factor",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per stochastic mode")
    parser.add_argument("--train-seed", type=int, default=7)
    args = parser.parse_args()

    tags = args.scenarios if args.scenarios else list(COLLISION_SCENARIO_BUILDERS.keys())
    for t in tags:
        if t not in COLLISION_SCENARIO_BUILDERS:
            raise SystemExit(f"unknown scenario tag: {t}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"collision_dropout_suite_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[suite] running {len(tags)} scenarios: {tags}")
    print(
        f"[suite] gamma={args.gamma} "
        f"d_safe_report={args.d_safe_report} d_safe_amp={args.d_safe_amp} "
        f"seeds={args.seeds} quick={args.quick}"
    )
    print(f"[suite] output: {out_dir}")

    scenario_summaries: list[dict[str, Any]] = []
    suite_t0 = time.perf_counter()
    for tag in tags:
        t0 = time.perf_counter()
        summary = _run_one_scenario(
            tag=tag,
            out_dir=out_dir,
            quick=bool(args.quick),
            gamma=float(args.gamma),
            d_safe_report=float(args.d_safe_report),
            d_safe_amp=float(args.d_safe_amp),
            n_stoch_seeds=int(args.seeds),
            train_seed=int(args.train_seed),
        )
        scenario_summaries.append(summary)
        print(
            "[suite] {tag} done in {sec:.1f}s "
            "(no_comm min={no_dm:.0f}m, full_comm min={full_dm:.0f}m)".format(
                tag=tag,
                sec=time.perf_counter() - t0,
                no_dm=next(m["median_min_d_min_m"] for m in summary["modes"] if m["mode"] == "no_comm"),
                full_dm=next(m["median_min_d_min_m"] for m in summary["modes"] if m["mode"] == "full_comm"),
            )
        )

    suite_wall = time.perf_counter() - suite_t0

    batch = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "d_safe_report_m": float(args.d_safe_report),
        "d_safe_amp_m": float(args.d_safe_amp),
        "gamma": float(args.gamma),
        "n_stoch_seeds": int(args.seeds),
        "quick": bool(args.quick),
        "wall_s": float(suite_wall),
        "scenarios": scenario_summaries,
    }
    dump_json(out_dir / "batch_summary.json", batch)
    dump_markdown(out_dir / "REPORT.md", _build_report_md(batch))

    _plot_dmin_grid(scenario_summaries, out_dir / "fig_dmin_grid.png", float(args.d_safe_report))
    _plot_degradation_curve(scenario_summaries, out_dir / "fig_degradation.png", float(args.d_safe_report))
    print(f"[suite] all done in {suite_wall:.1f}s. results: {out_dir}")


if __name__ == "__main__":
    main()
