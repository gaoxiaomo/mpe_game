from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import run_comm
from mpe_repro.config import AircraftParams, CommParams
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.simulator import EvalResult, MPECommSimulator
from run_many_to_many_collision_demo import (
    _build_6v3_crossing_scenario,
    _demo_learning_params,
    _run_method as run_collision_method,
)

matplotlib.use("Agg")


def sustained_capture_time(assigned_errors: np.ndarray, capture_radius: float, dt: float) -> float | None:
    """First time from which capture remains satisfied for the rest of the rollout."""
    if assigned_errors.size == 0:
        return None
    max_err = np.max(assigned_errors, axis=1)
    ok = max_err <= float(capture_radius)
    suffix_ok = np.logical_and.accumulate(ok[::-1])[::-1]
    idx = np.flatnonzero(suffix_ok)
    if idx.size == 0:
        return None
    return float(idx[0] * dt)


def relative_geometry_history(
    pursuer_traj: np.ndarray,
    assigned_targets: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Mean pairwise relative-displacement deviation over same-target pairs.

    For each time step t, define

        e_rel(t) = 1 / |G_t| sum_(j,k in G_t, j<k) || (p_j - p_k) - (r_k - r_j) ||_2

    where G_t contains pursuer pairs currently assigned to the same evader.
    Position channels only are used.
    """

    steps = min(pursuer_traj.shape[0], assigned_targets.shape[0])
    n_p = pursuer_traj.shape[1]
    hist = np.full(steps, np.nan, dtype=float)
    for t in range(steps):
        assignment = assigned_targets[t].astype(int, copy=False)
        pos = pursuer_traj[t, :, :3]
        r = np.asarray([displacements[j, assignment[j], :3] for j in range(n_p)], dtype=float)
        vals: list[float] = []
        for j in range(n_p):
            for k in range(j + 1, n_p):
                if int(assignment[j]) != int(assignment[k]):
                    continue
                # Tracking error is defined as x̃_j = x_j^p - x_e + r_j, so when
                # x̃_j → 0 and x̃_k → 0 for the same evader, we have
                # p_j - p_k → r_k - r_j.
                delta_jk = r[k] - r[j]
                vals.append(float(np.linalg.norm((pos[j] - pos[k]) - delta_jk)))
        if vals:
            hist[t] = float(np.mean(vals))
    return hist


def summarize_relative_geometry(
    result: EvalResult,
    displacements: np.ndarray,
) -> dict[str, Any]:
    hist = relative_geometry_history(
        pursuer_traj=result.pursuer_traj,
        assigned_targets=result.assigned_targets,
        displacements=displacements,
    )
    valid = hist[~np.isnan(hist)]
    return {
        "history": hist,
        "mean_relative_geometry_error": None if valid.size == 0 else float(np.mean(valid)),
        "final_relative_geometry_error": None if valid.size == 0 else float(valid[-1]),
    }


def build_standard_sim(case: run_comm.GeneralCase, gamma: float, d_safe: float, quick: bool = False) -> MPECommSimulator:
    scenario = run_comm.build_general_scenario(run_comm._scenario_spec(case, quick))
    return MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=run_comm._control_params(),
        learning_params=run_comm._learning_params(case.n_pursuers, case.n_evaders, quick),
        feature_params=run_comm._feature_params(),
        comm_params=run_comm._comm_params(gamma=gamma, comm_mode="full", d_safe=d_safe),
    )


def run_standard_case(case: run_comm.GeneralCase, gamma: float, d_safe: float) -> dict[str, Any]:
    sim = build_standard_sim(case, gamma=gamma, d_safe=d_safe, quick=False)
    dt = sim.learning.dt
    train = sim.train_policy(seed=case.seed, dynamic_graph=case.n_evaders > 1)
    eval_seed = case.seed + 1000
    full = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        dynamic_graph=case.n_evaders > 1,
        stop_on_capture=False,
        comm_params_override=CommParams(gamma=gamma, comm_mode="full", dropout_prob=0.0, d_safe=d_safe),
    )
    no = sim.evaluate_policy(
        weights=train.weights,
        seed=eval_seed,
        dynamic_graph=case.n_evaders > 1,
        stop_on_capture=False,
        comm_params_override=CommParams(gamma=0.0, comm_mode="none", dropout_prob=0.0, d_safe=0.0),
    )
    full_geom = summarize_relative_geometry(full, sim.displacements)
    no_geom = summarize_relative_geometry(no, sim.displacements)
    return {
        "case": asdict(case),
        "gamma": gamma,
        "d_safe": d_safe,
        "full": {
            "capture_time_s": sustained_capture_time(full.assigned_errors, sim.scenario.capture_radius, dt),
            "mean_assigned_error": float(np.mean(full.assigned_errors)),
            "min_d_min": float(np.min(full.d_min_history)),
            "mean_relative_geometry_error": full_geom["mean_relative_geometry_error"],
            "final_relative_geometry_error": full_geom["final_relative_geometry_error"],
        },
        "no": {
            "capture_time_s": sustained_capture_time(no.assigned_errors, sim.scenario.capture_radius, dt),
            "mean_assigned_error": float(np.mean(no.assigned_errors)),
            "min_d_min": float(np.min(no.d_min_history)),
            "mean_relative_geometry_error": no_geom["mean_relative_geometry_error"],
            "final_relative_geometry_error": no_geom["final_relative_geometry_error"],
        },
        "full_history": full_geom["history"].tolist(),
        "no_history": no_geom["history"].tolist(),
        "dt": sim.learning.dt,
    }


def plot_geometry_compare(
    t: np.ndarray,
    y_full: np.ndarray,
    y_no: np.ndarray,
    title: str,
    path: Path,
    full_label: str = "Full communication",
    no_label: str = "No communication",
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(t, y_full, linewidth=2.1, color="#1f77b4", label=full_label)
    ax.plot(t, y_no, linewidth=2.1, color="#d62728", linestyle="--", label=no_label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean relative displacement deviation (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def collision_geometry_summary(result: EvalResult, displacements: np.ndarray) -> dict[str, Any]:
    geom = summarize_relative_geometry(result, displacements)
    return {
        "capture_time_s": sustained_capture_time(result.assigned_errors, 180.0, _demo_learning_params().dt),
        "min_d_min": float(np.min(result.d_min_history)),
        "mean_relative_geometry_error": geom["mean_relative_geometry_error"],
        "final_relative_geometry_error": geom["final_relative_geometry_error"],
        "history": geom["history"].tolist(),
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    lines = [
        "# Relative Geometry Metrics",
        "",
        "## Metric",
        "The relative-geometry metric is defined over same-target pursuer pairs.",
        "",
        "e_rel(t)=1/|G_t| sum_(j,k in G_t, j<k) ||(p_j(t)-p_k(t))-(r_k(t)-r_j(t))||_2",
        "",
        "where G_t contains pursuer pairs assigned to the same evader at time t.",
        "",
        "## Standard Cases",
        "",
        "| Case | full sustained capture/s | no sustained capture/s | full mean assigned err | no mean assigned err | full final rel-geom err/m | no final rel-geom err/m | full min d_min/m | no min d_min/m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["standard_cases"]:
        case_name = f"{item['case']['n_pursuers']}v{item['case']['n_evaders']}"
        lines.append(
            f"| {case_name} | "
            f"{fmt(item['full']['capture_time_s'])} | {fmt(item['no']['capture_time_s'])} | "
            f"{item['full']['mean_assigned_error']:.3f} | {item['no']['mean_assigned_error']:.3f} | "
            f"{fmt(item['full']['final_relative_geometry_error'])} | "
            f"{fmt(item['no']['final_relative_geometry_error'])} | "
            f"{item['full']['min_d_min']:.2f} | {item['no']['min_d_min']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Collision Case",
            "",
            "| Method | sustained capture/s | final rel-geom err/m | min d_min/m |",
            "|---|---:|---:|---:|",
        ]
    )
    col = payload["collision_case"]
    lines.append(
        f"| no analytic | {fmt(col['no_analytic']['capture_time_s'])} | "
        f"{fmt(col['no_analytic']['final_relative_geometry_error'])} | {col['no_analytic']['min_d_min']:.3f} |"
    )
    lines.append(
        f"| smooth analytic | {fmt(col['smooth_analytic']['capture_time_s'])} | "
        f"{fmt(col['smooth_analytic']['final_relative_geometry_error'])} | {col['smooth_analytic']['min_d_min']:.3f} |"
    )
    return "\n".join(lines)


def main() -> None:
    out_dir = Path("outputs/relative_geometry_metrics_final")
    out_dir.mkdir(parents=True, exist_ok=True)

    standard_cases = [
        (run_comm.GeneralCase(3, 1, seed=35, assignment_mode="zero", layout_mode="structured"), 0.1),
        (run_comm.GeneralCase(3, 3, seed=17, assignment_mode="shifted", layout_mode="structured"), 0.3),
        (run_comm.GeneralCase(5, 3, seed=27, assignment_mode="shifted", layout_mode="structured"), 0.3),
        (run_comm.GeneralCase(6, 3, seed=37, assignment_mode="shifted", layout_mode="structured"), 0.3),
        (run_comm.GeneralCase(8, 4, seed=25, assignment_mode="shifted", layout_mode="structured"), 0.3),
    ]

    standard_payload: list[dict[str, Any]] = []
    for case, gamma in standard_cases:
        item = run_standard_case(case, gamma=gamma, d_safe=100.0)
        standard_payload.append(item)
        if case.name in {"3v1", "8v4"}:
            dt = item["dt"]
            t = np.arange(len(item["full_history"])) * dt
            plot_geometry_compare(
                t=t,
                y_full=np.asarray(item["full_history"], dtype=float),
                y_no=np.asarray(item["no_history"], dtype=float),
                title=f"{case.name} relative geometry deviation",
                path=out_dir / f"{case.name}_relative_geometry_compare.png",
            )

    collision_scenario = _build_6v3_crossing_scenario()
    eval_no, _ = run_collision_method(collision_scenario, gamma=0.0, d_safe=50.0, train_seed=2026, eval_seed=3026)
    eval_yes, _ = run_collision_method(collision_scenario, gamma=0.3, d_safe=50.0, train_seed=2026, eval_seed=3026)
    collision_no = collision_geometry_summary(eval_no, collision_scenario.displacement_matrix)
    collision_yes = collision_geometry_summary(eval_yes, collision_scenario.displacement_matrix)
    dt_col = _demo_learning_params().dt
    t_col = np.arange(len(collision_no["history"])) * dt_col
    plot_geometry_compare(
        t=t_col,
        y_full=np.asarray(collision_yes["history"], dtype=float),
        y_no=np.asarray(collision_no["history"], dtype=float),
        title="6v3 crossing relative geometry deviation",
        path=out_dir / "6v3_collision_relative_geometry_compare.png",
        full_label="Smooth analytic term",
        no_label="No analytic term",
    )

    payload = {
        "metric_definition": {
            "name": "mean_relative_geometry_error",
            "description": "Average same-group pairwise displacement deviation relative to the tracking-consistent reference offsets.",
        },
        "standard_cases": standard_payload,
        "collision_case": {
            "no_analytic": collision_no,
            "smooth_analytic": collision_yes,
        },
    }

    dump_json(out_dir / "relative_geometry_summary.json", payload)
    dump_markdown(out_dir / "REPORT.md", build_markdown_report(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
