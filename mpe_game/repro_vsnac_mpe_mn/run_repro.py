from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from mpe_repro.config import (
    AircraftParams,
    ControlParams,
    FeatureParams,
    LearningParams,
    ScenarioConfig,
    scenario_three_pursuer_one_evader,
    scenario_three_pursuer_three_evader,
)
from mpe_repro.plotting import (
    plot_assignment_timeline,
    plot_assigned_xyz_errors,
    plot_assigned_state_errors,
    plot_control_inputs,
    plot_old_new_errors,
    plot_saturation_comparison,
    plot_saturation_xyz_comparison,
    plot_team_error_compare,
    plot_trajectory_animation,
    plot_trajectory_3d,
    plot_trajectory_multiview,
    plot_weight_convergence,
)
from mpe_repro.report import dump_json, dump_markdown, eval_summary, network_summary, train_summary
from mpe_repro.report import dump_eval_log_markdown, dump_train_log_markdown
from mpe_repro.simulator import EvalResult, MPESimulator


def xy_distance_matrix(pursuers: np.ndarray, evaders: np.ndarray) -> np.ndarray:
    d = np.zeros((pursuers.shape[0], evaders.shape[0]), dtype=float)
    for j, p in enumerate(pursuers):
        for i, e in enumerate(evaders):
            d[j, i] = float(np.linalg.norm(p[:2] - e[:2]))
    return d


def team_error_series_with_fixed_assignment(
    pursuer_traj: np.ndarray,
    evader_traj: np.ndarray,
    assignment: np.ndarray,
    displacements: np.ndarray,
    nu: np.ndarray,
) -> np.ndarray:
    n_steps = min(pursuer_traj.shape[0], evader_traj.shape[0]) - 1
    out = np.zeros(n_steps, dtype=float)
    for t in range(n_steps):
        total = 0.0
        for j in range(pursuer_traj.shape[1]):
            i = int(assignment[j])
            diff = pursuer_traj[t, j] - evader_traj[t, i] + displacements[j, i]
            total += float(np.linalg.norm(nu * diff))
        out[t] = total
    return out


def switch_events(assigned_targets: np.ndarray, initial_assignment: np.ndarray, dt: float) -> list[float]:
    if assigned_targets.size == 0:
        return []
    events: list[int] = []
    if np.any(assigned_targets[0] != initial_assignment):
        events.append(0)
    changes = np.any(assigned_targets[1:] != assigned_targets[:-1], axis=1)
    events.extend((np.where(changes)[0] + 1).tolist())
    return [float(i * dt) for i in events]


def run_case(
    scenario: ScenarioConfig,
    control: ControlParams,
    learning: LearningParams,
    feature: FeatureParams,
    seed: int,
    dynamic_graph_train: bool,
    dynamic_graph_eval: bool,
    stop_on_capture: bool = True,
    record_logs: bool = False,
) -> tuple:
    sim = MPESimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
    )
    train = sim.train_policy(seed=seed, dynamic_graph=dynamic_graph_train)
    eval_result = sim.evaluate_policy(
        weights=train.weights,
        seed=seed + 1,
        dynamic_graph=dynamic_graph_eval,
        stop_on_capture=stop_on_capture,
        record_logs=record_logs,
    )
    return train, eval_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce V-SNAC MPE paper experiments.")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Short run for quick validation")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else project_root / "outputs" / f"repro_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_3v1 = FeatureParams(
        state_scale=(2600.0, 2600.0, 2600.0, 150.0, 150.0, 150.0),
        feature_gain=2.8,
    )
    feature_3v3 = FeatureParams(
        state_scale=(4200.0, 4200.0, 2800.0, 180.0, 180.0, 180.0),
        feature_gain=2.6,
    )
    learning = LearningParams(
        policy_iterations=18 if args.quick else 40,
        rollout_steps=100 if args.quick else 200,
        min_samples_per_evader=160 if args.quick else 400,
    )
    learning_3v1 = LearningParams(
        policy_iterations=36 if args.quick else 110,
        rollout_steps=110 if args.quick else 220,
        min_samples_per_evader=220 if args.quick else 520,
        convergence_tol=1e-5,
        critic_learning_rate=0.01,
        exploration_std_start=0.5,
        exploration_std_end=0.05,
    )

    s1 = scenario_three_pursuer_one_evader()
    c1 = ControlParams(
        q_diag=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        r1_diag=(1.0, 1.0, 1.0),
        r2_diag=(1.0, 1.0, 1.0),
        u_bar_p=25.0,
        u_bar_e=15.0,
        u_bar_p_policy=35.0,
        u_bar_e_policy=15.0,
        policy_gain=35.0,
        k_pos_p=0.02,
        k_vel_p=0.16,
        k_pos_e=0.01,
        k_vel_e=0.08,
    )
    train1, eval1 = run_case(
        scenario=s1,
        control=c1,
        learning=learning_3v1,
        feature=feature_3v1,
        seed=7,
        dynamic_graph_train=False,
        dynamic_graph_eval=False,
        record_logs=True,
    )

    s2 = scenario_three_pursuer_three_evader()
    c2 = ControlParams(
        q_diag=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        r1_diag=(1.0, 1.0, 1.0),
        r2_diag=(1.0, 1.0, 1.0),
        u_bar_p=20.0,
        u_bar_e=20.0,
        policy_gain=28.0,
        k_pos_p=0.015,
        k_vel_p=0.12,
        k_pos_e=0.0,
        k_vel_e=0.0,
    )
    learning_3v3 = LearningParams(
        policy_iterations=30 if args.quick else 90,
        rollout_steps=100 if args.quick else 240,
        min_samples_per_evader=160 if args.quick else 520,
        graph_update_interval=1,
        graph_update_start_step=0,
        critic_learning_rate=0.01,
    )
    # Strict apples-to-apples 3v3 comparison:
    # same trained weights + same initial state + same eval seed,
    # only dynamic_graph flag differs.
    sim2 = MPESimulator(
        scenario=s2,
        aircraft_params=AircraftParams(),
        control_params=c2,
        learning_params=learning_3v3,
        feature_params=feature_3v3,
    )
    train2 = sim2.train_policy(seed=13, dynamic_graph=True)
    eval_seed_3v3 = 113
    eval2_dynamic = sim2.evaluate_policy(
        weights=train2.weights,
        seed=eval_seed_3v3,
        dynamic_graph=True,
        stop_on_capture=True,
        record_logs=True,
    )
    eval2_fixed = sim2.evaluate_policy(
        weights=train2.weights,
        seed=eval_seed_3v3,
        dynamic_graph=False,
        stop_on_capture=True,
        record_logs=True,
    )

    # Fixed-reference evaluation under dynamic trajectory for cohesion contrast.
    fixed_ref_team = team_error_series_with_fixed_assignment(
        pursuer_traj=eval2_dynamic.pursuer_traj,
        evader_traj=eval2_dynamic.evader_traj,
        assignment=s2.initial_assignment,
        displacements=s2.displacement_matrix,
        nu=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    switches_s = switch_events(eval2_dynamic.assigned_targets, s2.initial_assignment, learning_3v3.dt)

    # Saturation comparison (scenario 1 style).
    sat_learning = LearningParams(
        policy_iterations=10 if args.quick else 22,
        rollout_steps=70 if args.quick else 120,
        min_samples_per_evader=120 if args.quick else 260,
        exploration_std_start=0.8,
        exploration_std_end=0.08,
    )
    s1_sat = replace(
        s1,
        t_final=200.0,
        capture_radius=160.0,
        # Use reactive evader for saturation study to obtain ordered outcomes:
        # u_p > u_e: capture fastest; u_p = u_e: capture slower; u_p < u_e: no capture.
        evader_motion_mode="reactive",
        evader_script_mix=1.0,
        evader_reactive_base=0.30,
        evader_reactive_response=1.00,
        evader_reactive_floor=0.20,
        evader_reactive_decay=0.012,
        evader_reactive_vel_gain=0.30,
    )
    sat_base = dict(
        q_diag=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        r1_diag=(1.0, 1.0, 1.0),
        r2_diag=(1.0, 1.0, 1.0),
        policy_gain=35.0,
        k_pos_p=0.02,
        k_vel_p=0.16,
        k_pos_e=0.02,
        k_vel_e=0.16,
    )
    sat_cases = {
        "u_p > u_e (25,20)": ControlParams(u_bar_p=25.0, u_bar_e=20.0, **sat_base),
        "u_p = u_e (20,20)": ControlParams(u_bar_p=20.0, u_bar_e=20.0, **sat_base),
        "u_p < u_e (15,20)": ControlParams(u_bar_p=15.0, u_bar_e=20.0, **sat_base),
    }
    sat_eval: Dict[str, EvalResult] = {}
    sat_train = {}
    for idx, (name, ctrl) in enumerate(sat_cases.items()):
        tr_sat, e = run_case(
            scenario=s1_sat,
            control=ctrl,
            learning=sat_learning,
            feature=feature_3v1,
            seed=31 + idx * 3,
            dynamic_graph_train=False,
            dynamic_graph_eval=False,
            stop_on_capture=False,
            record_logs=True,
        )
        sat_train[name] = tr_sat
        sat_eval[name] = e

    # Figures
    plot_trajectory_3d(eval1, out_dir / "fig1_trajectory_3v1.png", "3v1 trajectory")
    plot_trajectory_multiview(eval1, out_dir / "fig1b_trajectory_3v1_multiview.png", "3v1 trajectory multiview")
    plot_assigned_state_errors(eval1, learning_3v1.dt, out_dir / "fig2_errors_3v1.png", "3v1 assigned errors")
    plot_control_inputs(
        eval1,
        learning_3v1.dt,
        out_dir / "fig3_inputs_3v1.png",
        "3v1 input variation (paper style)",
        component_idx=0,
        y_lim=(-40.0, 40.0),
        use_tanh=False,
        paper_style_labels=True,
    )
    plot_control_inputs(
        eval1,
        learning_3v1.dt,
        out_dir / "fig3b_inputs_3v1_h_channel.png",
        "3v1 input variation (h-channel)",
        component_idx=2,
    )
    plot_weight_convergence(
        train1,
        out_dir / "fig4_weight_conv_3v1.png",
        "3v1 critic convergence",
        paper_target_start=0.389,
        paper_target_end=(0.369, 0.357, 0.3615),
    )

    plot_trajectory_3d(eval2_dynamic, out_dir / "fig6_trajectory_3v3_dynamic.png", "3v3 trajectory (dynamic graph)")
    plot_trajectory_multiview(
        eval2_dynamic,
        out_dir / "fig6b_trajectory_3v3_multiview.png",
        "3v3 trajectory multiview",
    )
    gif_ok = plot_trajectory_animation(
        eval2_dynamic,
        out_dir / "fig6_trajectory_3v3_dynamic.gif",
        "3v3 trajectory with assignment links",
    )
    plot_old_new_errors(
        eval2_dynamic,
        learning_3v3.dt,
        out_dir / "fig7_old_new_errors_3v3.png",
        "3v3 old/new target errors",
    )
    plot_assigned_xyz_errors(
        eval2_dynamic,
        learning_3v3.dt,
        out_dir / "fig7b_assigned_xyz_errors_3v3.png",
        "3v3 assigned target coordinate errors (x/y/h)",
    )
    plot_assignment_timeline(
        eval2_dynamic,
        learning_3v3.dt,
        out_dir / "fig8_assignment_switch_3v3.png",
        "3v3 target assignment switching timeline",
    )
    plot_assigned_state_errors(
        eval2_fixed,
        learning_3v3.dt,
        out_dir / "fig9_errors_3v3_fixed.png",
        "3v3 fixed-topology target errors ([26] style)",
    )
    plot_team_error_compare(
        eval2_dynamic.team_errors,
        eval2_fixed.team_errors,
        learning_3v3.dt,
        out_dir / "fig10_team_error_compare.png",
        "Team cohesion: proposed method vs [26]-style fixed graph",
    )
    plot_weight_convergence(
        train2,
        out_dir / "fig11_weight_conv_3v3.png",
        "3v3 critic convergence",
        paper_target_start=0.409,
        paper_target_end=(0.3902, 0.3750, 0.3904),
    )
    plot_saturation_comparison(
        sat_eval,
        learning_3v1.dt,
        out_dir / "fig5_saturation_compare.png",
        "Saturation constraint comparison",
    )
    plot_saturation_xyz_comparison(
        sat_eval,
        learning_3v1.dt,
        out_dir / "fig5b_saturation_xyz_compare.png",
        "Saturation comparison on x/y/h channels",
    )

    summary = {
        "scenario_3v1": {
            "train": train_summary(train1),
            "eval": eval_summary(eval1),
            "network": network_summary(s1.n_pursuers, s1.n_evaders),
            "initial_states": {
                "pursuers": s1.pursuer_init.tolist(),
                "evaders": s1.evader_init.tolist(),
            },
            "initial_xy_distance_matrix": xy_distance_matrix(s1.pursuer_init, s1.evader_init).tolist(),
        },
        "scenario_3v3_dynamic": {
            "train": train_summary(train2),
            "eval": eval_summary(eval2_dynamic),
            "eval_fixed_reference_same_trajectory": {
                "final_team_error": float(fixed_ref_team[-1]) if fixed_ref_team.size else None,
                "mean_team_error": float(np.mean(fixed_ref_team)) if fixed_ref_team.size else None,
            },
            "network": network_summary(s2.n_pursuers, s2.n_evaders),
            "initial_states": {
                "pursuers": s2.pursuer_init.tolist(),
                "evaders": s2.evader_init.tolist(),
            },
            "initial_xy_distance_matrix": xy_distance_matrix(s2.pursuer_init, s2.evader_init).tolist(),
            "switch_count": len(switches_s),
            "switch_times_s": switches_s,
            "initial_assignment": s2.initial_assignment.tolist(),
            "first_dynamic_assignment": (
                eval2_dynamic.assigned_targets[0].tolist() if eval2_dynamic.assigned_targets.size else []
            ),
            "dynamic_gif_generated": gif_ok,
        },
        "scenario_3v3_fixed_baseline": {
            "train": train_summary(train2),
            "eval": eval_summary(eval2_fixed),
            "same_weights_same_init_same_seed_as_dynamic": True,
        },
        "saturation_cases": {name: eval_summary(res) for name, res in sat_eval.items()},
    }
    dump_json(out_dir / "metrics_summary.json", summary)

    log_dir = out_dir / "logs"
    dump_train_log_markdown(log_dir / "scenario_3v1_train_log.md", "Scenario 3v1 Train Log", train1)
    dump_eval_log_markdown(log_dir / "scenario_3v1_eval_log.md", "Scenario 3v1 Eval Log", eval1)
    dump_train_log_markdown(log_dir / "scenario_3v3_train_log.md", "Scenario 3v3 Train Log", train2)
    dump_eval_log_markdown(
        log_dir / "scenario_3v3_dynamic_eval_log.md",
        "Scenario 3v3 Dynamic Eval Log",
        eval2_dynamic,
    )
    dump_eval_log_markdown(
        log_dir / "scenario_3v3_fixed_eval_log.md",
        "Scenario 3v3 Fixed Eval Log",
        eval2_fixed,
    )
    for name in sat_cases:
        safe = (
            name.replace(" ", "_")
            .replace(">", "gt")
            .replace("<", "lt")
            .replace("=", "eq")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
        )
        dump_train_log_markdown(
            log_dir / f"saturation_{safe}_train_log.md",
            f"Saturation {name} Train Log",
            sat_train[name],
        )
        dump_eval_log_markdown(
            log_dir / f"saturation_{safe}_eval_log.md",
            f"Saturation {name} Eval Log",
            sat_eval[name],
        )

    md = [
        "# V-SNAC MPE Reproduction Report",
        "",
        f"- Output folder: `{out_dir}`",
        f"- Quick mode: `{args.quick}`",
        "",
        "## Scenario 1 (3 pursuers, 1 evader)",
        f"- Final max assigned error: {summary['scenario_3v1']['eval']['final_max_assigned_error']:.3f}",
        f"- Final team error: {summary['scenario_3v1']['eval']['final_team_error']:.3f}",
        f"- Capture time (s): {summary['scenario_3v1']['eval']['capture_time_s']}",
        "",
        "## Scenario 2 (3 pursuers, 3 evaders)",
        "- Comparison protocol: same trained policy, same initial state, same eval seed; only dynamic-graph switch differs.",
        f"- Dynamic graph final team error: {summary['scenario_3v3_dynamic']['eval']['final_team_error']:.3f}",
        f"- Fixed graph final team error ([26] baseline run): {summary['scenario_3v3_fixed_baseline']['eval']['final_team_error']:.3f}",
        (
            "- Fixed-reference final team error (same trajectory): "
            f"{summary['scenario_3v3_dynamic']['eval_fixed_reference_same_trajectory']['final_team_error']:.3f}"
        ),
        f"- Dynamic graph switch count: {summary['scenario_3v3_dynamic']['switch_count']}",
        f"- Dynamic graph switch times (s): {summary['scenario_3v3_dynamic']['switch_times_s']}",
        f"- Dynamic GIF generated: {summary['scenario_3v3_dynamic']['dynamic_gif_generated']}",
        "",
        "## Network Count",
        f"- V-SNAC critics: {summary['scenario_3v3_dynamic']['network']['v_snac_networks']}",
        f"- AC estimated networks: {summary['scenario_3v3_dynamic']['network']['ac_networks_estimated']}",
        (
            f"- Estimated reduction: "
            f"{summary['scenario_3v3_dynamic']['network']['network_reduction_percent']:.2f}%"
        ),
        "",
        "## Files",
        "- `fig1_trajectory_3v1.png`",
        "- `fig1b_trajectory_3v1_multiview.png`",
        "- `fig2_errors_3v1.png`",
        "- `fig3_inputs_3v1.png`",
        "- `fig3b_inputs_3v1_h_channel.png`",
        "- `fig4_weight_conv_3v1.png`",
        "- `fig5_saturation_compare.png`",
        "- `fig5b_saturation_xyz_compare.png`",
        "- `fig6_trajectory_3v3_dynamic.png`",
        "- `fig6b_trajectory_3v3_multiview.png`",
        "- `fig6_trajectory_3v3_dynamic.gif`",
        "- `fig7_old_new_errors_3v3.png`",
        "- `fig7b_assigned_xyz_errors_3v3.png`",
        "- `fig8_assignment_switch_3v3.png`",
        "- `fig9_errors_3v3_fixed.png`",
        "- `fig10_team_error_compare.png`",
        "- `fig11_weight_conv_3v3.png`",
        "- `metrics_summary.json`",
    ]
    dump_markdown(out_dir / "REPORT.md", "\n".join(md))
    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
