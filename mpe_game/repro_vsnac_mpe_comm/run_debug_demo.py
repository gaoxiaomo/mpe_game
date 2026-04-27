from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mpe_repro.config import AircraftParams
from mpe_repro.general_scenarios import build_general_scenario
from mpe_repro.plotting import (
    plot_assigned_residual_norm,
    plot_assigned_state_errors,
    plot_assignment_timeline,
    plot_control_inputs,
    plot_d_min_history,
    plot_formation_error,
    plot_trajectory_animation_3d,
    plot_trajectory_multiview,
    plot_weight_convergence,
)
from mpe_repro.report import (
    dump_eval_log_markdown,
    dump_json,
    dump_markdown,
    dump_train_log_markdown,
    eval_summary,
    train_summary,
)
from mpe_repro.simulator import MPECommSimulator
from run_comm import (
    _comm_params,
    _control_params,
    _feature_params,
    _learning_params,
    _parse_case_token,
    _scenario_spec,
)


def _resolve_case(token: str, seed: int, assignment_mode: str, layout_mode: str):
    compact = token.lower().replace(" ", "")
    if compact.endswith(("x1", "v1", "*1", ":1")):
        assignment_mode = "zero"
    return _parse_case_token(
        token=token,
        seed=seed,
        assignment_mode=assignment_mode,
        layout_mode=layout_mode,
    )


def _source_map() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    return {
        "entry_demo": str(root / "run_debug_demo.py"),
        "entry_batch": str(root / "run_comm.py"),
        "simulator": str(root / "mpe_repro" / "simulator.py"),
        "controller": str(root / "mpe_repro" / "controller.py"),
        "comm_graph": str(root / "mpe_repro" / "comm_graph.py"),
        "critic_ls": str(root / "mpe_repro" / "offpolicy_ls.py"),
        "features": str(root / "mpe_repro" / "features.py"),
        "plotting": str(root / "mpe_repro" / "plotting.py"),
    }


def _debug_notes(case_name: str, gamma: float, d_safe: float) -> str:
    src = _source_map()
    lines = [
        f"# Debug Guide: {case_name}",
        "",
        "## First Read Path",
        f"1. Demo entry: `{src['entry_demo']}`",
        f"2. Main rollout step: `{src['simulator']}`",
        f"3. Control + analytic term: `{src['controller']}`",
        f"4. Communication topology: `{src['comm_graph']}`",
        f"5. Critic update: `{src['critic_ls']}`",
        f"6. Animation / plots: `{src['plotting']}`",
        "",
        "## Recommended Breakpoints",
        f"- `run_debug_demo.py`: inside `run_debug_demo(...)` before `sim.train_policy(...)` to inspect scenario/config.",
        f"- `{src['simulator']}`: `_step(...)` at adjacency build, control call, and next-state update.",
        f"- `{src['controller']}`: `coordination_potential_and_gradient(...)` to inspect `amp`, `pair_weight`, `gradients`.",
        f"- `{src['controller']}`: `policy(...)` after `total_grad += coordination_gradients` to inspect control-effective gradients.",
        f"- `{src['critic_ls']}`: `add_sample(...)` and `solve(...)` to inspect Bellman rows and LS solution.",
        "",
        "## Useful Watch Expressions",
        "- `assigned`",
        "- `A_p`",
        "- `delta_matrix[0, 1, :3]`",
        "- `value_terms_t`",
        "- `coordination_grads[0] if coordination_grads is not None else None`",
        "- `x_err[0]`",
        "- `u_p[0]`",
        "- `step.stage_costs`",
        "",
        "## Demo Parameters",
        f"- gamma = {gamma}",
        f"- d_safe = {d_safe}",
        "- evaluation runs with `record_logs=True` and `stop_on_capture=False` so you can inspect the full horizon.",
        "",
        "## Output Files",
        "- `summary.json`: compact train/eval summary",
        "- `step_logs.json`: per-step debug data for the rollout",
        "- `TRAIN_LOG.md`: per-iteration critic changes",
        "- `EVAL_LOG.md`: markdown view of step logs",
        "- `fig_trajectory_3d.gif`: animation for现场演示",
    ]
    return "\n".join(lines)


def run_debug_demo(
    *,
    case_token: str,
    output_dir: Path,
    seed: int,
    quick: bool,
    gamma: float,
    d_safe: float,
    assignment_mode: str,
    layout_mode: str,
    record_logs: bool,
) -> dict[str, Any]:
    case = _resolve_case(
        token=case_token,
        seed=seed,
        assignment_mode=assignment_mode,
        layout_mode=layout_mode,
    )
    scenario = build_general_scenario(_scenario_spec(case, quick))
    control = _control_params()
    learning = _learning_params(case.n_pursuers, case.n_evaders, quick)
    feature = _feature_params()
    comm = _comm_params(gamma=gamma, comm_mode="full", d_safe=d_safe)

    sim = MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        comm_params=comm,
    )

    t0 = time.perf_counter()
    train = sim.train_policy(seed=seed, dynamic_graph=scenario.n_evaders > 1)
    train_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_result = sim.evaluate_policy(
        weights=train.weights,
        seed=seed + 1000,
        dynamic_graph=scenario.n_evaders > 1,
        stop_on_capture=False,
        record_logs=record_logs,
        comm_params_override=comm,
    )
    eval_wall = time.perf_counter() - t1

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_train_log_markdown(output_dir / "TRAIN_LOG.md", f"{case.name} train log", train)
    dump_eval_log_markdown(output_dir / "EVAL_LOG.md", f"{case.name} eval log", eval_result)
    if eval_result.step_logs is not None:
        dump_json(output_dir / "step_logs.json", {"steps": eval_result.step_logs})

    plot_trajectory_multiview(
        eval_result,
        output_dir / "fig_trajectory_multiview.png",
        f"{case.name} debug demo trajectory",
    )
    plot_trajectory_animation_3d(
        eval_result,
        output_dir / "fig_trajectory_3d.gif",
        f"{case.name} debug demo",
        fps=12,
        max_frames=180,
    )
    plot_assigned_state_errors(
        eval_result,
        learning.dt,
        output_dir / "fig_assigned_errors.png",
        f"{case.name} assigned target errors",
    )
    plot_assigned_residual_norm(
        eval_result,
        learning.dt,
        scenario.displacement_matrix,
        output_dir / "fig_assigned_residual_norm.png",
        f"{case.name} assigned residual norms",
    )
    plot_control_inputs(
        eval_result,
        learning.dt,
        output_dir / "fig_control_inputs.png",
        f"{case.name} control inputs",
        component_idx=0,
        y_lim=(-40.0, 40.0),
        use_tanh=False,
        paper_style_labels=False,
    )
    plot_d_min_history(
        eval_result,
        learning.dt,
        output_dir / "fig_d_min.png",
        f"{case.name} d_min",
        d_min_threshold=d_safe,
    )
    plot_formation_error(
        eval_result,
        learning.dt,
        output_dir / "fig_formation_error.png",
        f"{case.name} formation error",
    )
    plot_weight_convergence(
        train,
        output_dir / "fig_weight_convergence.png",
        f"{case.name} critic convergence",
    )
    if scenario.n_evaders > 1:
        plot_assignment_timeline(
            eval_result,
            learning.dt,
            output_dir / "fig_assignment_timeline.png",
            f"{case.name} assignment timeline",
        )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case": {
            "name": case.name,
            "n_pursuers": case.n_pursuers,
            "n_evaders": case.n_evaders,
            "seed": seed,
            "quick": bool(quick),
            "assignment_mode": assignment_mode,
            "layout_mode": layout_mode,
        },
        "params": {
            "gamma": float(gamma),
            "d_safe": float(d_safe),
            "dt": float(learning.dt),
            "policy_iterations": int(learning.policy_iterations),
            "rollout_steps": int(learning.rollout_steps),
            "record_logs": bool(record_logs),
        },
        "runtime": {
            "train_wall_time_s": float(train_wall),
            "eval_wall_time_s": float(eval_wall),
            "total_wall_time_s": float(train_wall + eval_wall),
        },
        "train": train_summary(train),
        "eval": eval_summary(eval_result, comm_mode="full_comm"),
        "source_map": _source_map(),
    }
    dump_json(output_dir / "summary.json", summary)
    dump_markdown(output_dir / "DEBUG_GUIDE.md", _debug_notes(case.name, gamma, d_safe))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-case debug-friendly demo runner.")
    parser.add_argument("--case", type=str, default="3v1", help="Case token such as 3v1, 3v3, 6v3.")
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--quick", action="store_true", help="Use short training/eval settings for debugging.")
    parser.add_argument("--gamma", type=float, default=0.3, help="Communication coupling weight.")
    parser.add_argument("--d-safe", type=float, default=100.0, help="Safety distance for the smooth pair factor.")
    parser.add_argument(
        "--assignment-mode",
        type=str,
        default="shifted",
        choices=["zero", "cyclic", "shifted", "nearest", "random"],
        help="Initial pursuer-to-evader assignment.",
    )
    parser.add_argument(
        "--layout-mode",
        type=str,
        default="structured",
        choices=["structured", "random"],
        help="Initial state layout.",
    )
    parser.add_argument("--no-step-logs", action="store_true", help="Disable per-step JSON/markdown logs.")
    parser.add_argument("--output", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else root / "outputs" / f"debug_demo_{args.case}_{ts}"
    summary = run_debug_demo(
        case_token=args.case,
        output_dir=out_dir,
        seed=int(args.seed),
        quick=bool(args.quick),
        gamma=float(args.gamma),
        d_safe=float(args.d_safe),
        assignment_mode=args.assignment_mode,
        layout_mode=args.layout_mode,
        record_logs=not bool(args.no_step_logs),
    )
    print(f"Debug demo saved to: {out_dir}")
    print(f"Capture time: {summary['eval']['capture_time_s']}")


if __name__ == "__main__":
    main()
