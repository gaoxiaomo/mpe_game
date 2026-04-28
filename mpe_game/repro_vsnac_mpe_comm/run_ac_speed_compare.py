from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mpe_repro.ac_equivalent import TraditionalACEquivalent
from mpe_repro.comm_graph import CommunicationGraph
from mpe_repro.config import AircraftParams, CommParams
from mpe_repro.general_scenarios import build_general_scenario
from mpe_repro.report import dump_json, dump_markdown
from mpe_repro.simulator import MPECommSimulator
from run_comm import (
    GeneralCase,
    _collect_cases,
    _comm_params,
    _control_params,
    _feature_params,
    _learning_params,
    _scenario_spec,
)


@dataclass(frozen=True)
class PolicyTransition:
    pursuer_states: np.ndarray
    evader_states: np.ndarray
    assignment: np.ndarray
    next_pursuer_states: np.ndarray
    next_evader_states: np.ndarray
    next_assignment: np.ndarray


def _collect_policy_transitions(
    sim: MPECommSimulator,
    weights: list[np.ndarray],
    seed: int,
    dynamic_graph: bool,
) -> list[PolicyTransition]:
    rng = np.random.default_rng(seed)
    p, e = sim._reset_states(rng, perturb_scale=0.0)
    graph = sim._new_graph()

    snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    steps = int(sim.scenario.t_final / sim.learning.dt)
    policy_rng = np.random.default_rng(seed + 12345)

    for step_idx in range(steps):
        start_step = max(int(sim.learning.graph_update_start_step), 0)
        interval = max(int(sim.learning.graph_update_interval), 1)
        if dynamic_graph and (step_idx >= start_step) and ((step_idx - start_step) % interval == 0):
            p_for_graph, e_for_graph = sim._graph_states_for_update(p, e)
            graph.update(
                pursuer_states=p_for_graph,
                evader_states=e_for_graph,
                displacements=sim.displacements,
                switch_threshold=sim.scenario.swap_threshold,
                max_switch_worsening=sim.scenario.max_switch_worsening,
                nu=sim.nu_graph,
            )

        assigned = graph.assignment.copy()
        snapshots.append((p.copy(), e.copy(), assigned.copy()))

        u_p, u_e_virtual, _, _, _ = sim.controller.policy(
            pursuer_states=p,
            evader_states=e,
            assignment=assigned,
            displacements=sim.displacements,
            weights=weights,
            rng=policy_rng,
            exploration_std=0.0,
            gamma=0.0,
            A_p=None,
            delta_matrix=None,
            coordination_gradients=None,
        )
        u_e_applied = sim._applied_evader_inputs(
            step_idx=step_idx,
            u_e_virtual=u_e_virtual,
            pursuer_states=p,
            evader_states=e,
            assignment=assigned,
            u_p=u_p,
        )
        p = sim.dynamics.rk4_step_batch(p, u_p, sim.learning.dt)
        e = sim.dynamics.rk4_step_batch(e, u_e_applied, sim.learning.dt)

    transitions: list[PolicyTransition] = []
    for idx in range(max(len(snapshots) - 1, 0)):
        cur_p, cur_e, cur_assignment = snapshots[idx]
        next_p, next_e, next_assignment = snapshots[idx + 1]
        transitions.append(
            PolicyTransition(
                pursuer_states=cur_p,
                evader_states=cur_e,
                assignment=cur_assignment,
                next_pursuer_states=next_p,
                next_evader_states=next_e,
                next_assignment=next_assignment,
            )
        )
    return transitions


def _benchmark_policy_runtime(
    transitions: list[PolicyTransition],
    fn,
    warmup_passes: int,
    timed_passes: int,
) -> dict[str, float]:
    if not transitions:
        return {
            "total_s": 0.0,
            "ms_per_step": 0.0,
            "steps_per_s": 0.0,
        }

    for _ in range(max(warmup_passes, 0)):
        for transition in transitions:
            fn(transition)

    t0 = time.perf_counter()
    for _ in range(max(timed_passes, 1)):
        for transition in transitions:
            fn(transition)
    total_s = time.perf_counter() - t0

    total_steps = len(transitions) * max(timed_passes, 1)
    ms_per_step = 1000.0 * total_s / max(total_steps, 1)
    return {
        "total_s": float(total_s),
        "ms_per_step": float(ms_per_step),
        "steps_per_s": float(total_steps / max(total_s, 1e-12)),
    }


def _case_report_markdown(summary: dict[str, Any]) -> str:
    ratios = summary["ratios"]
    lines = [
        f"# AC Speed Compare: {summary['case']['name']}",
        "",
        "## Paper Alignment",
        "- Traditional AC is rebuilt from the local IEEE brief structure:",
        "  separate actor/critic banks for each pursuer and evader, action-dependent quadratic critics,",
        "  normalized critic updates, and actor updates extracted from critic blocks.",
        "- The paper's scalar/2D example is adapted here to the project's 6D local-error states and 3-axis controls.",
        "- Each critic therefore uses the augmented vector `z = [x, u, d] in R^12` and a 78-term quadratic basis.",
        "",
        "## Network Inventory",
        f"- `2 * (N_p + N_e) = {summary['ac_equivalent']['network_summary']['total_networks']}` networks.",
        f"- Actor parameters per network: {summary['ac_equivalent']['network_summary']['actor_parameters_per_network']}",
        f"- Critic parameters per network: {summary['ac_equivalent']['network_summary']['critic_parameters_per_network']}",
        "",
        "## Policy Runtime",
        f"- V-SNAC no-comm ms/step: {summary['timings']['vsnac_no_comm']['ms_per_step']:.6f}",
        f"- V-SNAC full-comm ms/step: {summary['timings']['vsnac_full_comm']['ms_per_step']:.6f}",
        f"- Paper AC policy-only ms/step: {summary['timings']['paper_ac_policy_only']['ms_per_step']:.6f}",
        f"- Paper AC online step ms/step: {summary['timings']['paper_ac_online']['ms_per_step']:.6f}",
        "",
        "## Ratios",
        f"- policy-only / V-SNAC no-comm: {ratios['paper_ac_policy_only_vs_vsnac_no_comm']:.3f}x",
        f"- online-step / V-SNAC no-comm: {ratios['paper_ac_online_vs_vsnac_no_comm']:.3f}x",
        f"- policy-only / V-SNAC full-comm: {ratios['paper_ac_policy_only_vs_vsnac_full_comm']:.3f}x",
        f"- online-step / V-SNAC full-comm: {ratios['paper_ac_online_vs_vsnac_full_comm']:.3f}x",
    ]
    return "\n".join(lines)


def _run_single_case(
    case: GeneralCase,
    out_dir: Path,
    quick: bool,
    gamma: float,
    warmup_passes: int,
    timed_passes: int,
) -> dict[str, Any]:
    scenario = build_general_scenario(_scenario_spec(case, quick))
    control = _control_params()
    learning = _learning_params(case.n_pursuers, case.n_evaders, quick)
    feature = _feature_params()

    sim = MPECommSimulator(
        scenario=scenario,
        aircraft_params=AircraftParams(),
        control_params=control,
        learning_params=learning,
        feature_params=feature,
        comm_params=_comm_params(gamma=gamma, comm_mode="full", d_safe=150.0),
    )

    dynamic_graph = scenario.n_evaders > 1
    train = sim.train_policy(seed=case.seed, dynamic_graph=dynamic_graph)

    full_comm_cfg = CommParams(gamma=gamma, comm_mode="full", dropout_prob=0.0, d_safe=150.0)
    full_comm_graph = CommunicationGraph(
        n_p=scenario.n_pursuers,
        comm_mode=full_comm_cfg.comm_mode,
        formation_ref_dist=full_comm_cfg.formation_ref_dist,
        d_safe=full_comm_cfg.d_safe,
    )
    transitions = _collect_policy_transitions(
        sim=sim,
        weights=train.weights,
        seed=case.seed + 1000,
        dynamic_graph=dynamic_graph,
    )

    ac_bank = TraditionalACEquivalent(
        state_scale=sim.features.state_scale,
        n_p=scenario.n_pursuers,
        n_e=scenario.n_evaders,
        u_bar_p=control.u_bar_p,
        u_bar_e=control.u_bar_e,
        q_diag=np.diag(control.q),
        r1_diag=np.diag(control.r1),
        r2_diag=np.diag(control.r2),
        seed=case.seed + 2024,
    )

    policy_rng = np.random.default_rng(case.seed + 4242)

    def vsnac_no_comm_step(transition: PolicyTransition) -> None:
        sim.controller.policy(
            pursuer_states=transition.pursuer_states,
            evader_states=transition.evader_states,
            assignment=transition.assignment,
            displacements=sim.displacements,
            weights=train.weights,
            rng=policy_rng,
            exploration_std=0.0,
            gamma=0.0,
            A_p=None,
            delta_matrix=None,
            coordination_gradients=None,
        )

    def vsnac_full_comm_step(transition: PolicyTransition) -> None:
        A_p, delta_matrix = sim._comm_structures(full_comm_graph, transition.assignment)
        _, coordination_grads = sim.controller.coordination_terms(
            pursuer_states=transition.pursuer_states,
            A_p=A_p,
            delta_matrix=delta_matrix,
            gamma=gamma,
            formation_ref_dist=full_comm_cfg.formation_ref_dist,
            d_safe=full_comm_cfg.d_safe,
        )
        sim.controller.policy(
            pursuer_states=transition.pursuer_states,
            evader_states=transition.evader_states,
            assignment=transition.assignment,
            displacements=sim.displacements,
            weights=train.weights,
            rng=policy_rng,
            exploration_std=0.0,
            gamma=gamma,
            A_p=A_p,
            delta_matrix=delta_matrix,
            formation_ref_dist=full_comm_cfg.formation_ref_dist,
            d_safe=full_comm_cfg.d_safe,
            coordination_gradients=coordination_grads,
        )

    def paper_ac_policy_only_step(transition: PolicyTransition) -> None:
        ac_bank.policy_only(
            pursuer_states=transition.pursuer_states,
            evader_states=transition.evader_states,
            assignment=transition.assignment,
            displacements=sim.displacements,
        )

    def paper_ac_online_step(transition: PolicyTransition) -> None:
        ac_bank.online_qlearning_step(
            pursuer_states=transition.pursuer_states,
            evader_states=transition.evader_states,
            assignment=transition.assignment,
            next_pursuer_states=transition.next_pursuer_states,
            next_evader_states=transition.next_evader_states,
            next_assignment=transition.next_assignment,
            displacements=sim.displacements,
            dt=learning.dt,
        )

    timings = {
        "vsnac_no_comm": _benchmark_policy_runtime(transitions, vsnac_no_comm_step, warmup_passes, timed_passes),
        "vsnac_full_comm": _benchmark_policy_runtime(transitions, vsnac_full_comm_step, warmup_passes, timed_passes),
        "paper_ac_policy_only": _benchmark_policy_runtime(
            transitions, paper_ac_policy_only_step, warmup_passes, timed_passes
        ),
        "paper_ac_online": _benchmark_policy_runtime(transitions, paper_ac_online_step, warmup_passes, timed_passes),
    }

    def _ratio(num: float, den: float) -> float:
        return float(num / max(den, 1e-12))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case": {
            "name": case.name,
            **asdict(case),
        },
        "trajectory_source": "no_comm_reference_rollout",
        "trajectory_steps": len(transitions),
        "warmup_passes": int(warmup_passes),
        "timed_passes": int(timed_passes),
        "vsnac_networks": int(scenario.n_pursuers),
        "ac_equivalent": {
            "network_summary": asdict(ac_bank.parameter_summary()),
            "assumption_note": (
                "Paper-inspired traditional baseline with one actor and one action-dependent critic per pursuer "
                "and per evader, matching the 2*(N_p+N_e) bank structure while adapting the paper's scalar/2D "
                "example to 6D local errors and 3-axis controls."
            ),
            "source_note": (
                "Implemented from the local PDF Adaptive Optimal Control via Q-Learning for Multi-Agent "
                "Pursuit-Evasion Games, using the action-dependent critic and online actor/critic updates from "
                "equations (23)-(36)."
            ),
        },
        "timings": timings,
        "ratios": {
            "paper_ac_policy_only_vs_vsnac_no_comm": _ratio(
                timings["paper_ac_policy_only"]["ms_per_step"], timings["vsnac_no_comm"]["ms_per_step"]
            ),
            "paper_ac_online_vs_vsnac_no_comm": _ratio(
                timings["paper_ac_online"]["ms_per_step"], timings["vsnac_no_comm"]["ms_per_step"]
            ),
            "paper_ac_policy_only_vs_vsnac_full_comm": _ratio(
                timings["paper_ac_policy_only"]["ms_per_step"], timings["vsnac_full_comm"]["ms_per_step"]
            ),
            "paper_ac_online_vs_vsnac_full_comm": _ratio(
                timings["paper_ac_online"]["ms_per_step"], timings["vsnac_full_comm"]["ms_per_step"]
            ),
        },
    }

    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    dump_json(case_dir / "summary.json", summary)
    dump_markdown(case_dir / "REPORT.md", _case_report_markdown(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runtime comparison between V-SNAC and a paper-inspired traditional actor-critic/Q-learning bank."
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Use shorter training schedules before benchmarking")
    parser.add_argument("--case", type=str, nargs="*", default=None, help="Case tokens such as 3v1 3v3 5v3")
    parser.add_argument("--assignment-mode", type=str, default="shifted", choices=["zero", "cyclic", "shifted", "nearest", "random"])
    parser.add_argument("--layout-mode", type=str, default="structured", choices=["structured", "random"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--timed-passes", type=int, default=12)
    args = parser.parse_args()

    cases = _collect_cases(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(__file__).resolve().parent
    out_dir = Path(args.output) if args.output else root / "outputs" / f"ac_speed_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for case in cases:
        all_summaries.append(
            _run_single_case(
                case=case,
                out_dir=out_dir,
                quick=bool(args.quick),
                gamma=float(args.gamma),
                warmup_passes=int(args.warmup_passes),
                timed_passes=int(args.timed_passes),
            )
        )

    batch = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": all_summaries,
    }
    dump_json(out_dir / "batch_summary.json", batch)

    lines = [
        "# AC Speed Compare Batch Report",
        "",
        "- Traditional baseline is implemented from the local PDF structure: separate actor/critic banks plus online Q-learning updates.",
        "- The benchmark uses a reference no-communication rollout for each case, then replays the same transitions.",
        "- `paper_ac_online` is the main traditional baseline because it includes the online critic and actor updates described in the paper.",
        "",
        "| case | V-SNAC no-comm ms/step | V-SNAC full-comm ms/step | paper AC policy-only ms/step | paper AC online ms/step | policy-only / V-SNAC no-comm | online / V-SNAC no-comm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in all_summaries:
        t = item["timings"]
        r = item["ratios"]
        lines.append(
            "| {case} | {v0:.6f} | {vf:.6f} | {a0:.6f} | {ac:.6f} | {r0:.3f}x | {rc:.3f}x |".format(
                case=item["case"]["name"],
                v0=t["vsnac_no_comm"]["ms_per_step"],
                vf=t["vsnac_full_comm"]["ms_per_step"],
                a0=t["paper_ac_policy_only"]["ms_per_step"],
                ac=t["paper_ac_online"]["ms_per_step"],
                r0=r["paper_ac_policy_only_vs_vsnac_no_comm"],
                rc=r["paper_ac_online_vs_vsnac_no_comm"],
            )
        )
    dump_markdown(out_dir / "REPORT.md", "\n".join(lines))
    print(f"Done. Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
