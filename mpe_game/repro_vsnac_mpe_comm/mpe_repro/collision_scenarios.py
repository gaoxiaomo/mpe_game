"""Collision-prone scenario suite for the dropout robustness study.

Each scenario is geometrically constructed so that the no-communication
(`gamma = 0`) baseline produces inter-pursuer crossings or dangerous
proximity events. The communication coordination layer is then expected to
drive lateral / vertical separation while preserving tracking convergence.

Six configurations span small (2v1, 3v1) and medium (4v1, 5v2) team sizes,
single- and multi-evader, planar and three-dimensional crossings:

- C1  Head-on 2v1            (paper baseline; canonical crossing)
- C2  Triangular 3v1         (three-way convergence at center)
- C3  X-pattern 4v1          (two simultaneous diagonal crossings)
- C4  Parallel-conflict 2v2  (intra-group crossings in two adjacent groups)
- C5  Vertical-funnel 3v1    (3D vertical crossing between altitude bands)
- C6  Asymmetric 5v2         (4 pursuers around one evader, 1 around the other)

All factories return a fully-populated ``ScenarioConfig`` ready for
``MPECommSimulator``. Initial speeds, formation displacements and evader
scripts are kept consistent across scenarios so that aggregated metrics are
comparable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .config import ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approach_velocity(pursuer_pos: np.ndarray, evader_pos: np.ndarray, speed: float) -> np.ndarray:
    """Unit vector from pursuer toward evader scaled by ``speed``."""
    direction = evader_pos - pursuer_pos
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.zeros(3, dtype=float)
    return float(speed) * direction / norm


def _mirror_displacement(pursuer_pos: np.ndarray, evader_pos: np.ndarray) -> np.ndarray:
    """Return r_j of length-6 such that the steady-state pursuer position is the
    mirror of ``pursuer_pos`` through ``evader_pos`` (i.e., 2*evader - pursuer)
    in the position channels and zero in the velocity channels.

    With the convention ``x_tilde = x_p - x_e + r``, ``x_tilde = 0`` implies
    ``x_p = x_e - r``. Setting ``r[:3] = pursuer_pos - evader_pos`` yields
    ``x_p_final = 2*evader_pos - pursuer_pos`` (the mirror point).
    """
    rel = pursuer_pos[:3] - evader_pos[:3]
    out = np.zeros(6, dtype=float)
    out[:3] = rel
    return out


def _displacement_from_target(pursuer_target_pos: np.ndarray, evader_pos: np.ndarray) -> np.ndarray:
    """Return r_j of length-6 so that the steady-state pursuer position equals
    ``pursuer_target_pos``. Velocity channels are zero."""
    out = np.zeros(6, dtype=float)
    out[:3] = evader_pos[:3] - pursuer_target_pos[:3]
    return out


# ---------------------------------------------------------------------------
# C1  Head-on 2v1
# ---------------------------------------------------------------------------


def build_C1_head_on_2v1() -> ScenarioConfig:
    """Two pursuers approaching a single evader from opposite lateral
    positions, each with formation offsets pointing to the opposite side
    of the evader. Their tracking-optimal paths physically cross."""
    n_p = 2
    evader = np.array([[0.0, 1500.0, 400.0, 25.0, 30.0, 0.0]], dtype=float)
    pursuers = np.array(
        [
            [-1800.0, -1500.0, 400.0, 0.0, 0.0, 0.0],
            [+1800.0, -1500.0, 400.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    speed = 70.0
    for j in range(n_p):
        pursuers[j, 3:6] = _approach_velocity(pursuers[j, :3], evader[0, :3], speed)

    displacements = np.zeros((n_p, 1, 6), dtype=float)
    for j in range(n_p):
        displacements[j, 0] = _mirror_displacement(pursuers[j], evader[0])

    return ScenarioConfig(
        name="collision_C1_head_on_2v1",
        pursuer_init=pursuers,
        evader_init=evader,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,
        max_switch_worsening=0.0,
        initial_assignment=np.zeros(n_p, dtype=int),
        t_final=70.0,
        capture_radius=180.0,
        evader_motion_mode="scripted",
        evader_script_amp=(3.0, 3.0, 1.0),
        evader_script_omega=0.18,
        evader_script_decay=0.02,
        evader_script_mix=0.5,
        swap_lookahead_time=0.0,
    )


# ---------------------------------------------------------------------------
# C2  Trio head-on 3v1
# ---------------------------------------------------------------------------


def build_C2_triangular_3v1() -> ScenarioConfig:
    """Three pursuers from south with all three pairs in genuine close
    approach: P0/P2 swap east<->west sides at the evader's altitude
    (horizontal head-on), and P1 descends from high altitude through the
    P0/P2 crossing region (vertical conflict with both). All three pairs
    contribute useful Phi gradient, avoiding the degree-dilution failure
    mode where a non-conflicting pursuer pollutes the average."""
    n_p = 3
    evader = np.array([[0.0, 1500.0, 500.0, 25.0, 30.0, 0.0]], dtype=float)
    pursuers = np.array(
        [
            [-1800.0, -1500.0, 500.0, 0.0, 0.0, 0.0],  # P0 SW at evader altitude
            [    0.0, -1500.0, 1100.0, 0.0, 0.0, 0.0],  # P1 mid-x HIGH altitude
            [+1800.0, -1500.0, 500.0, 0.0, 0.0, 0.0],  # P2 SE at evader altitude
        ],
        dtype=float,
    )
    # P0/P2 swap east<->west (horizontal head-on, stays at evader altitude).
    # P1 descends from z=1100 through to z=evader altitude, passing through
    # the P0/P2 collision zone in altitude.
    targets = np.array(
        [
            [+1800.0, 4500.0, 500.0],  # P0 final -> NE at evader altitude
            [    0.0, 4500.0,  500.0],  # P1 final -> straight N at evader altitude (descent)
            [-1800.0, 4500.0, 500.0],  # P2 final -> NW at evader altitude
        ],
        dtype=float,
    )
    speed = 70.0
    for j in range(n_p):
        pursuers[j, 3:6] = _approach_velocity(pursuers[j, :3], evader[0, :3], speed)

    displacements = np.zeros((n_p, 1, 6), dtype=float)
    for j in range(n_p):
        displacements[j, 0] = _displacement_from_target(targets[j], evader[0])

    return ScenarioConfig(
        name="collision_C2_trio_3v1",
        pursuer_init=pursuers,
        evader_init=evader,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,
        max_switch_worsening=0.0,
        initial_assignment=np.zeros(n_p, dtype=int),
        t_final=75.0,
        capture_radius=180.0,
        evader_motion_mode="scripted",
        evader_script_amp=(3.0, 3.0, 1.0),
        evader_script_omega=0.18,
        evader_script_decay=0.02,
        evader_script_mix=0.45,
        swap_lookahead_time=0.0,
    )


# ---------------------------------------------------------------------------
# C3  Stacked head-on 4v1
# ---------------------------------------------------------------------------


def build_C3_x_pattern_4v1() -> ScenarioConfig:
    """Two C1-style head-ons stacked at different altitudes. P0/P1 cross at
    the upper altitude, P2/P3 cross at the lower altitude. Both pairs share
    the evader-tracking objective so all four pursuers belong to one
    intra-group communication graph."""
    n_p = 4
    evader = np.array([[0.0, 1500.0, 500.0, 25.0, 30.0, 0.0]], dtype=float)
    pursuers = np.array(
        [
            [-1800.0, -1500.0, 750.0, 0.0, 0.0, 0.0],  # P0 high SW
            [+1800.0, -1500.0, 750.0, 0.0, 0.0, 0.0],  # P1 high SE
            [-1800.0, -1500.0, 250.0, 0.0, 0.0, 0.0],  # P2 low SW
            [+1800.0, -1500.0, 250.0, 0.0, 0.0, 0.0],  # P3 low SE
        ],
        dtype=float,
    )
    targets = np.array(
        [
            [+1800.0, 4500.0, 750.0],  # P0 -> NE high
            [-1800.0, 4500.0, 750.0],  # P1 -> NW high
            [+1800.0, 4500.0, 250.0],  # P2 -> NE low
            [-1800.0, 4500.0, 250.0],  # P3 -> NW low
        ],
        dtype=float,
    )
    speed = 70.0
    for j in range(n_p):
        pursuers[j, 3:6] = _approach_velocity(pursuers[j, :3], evader[0, :3], speed)

    displacements = np.zeros((n_p, 1, 6), dtype=float)
    for j in range(n_p):
        displacements[j, 0] = _displacement_from_target(targets[j], evader[0])

    return ScenarioConfig(
        name="collision_C3_stacked_4v1",
        pursuer_init=pursuers,
        evader_init=evader,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,
        max_switch_worsening=0.0,
        initial_assignment=np.zeros(n_p, dtype=int),
        t_final=75.0,
        capture_radius=200.0,
        evader_motion_mode="scripted",
        evader_script_amp=(3.0, 3.0, 1.0),
        evader_script_omega=0.16,
        evader_script_decay=0.02,
        evader_script_mix=0.45,
        swap_lookahead_time=0.0,
    )


# ---------------------------------------------------------------------------
# C4  Parallel-conflict 2v2
# ---------------------------------------------------------------------------


def build_C4_parallel_conflict_2v2() -> ScenarioConfig:
    """Two evaders close together (200 m apart along x). Each evader has two
    assigned pursuers whose formation offsets force an intra-group crossing.
    Because the two groups share airspace, the two crossings happen in
    overlapping volumes -- coordination must manage both simultaneously."""
    n_p = 4
    n_e = 2
    evaders = np.array(
        [
            [-200.0, 1800.0, 450.0, 0.0, 30.0, 0.0],
            [+200.0, 1800.0, 450.0, 0.0, 30.0, 0.0],
        ],
        dtype=float,
    )
    pursuers = np.array(
        [
            [-1700.0, -1500.0, 380.0, 0.0, 0.0, 0.0],   # P0 -> E0  (SW of E0)
            [+1300.0, +5100.0, 530.0, 0.0, 0.0, 0.0],   # P1 -> E0  (NE of E0)
            [+1700.0, -1500.0, 380.0, 0.0, 0.0, 0.0],   # P2 -> E1  (SE of E1)
            [-1300.0, +5100.0, 530.0, 0.0, 0.0, 0.0],   # P3 -> E1  (NW of E1)
        ],
        dtype=float,
    )
    speed = 70.0
    initial_assignment = np.array([0, 0, 1, 1], dtype=int)

    for j in range(n_p):
        pursuers[j, 3:6] = _approach_velocity(
            pursuers[j, :3], evaders[initial_assignment[j], :3], speed
        )

    displacements = np.zeros((n_p, n_e, 6), dtype=float)
    for j in range(n_p):
        i = int(initial_assignment[j])
        displacements[j, i] = _mirror_displacement(pursuers[j], evaders[i])

    return ScenarioConfig(
        name="collision_C4_parallel_conflict_2v2",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,
        max_switch_worsening=0.0,
        initial_assignment=initial_assignment,
        t_final=85.0,
        capture_radius=200.0,
        evader_motion_mode="scripted",
        evader_script_amp=(3.0, 3.0, 1.0),
        evader_script_omega=0.16,
        evader_script_decay=0.02,
        evader_script_mix=0.40,
        swap_lookahead_time=0.0,
    )


# ---------------------------------------------------------------------------
# C5  Vertical-funnel 3v1
# ---------------------------------------------------------------------------


def build_C5_vertical_funnel_3v1() -> ScenarioConfig:
    """Three pursuers in a vertical-emphasis trio. P0 (low) and P2 (high)
    swap altitudes (vertical head-on along z). P1 makes a lateral E-W
    crossing through the central column at evader altitude, so all three
    pairs contribute non-trivial Phi. The 3v1 form retains the C2-style
    triple conflict structure but along the vertical axis, where Phi is
    weaker due to feature-scale anisotropy in the V-SNAC basis -- this
    scenario therefore probes the separation effectiveness limit in the
    altitude channel."""
    n_p = 3
    evader = np.array([[0.0, 1500.0, 600.0, 25.0, 30.0, 0.0]], dtype=float)
    pursuers = np.array(
        [
            [    0.0, -1500.0,  100.0, 0.0, 0.0, 0.0],   # P0 low, central x
            [-1800.0, -1500.0,  600.0, 0.0, 0.0, 0.0],   # P1 mid altitude, west (will cross to east)
            [    0.0, -1500.0, 1100.0, 0.0, 0.0, 0.0],   # P2 high, central x
        ],
        dtype=float,
    )
    speed = 70.0
    for j in range(n_p):
        pursuers[j, 3:6] = _approach_velocity(pursuers[j, :3], evader[0, :3], speed)

    targets = np.array(
        [
            [    0.0, 4500.0, 1100.0],  # P0: ascend (vertical head-on with P2)
            [+1800.0, 4500.0,  600.0],  # P1: lateral W -> E crossing at evader altitude
            [    0.0, 4500.0,  100.0],  # P2: descend (vertical head-on with P0)
        ],
        dtype=float,
    )
    displacements = np.zeros((n_p, 1, 6), dtype=float)
    for j in range(n_p):
        displacements[j, 0] = _displacement_from_target(targets[j], evader[0])

    return ScenarioConfig(
        name="collision_C5_vertical_funnel_3v1",
        pursuer_init=pursuers,
        evader_init=evader,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,
        max_switch_worsening=0.0,
        initial_assignment=np.zeros(n_p, dtype=int),
        t_final=80.0,
        capture_radius=180.0,
        evader_motion_mode="scripted",
        evader_script_amp=(2.0, 2.0, 0.6),
        evader_script_omega=0.14,
        evader_script_decay=0.02,
        evader_script_mix=0.35,
        swap_lookahead_time=0.0,
    )


# ---------------------------------------------------------------------------
# C6  Asymmetric 5v2
# ---------------------------------------------------------------------------


def build_C6_asymmetric_5v2() -> ScenarioConfig:
    """Five pursuers, two evaders. E0 group has 4 pursuers in two stacked
    head-on pairs at different altitudes (the C3 pattern). E1 group has 1
    solo pursuer, no crossing risk. The asymmetry tests the framework on
    heterogeneous group sizes within a single run."""
    n_p = 5
    n_e = 2
    evaders = np.array(
        [
            [-3500.0, 1800.0, 500.0, 25.0, 30.0, 0.0],
            [+3500.0, 1800.0, 500.0, -20.0, 35.0, 0.0],
        ],
        dtype=float,
    )

    pursuers = np.array(
        [
            # E0 group (4): two altitude-stacked head-on pairs (= C3 pattern)
            [evaders[0, 0] - 1800.0, -1500.0, evaders[0, 2] + 250.0, 0.0, 0.0, 0.0],   # P0 high SW
            [evaders[0, 0] + 1800.0, -1500.0, evaders[0, 2] + 250.0, 0.0, 0.0, 0.0],   # P1 high SE
            [evaders[0, 0] - 1800.0, -1500.0, evaders[0, 2] - 250.0, 0.0, 0.0, 0.0],   # P2 low SW
            [evaders[0, 0] + 1800.0, -1500.0, evaders[0, 2] - 250.0, 0.0, 0.0, 0.0],   # P3 low SE
            # E1 group (1): solo straight-north approach
            [evaders[1, 0],          -1500.0, evaders[1, 2],         0.0, 0.0, 0.0],   # P4 solo for E1
        ],
        dtype=float,
    )
    targets = np.array(
        [
            [evaders[0, 0] + 1800.0, 4500.0, evaders[0, 2] + 250.0],   # P0 -> high NE (head-on with P1)
            [evaders[0, 0] - 1800.0, 4500.0, evaders[0, 2] + 250.0],   # P1 -> high NW (head-on with P0)
            [evaders[0, 0] + 1800.0, 4500.0, evaders[0, 2] - 250.0],   # P2 -> low NE  (head-on with P3)
            [evaders[0, 0] - 1800.0, 4500.0, evaders[0, 2] - 250.0],   # P3 -> low NW  (head-on with P2)
            [evaders[1, 0],          4500.0, evaders[1, 2]],            # P4 -> straight north
        ],
        dtype=float,
    )
    speed = 70.0
    initial_assignment = np.array([0, 0, 0, 0, 1], dtype=int)
    for j in range(n_p):
        pursuers[j, 3:6] = _approach_velocity(
            pursuers[j, :3], evaders[initial_assignment[j], :3], speed
        )

    displacements = np.zeros((n_p, n_e, 6), dtype=float)
    for j in range(n_p):
        i = int(initial_assignment[j])
        displacements[j, i] = _displacement_from_target(targets[j], evaders[i])

    return ScenarioConfig(
        name="collision_C6_asymmetric_5v2",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=1.0e9,  # keep assignments fixed to preserve crossing geometry
        max_switch_worsening=0.0,
        initial_assignment=initial_assignment,
        t_final=85.0,
        capture_radius=200.0,
        evader_motion_mode="scripted",
        evader_script_amp=(3.0, 3.0, 1.0),
        evader_script_omega=0.15,
        evader_script_decay=0.02,
        evader_script_mix=0.40,
        swap_lookahead_time=0.0,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


COLLISION_SCENARIO_BUILDERS: dict[str, Callable[[], ScenarioConfig]] = {
    "C1": build_C1_head_on_2v1,
    "C2": build_C2_triangular_3v1,
    "C3": build_C3_x_pattern_4v1,
    "C4": build_C4_parallel_conflict_2v2,
    "C5": build_C5_vertical_funnel_3v1,
    "C6": build_C6_asymmetric_5v2,
}


def list_collision_scenarios() -> list[str]:
    return list(COLLISION_SCENARIO_BUILDERS.keys())


def build_collision_scenario(tag: str) -> ScenarioConfig:
    if tag not in COLLISION_SCENARIO_BUILDERS:
        raise KeyError(f"unknown collision scenario tag: {tag!r}")
    return COLLISION_SCENARIO_BUILDERS[tag]()
