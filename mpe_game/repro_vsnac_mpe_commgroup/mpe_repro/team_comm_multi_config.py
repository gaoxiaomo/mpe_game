from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


AssignmentMode = Literal["zero", "cyclic", "shifted", "nearest", "random"]
LayoutMode = Literal["structured", "random"]


@dataclass(frozen=True)
class GroupCommunicationWindow:
    group_idx: int
    start_s: float
    end_s: float
    isolated_slots: Tuple[int, ...]


@dataclass(frozen=True)
class GroupCommunicationParams:
    windows: Tuple[GroupCommunicationWindow, ...] = ()
    windows_per_group: int = 1
    dropout_fraction: float = 0.45
    full_group_prob: float = 0.35
    randomize_each_episode: bool = True


@dataclass(frozen=True)
class GeneralTeamCommScenarioSpec:
    n_pursuers: int
    n_evaders: int
    seed: int = 0
    assignment_mode: AssignmentMode = "shifted"
    layout_mode: LayoutMode = "structured"
    t_final: float = 140.0
    capture_radius: float = 220.0
    swap_threshold: float = 5.0
    max_switch_worsening: float = 0.0
    evader_motion_mode: str = "scripted"
    evader_script_amp: tuple[float, float, float] = (8.0, 8.0, 3.0)
    evader_script_omega: float = 0.18
    evader_script_decay: float = 0.012
    evader_script_mix: float = 0.30
    swap_lookahead_time: float = 0.0
    comm_windows_per_group: int = 1
    comm_dropout_fraction: float = 0.45
    comm_full_group_prob: float = 0.35


@dataclass(frozen=True)
class TeamCommMultiScenario:
    name: str
    pursuer_init: np.ndarray
    evader_init: np.ndarray
    initial_assignment: np.ndarray
    slot_evaders: np.ndarray
    slot_local_indices: np.ndarray
    slot_displacements: np.ndarray
    group_slot_ids: tuple[np.ndarray, ...]
    t_final: float
    capture_radius: float
    nu_eval: np.ndarray
    communication: GroupCommunicationParams
    swap_threshold: float
    max_switch_worsening: float
    evader_motion_mode: str = "scripted"
    evader_script_amp: tuple[float, float, float] = (8.0, 8.0, 3.0)
    evader_script_omega: float = 0.18
    evader_script_decay: float = 0.012
    evader_script_mix: float = 0.30
    swap_lookahead_time: float = 0.0

    @property
    def n_pursuers(self) -> int:
        return int(self.pursuer_init.shape[0])

    @property
    def n_evaders(self) -> int:
        return int(self.evader_init.shape[0])

    @property
    def group_sizes(self) -> np.ndarray:
        return np.asarray([ids.shape[0] for ids in self.group_slot_ids], dtype=int)


def _formation_offsets(n_slots: int, xy_radius: float | None = None, h_radius: float | None = None) -> np.ndarray:
    if n_slots <= 0:
        return np.zeros((0, 6), dtype=float)
    if xy_radius is None:
        xy_radius = 85.0 + 14.0 * max(n_slots - 1, 0)
    if h_radius is None:
        h_radius = 30.0 + 6.0 * max(n_slots - 1, 0)
    angles = np.linspace(0.0, 2.0 * np.pi, n_slots, endpoint=False)
    offsets = np.zeros((n_slots, 6), dtype=float)
    offsets[:, 0] = float(xy_radius) * np.cos(angles)
    offsets[:, 1] = float(xy_radius) * np.sin(angles)
    offsets[:, 2] = float(h_radius) * np.sin(2.0 * angles)
    return offsets


def _sample_separated_positions(
    count: int,
    rng: np.random.Generator,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    h_range: tuple[float, float],
    min_xy: float,
    avoid_xyz: np.ndarray | None = None,
    min_avoid_xy: float = 0.0,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=float)

    points: list[np.ndarray] = []
    cur_min_xy = float(min_xy)
    cur_min_avoid = float(min_avoid_xy)
    attempts = 0
    relax_every = 600

    while len(points) < count:
        candidate = np.asarray(
            [
                rng.uniform(*x_range),
                rng.uniform(*y_range),
                rng.uniform(*h_range),
            ],
            dtype=float,
        )

        valid = True
        if points:
            existing = np.vstack(points)
            if np.any(np.linalg.norm(existing[:, :2] - candidate[:2], axis=1) < cur_min_xy):
                valid = False
        if valid and avoid_xyz is not None and avoid_xyz.size > 0 and cur_min_avoid > 0.0:
            if np.any(np.linalg.norm(avoid_xyz[:, :2] - candidate[:2], axis=1) < cur_min_avoid):
                valid = False

        if valid:
            points.append(candidate)
            attempts = 0
            continue

        attempts += 1
        if attempts >= relax_every:
            cur_min_xy *= 0.92
            cur_min_avoid *= 0.92
            attempts = 0

    return np.vstack(points)


def _random_assignment_balanced(n_p: int, n_e: int, rng: np.random.Generator) -> np.ndarray:
    if n_e <= 1:
        return np.zeros(n_p, dtype=int)
    if n_p >= n_e:
        labels = list(range(n_e))
        labels.extend(rng.integers(0, n_e, size=n_p - n_e).tolist())
        assignment = np.asarray(labels, dtype=int)
        rng.shuffle(assignment)
        return assignment
    return rng.integers(0, n_e, size=n_p, dtype=int)


def _assignment_template(n_p: int, n_e: int, mode: AssignmentMode, rng: np.random.Generator) -> np.ndarray:
    if n_e == 1 or mode == "zero":
        return np.zeros(n_p, dtype=int)
    if mode == "cyclic":
        return np.asarray([j % n_e for j in range(n_p)], dtype=int)
    if mode == "shifted":
        return np.asarray([(j + 1) % n_e for j in range(n_p)], dtype=int)
    if mode == "random":
        return _random_assignment_balanced(n_p, n_e, rng)
    raise ValueError(f"assignment template not available for mode: {mode}")


def _initial_assignment(
    pursuers: np.ndarray,
    evaders: np.ndarray,
    n_p: int,
    n_e: int,
    mode: AssignmentMode,
) -> np.ndarray:
    if n_e == 1 or mode == "zero":
        return np.zeros(n_p, dtype=int)
    if mode == "cyclic":
        return np.asarray([j % n_e for j in range(n_p)], dtype=int)
    if mode == "shifted":
        return np.asarray([(j + 1) % n_e for j in range(n_p)], dtype=int)
    if mode == "nearest":
        assignment = np.zeros(n_p, dtype=int)
        for j in range(n_p):
            d = np.linalg.norm(evaders[:, :3] - pursuers[j, :3], axis=1)
            assignment[j] = int(np.argmin(d))
        return assignment
    if mode == "random":
        return _random_assignment_balanced(n_p, n_e, np.random.default_rng(0))
    raise ValueError(f"unsupported assignment mode: {mode}")


def _random_evaders(n_e: int, rng: np.random.Generator) -> np.ndarray:
    evaders = np.zeros((n_e, 6), dtype=float)
    pos = _sample_separated_positions(
        n_e,
        rng,
        x_range=(-4200.0, 4200.0),
        y_range=(2600.0, 5200.0),
        h_range=(150.0, 520.0),
        min_xy=1800.0,
    )
    evaders[:, :3] = pos
    evaders[:, 3] = rng.uniform(35.0, 55.0, size=n_e)
    evaders[:, 4] = rng.uniform(52.0, 78.0, size=n_e)
    evaders[:, 5] = rng.uniform(-4.0, 4.0, size=n_e)
    return evaders


def _random_pursuers(n_p: int, rng: np.random.Generator, avoid_xyz: np.ndarray | None = None) -> np.ndarray:
    pursuers = np.zeros((n_p, 6), dtype=float)
    pos = _sample_separated_positions(
        n_p,
        rng,
        x_range=(-5200.0, 5200.0),
        y_range=(-6800.0, -1400.0),
        h_range=(160.0, 820.0),
        min_xy=1100.0,
        avoid_xyz=avoid_xyz,
        min_avoid_xy=3000.0,
    )
    pursuers[:, :3] = pos
    pursuers[:, 3] = rng.uniform(64.0, 88.0, size=n_p)
    pursuers[:, 4] = rng.uniform(8.0, 30.0, size=n_p)
    pursuers[:, 5] = rng.uniform(-8.0, 8.0, size=n_p)
    return pursuers


def _build_slot_layout(
    n_evaders: int,
    initial_assignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    slot_evaders: list[int] = []
    slot_local_indices: list[int] = []
    slot_displacements: list[np.ndarray] = []
    group_slot_ids: list[np.ndarray] = []

    cursor = 0
    for evader_idx in range(n_evaders):
        count = int(np.sum(initial_assignment == evader_idx))
        if count <= 0:
            raise ValueError("communication multi-team scenario requires each evader to have at least one pursuer")
        offsets = _formation_offsets(count)
        slot_ids = np.arange(cursor, cursor + count, dtype=int)
        group_slot_ids.append(slot_ids)
        for local_idx in range(count):
            slot_evaders.append(evader_idx)
            slot_local_indices.append(local_idx)
            slot_displacements.append(offsets[local_idx])
        cursor += count

    return (
        np.asarray(slot_evaders, dtype=int),
        np.asarray(slot_local_indices, dtype=int),
        np.asarray(slot_displacements, dtype=float),
        tuple(group_slot_ids),
    )


def sample_group_communication_windows(
    t_final: float,
    group_slot_ids: tuple[np.ndarray, ...],
    rng: np.random.Generator,
    windows_per_group: int,
    dropout_fraction: float,
    full_group_prob: float,
) -> GroupCommunicationParams:
    windows: list[GroupCommunicationWindow] = []
    n_windows = max(int(windows_per_group), 0)
    drop_frac = float(np.clip(dropout_fraction, 0.10, 1.00))
    full_prob = float(np.clip(full_group_prob, 0.0, 1.0))
    if n_windows == 0:
        return GroupCommunicationParams()

    base_centers = np.linspace(0.24, 0.70, n_windows)
    for group_idx, slot_ids in enumerate(group_slot_ids):
        group_size = int(slot_ids.shape[0])
        if group_size <= 1:
            continue
        for center in base_centers:
            duration = (0.10 + 0.06 * rng.random()) * t_final
            start = (center + rng.uniform(-0.045, 0.045)) * t_final
            start = float(np.clip(start, 0.08 * t_final, 0.86 * t_final))
            end = float(min(start + duration, 0.97 * t_final))

            if rng.random() < full_prob:
                isolated_count = group_size
            else:
                isolated_count = int(np.ceil(drop_frac * group_size))
                isolated_count = max(1, min(group_size - 1, isolated_count))
            isolated = tuple(sorted(rng.choice(group_size, size=isolated_count, replace=False).tolist()))
            windows.append(
                GroupCommunicationWindow(
                    group_idx=int(group_idx),
                    start_s=float(start),
                    end_s=float(end),
                    isolated_slots=isolated,
                )
            )

    windows.sort(key=lambda item: (item.group_idx, item.start_s, item.end_s))
    return tuple(windows)


def _build_group_communication(
    t_final: float,
    group_slot_ids: tuple[np.ndarray, ...],
    rng: np.random.Generator,
    windows_per_group: int,
    dropout_fraction: float,
    full_group_prob: float,
) -> GroupCommunicationParams:
    windows = sample_group_communication_windows(
        t_final=t_final,
        group_slot_ids=group_slot_ids,
        rng=rng,
        windows_per_group=windows_per_group,
        dropout_fraction=dropout_fraction,
        full_group_prob=full_group_prob,
    )
    return GroupCommunicationParams(
        windows=windows,
        windows_per_group=max(int(windows_per_group), 0),
        dropout_fraction=float(np.clip(dropout_fraction, 0.10, 1.00)),
        full_group_prob=float(np.clip(full_group_prob, 0.0, 1.0)),
        randomize_each_episode=True,
    )


def build_team_comm_multi_scenario(spec: GeneralTeamCommScenarioSpec) -> TeamCommMultiScenario:
    n_p = int(spec.n_pursuers)
    n_e = int(spec.n_evaders)
    if n_p <= 0 or n_e <= 0:
        raise ValueError("n_pursuers and n_evaders must be positive")
    if n_p < n_e:
        raise ValueError("communication multi-team mode expects n_pursuers >= n_evaders")

    rng = np.random.default_rng(spec.seed)
    if spec.layout_mode == "random":
        evaders = _random_evaders(n_e, rng)
    else:
        e_x = np.linspace(-2600.0, 2600.0, n_e) + 220.0 * np.sin(np.linspace(0.0, np.pi, n_e))
        e_y = 2400.0 + 420.0 * np.cos(np.linspace(0.0, np.pi, n_e))
        e_h = 220.0 + 120.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_e, endpoint=False))
        e_vx = 40.0 + 8.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_e, endpoint=False))
        e_vy = 60.0 + 6.0 * np.cos(np.linspace(0.0, np.pi, n_e))
        e_vh = 2.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_e, endpoint=False))
        evaders = np.column_stack((e_x, e_y, e_h, e_vx, e_vy, e_vh)).astype(float)

    assignment_mode = spec.assignment_mode if n_e > 1 else "zero"
    if assignment_mode == "nearest":
        template = _assignment_template(n_p, n_e, "shifted", rng)
    else:
        template = _assignment_template(n_p, n_e, assignment_mode, rng)

    if spec.layout_mode == "random":
        pursuers = _random_pursuers(n_p, rng, evaders[:, :3])
        assignment = _initial_assignment(pursuers, evaders, n_p, n_e, assignment_mode)
        counts = np.bincount(assignment, minlength=n_e)
        if np.any(counts == 0):
            assignment = template
    else:
        preferred_assignment = np.sort(template)
        pursuers = np.zeros((n_p, 6), dtype=float)
        for evader_idx in range(n_e):
            group = np.where(preferred_assignment == evader_idx)[0]
            if group.size == 0:
                continue
            angles = np.linspace(-0.7, 0.7, group.size)
            for local_idx, pursuer_idx in enumerate(group):
                ang = angles[local_idx]
                pursuers[pursuer_idx, 0] = evaders[evader_idx, 0] + 420.0 * np.sin(ang)
                pursuers[pursuer_idx, 1] = evaders[evader_idx, 1] - (3400.0 + 240.0 * local_idx)
                pursuers[pursuer_idx, 2] = evaders[evader_idx, 2] + 110.0 * np.cos(ang)
                pursuers[pursuer_idx, 3] = 72.0 + 6.0 * np.cos(ang + 0.3)
                pursuers[pursuer_idx, 4] = 18.0 + 5.0 * np.sin(ang + 0.5)
                pursuers[pursuer_idx, 5] = 6.0 * np.cos(ang)

        if assignment_mode == "nearest":
            assignment = _initial_assignment(pursuers, evaders, n_p, n_e, "nearest")
            counts = np.bincount(assignment, minlength=n_e)
            if np.any(counts == 0):
                assignment = template
        else:
            assignment = template

        pursuers += rng.normal(0.0, 15.0, size=pursuers.shape) * np.array([1.0, 1.0, 0.4, 0.05, 0.05, 0.05])
        evaders += rng.normal(0.0, 10.0, size=evaders.shape) * np.array([1.0, 1.0, 0.4, 0.05, 0.05, 0.05])

    slot_evaders, slot_local_indices, slot_displacements, group_slot_ids = _build_slot_layout(
        n_evaders=n_e,
        initial_assignment=assignment,
    )
    communication = _build_group_communication(
        t_final=float(spec.t_final),
        group_slot_ids=group_slot_ids,
        rng=rng,
        windows_per_group=spec.comm_windows_per_group,
        dropout_fraction=spec.comm_dropout_fraction,
        full_group_prob=spec.comm_full_group_prob,
    )

    return TeamCommMultiScenario(
        name=f"team_comm_{n_p}v{n_e}",
        pursuer_init=pursuers,
        evader_init=evaders,
        initial_assignment=assignment.astype(int),
        slot_evaders=slot_evaders,
        slot_local_indices=slot_local_indices,
        slot_displacements=slot_displacements,
        group_slot_ids=group_slot_ids,
        t_final=float(spec.t_final),
        capture_radius=float(spec.capture_radius),
        nu_eval=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
        communication=communication,
        swap_threshold=float(spec.swap_threshold if n_e > 1 else 1.0e9),
        max_switch_worsening=float(spec.max_switch_worsening),
        evader_motion_mode=spec.evader_motion_mode,
        evader_script_amp=spec.evader_script_amp,
        evader_script_omega=float(spec.evader_script_omega),
        evader_script_decay=float(spec.evader_script_decay),
        evader_script_mix=float(spec.evader_script_mix),
        swap_lookahead_time=float(spec.swap_lookahead_time),
    )
