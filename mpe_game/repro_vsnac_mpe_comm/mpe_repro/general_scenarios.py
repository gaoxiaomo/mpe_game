from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import ScenarioConfig


AssignmentMode = Literal["zero", "cyclic", "shifted", "nearest", "random"]
LayoutMode = Literal["structured", "random"]


@dataclass(frozen=True)
class GeneralScenarioSpec:
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


def _formation_offsets(n_p: int, xy_radius: float = 180.0, h_radius: float = 60.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_p, endpoint=False)
    offsets = np.zeros((n_p, 6), dtype=float)
    offsets[:, 0] = xy_radius * np.cos(angles)
    offsets[:, 1] = xy_radius * np.sin(angles)
    offsets[:, 2] = h_radius * np.sin(2.0 * angles)
    return offsets


def _build_displacement_matrix(n_p: int, n_e: int) -> np.ndarray:
    offsets = _formation_offsets(n_p)
    mat = np.zeros((n_p, n_e, 6), dtype=float)
    for j in range(n_p):
        mat[j, :, :] = offsets[j]
    return mat


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
    current_min_xy = float(min_xy)
    current_min_avoid_xy = float(min_avoid_xy)
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

        ok = True
        if points:
            existing = np.vstack(points)
            if np.any(np.linalg.norm(existing[:, :2] - candidate[:2], axis=1) < current_min_xy):
                ok = False
        if ok and avoid_xyz is not None and avoid_xyz.size > 0 and current_min_avoid_xy > 0.0:
            if np.any(np.linalg.norm(avoid_xyz[:, :2] - candidate[:2], axis=1) < current_min_avoid_xy):
                ok = False
        if ok:
            points.append(candidate)
            attempts = 0
            continue

        attempts += 1
        if attempts >= relax_every:
            current_min_xy *= 0.92
            current_min_avoid_xy *= 0.92
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


def _random_pursuers_many_to_one(
    n_p: int, rng: np.random.Generator, avoid_xyz: np.ndarray | None = None
) -> np.ndarray:
    pursuers = np.zeros((n_p, 6), dtype=float)
    pos = _sample_separated_positions(
        n_p,
        rng,
        x_range=(-4600.0, 4600.0),
        y_range=(-6200.0, -2200.0),
        h_range=(180.0, 760.0),
        min_xy=1200.0,
        avoid_xyz=avoid_xyz,
        min_avoid_xy=3200.0,
    )
    pursuers[:, :3] = pos
    pursuers[:, 3] = rng.uniform(62.0, 88.0, size=n_p)
    pursuers[:, 4] = rng.uniform(8.0, 28.0, size=n_p)
    pursuers[:, 5] = rng.uniform(-9.0, 9.0, size=n_p)
    return pursuers


def _random_pursuers_many_to_many(
    n_p: int, rng: np.random.Generator, avoid_xyz: np.ndarray | None = None
) -> np.ndarray:
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


def build_general_scenario(spec: GeneralScenarioSpec) -> ScenarioConfig:
    n_p = int(spec.n_pursuers)
    n_e = int(spec.n_evaders)
    if n_p <= 0 or n_e <= 0:
        raise ValueError("n_pursuers and n_evaders must be positive")

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
    if n_e == 1:
        if spec.layout_mode == "random":
            pursuers = _random_pursuers_many_to_one(n_p, rng, evaders[:, :3])
        else:
            p_x = np.linspace(-2600.0, 2600.0, n_p)
            p_y = -3200.0 - 450.0 * np.cos(np.linspace(0.0, np.pi, n_p))
            p_h = 420.0 + 220.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_p, endpoint=False))
            p_vx = 70.0 + 10.0 * np.cos(np.linspace(0.0, 2.0 * np.pi, n_p, endpoint=False))
            p_vy = 20.0 + 6.0 * np.sin(np.linspace(0.0, np.pi, n_p))
            p_vh = 8.0 * np.cos(np.linspace(0.0, 2.0 * np.pi, n_p, endpoint=False))
            pursuers = np.column_stack((p_x, p_y, p_h, p_vx, p_vy, p_vh)).astype(float)
        assignment = np.zeros(n_p, dtype=int)
    else:
        if assignment_mode == "nearest":
            template = _assignment_template(n_p, n_e, "shifted", rng)
        else:
            template = _assignment_template(n_p, n_e, assignment_mode, rng)

        if spec.layout_mode == "random":
            pursuers = _random_pursuers_many_to_many(n_p, rng, evaders[:, :3])
        else:
            # Pairwise-swap graph switching preserves each evader's indegree.
            # We therefore generate a preferred assignment with the same counts as the
            # initial template, but with pursuers spatially arranged near these
            # preferred targets so that switching has a meaningful benefit.
            preferred_assignment = np.sort(template)
            pursuers = np.zeros((n_p, 6), dtype=float)
            for i in range(n_e):
                group = np.where(preferred_assignment == i)[0]
                if group.size == 0:
                    continue
                angles = np.linspace(-0.7, 0.7, group.size)
                for local_idx, j in enumerate(group):
                    ang = angles[local_idx]
                    pursuers[j, 0] = evaders[i, 0] + 420.0 * np.sin(ang)
                    pursuers[j, 1] = evaders[i, 1] - (3400.0 + 240.0 * local_idx)
                    pursuers[j, 2] = evaders[i, 2] + 110.0 * np.cos(ang)
                    pursuers[j, 3] = 72.0 + 6.0 * np.cos(ang + 0.3)
                    pursuers[j, 4] = 18.0 + 5.0 * np.sin(ang + 0.5)
                    pursuers[j, 5] = 6.0 * np.cos(ang)

        if assignment_mode == "nearest":
            assignment = _initial_assignment(pursuers, evaders, n_p, n_e, "nearest")
        else:
            assignment = template

    # Small deterministic perturbation avoids symmetric dead-zones for larger teams.
    if spec.layout_mode == "structured":
        pursuers += rng.normal(0.0, 15.0, size=pursuers.shape) * np.array([1.0, 1.0, 0.4, 0.05, 0.05, 0.05])
        evaders += rng.normal(0.0, 10.0, size=evaders.shape) * np.array([1.0, 1.0, 0.4, 0.05, 0.05, 0.05])
    displacements = _build_displacement_matrix(n_p, n_e)

    return ScenarioConfig(
        name=f"scenario_{n_p}v{n_e}_generalized",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        swap_threshold=float(spec.swap_threshold if n_e > 1 else 1.0e9),
        max_switch_worsening=float(spec.max_switch_worsening),
        initial_assignment=assignment,
        t_final=float(spec.t_final),
        capture_radius=float(spec.capture_radius),
        evader_motion_mode=spec.evader_motion_mode,
        evader_script_amp=spec.evader_script_amp,
        evader_script_omega=float(spec.evader_script_omega),
        evader_script_decay=float(spec.evader_script_decay),
        evader_script_mix=float(spec.evader_script_mix),
        swap_lookahead_time=float(spec.swap_lookahead_time),
    )
