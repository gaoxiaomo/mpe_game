from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class TeamFeatureParams:
    state_scale: Tuple[float, ...] = (2600.0, 2600.0, 2600.0, 150.0, 150.0, 150.0)
    local_gain: float = 2.8
    cross_gain: float = 1.2

    @property
    def n_features(self) -> int:
        return 27


@dataclass(frozen=True)
class CommunicationWindow:
    start_s: float
    end_s: float
    isolated_agents: Tuple[int, ...]


@dataclass(frozen=True)
class CommunicationParams:
    windows: Tuple[CommunicationWindow, ...] = ()


@dataclass(frozen=True)
class TeamCommScenario:
    name: str
    pursuer_init: np.ndarray
    evader_init: np.ndarray
    displacement_matrix: np.ndarray
    t_final: float
    capture_radius: float
    nu_eval: np.ndarray
    communication: CommunicationParams
    evader_script_amp: Tuple[float, float, float] = (10.0, 8.0, 4.0)
    evader_script_omega: float = 0.58
    evader_script_decay: float = 0.075
    evader_script_mix: float = 1.0

    @property
    def n_pursuers(self) -> int:
        return int(self.pursuer_init.shape[0])

    @property
    def n_evaders(self) -> int:
        return int(self.evader_init.shape[0])


def communication_three_pursuer_one_evader_scenario() -> TeamCommScenario:
    pursuers = np.array(
        [
            [600.0, -1900.0, 300.0, 78.0, 2.0, 18.0],
            [1200.0, 500.0, 2200.0, 2.0, 20.0, 15.0],
            [200.0, 600.0, 800.0, 5.0, 10.0, 125.0],
        ],
        dtype=float,
    )
    evaders = np.array([[500.0, 1500.0, 200.0, 50.0, 80.0, 10.0]], dtype=float)

    displacements = np.array(
        [
            [[50.0, 10.0, 0.0, 0.0, 0.0, 0.0]],
            [[10.0, 50.0, 0.0, 0.0, 0.0, 0.0]],
            [[-10.0, 0.0, -50.0, 0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )

    communication = CommunicationParams(
        windows=(
            CommunicationWindow(start_s=15.0, end_s=40.0, isolated_agents=(0, 1, 2)),
        )
    )

    return TeamCommScenario(
        name="team_comm_3v1",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        t_final=90.0,
        capture_radius=180.0,
        nu_eval=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
        communication=communication,
        evader_script_amp=(10.0, 8.0, 4.0),
        evader_script_omega=0.58,
        evader_script_decay=0.075,
        evader_script_mix=1.0,
    )


def _general_offsets(n_pursuers: int, xy_radius: float = 80.0, h_radius: float = 35.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_pursuers, endpoint=False)
    offsets = np.zeros((n_pursuers, 6), dtype=float)
    offsets[:, 0] = xy_radius * np.cos(angles)
    offsets[:, 1] = xy_radius * np.sin(angles)
    offsets[:, 2] = h_radius * np.sin(2.0 * angles)
    return offsets


def communication_many_pursuer_one_evader_scenario(
    n_pursuers: int,
    seed: int = 0,
    t_final: float = 95.0,
) -> TeamCommScenario:
    if n_pursuers <= 0:
        raise ValueError("n_pursuers must be positive")

    rng = np.random.default_rng(seed)

    evaders = np.array([[900.0, 2100.0, 220.0, 48.0, 78.0, 10.0]], dtype=float)
    displacements = _general_offsets(n_pursuers)[:, None, :]

    pursuers = np.zeros((n_pursuers, 6), dtype=float)
    angles = np.linspace(-0.9, 0.9, n_pursuers)
    for j, ang in enumerate(angles):
        pursuers[j, 0] = evaders[0, 0] + 520.0 * np.sin(ang)
        pursuers[j, 1] = evaders[0, 1] - (3200.0 + 220.0 * j)
        pursuers[j, 2] = evaders[0, 2] + 160.0 * np.cos(ang)
        pursuers[j, 3] = 72.0 + 7.0 * np.cos(ang + 0.3)
        pursuers[j, 4] = 18.0 + 5.0 * np.sin(ang + 0.5)
        pursuers[j, 5] = 8.0 * np.cos(ang)

    pursuers += rng.normal(0.0, 12.0, size=pursuers.shape) * np.array([1.0, 1.0, 0.4, 0.05, 0.05, 0.05])
    evaders += rng.normal(0.0, 8.0, size=evaders.shape) * np.array([1.0, 1.0, 0.4, 0.05, 0.05, 0.05])

    # Deterministic random communication blackout window for fair repeated
    # comparison. We randomize the start/duration, but isolate the whole
    # pursuer team so the effect is unambiguous: decentralized execution
    # must rely only on local information during the dropout interval.
    start_1 = 0.14 * t_final + rng.uniform(-0.015, 0.015) * t_final
    dur_1 = 0.14 * t_final + rng.uniform(0.02, 0.04) * t_final
    agents_1 = tuple(range(n_pursuers))
    communication = CommunicationParams(
        windows=(
            CommunicationWindow(start_s=float(start_1), end_s=float(start_1 + dur_1), isolated_agents=agents_1),
        )
    )

    return TeamCommScenario(
        name=f"team_comm_{n_pursuers}v1",
        pursuer_init=pursuers,
        evader_init=evaders,
        displacement_matrix=displacements,
        t_final=float(t_final),
        capture_radius=180.0,
        nu_eval=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
        communication=communication,
        evader_script_amp=(10.0, 8.0, 4.0),
        evader_script_omega=0.52,
        evader_script_decay=0.060,
        evader_script_mix=1.0,
    )

