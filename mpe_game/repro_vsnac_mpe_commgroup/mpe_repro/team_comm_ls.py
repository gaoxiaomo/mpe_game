from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class TeamLSSolveStats:
    sample_count: int
    residual_rms: float


class TeamReplayLeastSquares:
    """Recent-window Bellman-difference least-squares solver for the team critic."""

    def __init__(
        self,
        n_features: int,
        capacity: int,
        local_feature_count: int,
        local_bounds: Tuple[float, float],
        cross_bounds: Tuple[float, float],
        recent_weight_floor: float = 0.55,
    ) -> None:
        self.n_features = n_features
        self.local_feature_count = local_feature_count
        self.local_bounds = tuple(float(v) for v in local_bounds)
        self.cross_bounds = tuple(float(v) for v in cross_bounds)
        self.recent_weight_floor = float(np.clip(recent_weight_floor, 0.05, 1.0))
        self.a_buf: Deque[np.ndarray] = deque(maxlen=capacity)
        self.b_buf: Deque[float] = deque(maxlen=capacity)

    def add_sample(self, phi_t: np.ndarray, phi_tp1: np.ndarray, stage_cost: float, dt: float) -> None:
        a_row = phi_tp1 - phi_t
        b_val = -stage_cost * dt
        self.a_buf.append(np.asarray(a_row, dtype=float))
        self.b_buf.append(float(b_val))

    def solve(
        self,
        current_weights: np.ndarray,
        ridge_lambda: float,
        min_samples: int,
    ) -> tuple[np.ndarray, TeamLSSolveStats]:
        sample_count = len(self.a_buf)
        if sample_count < min_samples:
            return current_weights.copy(), TeamLSSolveStats(sample_count=sample_count, residual_rms=np.nan)

        a = np.vstack(self.a_buf)
        b = np.asarray(self.b_buf, dtype=float)

        col_scale = np.maximum(np.std(a, axis=0), 1e-6)
        a_scaled = a / col_scale[None, :]

        # Emphasize the most recent Bellman samples so the LS target tracks the
        # current policy instead of averaging over too much stale rollout data.
        weights = np.linspace(self.recent_weight_floor, 1.0, sample_count, dtype=float)
        sqrt_w = np.sqrt(weights)
        aw = a_scaled * sqrt_w[:, None]
        bw = b * sqrt_w

        lhs = aw.T @ aw + ridge_lambda * np.eye(self.n_features)
        rhs = aw.T @ bw + ridge_lambda * (current_weights * col_scale)
        w_scaled = np.linalg.solve(lhs, rhs)
        w = w_scaled / col_scale

        local_count = self.local_feature_count
        if local_count > 0:
            w[:local_count] = np.clip(w[:local_count], self.local_bounds[0], self.local_bounds[1])
        if local_count < self.n_features:
            w[local_count:] = np.clip(w[local_count:], self.cross_bounds[0], self.cross_bounds[1])

        residual = a @ w - b
        rms = float(np.sqrt(np.mean(residual * residual)))
        return w, TeamLSSolveStats(sample_count=sample_count, residual_rms=rms)
