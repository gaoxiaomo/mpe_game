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
    """Recent-window Bellman-difference least-squares solver for the team critic.

    Uses truncated SVD to handle ill-conditioned systems that arise when
    cross-features couple multiple pursuers in the same group.
    """

    def __init__(
        self,
        n_features: int,
        capacity: int,
        local_feature_count: int,
        local_bounds: Tuple[float, float],
        cross_bounds: Tuple[float, float],
        recent_weight_floor: float = 0.55,
        cross_ridge_scale: float = 4.0,
        svd_threshold: float = 1e-4,
    ) -> None:
        self.n_features = n_features
        self.local_feature_count = local_feature_count
        self.local_bounds = tuple(float(v) for v in local_bounds)
        self.cross_bounds = tuple(float(v) for v in cross_bounds)
        self.recent_weight_floor = float(np.clip(recent_weight_floor, 0.05, 1.0))
        self.cross_ridge_scale = float(cross_ridge_scale)
        self.svd_threshold = float(svd_threshold)
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

        # Diagonal Tikhonov: stronger ridge on cross features to stabilise
        # the coupled system when group size > 2.
        ridge_diag = np.full(self.n_features, ridge_lambda, dtype=float)
        local_count = self.local_feature_count
        if local_count < self.n_features:
            ridge_diag[local_count:] = ridge_lambda * self.cross_ridge_scale

        lhs = aw.T @ aw + np.diag(ridge_diag)
        rhs = aw.T @ bw + ridge_diag * (current_weights * col_scale)

        # SVD-based solve: truncate near-zero singular values to avoid
        # amplifying noise in ill-conditioned directions.
        u, s, vt = np.linalg.svd(lhs, full_matrices=False)
        s_max = float(np.max(s))
        threshold = self.svd_threshold * s_max
        inv_s = np.where(s > threshold, 1.0 / s, 0.0)
        w_scaled = vt.T @ (inv_s * (u.T @ rhs))
        w = w_scaled / col_scale

        if local_count > 0:
            w[:local_count] = np.clip(w[:local_count], self.local_bounds[0], self.local_bounds[1])
        if local_count < self.n_features:
            w[local_count:] = np.clip(w[local_count:], self.cross_bounds[0], self.cross_bounds[1])

        residual = a @ w - b
        rms = float(np.sqrt(np.mean(residual * residual)))
        return w, TeamLSSolveStats(sample_count=sample_count, residual_rms=rms)
