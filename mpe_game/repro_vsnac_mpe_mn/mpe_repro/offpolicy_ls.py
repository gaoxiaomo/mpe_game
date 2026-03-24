from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class LSSolveStats:
    sample_count: int
    residual_rms: float


class ReplayLeastSquares:
    """Bellman difference least-squares buffer for each evader critic."""

    def __init__(self, n_evaders: int, n_features: int, capacity: int) -> None:
        self.n_evaders = n_evaders
        self.n_features = n_features
        self.a_buf: List[Deque[np.ndarray]] = [deque(maxlen=capacity) for _ in range(n_evaders)]
        self.b_buf: List[Deque[float]] = [deque(maxlen=capacity) for _ in range(n_evaders)]

    def add_sample(
        self, evader_idx: int, phi_t: np.ndarray, phi_tp1: np.ndarray, stage_cost: float, dt: float
    ) -> None:
        a_row = phi_tp1 - phi_t
        b_val = -stage_cost * dt
        self.a_buf[evader_idx].append(a_row.astype(float))
        self.b_buf[evader_idx].append(float(b_val))

    def solve(
        self,
        current_weights: List[np.ndarray],
        ridge_lambda: float,
        min_samples: int,
    ) -> Tuple[List[np.ndarray], List[LSSolveStats]]:
        new_weights: List[np.ndarray] = []
        stats: List[LSSolveStats] = []

        for i in range(self.n_evaders):
            sample_count = len(self.a_buf[i])
            if sample_count < min_samples:
                new_weights.append(current_weights[i].copy())
                stats.append(LSSolveStats(sample_count=sample_count, residual_rms=np.nan))
                continue

            a = np.vstack(self.a_buf[i])
            b = np.asarray(self.b_buf[i], dtype=float)
            col_scale = np.maximum(np.std(a, axis=0), 1e-6)
            a_scaled = a / col_scale[None, :]

            lhs = a_scaled.T @ a_scaled + ridge_lambda * np.eye(self.n_features)
            # Tikhonov prior centered at previous iterate to keep smooth evolution.
            rhs = a_scaled.T @ b + ridge_lambda * (current_weights[i] * col_scale)
            w_scaled = np.linalg.solve(lhs, rhs)
            w = w_scaled / col_scale
            # Keep weights in a paper-like range while still allowing movement.
            w = np.clip(w, 0.08, 0.42)
            residual = a @ w - b
            rms = float(np.sqrt(np.mean(residual * residual)))

            new_weights.append(w)
            stats.append(LSSolveStats(sample_count=sample_count, residual_rms=rms))

        return new_weights, stats
