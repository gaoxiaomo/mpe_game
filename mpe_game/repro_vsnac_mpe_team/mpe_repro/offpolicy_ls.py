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
    """Bellman difference least-squares buffer -- one buffer per critic."""

    def __init__(self, n_critics: int, n_features: int, capacity: int) -> None:
        self.n_critics = n_critics
        self.n_features = n_features
        self.a_buf: List[Deque[np.ndarray]] = [deque(maxlen=capacity) for _ in range(n_critics)]
        self.b_buf: List[Deque[float]] = [deque(maxlen=capacity) for _ in range(n_critics)]

    def add_sample(
        self, critic_idx: int, phi_t: np.ndarray, phi_tp1: np.ndarray, stage_cost: float, dt: float
    ) -> None:
        a_row = phi_tp1 - phi_t
        b_val = -stage_cost * dt
        self.a_buf[critic_idx].append(a_row.astype(float))
        self.b_buf[critic_idx].append(float(b_val))

    def solve(
        self,
        current_weights: List[np.ndarray],
        ridge_lambda: float,
        min_samples: int,
    ) -> Tuple[List[np.ndarray], List[LSSolveStats]]:
        new_weights: List[np.ndarray] = []
        stats: List[LSSolveStats] = []

        for c in range(self.n_critics):
            sample_count = len(self.a_buf[c])
            if sample_count < min_samples:
                new_weights.append(current_weights[c].copy())
                stats.append(LSSolveStats(sample_count=sample_count, residual_rms=np.nan))
                continue

            a = np.vstack(self.a_buf[c])
            b = np.asarray(self.b_buf[c], dtype=float)
            col_scale = np.maximum(np.std(a, axis=0), 1e-6)
            a_scaled = a / col_scale[None, :]

            lhs = a_scaled.T @ a_scaled + ridge_lambda * np.eye(self.n_features)
            rhs = a_scaled.T @ b + ridge_lambda * (current_weights[c] * col_scale)
            w_scaled = np.linalg.solve(lhs, rhs)
            w = w_scaled / col_scale
            w = np.clip(w, 0.08, 0.42)
            residual = a @ w - b
            rms = float(np.sqrt(np.mean(residual * residual)))

            new_weights.append(w)
            stats.append(LSSolveStats(sample_count=sample_count, residual_rms=rms))

        return new_weights, stats


class CouplingReplayLS:
    """Shared LS buffer for the coupling weight W_c (Stage 2).

    All pursuers contribute samples to one buffer.
    Bellman: (ψ_{t+1} - ψ_t)^T W_c = -γ S_j dt
    """

    def __init__(self, n_features: int, capacity: int) -> None:
        self.n_features = n_features
        self.a_buf: Deque[np.ndarray] = deque(maxlen=capacity)
        self.b_buf: Deque[float] = deque(maxlen=capacity)

    def clear(self) -> None:
        self.a_buf.clear()
        self.b_buf.clear()

    def add_sample(self, psi_t: np.ndarray, psi_tp1: np.ndarray, sep_penalty: float, gamma: float, dt: float) -> None:
        a_row = psi_tp1 - psi_t
        b_val = -gamma * sep_penalty * dt
        self.a_buf.append(a_row.astype(float))
        self.b_buf.append(float(b_val))

    def solve(
        self,
        current_w: np.ndarray,
        ridge_lambda: float,
        min_samples: int,
        w_min: float = -0.30,
        w_max: float = 0.30,
    ) -> Tuple[np.ndarray, LSSolveStats]:
        n = len(self.a_buf)
        if n < min_samples:
            return current_w.copy(), LSSolveStats(sample_count=n, residual_rms=np.nan)

        A = np.vstack(self.a_buf)
        b = np.asarray(self.b_buf, dtype=float)
        col_scale = np.maximum(np.std(A, axis=0), 1e-6)
        A_sc = A / col_scale[None, :]

        lhs = A_sc.T @ A_sc + ridge_lambda * np.eye(self.n_features)
        rhs = A_sc.T @ b + ridge_lambda * (current_w * col_scale)
        w_sc = np.linalg.solve(lhs, rhs)
        w = w_sc / col_scale
        w = np.clip(w, w_min, w_max)

        residual = A @ w - b
        rms = float(np.sqrt(np.mean(residual * residual)))
        return w, LSSolveStats(sample_count=n, residual_rms=rms)
