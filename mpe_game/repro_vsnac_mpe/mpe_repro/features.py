from __future__ import annotations

import numpy as np

from .config import FeatureParams


class SigmoidFeatureMap:
    """Paper-aligned 6-term coupled basis used by each critic."""

    def __init__(self, params: FeatureParams, state_dim: int = 6) -> None:
        self.n_features = params.n_features
        self.state_scale = np.asarray(params.state_scale, dtype=float)
        self.gain = float(params.feature_gain)
        if self.state_scale.shape != (state_dim,):
            raise ValueError("state_scale must match state dimension")

        if self.n_features != 6:
            raise ValueError("This implementation expects n_features=6")

        # Position scaling for coupled basis (velocity kept unscaled to keep control-effective gradients).
        self.pos_scale = np.maximum(self.state_scale[:3] / 260.0, 1.0)

    def _preact(self, x: np.ndarray) -> np.ndarray:
        p = x[:3] / self.pos_scale
        v = x[3:]
        z = np.array(
            [
                p[0] * v[0],
                p[1] * v[1],
                p[2] * v[2],
                0.5 * v[0] * v[0],
                0.5 * v[1] * v[1],
                0.5 * v[2] * v[2],
            ],
            dtype=float,
        )
        z = self.gain * z
        return np.clip(z, -2.0e4, 2.0e4)

    def phi(self, x: np.ndarray) -> np.ndarray:
        return self._preact(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        p = x[:3] / self.pos_scale
        v = x[3:]
        j = np.zeros((6, 6), dtype=float)
        # d(p_i v_i)/dx_i and d(p_i v_i)/dv_i
        j[0, 0] = v[0] / self.pos_scale[0]
        j[0, 3] = p[0]
        j[1, 1] = v[1] / self.pos_scale[1]
        j[1, 4] = p[1]
        j[2, 2] = v[2] / self.pos_scale[2]
        j[2, 5] = p[2]
        # d(0.5 v_i^2)/dv_i
        j[3, 3] = v[0]
        j[4, 4] = v[1]
        j[5, 5] = v[2]
        return self.gain * j
