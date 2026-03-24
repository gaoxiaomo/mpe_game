from __future__ import annotations

import numpy as np

from .team_comm_config import TeamFeatureParams


class SummedStateFeatureMap:
    """Feature map on summed error state (original paper's approach).

    Instead of stacking individual errors (e_1, ..., e_n) into a 6n-dim state,
    sums them: tilde_x = sum_j e_j, giving a 6-dim state.

    Features: [p*v_a, p*v_b, p*v_h, v_a^2/2, v_b^2/2, v_h^2/2] (6 features)

    All pursuers share the SAME gradient -- no personalization, no communication needed.
    """

    def __init__(self, params: TeamFeatureParams) -> None:
        self.n_features = 6
        self.local_feature_count = 6
        self.cross_feature_count = 0
        self.local_gain = float(params.local_gain)
        self.state_scale = np.asarray(params.state_scale, dtype=float)
        if self.state_scale.shape != (6,):
            raise ValueError("state_scale must have 6 elements")
        self.pos_scale = np.maximum(self.state_scale[:3] / 260.0, 1.0)

    def phi(self, summed_state: np.ndarray) -> np.ndarray:
        """Compute features on the 6D summed error state.

        Parameters
        ----------
        summed_state : (6,) array
            tilde_x = sum_j e_j, the summed error across all pursuers in the group.

        Returns
        -------
        (6,) array of features.
        """
        x = np.asarray(summed_state, dtype=float).ravel()[:6]
        p = x[:3] / self.pos_scale
        v = x[3:]
        pv = p * v
        vv = 0.5 * v * v
        feats = self.local_gain * np.concatenate([pv, vv])
        return np.clip(feats, -1.0e5, 1.0e5)

    def gradient(self, summed_state: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Gradient of V = w^T phi with respect to summed_state.

        Parameters
        ----------
        summed_state : (6,) array
        weights : (6,) array

        Returns
        -------
        (6,) gradient dV/d(tilde_x).
        """
        x = np.asarray(summed_state, dtype=float).ravel()[:6]
        w = np.asarray(weights, dtype=float).ravel()[:6]
        p = x[:3] / self.pos_scale
        v = x[3:]
        w_pv = w[:3]
        w_vv = w[3:]

        # d(p*v * w_pv)/dx = [v*w_pv / pos_scale,  p*w_pv + v*w_vv]
        grad_pos = self.local_gain * v / self.pos_scale * w_pv
        grad_vel = self.local_gain * (p * w_pv + v * w_vv)
        return np.concatenate([grad_pos, grad_vel])

    def jacobian(self, summed_state: np.ndarray) -> np.ndarray:
        """Jacobian d(phi)/d(tilde_x), shape (6, 6)."""
        x = np.asarray(summed_state, dtype=float).ravel()[:6]
        p = x[:3] / self.pos_scale
        v = x[3:]
        jac = np.zeros((6, 6), dtype=float)

        # Features 0..2: local_gain * p_c * v_c
        # d/d(x_c) = local_gain * v_c / pos_scale_c   (position axes c=0,1,2)
        # d/d(x_{c+3}) = local_gain * p_c              (velocity axes)
        for c in range(3):
            jac[c, c] = self.local_gain * v[c] / self.pos_scale[c]
            jac[c, c + 3] = self.local_gain * p[c]

        # Features 3..5: local_gain * 0.5 * v_c^2
        # d/d(x_{c+3}) = local_gain * v_c
        for c in range(3):
            jac[c + 3, c + 3] = self.local_gain * v[c]

        return jac
