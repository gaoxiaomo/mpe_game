from __future__ import annotations

from itertools import combinations

import numpy as np

from .team_comm_config import TeamFeatureParams


class TeamCommunicationFeatureMap:
    """Feature map for the stacked n-pursuer/1-evader team state."""

    def __init__(self, params: TeamFeatureParams, n_pursuers: int = 3, block_dim: int = 6) -> None:
        self.n_pursuers = n_pursuers
        self.block_dim = block_dim
        self.state_dim = n_pursuers * block_dim
        self.local_feature_count = 6 * n_pursuers
        self.cross_feature_count = 3 * (n_pursuers * (n_pursuers - 1) // 2)
        self.n_features = self.local_feature_count + self.cross_feature_count
        self.local_gain = float(params.local_gain)
        self.cross_gain = float(params.cross_gain)
        self.state_scale = np.asarray(params.state_scale, dtype=float)
        if self.state_scale.shape != (block_dim,):
            raise ValueError("state_scale must match block_dim")
        self.pos_scale = np.maximum(self.state_scale[:3] / 260.0, 1.0)
        self.pairs = list(combinations(range(n_pursuers), 2))

    def _split(self, team_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(team_state, dtype=float).reshape(self.n_pursuers, self.block_dim)
        p = x[:, :3] / self.pos_scale[None, :]
        v = x[:, 3:]
        return p, v

    def _resolve_visibility(self, visibility_mask: np.ndarray | None) -> np.ndarray:
        if visibility_mask is None:
            return np.ones(self.n_pursuers, dtype=float)
        vis = np.asarray(visibility_mask, dtype=float).reshape(self.n_pursuers)
        return np.clip(vis, 0.0, 1.0)

    def phi(self, team_state: np.ndarray, visibility_mask: np.ndarray | None = None) -> np.ndarray:
        p, v = self._split(team_state)
        vis = self._resolve_visibility(visibility_mask)
        feats: list[float] = []
        for j in range(self.n_pursuers):
            local = np.array(
                [
                    p[j, 0] * v[j, 0],
                    p[j, 1] * v[j, 1],
                    p[j, 2] * v[j, 2],
                    0.5 * v[j, 0] * v[j, 0],
                    0.5 * v[j, 1] * v[j, 1],
                    0.5 * v[j, 2] * v[j, 2],
                ],
                dtype=float,
            )
            feats.extend((self.local_gain * vis[j] * local).tolist())

        for a, b in self.pairs:
            delta_p = p[a] - p[b]
            delta_v = v[a] - v[b]
            pair_vis = vis[a] * vis[b]
            mixed = self.cross_gain * pair_vis * delta_p * delta_v
            feats.extend(mixed.tolist())

        out = np.asarray(feats, dtype=float)
        return np.clip(out, -1.0e5, 1.0e5)

    def jacobian(self, team_state: np.ndarray, visibility_mask: np.ndarray | None = None) -> np.ndarray:
        p, v = self._split(team_state)
        vis = self._resolve_visibility(visibility_mask)
        jac = np.zeros((self.n_features, self.state_dim), dtype=float)

        row = 0
        for j in range(self.n_pursuers):
            base = j * self.block_dim
            scale = self.local_gain * vis[j]
            jac[row + 0, base + 0] = scale * v[j, 0] / self.pos_scale[0]
            jac[row + 0, base + 3] = scale * p[j, 0]
            jac[row + 1, base + 1] = scale * v[j, 1] / self.pos_scale[1]
            jac[row + 1, base + 4] = scale * p[j, 1]
            jac[row + 2, base + 2] = scale * v[j, 2] / self.pos_scale[2]
            jac[row + 2, base + 5] = scale * p[j, 2]
            jac[row + 3, base + 3] = scale * v[j, 0]
            jac[row + 4, base + 4] = scale * v[j, 1]
            jac[row + 5, base + 5] = scale * v[j, 2]
            row += 6

        for a, b in self.pairs:
            base_a = a * self.block_dim
            base_b = b * self.block_dim
            delta_p = p[a] - p[b]
            delta_v = v[a] - v[b]
            scale = self.cross_gain * vis[a] * vis[b]
            for c in range(3):
                jac[row + c, base_a + c] = scale * delta_v[c] / self.pos_scale[c]
                jac[row + c, base_a + 3 + c] = scale * delta_p[c]
                jac[row + c, base_b + c] = -scale * delta_v[c] / self.pos_scale[c]
                jac[row + c, base_b + 3 + c] = -scale * delta_p[c]
            row += 3

        return jac

    def gradient(self, team_state: np.ndarray, weights: np.ndarray, visibility_mask: np.ndarray | None = None) -> np.ndarray:
        return self.jacobian(team_state, visibility_mask=visibility_mask).T @ weights

    def block_gradient(
        self,
        team_state: np.ndarray,
        weights: np.ndarray,
        block_idx: int,
        visibility_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        grad = self.gradient(team_state, weights, visibility_mask=visibility_mask)
        start = block_idx * self.block_dim
        end = start + self.block_dim
        return grad[start:end]
