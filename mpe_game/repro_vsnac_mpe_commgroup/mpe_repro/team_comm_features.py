from __future__ import annotations

from itertools import combinations

import numpy as np

from .team_comm_config import TeamFeatureParams


class TeamCommunicationFeatureMap:
    """Feature map for the stacked n-pursuer/1-evader team state."""

    def __init__(
        self,
        params: TeamFeatureParams,
        n_pursuers: int = 3,
        block_dim: int = 6,
        formation_offsets: np.ndarray | None = None,
    ) -> None:
        self.n_pursuers = n_pursuers
        self.block_dim = block_dim
        self.state_dim = n_pursuers * block_dim
        self.local_feature_count = 6 * n_pursuers
        self.cross_feature_count = 3 * (n_pursuers * (n_pursuers - 1) // 2)
        self.local_gain = float(params.local_gain)
        self.cross_gain = float(params.cross_gain)
        self.state_scale = np.asarray(params.state_scale, dtype=float)
        if self.state_scale.shape != (block_dim,):
            raise ValueError("state_scale must match block_dim")
        self.pos_scale = np.maximum(self.state_scale[:3] / 260.0, 1.0)
        self.pairs = list(combinations(range(n_pursuers), 2))
        if self.pairs:
            self._pair_a = np.array([a for a, _ in self.pairs], dtype=int)
            self._pair_b = np.array([b for _, b in self.pairs], dtype=int)
        else:
            self._pair_a = np.empty(0, dtype=int)
            self._pair_b = np.empty(0, dtype=int)

        # Coordination features: one scalar ||delta_p_jk||^2 per pair
        self.formation_offsets = None
        self.coord_feature_count = 0
        if formation_offsets is not None and len(self.pairs) > 0:
            self.formation_offsets = np.asarray(formation_offsets, dtype=float)
            if self.formation_offsets.shape != (n_pursuers, block_dim):
                raise ValueError(
                    f"formation_offsets shape {self.formation_offsets.shape} "
                    f"does not match ({n_pursuers}, {block_dim})"
                )
            self.coord_feature_count = len(self.pairs)
            # Precompute position formation offset differences (r_j[:3] - r_k[:3])
            self._coord_d_jk = (
                self.formation_offsets[self._pair_a, :3]
                - self.formation_offsets[self._pair_b, :3]
            )

        self.n_features = self.local_feature_count + self.cross_feature_count + self.coord_feature_count

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
        pv = p * v
        vv = 0.5 * v * v
        local = (self.local_gain * vis[:, None]) * np.column_stack([pv, vv])
        parts = [local.ravel()]
        if self.cross_feature_count > 0:
            dp = p[self._pair_a] - p[self._pair_b]
            dv = v[self._pair_a] - v[self._pair_b]
            pair_vis = vis[self._pair_a] * vis[self._pair_b]
            cross = self.cross_gain * pair_vis[:, None] * dp * dv
            parts.append(cross.ravel())
        if self.coord_feature_count > 0:
            # Coordination features: ||delta_p_jk||^2 per pair
            # delta_p_jk = (p_j - p_k) - (r_j[:3] - r_k[:3]) / pos_scale
            # where p_j, p_k are already position / pos_scale from _split
            pair_vis = vis[self._pair_a] * vis[self._pair_b]
            delta_pos = (
                p[self._pair_a] - p[self._pair_b]
                - self._coord_d_jk / self.pos_scale
            )
            coord = self.cross_gain * pair_vis * np.sum(delta_pos * delta_pos, axis=1)
            parts.append(coord)
        return np.clip(np.concatenate(parts), -1.0e5, 1.0e5)

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

        if self.coord_feature_count > 0:
            for pair_idx, (a, b) in enumerate(self.pairs):
                base_a = a * self.block_dim
                base_b = b * self.block_dim
                delta_pos = (
                    p[a] - p[b]
                    - self._coord_d_jk[pair_idx] / self.pos_scale
                )
                scale = self.cross_gain * vis[a] * vis[b]
                # d/d(pos_a_c) of (scale * ||delta_pos||^2) = 2 * scale * delta_pos_c / pos_scale_c
                for c in range(3):
                    jac[row, base_a + c] = 2.0 * scale * delta_pos[c] / self.pos_scale[c]
                    jac[row, base_b + c] = -2.0 * scale * delta_pos[c] / self.pos_scale[c]
                row += 1

        return jac

    def gradient(self, team_state: np.ndarray, weights: np.ndarray, visibility_mask: np.ndarray | None = None) -> np.ndarray:
        p, v = self._split(team_state)
        vis = self._resolve_visibility(visibility_mask)
        w_local = weights[: self.local_feature_count].reshape(self.n_pursuers, 6)
        scale = self.local_gain * vis
        grad_pos = scale[:, None] * v / self.pos_scale[None, :] * w_local[:, :3]
        grad_vel = scale[:, None] * (p * w_local[:, :3] + v * w_local[:, 3:])
        grad_blocks = np.column_stack([grad_pos, grad_vel])
        if self.cross_feature_count > 0:
            cross_end = self.local_feature_count + self.cross_feature_count
            w_cross = weights[self.local_feature_count : cross_end].reshape(-1, 3)
            dp = p[self._pair_a] - p[self._pair_b]
            dv = v[self._pair_a] - v[self._pair_b]
            pair_vis = vis[self._pair_a] * vis[self._pair_b]
            sc = self.cross_gain * pair_vis
            pos_c = sc[:, None] * dv / self.pos_scale[None, :] * w_cross
            vel_c = sc[:, None] * dp * w_cross
            np.add.at(grad_blocks[:, :3], self._pair_a, pos_c)
            np.add.at(grad_blocks[:, 3:], self._pair_a, vel_c)
            np.add.at(grad_blocks[:, :3], self._pair_b, -pos_c)
            np.add.at(grad_blocks[:, 3:], self._pair_b, -vel_c)
        if self.coord_feature_count > 0:
            coord_start = self.local_feature_count + self.cross_feature_count
            w_coord = weights[coord_start:]
            pair_vis = vis[self._pair_a] * vis[self._pair_b]
            delta_pos = (
                p[self._pair_a] - p[self._pair_b]
                - self._coord_d_jk / self.pos_scale
            )
            sc = self.cross_gain * pair_vis
            # d/d(pos_j_c) = 2 * sc * w_coord * delta_pos_c / pos_scale_c
            coord_grad_pos = 2.0 * (sc * w_coord)[:, None] * delta_pos / self.pos_scale[None, :]
            np.add.at(grad_blocks[:, :3], self._pair_a, coord_grad_pos)
            np.add.at(grad_blocks[:, :3], self._pair_b, -coord_grad_pos)
        return grad_blocks.ravel()

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
