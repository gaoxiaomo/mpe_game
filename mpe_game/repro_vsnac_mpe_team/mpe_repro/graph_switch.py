from __future__ import annotations

import numpy as np


class DynamicTargetGraph:
    """Algorithm 1: dynamic target assignment by pairwise swap."""

    def __init__(self, n_p: int, n_e: int, initial_assignment: np.ndarray) -> None:
        self.n_p = n_p
        self.n_e = n_e
        if initial_assignment.shape != (n_p,):
            raise ValueError("initial_assignment shape mismatch")
        self.assignment = initial_assignment.astype(int).copy()
        self.A_pe = np.zeros((n_p, n_e), dtype=float)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        self.A_pe.fill(0.0)
        for j, i in enumerate(self.assignment):
            self.A_pe[j, i] = 1.0

    def set_assignment(self, new_assignment: np.ndarray) -> None:
        if new_assignment.shape != (self.n_p,):
            raise ValueError("new_assignment shape mismatch")
        self.assignment = new_assignment.astype(int).copy()
        self._rebuild_matrix()

    def A_ep(self) -> np.ndarray:
        return self.A_pe.T.copy()

    def evader_indegree(self) -> np.ndarray:
        return np.sum(self.A_ep(), axis=1)

    def _weighted_distance(
        self,
        pursuer_state: np.ndarray,
        evader_state: np.ndarray,
        displacement: np.ndarray,
        nu: np.ndarray,
    ) -> float:
        diff = pursuer_state - evader_state + displacement
        return float(np.linalg.norm(nu * diff))

    def weighted_distance_matrix(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        displacements: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        diff = pursuer_states[:, None, :] - evader_states[None, :, :] + displacements
        return np.linalg.norm(diff * nu[None, None, :], axis=2)

    def update(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        displacements: np.ndarray,
        switch_threshold: float,
        max_switch_worsening: float,
        nu: np.ndarray,
    ) -> None:
        if self.n_e <= 1:
            return

        costs = self.weighted_distance_matrix(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            displacements=displacements,
            nu=nu,
        )

        swapped = True
        while swapped:
            swapped = False
            for j in range(self.n_p):
                for jp in range(j + 1, self.n_p):
                    i = self.assignment[j]
                    ip = self.assignment[jp]
                    if i == ip:
                        continue

                    g_ji = costs[j, i]
                    g_jpip = costs[jp, ip]
                    g_jpi = costs[jp, i]
                    g_jip = costs[j, ip]

                    old_sum = g_ji + g_jpip
                    new_sum = g_jpi + g_jip
                    # Paper Algorithm 1 swap condition: old_sum - new_sum > threshold.
                    # Keep the argument for compatibility, but do not gate by max worsening.
                    if old_sum - new_sum > switch_threshold:
                        self.assignment[j], self.assignment[jp] = ip, i
                        swapped = True
            if swapped:
                self._rebuild_matrix()

    def team_error(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        displacements: np.ndarray,
        nu: np.ndarray,
        pairwise_costs: np.ndarray | None = None,
    ) -> float:
        costs = pairwise_costs
        if costs is None:
            costs = self.weighted_distance_matrix(
                pursuer_states=pursuer_states,
                evader_states=evader_states,
                displacements=displacements,
                nu=nu,
            )
        return float(np.sum(costs[np.arange(self.n_p), self.assignment]))
