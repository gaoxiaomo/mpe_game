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

        swapped = True
        while swapped:
            swapped = False
            for j in range(self.n_p):
                for jp in range(j + 1, self.n_p):
                    i = self.assignment[j]
                    ip = self.assignment[jp]
                    if i == ip:
                        continue

                    g_ji = self._weighted_distance(
                        pursuer_states[j], evader_states[i], displacements[j, i], nu
                    )
                    g_jpip = self._weighted_distance(
                        pursuer_states[jp], evader_states[ip], displacements[jp, ip], nu
                    )
                    g_jpi = self._weighted_distance(
                        pursuer_states[jp], evader_states[i], displacements[jp, i], nu
                    )
                    g_jip = self._weighted_distance(
                        pursuer_states[j], evader_states[ip], displacements[j, ip], nu
                    )

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
    ) -> float:
        total = 0.0
        for j in range(self.n_p):
            for i in range(self.n_e):
                if self.A_pe[j, i] <= 0.0:
                    continue
                diff = pursuer_states[j] - evader_states[i] + displacements[j, i]
                total += float(np.linalg.norm(nu * diff))
        return total
