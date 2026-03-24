from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlotGraphSnapshot:
    assignment: np.ndarray
    pursuer_slot: np.ndarray
    slot_owner: np.ndarray


class DynamicGroupSlotGraph:
    """Pairwise-swap target graph with fixed per-evader communication slots.

    Each evader owns a fixed number of ordered slots. Dynamic reassignment swaps
    pursuers between slots, which preserves each evader group's size while also
    preserving the local slot role used by the communication-aware team critic.
    """

    def __init__(
        self,
        initial_assignment: np.ndarray,
        slot_evaders: np.ndarray,
        group_slot_ids: tuple[np.ndarray, ...],
    ) -> None:
        self.initial_assignment = np.asarray(initial_assignment, dtype=int).copy()
        self.slot_evaders = np.asarray(slot_evaders, dtype=int).copy()
        self.group_slot_ids = tuple(np.asarray(ids, dtype=int).copy() for ids in group_slot_ids)

        self.n_pursuers = int(self.initial_assignment.shape[0])
        self.n_slots = int(self.slot_evaders.shape[0])
        self.n_evaders = int(len(self.group_slot_ids))
        if self.n_slots != self.n_pursuers:
            raise ValueError("slot count must equal pursuer count")

        self.slot_owner = np.full(self.n_slots, -1, dtype=int)
        self.pursuer_slot = np.full(self.n_pursuers, -1, dtype=int)

        for evader_idx, slot_ids in enumerate(self.group_slot_ids):
            members = np.where(self.initial_assignment == evader_idx)[0]
            if members.shape[0] != slot_ids.shape[0]:
                raise ValueError("initial assignment counts do not match slot layout")
            ordered_members = np.sort(members)
            for local_pos, slot_id in enumerate(slot_ids):
                owner = int(ordered_members[local_pos])
                self.slot_owner[int(slot_id)] = owner
                self.pursuer_slot[owner] = int(slot_id)

        if np.any(self.slot_owner < 0) or np.any(self.pursuer_slot < 0):
            raise ValueError("failed to initialize slot ownership")

    @property
    def assignment(self) -> np.ndarray:
        return self.slot_evaders[self.pursuer_slot].copy()

    def snapshot(self) -> SlotGraphSnapshot:
        return SlotGraphSnapshot(
            assignment=self.assignment,
            pursuer_slot=self.pursuer_slot.copy(),
            slot_owner=self.slot_owner.copy(),
        )

    def group_members(self, evader_idx: int) -> np.ndarray:
        slot_ids = self.group_slot_ids[int(evader_idx)]
        return self.slot_owner[slot_ids].copy()

    def group_slots(self, evader_idx: int) -> np.ndarray:
        return self.group_slot_ids[int(evader_idx)].copy()

    def assigned_slots(self) -> np.ndarray:
        return self.pursuer_slot.copy()

    def assigned_costs(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        slot_displacements: np.ndarray,
        nu: np.ndarray,
        slot_costs: np.ndarray | None = None,
    ) -> np.ndarray:
        costs = slot_costs
        if costs is None:
            costs = self.cost_matrix_to_slots(
                pursuer_states=pursuer_states,
                evader_states=evader_states,
                slot_displacements=slot_displacements,
                nu=nu,
            )
        return costs[np.arange(self.n_pursuers), self.pursuer_slot]

    def team_error(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        slot_displacements: np.ndarray,
        nu: np.ndarray,
        slot_costs: np.ndarray | None = None,
    ) -> float:
        return float(
            np.sum(
                self.assigned_costs(
                    pursuer_states=pursuer_states,
                    evader_states=evader_states,
                    slot_displacements=slot_displacements,
                    nu=nu,
                    slot_costs=slot_costs,
                )
            )
        )

    def cost_matrix_to_slots(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        slot_displacements: np.ndarray,
        nu: np.ndarray,
    ) -> np.ndarray:
        slot_evader_states = evader_states[self.slot_evaders]
        diff = pursuer_states[:, None, :] - slot_evader_states[None, :, :] + slot_displacements[None, :, :]
        return np.linalg.norm(diff * nu[None, None, :], axis=2)

    def _swap_slots(self, pursuer_a: int, pursuer_b: int) -> None:
        slot_a = int(self.pursuer_slot[pursuer_a])
        slot_b = int(self.pursuer_slot[pursuer_b])
        self.pursuer_slot[pursuer_a], self.pursuer_slot[pursuer_b] = slot_b, slot_a
        self.slot_owner[slot_a], self.slot_owner[slot_b] = pursuer_b, pursuer_a

    def update(
        self,
        pursuer_states: np.ndarray,
        evader_states: np.ndarray,
        slot_displacements: np.ndarray,
        switch_threshold: float,
        max_switch_worsening: float,
        nu: np.ndarray,
    ) -> None:
        del max_switch_worsening  # kept for interface compatibility
        if self.n_evaders <= 1:
            return

        costs = self.cost_matrix_to_slots(
            pursuer_states=pursuer_states,
            evader_states=evader_states,
            slot_displacements=slot_displacements,
            nu=nu,
        )

        swapped = True
        while swapped:
            swapped = False
            for pursuer_a in range(self.n_pursuers):
                slot_a = int(self.pursuer_slot[pursuer_a])
                evader_a = int(self.slot_evaders[slot_a])
                for pursuer_b in range(pursuer_a + 1, self.n_pursuers):
                    slot_b = int(self.pursuer_slot[pursuer_b])
                    evader_b = int(self.slot_evaders[slot_b])
                    if evader_a == evader_b:
                        continue

                    old_sum = float(costs[pursuer_a, slot_a] + costs[pursuer_b, slot_b])
                    new_sum = float(costs[pursuer_a, slot_b] + costs[pursuer_b, slot_a])
                    if old_sum - new_sum > switch_threshold:
                        self._swap_slots(pursuer_a, pursuer_b)
                        swapped = True

