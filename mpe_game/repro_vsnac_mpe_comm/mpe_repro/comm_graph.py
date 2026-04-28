"""Pursuer communication graph: adjacency, Laplacian, dropout, delta_{jk}."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Dropout patterns
# ---------------------------------------------------------------------------
#
# Each pattern exposes ``apply(base_A, step_idx, dt, rng) -> A``.  The base
# adjacency is the intra-group all-to-all topology produced by
# ``CommunicationGraph.build_adjacency``; the pattern returns a possibly-
# modified adjacency representing the realised communication channel at the
# requested time index.
#
# ``IIDBernoulliDropout`` is the legacy independent per-edge per-step model.
# ``PersistentEdgeDropout`` removes a random subset of edges once and keeps
# them removed for the entire evaluation (permanent link failure).
# ``PeriodicOutageDropout`` performs duty-cycle outages of the whole graph
# (jamming windows or terrain shadowing).


class DropoutPattern:
    """Abstract dropout pattern. Subclasses must implement ``apply``."""

    name: str = "base"

    def apply(
        self,
        base_A: np.ndarray,
        step_idx: int,
        dt: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        raise NotImplementedError


class IIDBernoulliDropout(DropoutPattern):
    """Independent per-edge per-step Bernoulli dropout, symmetrised."""

    name = "iid_bernoulli"

    def __init__(self, prob: float) -> None:
        self.prob = float(np.clip(prob, 0.0, 1.0))

    def apply(
        self,
        base_A: np.ndarray,
        step_idx: int,
        dt: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        if self.prob <= 0.0 or rng is None:
            return base_A.copy()
        n = base_A.shape[0]
        mask = rng.random((n, n)) > self.prob
        mask = mask & mask.T
        np.fill_diagonal(mask, False)
        return base_A * mask.astype(float)


class PersistentEdgeDropout(DropoutPattern):
    """A fixed random subset of edges is removed for the entire evaluation.

    Edges are sampled from a Bernoulli(``prob``) once at first ``apply`` call
    and the resulting mask is reused at every subsequent step. Both endpoints
    of an edge are dropped together (symmetric).
    """

    name = "persistent_edge"

    def __init__(self, prob: float, seed: int = 0) -> None:
        self.prob = float(np.clip(prob, 0.0, 1.0))
        self._seed = int(seed)
        self._mask: np.ndarray | None = None

    def _build_mask(self, n: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        mask = rng.random((n, n)) > self.prob
        mask = mask & mask.T
        np.fill_diagonal(mask, False)
        return mask.astype(float)

    def apply(
        self,
        base_A: np.ndarray,
        step_idx: int,
        dt: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        if self.prob <= 0.0:
            return base_A.copy()
        n = base_A.shape[0]
        if self._mask is None or self._mask.shape[0] != n:
            self._mask = self._build_mask(n)
        return base_A * self._mask


class PeriodicOutageDropout(DropoutPattern):
    """Periodic full-graph outage: every ``period_s`` seconds, the entire
    graph is severed for the first ``off_fraction * period_s`` seconds, then
    fully restored. Models periodic terrain shadowing or scheduled jamming.
    """

    name = "periodic_outage"

    def __init__(self, period_s: float, off_fraction: float) -> None:
        self.period_s = float(max(period_s, 1e-3))
        self.off_fraction = float(np.clip(off_fraction, 0.0, 1.0))

    def apply(
        self,
        base_A: np.ndarray,
        step_idx: int,
        dt: float,
        rng: np.random.Generator | None,
    ) -> np.ndarray:
        if self.off_fraction <= 0.0:
            return base_A.copy()
        t = float(step_idx) * float(dt)
        phase = (t % self.period_s) / self.period_s
        if phase < self.off_fraction:
            return np.zeros_like(base_A)
        return base_A.copy()


def make_dropout_pattern(
    kind: str,
    prob: float = 0.0,
    *,
    seed: int = 0,
    period_s: float = 2.0,
    off_fraction: float = 0.25,
) -> DropoutPattern:
    """Factory that constructs a ``DropoutPattern`` from a string tag."""
    kind = (kind or "iid").lower()
    if kind in {"iid", "iid_bernoulli", "bernoulli"}:
        return IIDBernoulliDropout(prob=prob)
    if kind in {"persistent", "persistent_edge"}:
        return PersistentEdgeDropout(prob=prob, seed=seed)
    if kind in {"periodic", "periodic_outage"}:
        return PeriodicOutageDropout(period_s=period_s, off_fraction=off_fraction)
    raise ValueError(f"unknown dropout pattern kind: {kind!r}")


class CommunicationGraph:
    """Manages the inter-pursuer communication topology used in the
    communication-augmented V-SNAC error state."""

    def __init__(
        self,
        n_p: int,
        comm_mode: str = "full",
        formation_ref_dist: float = 500.0,
        d_safe: float = 0.0,
        degree_norm: str = "linear",
    ) -> None:
        self.n_p = n_p
        self.comm_mode = comm_mode
        self.formation_ref_dist = formation_ref_dist
        self.d_safe = d_safe
        self.degree_norm = degree_norm

    def build_adjacency(
        self,
        pursuer_states: np.ndarray | None = None,
        assignment: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build n_p x n_p adjacency matrix.

        When *assignment* is provided, only pursuers targeting the same
        evader are connected (intra-group communication).  This is
        physically meaningful: formation coordination applies within a
        pursuit group, not between groups chasing different targets.
        """
        n = self.n_p
        if self.comm_mode == "none":
            return np.zeros((n, n), dtype=float)
        if assignment is not None:
            A = np.zeros((n, n), dtype=float)
            for j in range(n):
                for k in range(j + 1, n):
                    if int(assignment[j]) == int(assignment[k]):
                        A[j, k] = 1.0
                        A[k, j] = 1.0
            return A
        # "full" without assignment: all-to-all
        return np.ones((n, n), dtype=float) - np.eye(n, dtype=float)

    def apply_dropout(self, A: np.ndarray, rng: np.random.Generator, prob: float) -> np.ndarray:
        """Randomly zero edges with given probability."""
        if prob <= 0.0:
            return A.copy()
        mask = rng.random((self.n_p, self.n_p)) > prob
        # Symmetrize: both (j,k) and (k,j) drop together
        mask = mask & mask.T
        np.fill_diagonal(mask, False)
        return A * mask.astype(float)

    def laplacian_and_degree(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (degree_vector, Laplacian matrix).

        Parameters
        ----------
        A : ndarray, shape (n_p, n_p)

        Returns
        -------
        d : ndarray, shape (n_p,) -- degree vector
        L : ndarray, shape (n_p, n_p) -- graph Laplacian
        """
        d = A.sum(axis=1)  # shape (n_p,)
        L = np.diag(d) - A
        return d, L

    def compute_delta_matrix(self, displacements: np.ndarray, assignment: np.ndarray) -> np.ndarray:
        """Compute Delta_{jk} = r_k - r_j where r_j = displacements[j, sigma(j)].

        Parameters
        ----------
        displacements : ndarray, shape (n_p, n_e, 6)
        assignment : ndarray, shape (n_p,)

        Returns
        -------
        delta : ndarray, shape (n_p, n_p, 6)
        """
        n_p = self.n_p
        # r_j = displacements[j, assignment[j], :]
        r = np.array([displacements[j, int(assignment[j])] for j in range(n_p)])  # (n_p, 6)
        # Tracking error is x̃_j = x_j^p - x_e + r_j, so x̃_j -> 0 implies
        # p_j -> p_e - r_j. For two pursuers tracking the same evader,
        # p_j - p_k -> r_k - r_j.
        delta = r[None, :, :] - r[:, None, :]  # (n_p, n_p, 6), delta[j, k] = r_k - r_j
        return delta

    def augmented_error(
        self,
        j: int,
        tracking_error: np.ndarray,
        pursuer_states: np.ndarray,
        A: np.ndarray,
        delta_matrix: np.ndarray,
        gamma: float,
        b_j: float = 1.0,
    ) -> np.ndarray:
        """Compute communication-augmented error state.

        x_tilde_j^c = b_j * tracking_error
                     + gamma * (1/d_j) * sum_k a_{jk}
                       * [(x_j - x_k) - Delta_{jk}]

        The formation contribution is degree-normalized so its scale
        is independent of team size.

        Parameters
        ----------
        j : int
            Pursuer index.
        tracking_error : ndarray, shape (6,)
            Basic tracking error x_j^p - x_e + r_j.
        pursuer_states : ndarray, shape (n_p, 6)
        A : ndarray, shape (n_p, n_p)
            Adjacency matrix.
        delta_matrix : ndarray, shape (n_p, n_p, 6)
            Desired inter-pursuer displacement induced by the tracking offsets.
        gamma : float
            Formation coupling weight.
        b_j : float
            Pinning gain (default 1.0).

        Returns
        -------
        x_tilde : ndarray, shape (6,)
        """
        x_tilde = b_j * tracking_error.copy()
        if gamma > 0.0:
            d_j = float(A[j].sum())
            if d_j > 0.0:
                # Adaptive coupling: activate formation only when near the
                # evader so that tracking dominates the approach phase.
                track_pos_norm = float(np.linalg.norm(tracking_error[:3]))
                activation = min(1.0, self.formation_ref_dist / max(track_pos_norm, 1.0))
                gamma_eff = gamma * activation
                form_sum = np.zeros(6, dtype=float)
                for k in range(self.n_p):
                    if k == j or A[j, k] <= 0.0:
                        continue
                    form_sum += A[j, k] * ((pursuer_states[j] - pursuer_states[k]) - delta_matrix[j, k])
                x_tilde += (gamma_eff / d_j) * form_sum
        return x_tilde

    def formation_error_norm(
        self,
        j: int,
        pursuer_states: np.ndarray,
        A: np.ndarray,
        delta_matrix: np.ndarray,
    ) -> float:
        """Position-only formation error norm for pursuer j."""
        d_j = float(A[j].sum())
        if d_j <= 0.0:
            return 0.0
        form = np.zeros(3, dtype=float)
        for k in range(self.n_p):
            if k == j or A[j, k] <= 0.0:
                continue
            dp = (pursuer_states[j, :3] - pursuer_states[k, :3]) - delta_matrix[j, k, :3]
            form += A[j, k] * dp
        return float(np.linalg.norm(form / d_j))
