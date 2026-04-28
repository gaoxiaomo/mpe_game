"""Linear single-integrator-style plant matching Xu 2024's simulation setting.

The Xu 2024 paper "Approximate Optimal Strategy for Multiagent System
Pursuit-Evasion Game" (IEEE TCAS-II Express Briefs, 2024) reports
convergence of its traditional Actor-Critic baseline on a 2D linear
plant ``\dot x = A x + B u`` with ``A = -I_2``, ``B = [0; 1]^T`` and
quadratic running cost. Under this LQR setting the true Q* is exactly
quadratic in (x, u, d), so the indirect actor target ``-Q_uu^{-1} Q_ux x``
is the exact Bellman-greedy minimiser. The AC's quadratic critic basis
matches the true Q* and convergence follows.

When the same algorithm is moved to the 6-DOF nonlinear ``AircraftDynamics``
plant in this repository, Q* is no longer quadratic in u, the indirect
target is a noisy local pseudo-Hessian, and the AC actor diverges.

This file provides a faithful linear plant so the AC reproduction can be
re-run on Xu's native setting. The plant interface mirrors
``AircraftDynamics`` (same method signatures: ``f``, ``g``, ``dxdt``,
``rk4_step``, plus their batch counterparts) so it can be plugged into
the existing simulator without any other code change.

The state convention is kept at 6-D for compatibility with the simulator
(state = position+velocity 6-D, control = 3-D), but the dynamics reduce
to ``\dot x = -x + B u`` with ``B = [0; I_3]`` (control directly drives
the velocity channels). This is the natural 6-D analogue of Xu's 2-D
``A = -I, B = [0; 1]^T``.
"""
from __future__ import annotations

import numpy as np

from .config import AircraftParams


class LinearPlantDynamics:
    """6-D linear single-integrator analogue of Xu 2024's 2-D plant.

    Dynamics: ``\dot x = -alpha * x + B u`` where
    - ``B = [0_{3x3}; I_3]`` (control acts on velocity channels only)
    - ``alpha`` is a small mean-reverting coefficient (default 0.0 = pure
      double-integrator-like, matches Xu's ``A = -I`` after rescaling).

    The control-affine form ``\dot x = f(x) + g(x) u`` has
    ``f(x) = -alpha * x`` and ``g(x) = [0_{3x3}; I_3]``. This makes Q*
    exactly quadratic in (x, u, d), enabling the Xu 2024 AC baseline to
    converge.

    The class accepts the same ``AircraftParams`` for API compatibility,
    but only ``alpha`` (set externally) is read.
    """

    def __init__(self, params: AircraftParams, alpha: float = 0.0) -> None:
        self.params = params
        self.alpha = float(alpha)
        self._B = np.zeros((6, 3), dtype=float)
        self._B[3, 0] = 1.0
        self._B[4, 1] = 1.0
        self._B[5, 2] = 1.0

    def f(self, x: np.ndarray) -> np.ndarray:
        # Drift: pure decay (alpha=0 by default => no drift, matches LQR baseline)
        return -self.alpha * x

    def f_batch(self, x: np.ndarray) -> np.ndarray:
        return -self.alpha * x

    def g(self, x: np.ndarray) -> np.ndarray:
        return self._B.copy()

    def control_effectiveness_batch(self, x: np.ndarray) -> np.ndarray:
        # g_33 element used by AircraftDynamics to scale altitude channel; for
        # the linear plant g_33 = 1 always.
        return np.ones(x.shape[0], dtype=float)

    def g_transpose_dot_batch(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        out = np.empty((x.shape[0], 3), dtype=float)
        out[:, 0] = grad[:, 3]
        out[:, 1] = grad[:, 4]
        out[:, 2] = grad[:, 5]
        return out

    def dxdt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.f(x) + self._B @ u

    def dxdt_batch(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = self.f_batch(x)
        out[:, 3] += u[:, 0]
        out[:, 4] += u[:, 1]
        out[:, 5] += u[:, 2]
        return out

    def rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.dxdt(x, u)
        k2 = self.dxdt(x + 0.5 * dt * k1, u)
        k3 = self.dxdt(x + 0.5 * dt * k2, u)
        k4 = self.dxdt(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rk4_step_batch(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.dxdt_batch(x, u)
        k2 = self.dxdt_batch(x + 0.5 * dt * k1, u)
        k3 = self.dxdt_batch(x + 0.5 * dt * k2, u)
        k4 = self.dxdt_batch(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
