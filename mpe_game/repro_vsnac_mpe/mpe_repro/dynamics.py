from __future__ import annotations

import numpy as np

from .config import AircraftParams


class AircraftDynamics:
    """Nonlinear aircraft model from paper equations (53) and (54)."""

    def __init__(self, params: AircraftParams) -> None:
        self.params = params

    def _aero_terms(self, x: np.ndarray) -> tuple[float, float, float, float, float]:
        _, xb, h, va, vb, vh = x
        speed = float(np.sqrt(va * va + vb * vb + vh * vh))
        speed = max(speed, 1e-6)

        rho = self.params.c_rho * np.exp(-h / 24000.0)
        q_bar = 0.5 * rho * speed * speed
        alpha = float(np.arctan2(vh, va + 1e-9))
        beta = float(np.arcsin(np.clip(vb / speed, -1.0, 1.0)))

        c_t = self.params.c_beta * beta
        thrust = q_bar * self.params.S * c_t
        drag = q_bar * self.params.S * self.params.c_d0
        lift = q_bar * self.params.S * self.params.c_l0
        moment = q_bar * self.params.S * self.params.c_m0
        g33 = 1.0 + 0.25 * (np.sin(va * xb) ** 2)
        return thrust, drag, lift, moment, alpha, g33

    def f(self, x: np.ndarray) -> np.ndarray:
        thrust, drag, lift, moment, alpha, _ = self._aero_terms(x)
        va = x[3]
        vb = x[4]
        vh = x[5]
        out = np.zeros(6, dtype=float)
        out[0] = va
        out[1] = vb
        out[2] = vh
        out[3] = (thrust * np.cos(alpha) - drag) / self.params.m
        out[4] = (lift + thrust * np.sin(alpha)) / self.params.m
        out[5] = (moment + thrust * self.params.y_d) / self.params.m
        return out

    def g(self, x: np.ndarray) -> np.ndarray:
        _, _, _, _, _, g33 = self._aero_terms(x)
        out = np.zeros((6, 3), dtype=float)
        out[3, 0] = 1.0
        out[4, 1] = 1.0
        out[5, 2] = g33
        return out

    def dxdt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.f(x) + self.g(x) @ u

    def rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.dxdt(x, u)
        k2 = self.dxdt(x + 0.5 * dt * k1, u)
        k3 = self.dxdt(x + 0.5 * dt * k2, u)
        k4 = self.dxdt(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

