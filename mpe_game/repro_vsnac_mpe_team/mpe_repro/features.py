from __future__ import annotations

import numpy as np

from .config import FeatureParams


class SigmoidFeatureMap:
    """Paper-aligned 6-term coupled basis used by each individual critic."""

    def __init__(self, params: FeatureParams, state_dim: int = 6) -> None:
        self.n_features = params.n_features
        self.state_scale = np.asarray(params.state_scale, dtype=float)
        self.gain = float(params.feature_gain)
        if self.state_scale.shape != (state_dim,):
            raise ValueError("state_scale must match state dimension")
        if self.n_features != 6:
            raise ValueError("This implementation expects n_features=6")
        self.pos_scale = np.maximum(self.state_scale[:3] / 260.0, 1.0)

    def phi(self, x: np.ndarray) -> np.ndarray:
        p = x[:3] / self.pos_scale
        v = x[3:]
        z = self.gain * np.array([
            p[0]*v[0], p[1]*v[1], p[2]*v[2],
            0.5*v[0]*v[0], 0.5*v[1]*v[1], 0.5*v[2]*v[2],
        ], dtype=float)
        return np.clip(z, -2.0e4, 2.0e4)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        p = x[:3] / self.pos_scale
        v = x[3:]
        j = np.zeros((6, 6), dtype=float)
        j[0, 0] = v[0] / self.pos_scale[0]; j[0, 3] = p[0]
        j[1, 1] = v[1] / self.pos_scale[1]; j[1, 4] = p[1]
        j[2, 2] = v[2] / self.pos_scale[2]; j[2, 5] = p[2]
        j[3, 3] = v[0]; j[4, 4] = v[1]; j[5, 5] = v[2]
        return self.gain * j


class CouplingFeatureMap:
    """9-term coupling basis: 6 state + 3 control coupling.

    phi_c(Δx, u_k) = [phi_s(Δx);  phi_u(Δv, u_k)]

    State coupling (6 terms) — inter-pursuer relative motion:
        gamma_s * [Δp₁Δv₁/σ₁, Δp₂Δv₂/σ₂, Δp₃Δv₃/σ₃, ½Δv₁², ½Δv₂², ½Δv₃²]

    Control coupling (3 terms) — teammate intent via Δv·u_k:
        gamma_u * [Δv₁·u_{k,1}, Δv₂·u_{k,2}, Δv₃·u_{k,3}]

    The Δv·u_k form is chosen so that the gradient w.r.t. x_j falls in
    velocity components (rows 3-5), which survive multiplication by g(x)^T
    and thus actually enter the control law.
    """

    def __init__(
        self,
        state_gain: float = 2.5,
        control_gain: float = 1.5,
        pos_scale: np.ndarray | None = None,
    ) -> None:
        self.n_features = 9
        self.state_gain = float(state_gain)
        self.control_gain = float(control_gain)
        self.pos_scale = np.asarray(pos_scale, dtype=float) if pos_scale is not None else np.array([500.0, 500.0, 300.0])

    # ---- single pair ----

    def phi_c(self, delta_x: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """9-dim coupling feature for one (j, k) pair."""
        dp = delta_x[:3] / self.pos_scale
        dv = delta_x[3:]
        gs, gu = self.state_gain, self.control_gain
        z = np.array([
            gs * dp[0] * dv[0], gs * dp[1] * dv[1], gs * dp[2] * dv[2],
            gs * 0.5 * dv[0] * dv[0], gs * 0.5 * dv[1] * dv[1], gs * 0.5 * dv[2] * dv[2],
            gu * dv[0] * u_k[0], gu * dv[1] * u_k[1], gu * dv[2] * u_k[2],
        ], dtype=float)
        return np.clip(z, -2.0e4, 2.0e4)

    def jac_c(self, delta_x: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """9×6 Jacobian of phi_c w.r.t. Δx (u_k treated as constant)."""
        dp = delta_x[:3] / self.pos_scale
        dv = delta_x[3:]
        gs, gu = self.state_gain, self.control_gain
        J = np.zeros((9, 6), dtype=float)
        for l in range(3):
            # state coupling rows 0-2:  d(dp_l * dv_l)/d(Δp_l) and d(...)/d(Δv_l)
            J[l, l] = gs * dv[l] / self.pos_scale[l]
            J[l, l + 3] = gs * dp[l]
            # state coupling rows 3-5:  d(½ dv_l²)/d(Δv_l)
            J[l + 3, l + 3] = gs * dv[l]
            # control coupling rows 6-8: d(dv_l * u_{k,l})/d(Δv_l)
            J[l + 6, l + 3] = gu * u_k[l]
        return J

    # ---- averaged over teammates ----

    def psi(
        self,
        pursuer_j: np.ndarray,
        teammates: np.ndarray,
        teammate_controls: np.ndarray,
        state_only: bool = False,
    ) -> np.ndarray:
        """Averaged 9-dim coupling feature for pursuer j."""
        K = teammates.shape[0]
        if K == 0:
            return np.zeros(self.n_features, dtype=float)
        agg = np.zeros(self.n_features, dtype=float)
        for idx in range(K):
            u_k = np.zeros(3, dtype=float) if state_only else teammate_controls[idx]
            agg += self.phi_c(pursuer_j - teammates[idx], u_k)
        return agg / K

    def jac_psi(
        self,
        pursuer_j: np.ndarray,
        teammates: np.ndarray,
        teammate_controls: np.ndarray,
        state_only: bool = False,
    ) -> np.ndarray:
        """9×6 Jacobian of psi w.r.t. pursuer_j (x_j^p)."""
        K = teammates.shape[0]
        if K == 0:
            return np.zeros((self.n_features, 6), dtype=float)
        agg = np.zeros((self.n_features, 6), dtype=float)
        for idx in range(K):
            u_k = np.zeros(3, dtype=float) if state_only else teammate_controls[idx]
            agg += self.jac_c(pursuer_j - teammates[idx], u_k)
        return agg / K
